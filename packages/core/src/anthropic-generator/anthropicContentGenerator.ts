/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ContentGenerator } from '../core/contentGenerator.js';
import type { Config } from '../config/config.js';
import type { AnthropicCompatibleProvider } from './provider/types.js';
import type {
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
  Content,
  Part,
  Tool,
  CallableTool,
} from '@google/genai';
import type { PipelineConfig } from './pipeline.js';
import { ContentGenerationPipeline } from './pipeline.js';
import { DefaultTelemetryService } from './telemetryService.js';
import { EnhancedErrorHandler } from './errorHandler.js';
import type { ContentGeneratorConfig } from '../core/contentGenerator.js';
import type Anthropic from '@anthropic-ai/sdk';

export class AnthropicContentGenerator implements ContentGenerator {
  protected pipeline: ContentGenerationPipeline;

  constructor(
    contentGeneratorConfig: ContentGeneratorConfig,
    cliConfig: Config,
    provider: AnthropicCompatibleProvider,
  ) {
    const pipelineConfig: PipelineConfig = {
      cliConfig,
      provider,
      contentGeneratorConfig,
      telemetryService: new DefaultTelemetryService(),
      errorHandler: new EnhancedErrorHandler(),
    };

    this.pipeline = new ContentGenerationPipeline(pipelineConfig);
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    return this.pipeline.execute(request, userPromptId);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    return this.pipeline.executeStream(request, userPromptId);
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Convert Gemini request to Anthropic format for token counting
    const anthropicRequest =
      await this.convertToAnthropicTokenCountRequest(request);

    // Use Anthropic's countTokens API for accurate token counting
    const tokenCount =
      await this.pipeline.client.messages.countTokens(anthropicRequest);

    return { totalTokens: tokenCount.input_tokens };
  }

  private async convertToAnthropicTokenCountRequest(
    request: CountTokensParameters,
  ): Promise<Anthropic.Messages.MessageCountTokensParams> {
    // For token counting, we need to convert the Gemini format to Anthropic format
    // This is similar to the conversion in the converter, but simplified for token counting

    const contents = Array.isArray(request.contents)
      ? request.contents
      : [request.contents];

    const messages: Anthropic.Messages.MessageParam[] = contents
      .filter(
        (content): content is Content =>
          typeof content === 'object' && 'role' in content,
      )
      .map((content: Content) => {
        const role = content.role === 'model' ? 'assistant' : 'user';
        const contentBlocks: Anthropic.Messages.ContentBlockParam[] = [];

        for (const part of content.parts ?? []) {
          if ('text' in part && part.text) {
            contentBlocks.push({ type: 'text', text: part.text });
          } else if ('functionCall' in part) {
            contentBlocks.push({
              type: 'tool_use',
              id: part.functionCall?.id || `tool_use_${Date.now()}`,
              name: part.functionCall?.name || '',
              input: part.functionCall?.args || {},
            });
          } else if ('functionResponse' in part) {
            contentBlocks.push({
              type: 'tool_result',
              tool_use_id: part.functionResponse?.id || '',
              content: JSON.stringify(part.functionResponse?.response),
            });
          }
        }
        return { role, content: contentBlocks };
      });

    const tokenCountRequest: Anthropic.Messages.MessageCountTokensParams = {
      model: this.pipeline.contentGeneratorConfig.model!,
      messages,
    };

    // Add system instruction if present
    if (request.config?.systemInstruction) {
      const systemInstruction = request.config.systemInstruction;
      if (typeof systemInstruction === 'string') {
        tokenCountRequest.system = systemInstruction;
      } else if ('text' in systemInstruction && systemInstruction.text) {
        tokenCountRequest.system = systemInstruction.text;
      } else if (Array.isArray(systemInstruction)) {
        const textParts = systemInstruction
          .filter(
            (part): part is Part =>
              typeof part === 'object' && 'text' in part && !!part.text,
          )
          .map((part) => part.text);
        tokenCountRequest.system = textParts.join('');
      }
    }

    // Add tools if present
    if (request.config?.tools) {
      // We need to extract tools similar to the converter
      const tools: Anthropic.Messages.Tool[] = [];
      for (const tool of request.config.tools) {
        let actualTool: Tool;

        // Handle CallableTool vs Tool
        if ('tool' in tool) {
          // This is a CallableTool
          actualTool = await (tool as CallableTool).tool();
        } else {
          // This is already a Tool
          actualTool = tool as Tool;
        }

        if (actualTool.functionDeclarations) {
          for (const func of actualTool.functionDeclarations) {
            if (func.name && func.description) {
              let inputSchema: Anthropic.Messages.Tool.InputSchema;

              // Handle both Gemini tools (parameters) and MCP tools (parametersJsonSchema)
              if (func.parametersJsonSchema) {
                // MCP tool format - use parametersJsonSchema directly
                inputSchema = {
                  type: 'object',
                  ...(func.parametersJsonSchema as Record<string, unknown>),
                };
              } else if (func.parameters) {
                // Gemini tool format - convert parameters to Anthropic format
                inputSchema = this.convertGeminiToolParametersToAnthropic(
                  func.parameters as Record<string, unknown>,
                );
              } else {
                // Default schema if no parameters specified
                inputSchema = {
                  type: 'object',
                  properties: {},
                  required: [],
                };
              }

              tools.push({
                name: func.name,
                description: func.description,
                input_schema: inputSchema,
              });
            }
          }
        }
      }

      if (tools.length > 0) {
        tokenCountRequest.tools = tools;
      }
    }

    return tokenCountRequest;
  }

  private convertGeminiToolParametersToAnthropic(
    geminiParameters: Record<string, unknown>,
  ): Anthropic.Messages.Tool.InputSchema {
    // Convert Gemini tool parameters to Anthropic JSON schema format
    const properties: Record<string, unknown> = {};
    const required: string[] = [];

    if (
      geminiParameters['properties'] &&
      typeof geminiParameters['properties'] === 'object'
    ) {
      for (const [key, prop] of Object.entries(
        geminiParameters['properties'] as Record<string, unknown>,
      )) {
        if (typeof prop === 'object' && prop !== null) {
          properties[key] = {
            ...(prop as Record<string, unknown>),
          };
        }
      }
    }

    if (Array.isArray(geminiParameters['required'])) {
      required.push(...geminiParameters['required'].map(String));
    }

    return {
      type: 'object',
      properties: Object.keys(properties).length > 0 ? properties : undefined,
      required: required.length > 0 ? required : undefined,
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // TODO: Implement embedding for Anthropic if available
    throw new Error(
      'Embedding not implemented for Anthropic-compatible models.',
    );
  }
}
