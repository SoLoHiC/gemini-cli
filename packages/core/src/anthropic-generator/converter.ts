/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type GenerateContentParameters,
  GenerateContentResponse,
  type Part,
  type Content,
  FinishReason,
  type Tool,
  type CallableTool,
} from '@google/genai';
import type Anthropic from '@anthropic-ai/sdk';
import { StreamingToolCallParser } from './streamingToolCallParser.js';

export class AnthropicContentConverter {
  private model: string;
  private streamingToolCallParser = new StreamingToolCallParser();

  constructor(model: string) {
    this.model = model;
  }

  resetStreamingToolCalls(): void {
    this.streamingToolCallParser.reset();
  }

  private extractSystemInstruction(
    request: GenerateContentParameters,
  ): string | undefined {
    if (!request.config?.systemInstruction) return undefined;

    // Handle different types of system instruction content
    const systemInstruction = request.config.systemInstruction;

    if (typeof systemInstruction === 'string') {
      return systemInstruction;
    }

    if ('text' in systemInstruction && systemInstruction.text) {
      return systemInstruction.text;
    }

    if (Array.isArray(systemInstruction)) {
      // Extract text from all text parts
      const textParts = systemInstruction
        .filter(
          (part): part is Part =>
            typeof part === 'object' && 'text' in part && !!part.text,
        )
        .map((part) => part.text);
      return textParts.join('');
    }

    return undefined;
  }

  private async extractTools(
    request: GenerateContentParameters,
  ): Promise<Anthropic.Messages.Tool[]> {
    const tools: Anthropic.Messages.Tool[] = [];

    if (!request.config?.tools) {
      return tools;
    }

    // Convert Gemini tools to Anthropic format
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
            let inputSchema: Anthropic.Messages.Tool.InputSchema | undefined;

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

    return tools;
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

  private extractMaxTokens(request: GenerateContentParameters): number {
    // Extract max output tokens from generation config
    if (
      request.config?.maxOutputTokens &&
      typeof request.config.maxOutputTokens === 'number'
    ) {
      return request.config.maxOutputTokens;
    }

    // Default value
    return 4096;
  }

  private extractSamplingParameters(
    request: GenerateContentParameters,
  ): Partial<Anthropic.Messages.MessageCreateParams> {
    const params: Partial<Anthropic.Messages.MessageCreateParams> = {};

    // Temperature
    if (
      request.config?.temperature !== undefined &&
      typeof request.config.temperature === 'number'
    ) {
      params.temperature = Math.max(
        0.0,
        Math.min(1.0, request.config.temperature),
      );
    }

    // Top P
    if (
      request.config?.topP !== undefined &&
      typeof request.config.topP === 'number'
    ) {
      params.top_p = Math.max(0.0, Math.min(1.0, request.config.topP));
    }

    // Top K
    if (
      request.config?.topK !== undefined &&
      typeof request.config.topK === 'number'
    ) {
      params.top_k = Math.max(1, request.config.topK);
    }

    // Stop sequences
    if (
      request.config?.stopSequences &&
      Array.isArray(request.config.stopSequences)
    ) {
      params.stop_sequences = request.config.stopSequences.filter(
        (seq): seq is string => typeof seq === 'string',
      );
    }

    return params;
  }

  async convertGeminiRequestToAnthropic(
    request: GenerateContentParameters,
  ): Promise<Anthropic.Messages.MessageCreateParams> {
    // Extract system instruction from generation config
    const system_prompt = this.extractSystemInstruction(request);

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

    // Extract tools from generation config
    const tools: Anthropic.Messages.Tool[] = await this.extractTools(request);
    const max_tokens = this.extractMaxTokens(request);

    const params: Anthropic.Messages.MessageCreateParams = {
      model: this.model,
      system: system_prompt,
      messages,
      max_tokens,
    };

    // Add sampling parameters
    const samplingParams = this.extractSamplingParameters(request);
    Object.assign(params, samplingParams);

    if (tools.length > 0) {
      params.tools = tools;
    }

    return params;
  }

  convertAnthropicResponseToGemini(
    anthropicResponse: Anthropic.Messages.Message,
  ): GenerateContentResponse {
    const response = new GenerateContentResponse();
    response.responseId = anthropicResponse.id;
    response.modelVersion = anthropicResponse.model;

    const parts: Part[] = [];
    for (const block of anthropicResponse.content) {
      if (block.type === 'text') {
        parts.push({ text: block.text });
      } else if (block.type === 'tool_use') {
        parts.push({
          functionCall: {
            name: block.name,
            args: block.input as { [key: string]: unknown },
            id: block.id,
          },
        });
      }
    }

    response.candidates = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason: this.mapAnthropicFinishReasonToGemini(
          anthropicResponse.stop_reason,
        ),
        index: 0,
        safetyRatings: [],
      },
    ];

    if (anthropicResponse.usage) {
      response.usageMetadata = {
        promptTokenCount: anthropicResponse.usage.input_tokens,
        candidatesTokenCount: anthropicResponse.usage.output_tokens,
        totalTokenCount:
          anthropicResponse.usage.input_tokens +
          anthropicResponse.usage.output_tokens,
      };
    }

    return response;
  }

  convertAnthropicChunkToGemini(
    chunk: Anthropic.Messages.MessageStreamEvent,
  ): GenerateContentResponse | null {
    const response = new GenerateContentResponse();
    const parts: Part[] = [];
    let finishReason: FinishReason | undefined = undefined;

    switch (chunk.type) {
      case 'content_block_delta':
        if (chunk.delta.type === 'text_delta') {
          parts.push({ text: chunk.delta.text });
        } else if (chunk.delta.type === 'input_json_delta') {
          const toolCallPart = this.streamingToolCallParser.parse(
            chunk as unknown as Record<string, unknown>,
          );
          if (toolCallPart) {
            parts.push(toolCallPart);
          }
        }
        break;

      case 'message_delta':
        finishReason = this.mapAnthropicFinishReasonToGemini(
          chunk.delta.stop_reason,
        );
        if (chunk.usage) {
          response.usageMetadata = {
            promptTokenCount: 0, // Not available in stream delta
            candidatesTokenCount: chunk.usage.output_tokens,
            totalTokenCount: 0, // Not available in stream delta
          };
        }
        break;

      case 'message_start':
        response.responseId = chunk.message.id;
        response.modelVersion = chunk.message.model;
        if (chunk.message.usage) {
          response.usageMetadata = {
            promptTokenCount: chunk.message.usage.input_tokens,
            candidatesTokenCount: chunk.message.usage.output_tokens,
            totalTokenCount:
              chunk.message.usage.input_tokens +
              chunk.message.usage.output_tokens,
          };
        }
        break;

      case 'content_block_start':
        if (chunk.content_block.type === 'tool_use') {
          const toolCallPart = this.streamingToolCallParser.parse(
            chunk as unknown as Record<string, unknown>,
          );
          if (toolCallPart) {
            parts.push(toolCallPart);
          }
        }
        break;

      case 'content_block_stop':
      case 'message_stop':
      default:
        // These events don't carry data we need to yield, so we can ignore them.
        return null;
    }

    if (parts.length === 0 && !finishReason) {
      return null;
    }

    response.candidates = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason,
        index: 0,
        safetyRatings: [],
      },
    ];

    return response;
  }

  private mapAnthropicFinishReasonToGemini(
    anthropicReason: Anthropic.Messages.Message['stop_reason'],
  ): FinishReason {
    if (!anthropicReason) return FinishReason.FINISH_REASON_UNSPECIFIED;
    switch (anthropicReason) {
      case 'end_turn':
        return FinishReason.STOP;
      case 'max_tokens':
        return FinishReason.MAX_TOKENS;
      case 'stop_sequence':
        return FinishReason.STOP;
      case 'tool_use':
        return FinishReason.OTHER; // No direct mapping for TOOL_USE, map to OTHER
      case 'refusal':
        return FinishReason.SAFETY;
      default:
        return FinishReason.OTHER;
    }
  }
}
