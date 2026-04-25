/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import Anthropic from '@anthropic-ai/sdk';
import type { GenerateContentParameters, Part } from '@google/genai';
import type { Config } from '../../config/config.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import { debugLogger } from '../../utils/debugLogger.js';
import { DefaultAnthropicCompatibleProvider } from './default.js';

const UNSUPPORTED_CONTENT_TYPES = new Set([
  'image',
  'document',
  'search_result',
  'redacted_thinking',
  'server_tool_use',
  'web_search_tool_result',
  'code_execution_tool_result',
  'mcp_tool_use',
  'mcp_tool_result',
  'container_upload',
]);

function sanitizeToolResultContent(value: unknown, context: string): unknown {
  if (!Array.isArray(value)) {
    return value;
  }

  return value.map((item) => sanitizeContentBlock(item, context));
}

function validateGeminiPart(part: Part): void {
  if ('inlineData' in part && part.inlineData) {
    throw new Error(
      'DeepSeek Anthropic endpoint does not support image/document inlineData content.',
    );
  }

  if ('fileData' in part && part.fileData) {
    throw new Error(
      'DeepSeek Anthropic endpoint does not support image/document fileData content.',
    );
  }
}

function sanitizeContentBlock(block: unknown, context: string): unknown {
  if (typeof block !== 'object' || block === null || !('type' in block)) {
    return block;
  }

  const typedBlock = {
    ...(block as Record<string, unknown>),
  };
  const type = String(typedBlock['type']);

  if (UNSUPPORTED_CONTENT_TYPES.has(type)) {
    throw new Error(
      `DeepSeek Anthropic endpoint does not support content block type '${type}' in ${context}.`,
    );
  }

  delete typedBlock['cache_control'];
  delete typedBlock['is_error'];

  if (type === 'tool_result' && typedBlock['content'] !== undefined) {
    typedBlock['content'] = sanitizeToolResultContent(
      typedBlock['content'],
      `${context}.tool_result`,
    );
  }

  return typedBlock;
}

function isContentBlockParam(
  block: unknown,
): block is Anthropic.Messages.ContentBlockParam {
  return typeof block === 'object' && block !== null && 'type' in block;
}

function sanitizeContentBlockToContentBlock(
  block: unknown,
  context: string,
): Anthropic.Messages.ContentBlockParam {
  const result = sanitizeContentBlock(block, context);
  if (isContentBlockParam(result)) {
    return result;
  }
  // Fallback for unexpected types
  return { type: 'text', text: String(result) };
}

function toRecord(obj: unknown): Record<string, unknown> {
  if (typeof obj !== 'object' || obj === null) {
    return {};
  }
  return Object.fromEntries(Object.entries(obj));
}

function isThinkingDisabledInRequest(
  request: Anthropic.Messages.MessageCreateParams,
): boolean {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const thinking = (request as unknown as Record<string, unknown>)['thinking'];
  if (!thinking || typeof thinking !== 'object') {
    return false;
  }

  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  return (thinking as Record<string, unknown>)['type'] === 'disabled';
}

export class DeepSeekAnthropicCompatibleProvider extends DefaultAnthropicCompatibleProvider {
  constructor(
    contentGeneratorConfig: ContentGeneratorConfig,
    cliConfig: Config,
  ) {
    super(contentGeneratorConfig, cliConfig);
  }

  static isDeepSeekProvider(
    contentGeneratorConfig: ContentGeneratorConfig,
  ): boolean {
    const baseUrl = contentGeneratorConfig.baseUrl ?? '';

    return baseUrl.toLowerCase().includes('api.deepseek.com');
  }

  static isDeepSeekModel(
    contentGeneratorConfig: ContentGeneratorConfig,
  ): boolean {
    const model = contentGeneratorConfig.model ?? '';

    return model.toLowerCase().startsWith('deepseek-');
  }

  validateGeminiRequest(request: GenerateContentParameters): void {
    const contents = Array.isArray(request.contents)
      ? request.contents
      : [request.contents];

    for (const content of contents) {
      if (
        typeof content !== 'object' ||
        content === null ||
        !('parts' in content)
      ) {
        continue;
      }

      for (const part of content.parts ?? []) {
        if (typeof part === 'object' && part !== null) {
          validateGeminiPart(part);
        }
      }
    }
  }

  override buildHeaders(): Record<string, string | undefined> {
    const headers: Record<string, string | undefined> = {
      ...super.buildHeaders(),
      'x-api-key': this.contentGeneratorConfig.apiKey,
    };

    delete headers['anthropic-beta'];
    delete headers['anthropic-version'];

    return headers;
  }

  override buildClient(): Anthropic {
    const {
      baseUrl,
      timeout = 600000,
      maxRetries = 3,
    } = this.contentGeneratorConfig;

    return new Anthropic({
      apiKey: '',
      baseURL: baseUrl,
      timeout,
      maxRetries,
      defaultHeaders: this.buildHeaders(),
    });
  }

  override buildRequest(
    request: Anthropic.Messages.MessageCreateParams,
    userPromptId: string,
  ): Anthropic.Messages.MessageCreateParams {
    const baseRequest = super.buildRequest(request, userPromptId);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const extendedRequest = baseRequest as unknown as Record<string, unknown>;

    const model = extendedRequest['model'];
    if (typeof model === 'string' && !model.startsWith('deepseek-')) {
      debugLogger.warn(
        `DeepSeek Anthropic endpoint is being used with non-DeepSeek model '${model}'. The request will be forwarded unchanged.`,
      );
    }

    const temperature = extendedRequest['temperature'];
    if (typeof temperature === 'number') {
      extendedRequest['temperature'] = Math.max(
        0.0,
        Math.min(2.0, temperature),
      );
    }

    // Auto-inject output_config for DeepSeek V4 models (Anthropic format)
    const modelStr = typeof model === 'string' ? model : '';
    if (modelStr.toLowerCase().startsWith('deepseek-v4-')) {
      if (extendedRequest['thinking'] === undefined) {
        extendedRequest['thinking'] = { type: 'enabled' };
      }

      const isThinkingDisabled = isThinkingDisabledInRequest(
        baseRequest as Anthropic.Messages.MessageCreateParams,
      );
      if (
        !isThinkingDisabled &&
        extendedRequest['output_config'] === undefined
      ) {
        extendedRequest['output_config'] = { effort: 'high' };
      }
    }

    delete extendedRequest['top_k'];
    delete extendedRequest['disable_parallel_tool_use'];

    const messages = extendedRequest['messages'];
    if (Array.isArray(messages)) {
      const sanitizedMessages: unknown[] = messages.map(
        (message: unknown, messageIndex: number) => {
          const sanitizedMessage: Record<string, unknown> = {
            ...toRecord(message),
          };

          delete sanitizedMessage['cache_control'];

          const content = sanitizedMessage['content'];
          if (Array.isArray(content)) {
            sanitizedMessage['content'] = content.map((block, blockIndex) =>
              sanitizeContentBlockToContentBlock(
                block,
                `messages[${messageIndex}].content[${blockIndex}]`,
              ),
            );
          }

          return sanitizedMessage;
        },
      );
      extendedRequest['messages'] = sanitizedMessages;
    }

    const tools = extendedRequest['tools'];
    if (Array.isArray(tools)) {
      const sanitizedTools: unknown[] = tools.map((tool: unknown) => {
        const sanitizedTool: Record<string, unknown> = {
          ...toRecord(tool),
        };
        delete sanitizedTool['cache_control'];
        return sanitizedTool;
      });
      extendedRequest['tools'] = sanitizedTools;
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    return extendedRequest as unknown as Anthropic.Messages.MessageCreateParams;
  }
}
