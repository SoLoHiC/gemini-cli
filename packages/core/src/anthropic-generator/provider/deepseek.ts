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

function sanitizeToolResultContent(
  value: unknown,
  context: string,
): unknown {
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

function sanitizeContentBlock(
  block: unknown,
  context: string,
): unknown {
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

  validateGeminiRequest(request: GenerateContentParameters): void {
    const contents = Array.isArray(request.contents)
      ? request.contents
      : [request.contents];

    for (const content of contents) {
      if (typeof content !== 'object' || content === null || !('parts' in content)) {
        continue;
      }

      for (const part of content.parts ?? []) {
        if (typeof part === 'object' && part !== null) {
          validateGeminiPart(part as Part);
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
    const baseRequest = super.buildRequest(request, userPromptId) as
      & Anthropic.Messages.MessageCreateParams
      & Record<string, unknown>;

    if (
      typeof baseRequest.model === 'string' &&
      !baseRequest.model.startsWith('deepseek-')
    ) {
      debugLogger.warn(
        `DeepSeek Anthropic endpoint is being used with non-DeepSeek model '${baseRequest.model}'. The request will be forwarded unchanged.`,
      );
    }

    if (baseRequest.temperature !== undefined) {
      baseRequest.temperature = Math.max(
        0.0,
        Math.min(2.0, baseRequest.temperature),
      );
    }

    delete baseRequest['top_k'];
    delete baseRequest['disable_parallel_tool_use'];

    if (Array.isArray(baseRequest.messages)) {
      baseRequest.messages = baseRequest.messages.map((message, messageIndex) => {
        const sanitizedMessage = {
          ...message,
        } as Anthropic.Messages.MessageParam & Record<string, unknown>;

        delete sanitizedMessage['cache_control'];

        if (Array.isArray(sanitizedMessage.content)) {
          sanitizedMessage.content = sanitizedMessage.content.map((block, blockIndex) =>
            sanitizeContentBlock(
              block,
              `messages[${messageIndex}].content[${blockIndex}]`,
            ),
          ) as Anthropic.Messages.ContentBlockParam[];
        }

        return sanitizedMessage;
      });
    }

    if (Array.isArray(baseRequest.tools)) {
      baseRequest.tools = baseRequest.tools.map((tool) => {
        const sanitizedTool = {
          ...tool,
        } as Anthropic.Messages.Tool & Record<string, unknown>;
        delete sanitizedTool['cache_control'];
        return sanitizedTool;
      });
    }

    return baseRequest;
  }
}
