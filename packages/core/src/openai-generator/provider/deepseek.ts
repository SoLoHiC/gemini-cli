/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type OpenAI from 'openai';
import type { Config } from '../../config/config.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import { debugLogger } from '../../utils/debugLogger.js';
import { DefaultOpenAICompatibleProvider } from './default.js';
import { tokenLimit } from '../../core/tokenLimits.js';

export class DeepSeekOpenAICompatibleProvider extends DefaultOpenAICompatibleProvider {
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

  override buildRequest(
    request: OpenAI.Chat.ChatCompletionCreateParams,
    userPromptId: string,
  ): OpenAI.Chat.ChatCompletionCreateParams {
    const baseRequest = super.buildRequest(request, userPromptId);
    if (baseRequest.temperature === undefined) {
      baseRequest.temperature = 0;
    }
    const outputLimit = tokenLimit(
      baseRequest.model || this.contentGeneratorConfig.model || '',
      this.cliConfig,
      'output',
    );
    if (
      typeof baseRequest.max_tokens === 'number' &&
      baseRequest.max_tokens > outputLimit
    ) {
      debugLogger.warn(
        `DeepSeek max_tokens ${baseRequest.max_tokens} exceeds provider limit ${outputLimit}; clamping the request.`,
      );
      baseRequest.max_tokens = outputLimit;
    }
    if (!baseRequest.messages?.length) {
      return baseRequest;
    }

    const messages = baseRequest.messages.map((message) => {
      if (!('content' in message)) {
        return message;
      }

      const { content } = message;

      if (
        typeof content === 'string' ||
        content === null ||
        content === undefined
      ) {
        return message;
      }

      if (!Array.isArray(content)) {
        return message;
      }

      const text = content
        .map((part) => {
          if (part.type !== 'text') {
            throw new Error(
              `DeepSeek provider only supports text content. Found non-text part of type '${part.type}' in message with role '${message.role}'.`,
            );
          }

          return part.text ?? '';
        })
        .join('');

      return {
        ...message,
        content: text,
      } as OpenAI.Chat.ChatCompletionMessageParam;
    });

    return {
      ...baseRequest,
      messages,
    };
  }
}
