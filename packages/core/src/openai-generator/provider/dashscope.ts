/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import type { Config } from '../../config/config.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import { tokenLimit } from '../../core/tokenLimits.js';
import { DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES } from '../constants.js';
import type {
  OpenAICompatibleProvider,
  DashScopeRequestMetadata,
  ChatCompletionContentPartTextWithCache,
  ChatCompletionContentPartWithCache,
  ChatCompletionToolWithCache,
} from './types.js';

export class DashScopeOpenAICompatibleProvider
  implements OpenAICompatibleProvider
{
  private contentGeneratorConfig: ContentGeneratorConfig;
  private cliConfig: Config;

  constructor(
    contentGeneratorConfig: ContentGeneratorConfig,
    cliConfig: Config,
  ) {
    this.cliConfig = cliConfig;
    this.contentGeneratorConfig = contentGeneratorConfig;
  }

  static isDashScopeProvider(
    contentGeneratorConfig: ContentGeneratorConfig,
  ): boolean {
    const baseUrl = contentGeneratorConfig.baseUrl;
    return (
      baseUrl === 'https://dashscope.aliyuncs.com/compatible-mode/v1' ||
      baseUrl === 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    );
  }

  buildHeaders(): Record<string, string | undefined> {
    const version = 'unknown'; // this.cliConfig.getCliVersion() || 'unknown';
    const userAgent = `GeminiCLI/${version} (${process.platform}; ${process.arch})`;
    const { authType } = this.contentGeneratorConfig;
    return {
      'User-Agent': userAgent,
      'X-DashScope-CacheControl': 'enable',
      'X-DashScope-UserAgent': userAgent,
      'X-DashScope-AuthType': authType,
    };
  }

  buildClient(): OpenAI {
    const {
      apiKey,
      baseUrl,
      timeout = DEFAULT_TIMEOUT,
      maxRetries = DEFAULT_MAX_RETRIES,
    } = this.contentGeneratorConfig;
    const defaultHeaders = this.buildHeaders();
    return new OpenAI({
      apiKey,
      baseURL: baseUrl,
      timeout,
      maxRetries,
      defaultHeaders,
    });
  }

  buildRequest(
    request: OpenAI.Chat.ChatCompletionCreateParams,
    userPromptId: string,
  ): OpenAI.Chat.ChatCompletionCreateParams {
    let messages = request.messages;
    let tools = request.tools;

    if (!this.shouldDisableCacheControl()) {
      const { messages: updatedMessages, tools: updatedTools } =
        this.addDashScopeCacheControl(
          request,
          request.stream ? 'all' : 'system_only',
        );
      messages = updatedMessages;
      tools = updatedTools;
    }

    const requestWithTokenLimits = this.applyOutputTokenLimit(
      request,
      request.model,
    );

    if (this.isVisionModel(request.model)) {
      return {
        ...requestWithTokenLimits,
        messages,
        ...(tools ? { tools } : {}),
        ...(this.buildMetadata(userPromptId) || {}),
        /* @ts-expect-error dashscope exclusive */
        vl_high_resolution_images: true,
      } as OpenAI.Chat.ChatCompletionCreateParams;
    }

    return {
      ...requestWithTokenLimits,
      messages,
      ...(tools ? { tools } : {}),
      ...(this.buildMetadata(userPromptId) || {}),
    } as OpenAI.Chat.ChatCompletionCreateParams;
  }

  buildMetadata(userPromptId: string): DashScopeRequestMetadata {
    return {
      metadata: {
        sessionId: this.cliConfig.getSessionId?.(),
        promptId: userPromptId,
      },
    };
  }

  private addDashScopeCacheControl(
    request: OpenAI.Chat.ChatCompletionCreateParams,
    cacheControl: 'system_only' | 'all',
  ): {
    messages: OpenAI.Chat.ChatCompletionMessageParam[];
    tools?: ChatCompletionToolWithCache[];
  } {
    const messages = request.messages;

    const systemIndex = messages.findIndex(
      (msg: OpenAI.Chat.ChatCompletionMessageParam) => msg.role === 'system',
    );
    const lastIndex = messages.length - 1;

    const updatedMessages =
      messages.length === 0
        ? messages
        : messages.map(
            (
              message: OpenAI.Chat.ChatCompletionMessageParam,
              index: number,
            ) => {
              const shouldAddCacheControl = Boolean(
                (index === systemIndex && systemIndex !== -1) ||
                  (index === lastIndex && cacheControl === 'all'),
              );

              if (
                !shouldAddCacheControl ||
                !('content' in message) ||
                message.content === null ||
                message.content === undefined
              ) {
                return message;
              }

              return {
                ...message,
                content: this.addCacheControlToContent(message.content),
              } as OpenAI.Chat.ChatCompletionMessageParam;
            },
          );

    const updatedTools =
      cacheControl === 'all' && request.tools?.length
        ? this.addCacheControlToTools(request.tools)
        : (request.tools as ChatCompletionToolWithCache[] | undefined);

    return {
      messages: updatedMessages,
      tools: updatedTools,
    };
  }

  private addCacheControlToTools(
    tools: OpenAI.Chat.ChatCompletionTool[],
  ): ChatCompletionToolWithCache[] {
    if (tools.length === 0) {
      return tools as ChatCompletionToolWithCache[];
    }

    const updatedTools = [...tools] as ChatCompletionToolWithCache[];
    const lastToolIndex = tools.length - 1;
    updatedTools[lastToolIndex] = {
      ...updatedTools[lastToolIndex],
      cache_control: { type: 'ephemeral' },
    };

    return updatedTools;
  }

  private addCacheControlToContent(
    content: NonNullable<OpenAI.Chat.ChatCompletionMessageParam['content']>,
  ): ChatCompletionContentPartWithCache[] {
    const contentArray = this.normalizeContentToArray(content);

    return this.addCacheControlToContentArray(contentArray);
  }

  private normalizeContentToArray(
    content: NonNullable<OpenAI.Chat.ChatCompletionMessageParam['content']>,
  ): ChatCompletionContentPartWithCache[] {
    if (typeof content === 'string') {
      return [
        {
          type: 'text',
          text: content,
        } as ChatCompletionContentPartTextWithCache,
      ];
    }
    return [...content] as ChatCompletionContentPartWithCache[];
  }

  private addCacheControlToContentArray(
    contentArray: ChatCompletionContentPartWithCache[],
  ): ChatCompletionContentPartWithCache[] {
    if (contentArray.length === 0) {
      return [
        {
          type: 'text',
          text: '',
          cache_control: { type: 'ephemeral' },
        } as ChatCompletionContentPartTextWithCache,
      ];
    }

    const lastItem = contentArray[contentArray.length - 1];

    if (lastItem.type === 'text') {
      contentArray[contentArray.length - 1] = {
        ...lastItem,
        cache_control: { type: 'ephemeral' },
      } as ChatCompletionContentPartTextWithCache;
    } else {
      contentArray.push({
        type: 'text',
        text: '',
        cache_control: { type: 'ephemeral' },
      } as ChatCompletionContentPartTextWithCache);
    }

    return contentArray;
  }

  private isVisionModel(model: string | undefined): boolean {
    if (!model) {
      return false;
    }

    const normalized = model.toLowerCase();

    if (normalized === 'vision-model') {
      return true;
    }

    if (normalized.startsWith('qwen-vl')) {
      return true;
    }

    if (normalized.startsWith('qwen3-vl-plus')) {
      return true;
    }

    return false;
  }

  private applyOutputTokenLimit<T extends { max_tokens?: number | null }>(
    request: T,
    model: string,
  ): T {
    const currentMaxTokens = request.max_tokens;

    if (currentMaxTokens === undefined || currentMaxTokens === null) {
      return request;
    }

    const modelLimit = tokenLimit(model, this.cliConfig);

    if (currentMaxTokens > modelLimit) {
      return {
        ...request,
        max_tokens: modelLimit,
      };
    }

    return request;
  }

  private shouldDisableCacheControl(): boolean {
    return (
      this.cliConfig.getContentGeneratorConfig()?.disableCacheControl === true
    );
  }
}
