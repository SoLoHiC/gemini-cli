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
    const baseUrl = (contentGeneratorConfig.baseUrl || '')
      .toLowerCase()
      .replace(/\/+$/, '');
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
      const visionRequest: OpenAI.Chat.ChatCompletionCreateParams & {
        vl_high_resolution_images: true;
      } = {
        ...requestWithTokenLimits,
        messages,
        ...(tools ? { tools } : {}),
        ...(this.buildMetadata(userPromptId) || {}),
        // DashScope supports this provider-specific flag in compatible mode.
        vl_high_resolution_images: true,
      };

      return visionRequest;
    }

    return {
      ...requestWithTokenLimits,
      messages,
      ...(tools ? { tools } : {}),
      ...(this.buildMetadata(userPromptId) || {}),
    };
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
                message.role === 'function' ||
                !('content' in message) ||
                message.content === null ||
                message.content === undefined
              ) {
                return message;
              }

              return this.addCacheControlToMessage(message);
            },
          );

    const updatedTools =
      cacheControl === 'all' && request.tools?.length
        ? this.addCacheControlToTools(request.tools)
        : request.tools;

    return {
      messages: updatedMessages,
      tools: updatedTools,
    };
  }

  private addCacheControlToTools(
    tools: OpenAI.Chat.ChatCompletionTool[],
  ): ChatCompletionToolWithCache[] {
    if (tools.length === 0) {
      return tools;
    }

    const updatedTools: ChatCompletionToolWithCache[] = [...tools];
    const lastToolIndex = tools.length - 1;
    updatedTools[lastToolIndex] = {
      ...updatedTools[lastToolIndex],
      cache_control: { type: 'ephemeral' },
    };

    return updatedTools;
  }

  private addCacheControlToMessage(
    message:
      | OpenAI.Chat.ChatCompletionDeveloperMessageParam
      | OpenAI.Chat.ChatCompletionSystemMessageParam
      | OpenAI.Chat.ChatCompletionUserMessageParam
      | OpenAI.Chat.ChatCompletionAssistantMessageParam
      | OpenAI.Chat.ChatCompletionToolMessageParam,
  ): OpenAI.Chat.ChatCompletionMessageParam {
    switch (message.role) {
      case 'developer':
      case 'system':
      case 'tool':
        return {
          ...message,
          content: this.addCacheControlToTextContent(message.content),
        };
      case 'assistant':
        if (message.content === undefined || message.content === null) {
          return message;
        }
        return {
          ...message,
          content: this.addCacheControlToAssistantContent(message.content),
        };
      case 'user':
        return {
          ...message,
          content: this.addCacheControlToUserContent(message.content),
        };
      default:
        return message;
    }
  }

  private createTextPart(
    text: string,
  ): OpenAI.Chat.ChatCompletionContentPartText {
    return {
      type: 'text',
      text,
    };
  }

  private createEphemeralTextPart(
    text: string,
  ): ChatCompletionContentPartTextWithCache {
    return {
      type: 'text',
      text,
      cache_control: { type: 'ephemeral' },
    };
  }

  private addCacheControlToTextContent(
    content: string | OpenAI.Chat.ChatCompletionContentPartText[],
  ): OpenAI.Chat.ChatCompletionContentPartText[] {
    const contentArray =
      typeof content === 'string'
        ? [this.createTextPart(content)]
        : [...content];

    return this.addCacheControlToTextContentArray(contentArray);
  }

  private addCacheControlToAssistantContent(
    content:
      | string
      | Array<
          | OpenAI.Chat.ChatCompletionContentPartText
          | OpenAI.Chat.ChatCompletionContentPartRefusal
        >,
  ):
    | string
    | Array<
        | OpenAI.Chat.ChatCompletionContentPartText
        | OpenAI.Chat.ChatCompletionContentPartRefusal
      > {
    const contentArray =
      typeof content === 'string'
        ? [this.createTextPart(content)]
        : [...content];

    return this.addCacheControlToAssistantContentArray(contentArray);
  }

  private addCacheControlToUserContent(
    content: string | OpenAI.Chat.ChatCompletionContentPart[],
  ): OpenAI.Chat.ChatCompletionContentPart[] {
    const contentArray =
      typeof content === 'string'
        ? [this.createTextPart(content)]
        : [...content];

    return this.addCacheControlToUserContentArray(contentArray);
  }

  private addCacheControlToTextContentArray(
    contentArray: OpenAI.Chat.ChatCompletionContentPartText[],
  ): OpenAI.Chat.ChatCompletionContentPartText[] {
    if (contentArray.length === 0) {
      return [this.createEphemeralTextPart('')];
    }

    const lastItem = contentArray[contentArray.length - 1];
    contentArray[contentArray.length - 1] = this.createEphemeralTextPart(
      lastItem.text,
    );

    return contentArray;
  }

  private addCacheControlToAssistantContentArray(
    contentArray: Array<
      | OpenAI.Chat.ChatCompletionContentPartText
      | OpenAI.Chat.ChatCompletionContentPartRefusal
    >,
  ): Array<
    | OpenAI.Chat.ChatCompletionContentPartText
    | OpenAI.Chat.ChatCompletionContentPartRefusal
  > {
    if (contentArray.length === 0) {
      return [this.createEphemeralTextPart('')];
    }

    const lastItem = contentArray[contentArray.length - 1];

    if (lastItem.type === 'text') {
      contentArray[contentArray.length - 1] = this.createEphemeralTextPart(
        lastItem.text,
      );
    } else {
      contentArray.push(this.createEphemeralTextPart(''));
    }

    return contentArray;
  }

  private addCacheControlToUserContentArray(
    contentArray: OpenAI.Chat.ChatCompletionContentPart[],
  ): OpenAI.Chat.ChatCompletionContentPart[] {
    if (contentArray.length === 0) {
      return [this.createEphemeralTextPart('')];
    }

    const lastItem = contentArray[contentArray.length - 1];

    if (lastItem.type === 'text') {
      contentArray[contentArray.length - 1] = this.createEphemeralTextPart(
        lastItem.text,
      );
    } else {
      contentArray.push(this.createEphemeralTextPart(''));
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

    const modelLimit = tokenLimit(model, this.cliConfig, 'output');

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
