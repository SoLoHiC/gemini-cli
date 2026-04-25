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

/**
 * Check if thinking mode is explicitly disabled in the request's extra_body.
 * Returns `false` if `extra_body` or `thinking` are absent (i.e. no explicit
 * enablement), meaning thinking is NOT considered "disabled" — it is just not
 * opted in. The caller should interpret the absence of explicit enablement
 * according to its own context (e.g. V4 models auto-enable it).
 */
function isThinkingDisabledInRequest(
  request: OpenAI.Chat.ChatCompletionCreateParams,
): boolean {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const extendedRequest = request as unknown as Record<string, unknown>;
  const extraBody = extendedRequest['extra_body'];
  if (!extraBody || typeof extraBody !== 'object') return false;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const thinking = (extraBody as Record<string, unknown>)['thinking'];
  if (!thinking || typeof thinking !== 'object') return false;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  return (thinking as Record<string, unknown>)['type'] === 'disabled';
}

/**
 * Check if thinking mode is explicitly enabled in the request's extra_body.
 * Returns `true` only when `extra_body.thinking.type` is present and its value
 * is anything other than `'disabled'`.  Returns `false` when no explicit
 * thinking configuration exists at all (i.e. thinking was not opted in).
 */
function isThinkingExplicitlyEnabled(
  request: OpenAI.Chat.ChatCompletionCreateParams,
): boolean {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const extendedRequest = request as unknown as Record<string, unknown>;
  const extraBody = extendedRequest['extra_body'];
  if (!extraBody || typeof extraBody !== 'object') return false;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const thinking = (extraBody as Record<string, unknown>)['thinking'];
  if (!thinking || typeof thinking !== 'object') return false;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const type = (thinking as Record<string, unknown>)['type'];
  // Explicitly set to a value that is not 'disabled' means thinking is on.
  return type !== undefined && type !== 'disabled';
}

/**
 * Ensure every assistant message in the array has a `reasoning_content` field.
 * When DeepSeek's thinking mode is enabled the API requires this field on every
 * assistant message — even if it is an empty string.  Chat compression can
 * strip thought parts, causing the 400 error this function prevents.
 */
function ensureAssistantReasoningContent(
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  return messages.map((message) => {
    if (message.role === 'assistant') {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const extendedMsg = message as unknown as Record<string, unknown>;
      if (extendedMsg['reasoning_content'] === undefined) {
        extendedMsg['reasoning_content'] = '';
      }
    }
    return message;
  });
}

/**
 * Remove the `reasoning_content` field from every assistant message.
 * When thinking mode is disabled, pre-existing reasoning_content in the
 * history (from previous turns that had thinking enabled) can cause the
 * API to continue returning thinking content.  Stripping it prevents that.
 */
function stripAssistantReasoningContent(
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  return messages.map((message) => {
    if (message.role === 'assistant') {
      const { reasoning_content: _, ...rest } = message as OpenAI.Chat.ChatCompletionAssistantMessageParam & { reasoning_content?: string };
      return rest as OpenAI.Chat.ChatCompletionMessageParam;
    }
    return message;
  });
}

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

  static isDeepSeekModel(
    contentGeneratorConfig: ContentGeneratorConfig,
  ): boolean {
    const model = contentGeneratorConfig.model ?? '';

    return model.toLowerCase().startsWith('deepseek-');
  }

  override buildRequest(
    request: OpenAI.Chat.ChatCompletionCreateParams,
    userPromptId: string,
  ): OpenAI.Chat.ChatCompletionCreateParams {
    const baseRequest = super.buildRequest(request, userPromptId);
    if (baseRequest.temperature === undefined) {
      baseRequest.temperature = 0;
    }

    // Auto-inject thinking mode and reasoning_effort for DeepSeek V4 models
    const model = baseRequest.model || this.contentGeneratorConfig.model || '';
    if (model.toLowerCase().startsWith('deepseek-v4-')) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const extendedRequest = baseRequest as OpenAI.Chat.ChatCompletionCreateParams &
        Record<string, unknown>;

      // Auto-enable thinking mode if not explicitly configured
      if (extendedRequest['extra_body'] === undefined) {
        extendedRequest['extra_body'] = {
          thinking: { type: 'enabled' },
        };
      } else if (
        // eslint-disable-next-line no-restricted-syntax
        typeof extendedRequest['extra_body'] === 'object' &&
        extendedRequest['extra_body'] !== null
      ) {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        const extraBody = extendedRequest['extra_body'] as Record<
          string,
          unknown
        >;
        if (extraBody['thinking'] === undefined) {
          extraBody['thinking'] = { type: 'enabled' };
        }
      }

      // Determine if thinking is explicitly disabled
      const isThinkingDisabled = isThinkingDisabledInRequest(baseRequest);

      // Auto-set reasoning_effort to 'high' if not already configured,
      // but only when thinking is not explicitly disabled
      if (
        !isThinkingDisabled &&
        extendedRequest['reasoning_effort'] === undefined &&
        this.contentGeneratorConfig.reasoningEffort === undefined
      ) {
        extendedRequest['reasoning_effort'] =
          'high' as OpenAI.Chat.ChatCompletionCreateParams['reasoning_effort'];
      }
    }

    // DeepSeek API requires that when thinking mode is enabled, every
    // assistant message must include a reasoning_content field (even if
    // empty string). Chat compression can strip thought parts from model
    // responses, causing a 400 error. Ensure all assistant messages have
    // reasoning_content when thinking is explicitly enabled.
    //
    // Conversely, when thinking is explicitly disabled, strip any pre-existing
    // reasoning_content from the history.  Otherwise a previous turn that
    // had thinking enabled could leave reasoning_content in the messages,
    // causing the API to continue returning thinking content.
    //
    // When thinking is neither explicitly enabled nor disabled (i.e. not
    // configured at all), leave the messages untouched.  Non-reasoning
    // models like `deepseek-chat` do not support the reasoning_content
    // field, and injecting it would cause a 400 error.
    //
    // This applies to ALL DeepSeek models, not just V4.
    const thinkingExplicitlyEnabled = isThinkingExplicitlyEnabled(baseRequest);
    const thinkingExplicitlyDisabled = isThinkingDisabledInRequest(baseRequest);
    if (Array.isArray(baseRequest.messages)) {
      if (thinkingExplicitlyEnabled) {
        baseRequest.messages =
          ensureAssistantReasoningContent(baseRequest.messages);
      } else if (thinkingExplicitlyDisabled) {
        baseRequest.messages =
          stripAssistantReasoningContent(baseRequest.messages);
      }
      // If neither explicitly enabled nor disabled, leave messages as-is.
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
