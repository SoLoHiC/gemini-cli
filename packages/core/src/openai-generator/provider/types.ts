/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type OpenAI from 'openai';

// Extended types to support cache_control for DashScope
export interface ChatCompletionContentPartTextWithCache
  extends OpenAI.Chat.ChatCompletionContentPartText {
  cache_control?: { type: 'ephemeral' };
}

export type ChatCompletionContentPartWithCache =
  | ChatCompletionContentPartTextWithCache
  | OpenAI.Chat.ChatCompletionContentPartImage
  | OpenAI.Chat.ChatCompletionContentPartRefusal;

export type ChatCompletionToolWithCache = OpenAI.Chat.ChatCompletionTool & {
  cache_control?: { type: 'ephemeral' };
};

export interface OpenAICompatibleProvider {
  buildHeaders(): Record<string, string | undefined>;
  buildClient(): OpenAI;
  buildRequest(
    request: OpenAI.Chat.ChatCompletionCreateParams,
    userPromptId: string,
  ): OpenAI.Chat.ChatCompletionCreateParams;
}

export type DashScopeRequestMetadata = {
  metadata: {
    sessionId?: string;
    promptId: string;
  };
};
