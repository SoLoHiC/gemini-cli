/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type Anthropic from '@anthropic-ai/sdk';
import type { GenerateContentParameters } from '@google/genai';

export interface AnthropicCompatibleProvider {
  validateGeminiRequest?(
    request: GenerateContentParameters,
  ): void | Promise<void>;
  buildClient(): Anthropic;
  buildRequest(
    request: Anthropic.Messages.MessageCreateParams,
    userPromptId: string,
  ): Anthropic.Messages.MessageCreateParams;
}
