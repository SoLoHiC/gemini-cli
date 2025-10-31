/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type Anthropic from '@anthropic-ai/sdk';

export interface AnthropicCompatibleProvider {
  buildClient(): Anthropic;
  buildRequest(
    request: Anthropic.Messages.MessageCreateParams,
    userPromptId: string,
  ): Anthropic.Messages.MessageCreateParams;
}
