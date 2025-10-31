/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type Config,
  isGoogleAuthType,
  tokenLimit,
} from '@google/gemini-cli-core';

export function getEffectiveContextLimit(
  model: string | undefined,
  config: Config,
): number {
  if (!model || typeof model !== 'string' || model.length === 0) {
    return 0;
  }

  const inputLimit = tokenLimit(model, config);
  const authType = config.getContentGeneratorConfig?.()?.authType;

  if (!authType || isGoogleAuthType(authType)) {
    return inputLimit;
  }

  const configuredMaxTokens =
    config.getContentGeneratorConfig?.()?.samplingParams?.max_tokens;
  const reservedCompletionTokens =
    typeof configuredMaxTokens === 'number' && configuredMaxTokens > 0
      ? Math.min(configuredMaxTokens, tokenLimit(model, config, 'output'))
      : tokenLimit(model, config, 'output');

  return Math.max(1, inputLimit - reservedCompletionTokens);
}

export function getContextUsagePercentage(
  promptTokenCount: number,
  model: string | undefined,
  config: Config,
): number {
  if (!model || typeof model !== 'string' || model.length === 0) {
    return 0;
  }
  const limit = getEffectiveContextLimit(model, config);
  if (limit <= 0) {
    return 0;
  }
  return promptTokenCount / limit;
}

export function isContextUsageHigh(
  promptTokenCount: number,
  model: string | undefined,
  config: Config,
  threshold = 0.6,
): boolean {
  return getContextUsagePercentage(promptTokenCount, model, config) > threshold;
}
