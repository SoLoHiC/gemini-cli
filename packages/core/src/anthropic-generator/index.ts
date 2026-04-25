/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Config } from '../config/config.js';
import type { ContentGeneratorConfig } from '../core/contentGenerator.js';
import type { AnthropicCompatibleProvider } from './provider/types.js';
import { DefaultAnthropicCompatibleProvider } from './provider/default.js';
import { DeepSeekAnthropicCompatibleProvider } from './provider/deepseek.js';

export function createAnthropicCompatibleProvider(
  contentGeneratorConfig: ContentGeneratorConfig,
  cliConfig: Config,
): AnthropicCompatibleProvider {
  const config =
    contentGeneratorConfig || cliConfig.getContentGeneratorConfig();

  // Check for DeepSeek provider
  if (DeepSeekAnthropicCompatibleProvider.isDeepSeekProvider(config)) {
    return new DeepSeekAnthropicCompatibleProvider(
      contentGeneratorConfig,
      cliConfig,
    );
  }

  // Also detect DeepSeek models by name to support third-party providers
  // hosting DeepSeek V4 under their own base URL
  if (DeepSeekAnthropicCompatibleProvider.isDeepSeekModel(config)) {
    return new DeepSeekAnthropicCompatibleProvider(
      contentGeneratorConfig,
      cliConfig,
    );
  }

  // Default provider for standard Anthropic-compatible APIs
  return new DefaultAnthropicCompatibleProvider(
    contentGeneratorConfig,
    cliConfig,
  );
}
