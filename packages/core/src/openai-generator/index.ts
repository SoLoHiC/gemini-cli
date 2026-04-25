/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  ContentGenerator,
  ContentGeneratorConfig,
} from '../core/contentGenerator.js';
import type { Config } from '../config/config.js';
import { OpenAIContentGenerator } from './openaiContentGenerator.js';
import {
  DeepSeekOpenAICompatibleProvider,
  OpenRouterOpenAICompatibleProvider,
  type OpenAICompatibleProvider,
  DefaultOpenAICompatibleProvider,
  DashScopeOpenAICompatibleProvider,
} from './provider/index.js';

export { OpenAIContentGenerator } from './openaiContentGenerator.js';
export { ContentGenerationPipeline, type PipelineConfig } from './pipeline.js';

export {
  type OpenAICompatibleProvider,
  DeepSeekOpenAICompatibleProvider,
  OpenRouterOpenAICompatibleProvider,
} from './provider/index.js';

export { OpenAIContentConverter } from './converter.js';

/**
 * Create an OpenAI-compatible content generator with the appropriate provider
 */
export function createOpenAIContentGenerator(
  contentGeneratorConfig: ContentGeneratorConfig,
  cliConfig: Config,
): ContentGenerator {
  const provider = determineProvider(contentGeneratorConfig, cliConfig);
  return new OpenAIContentGenerator(
    contentGeneratorConfig,
    cliConfig,
    provider,
  );
}

/**
 * Determine the appropriate provider based on configuration
 */
export function determineProvider(
  contentGeneratorConfig: ContentGeneratorConfig,
  cliConfig: Config,
): OpenAICompatibleProvider {
  const config =
    contentGeneratorConfig || cliConfig.getContentGeneratorConfig();

  if (DeepSeekOpenAICompatibleProvider.isDeepSeekProvider(config)) {
    return new DeepSeekOpenAICompatibleProvider(
      contentGeneratorConfig,
      cliConfig,
    );
  }

  // Also detect DeepSeek models by name to support third-party providers
  // hosting DeepSeek V4 under their own base URL (e.g., OpenRouter, Together AI)
  if (DeepSeekOpenAICompatibleProvider.isDeepSeekModel(config)) {
    return new DeepSeekOpenAICompatibleProvider(
      contentGeneratorConfig,
      cliConfig,
    );
  }

  if (DashScopeOpenAICompatibleProvider.isDashScopeProvider(config)) {
    return new DashScopeOpenAICompatibleProvider(
      contentGeneratorConfig,
      cliConfig,
    );
  }

  // Check for OpenRouter provider
  if (OpenRouterOpenAICompatibleProvider.isOpenRouterProvider(config)) {
    return new OpenRouterOpenAICompatibleProvider(
      contentGeneratorConfig,
      cliConfig,
    );
  }

  // Default provider for standard OpenAI-compatible APIs
  return new DefaultOpenAICompatibleProvider(contentGeneratorConfig, cliConfig);
}

// Services
export {
  type TelemetryService,
  type RequestContext,
  DefaultTelemetryService,
} from './telemetryService.js';

export { type ErrorHandler, EnhancedErrorHandler } from './errorHandler.js';
