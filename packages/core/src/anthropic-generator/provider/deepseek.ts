/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import Anthropic from '@anthropic-ai/sdk';
import type { Config } from '../../config/config.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import { DefaultAnthropicCompatibleProvider } from './default.js';

export class DeepSeekAnthropicCompatibleProvider extends DefaultAnthropicCompatibleProvider {
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

  override buildHeaders(): Record<string, string | undefined> {
    const defaultHeaders = super.buildHeaders();

    // DeepSeek Anthropic API requires x-api-key header instead of Authorization
    return {
      ...defaultHeaders,
      'x-api-key': this.contentGeneratorConfig.apiKey,
    };
  }

  override buildClient(): Anthropic {
    const {
      baseUrl,
      timeout = 600000, // DeepSeek recommends 600000ms timeout
      maxRetries = 3,
    } = this.contentGeneratorConfig;

    const headers = this.buildHeaders();

    return new Anthropic({
      apiKey: '', // Use headers for authentication
      baseURL: baseUrl,
      timeout,
      maxRetries,
      defaultHeaders: headers,
    });
  }

  override buildRequest(
    request: Anthropic.Messages.MessageCreateParams,
    userPromptId: string,
  ): Anthropic.Messages.MessageCreateParams {
    const baseRequest = super.buildRequest(request, userPromptId);

    // DeepSeek-specific request modifications
    // - Ensure temperature is within [0.0, 2.0] range
    // - Handle any DeepSeek-specific parameter requirements

    if (baseRequest.temperature !== undefined) {
      baseRequest.temperature = Math.max(
        0.0,
        Math.min(2.0, baseRequest.temperature),
      );
    }

    return baseRequest;
  }
}
