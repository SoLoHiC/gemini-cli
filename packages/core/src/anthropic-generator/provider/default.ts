/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import Anthropic from '@anthropic-ai/sdk';
import type { Config } from '../../config/config.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import {
  DEFAULT_TIMEOUT,
  DEFAULT_MAX_RETRIES,
} from '../../openai-generator/constants.js'; // Re-using constants
import type { AnthropicCompatibleProvider } from './types.js';
import { tokenLimit } from '../../core/tokenLimits.js';

export class DefaultAnthropicCompatibleProvider
  implements AnthropicCompatibleProvider
{
  protected contentGeneratorConfig: ContentGeneratorConfig;
  protected cliConfig: Config;

  constructor(
    contentGeneratorConfig: ContentGeneratorConfig,
    cliConfig: Config,
  ) {
    this.cliConfig = cliConfig;
    this.contentGeneratorConfig = contentGeneratorConfig;
  }

  buildHeaders(): Record<string, string | undefined> {
    const version = 'unknown'; // this.cliConfig.getCliVersion() || 'unknown';
    const userAgent = `GeminiCLI/${version} (${process.platform}; ${process.arch})`;
    return {
      'User-Agent': userAgent,
      ...this.contentGeneratorConfig.customHeaders,
    };
  }

  buildClient(): Anthropic {
    const {
      apiKey,
      baseUrl,
      timeout = DEFAULT_TIMEOUT,
      maxRetries = DEFAULT_MAX_RETRIES,
    } = this.contentGeneratorConfig;
    const defaultHeaders = this.buildHeaders();
    return new Anthropic({
      apiKey,
      baseURL: baseUrl,
      timeout,
      maxRetries,
      defaultHeaders,
    });
  }

  buildRequest(
    request: Anthropic.Messages.MessageCreateParams,
    _userPromptId: string,
  ): Anthropic.Messages.MessageCreateParams {
    const mergedRequest = {
      ...request,
    } as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;
    const samplingParams = this.contentGeneratorConfig.samplingParams;

    if (
      samplingParams?.temperature !== undefined &&
      mergedRequest.temperature === undefined
    ) {
      mergedRequest.temperature = samplingParams.temperature;
    }
    if (
      samplingParams?.top_p !== undefined &&
      mergedRequest.top_p === undefined
    ) {
      mergedRequest.top_p = samplingParams.top_p;
    }
    if (
      samplingParams?.top_k !== undefined &&
      mergedRequest.top_k === undefined
    ) {
      mergedRequest.top_k = samplingParams.top_k;
    }
    if (
      samplingParams?.max_tokens !== undefined &&
      mergedRequest.max_tokens === undefined
    ) {
      mergedRequest.max_tokens = samplingParams.max_tokens;
    }
    if (
      this.contentGeneratorConfig.thinking !== undefined &&
      mergedRequest['thinking'] === undefined
    ) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      mergedRequest['thinking'] = this.contentGeneratorConfig
        .thinking as unknown as Anthropic.ThinkingConfigParam;
    }
    if (
      this.contentGeneratorConfig.reasoning !== undefined &&
      mergedRequest['thinking'] === undefined
    ) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      mergedRequest['thinking'] = this.contentGeneratorConfig
        .reasoning as unknown as Anthropic.ThinkingConfigParam;
    }
    if (
      this.contentGeneratorConfig.outputConfig !== undefined &&
      mergedRequest['output_config'] === undefined
    ) {
      mergedRequest['output_config'] = this.contentGeneratorConfig.outputConfig;
    }
    if (typeof mergedRequest.max_tokens === 'number') {
      const outputLimit = tokenLimit(
        mergedRequest.model || this.contentGeneratorConfig.model || '',
        this.cliConfig,
        'output',
      );
      if (mergedRequest.max_tokens > outputLimit) {
        mergedRequest.max_tokens = outputLimit;
      }
    }

    return mergedRequest;
  }
}
