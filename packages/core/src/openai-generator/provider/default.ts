/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import type { Config } from '../../config/config.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import { DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES } from '../constants.js';
import type { OpenAICompatibleProvider } from './types.js';
import { tokenLimit } from '../../core/tokenLimits.js';

/**
 * Default provider for standard OpenAI-compatible APIs
 */
export class DefaultOpenAICompatibleProvider
  implements OpenAICompatibleProvider
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
    _userPromptId: string,
  ): OpenAI.Chat.ChatCompletionCreateParams {
    const mergedRequest = {
      ...request,
    } as OpenAI.Chat.ChatCompletionCreateParams & Record<string, unknown>;
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
      samplingParams?.presence_penalty !== undefined &&
      mergedRequest.presence_penalty === undefined
    ) {
      mergedRequest.presence_penalty = samplingParams.presence_penalty;
    }
    if (
      samplingParams?.frequency_penalty !== undefined &&
      mergedRequest.frequency_penalty === undefined
    ) {
      mergedRequest.frequency_penalty = samplingParams.frequency_penalty;
    }
    if (
      samplingParams?.max_tokens !== undefined &&
      mergedRequest.max_tokens === undefined
    ) {
      mergedRequest.max_tokens = samplingParams.max_tokens;
    }
    if (
      this.contentGeneratorConfig.reasoning !== undefined &&
      mergedRequest['reasoning'] === undefined
    ) {
      mergedRequest['reasoning'] = this.contentGeneratorConfig.reasoning;
    }
    if (
      this.contentGeneratorConfig.reasoningEffort !== undefined &&
      mergedRequest['reasoning_effort'] === undefined
    ) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      mergedRequest['reasoning_effort'] = this.contentGeneratorConfig
        .reasoningEffort as OpenAI.Chat.ChatCompletionCreateParams['reasoning_effort'];
    }
    if (
      this.contentGeneratorConfig.modalities !== undefined &&
      mergedRequest['modalities'] === undefined
    ) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      mergedRequest['modalities'] = this.contentGeneratorConfig
        .modalities as Array<'text' | 'audio'>;
    }
    if (
      this.contentGeneratorConfig.extra_body !== undefined &&
      mergedRequest['extra_body'] === undefined
    ) {
      mergedRequest['extra_body'] = this.contentGeneratorConfig.extra_body;
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
