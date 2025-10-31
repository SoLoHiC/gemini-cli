/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type Anthropic from '@anthropic-ai/sdk';
import type { GenerateContentResponse } from '@google/genai';
import { type GenerateContentParameters } from '@google/genai';
import type { Config } from '../config/config.js';
import type { ContentGeneratorConfig } from '../core/contentGenerator.js';
import type { AnthropicCompatibleProvider } from './provider/types.js';
import { AnthropicContentConverter } from './converter.js';
import type { TelemetryService } from './telemetryService.js';
import type { ErrorHandler } from './errorHandler.js';

export interface PipelineConfig {
  cliConfig: Config;
  provider: AnthropicCompatibleProvider;
  contentGeneratorConfig: ContentGeneratorConfig;
  telemetryService: TelemetryService;
  errorHandler: ErrorHandler;
}

export class ContentGenerationPipeline {
  client: Anthropic;
  private converter: AnthropicContentConverter;
  readonly contentGeneratorConfig: ContentGeneratorConfig;

  constructor(private config: PipelineConfig) {
    this.contentGeneratorConfig = config.contentGeneratorConfig;
    this.client = this.config.provider.buildClient();
    this.converter = new AnthropicContentConverter(
      this.contentGeneratorConfig.model!,
    );
  }

  async execute(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const anthropicRequest =
      await this.converter.convertGeminiRequestToAnthropic(request);
    const processedRequest = this.config.provider.buildRequest(
      anthropicRequest,
      userPromptId,
    );
    const anthropicResponse =
      await this.client.messages.create(processedRequest);
    return this.converter.convertAnthropicResponseToGemini(
      anthropicResponse as Anthropic.Message,
    );
  }

  async executeStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    this.converter.resetStreamingToolCalls();
    const anthropicRequest =
      await this.converter.convertGeminiRequestToAnthropic(request);
    const processedRequest = this.config.provider.buildRequest(
      anthropicRequest,
      userPromptId,
    );
    const stream = this.client.messages.stream(processedRequest);

    const generate = async function* (
      this: ContentGenerationPipeline,
    ): AsyncGenerator<GenerateContentResponse> {
      for await (const chunk of stream) {
        const geminiResponse =
          this.converter.convertAnthropicChunkToGemini(chunk);
        if (geminiResponse) {
          yield geminiResponse;
        }
      }
    };

    return generate.call(this);
  }
}
