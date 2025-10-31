/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { GenerateContentResponse } from '@google/genai';

export interface RequestContext {
  userPromptId: string;
  model: string;
  authType: string;
  startTime: number;
  duration: number;
  isStreaming: boolean;
}

export interface TelemetryService {
  logSuccess(
    context: RequestContext,
    response: GenerateContentResponse,
  ): Promise<void>;

  logError(context: RequestContext, error: unknown): Promise<void>;

  logStreamingSuccess(
    context: RequestContext,
    responses: GenerateContentResponse[],
  ): Promise<void>;
}

export class DefaultTelemetryService implements TelemetryService {
  constructor() {}

  async logSuccess() // context: RequestContext,
  // response: GenerateContentResponse,
  : Promise<void> {
    // Stubbed method. Does nothing.
    return Promise.resolve();
  }

  async logError() // context: RequestContext,
  // error: unknown,
  : Promise<void> {
    // Stubbed method. Does nothing.
    return Promise.resolve();
  }

  async logStreamingSuccess() // context: RequestContext,
  // responses: GenerateContentResponse[],
  : Promise<void> {
    // Stubbed method. Does nothing.
    return Promise.resolve();
  }
}
