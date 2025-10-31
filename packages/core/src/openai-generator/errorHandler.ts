/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { GenerateContentParameters } from '@google/genai';
import type { RequestContext } from './telemetryService.js';
import { debugLogger } from '../utils/debugLogger.js';

export interface ErrorHandler {
  handle(
    error: unknown,
    context: RequestContext,
    request: GenerateContentParameters,
  ): never;
  shouldSuppressErrorLogging(
    error: unknown,
    request: GenerateContentParameters,
  ): boolean;
}

export class EnhancedErrorHandler implements ErrorHandler {
  constructor(
    private shouldSuppressLogging: (
      error: unknown,
      request: GenerateContentParameters,
    ) => boolean = () => false,
  ) {}

  handle(
    error: unknown,
    context: RequestContext,
    request: GenerateContentParameters,
  ): never {
    const errorMessage = this.getErrorMessage(error);

    // Check if we should suppress logging
    if (!this.shouldSuppressErrorLogging(error, request)) {
      this.logError(errorMessage, context.isStreaming);
    }

    // Check if it's a timeout error
    if (this.isTimeoutError(error)) {
      const enhancedMessage = this.buildTimeoutErrorMessage(
        errorMessage,
        context,
      );
      throw new Error(enhancedMessage);
    }

    // For non-timeout errors, throw the original error
    if (error instanceof Error) {
      throw error;
    }

    // For non-Error objects, convert to string and throw
    throw new Error(String(error));
  }

  shouldSuppressErrorLogging(
    error: unknown,
    request: GenerateContentParameters,
  ): boolean {
    return this.shouldSuppressLogging(error, request);
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    if (typeof error === 'string') {
      return error;
    }
    if (error && typeof error === 'object' && 'message' in error) {
      return String((error as { message: unknown }).message);
    }
    if (error && typeof error === 'object' && 'toString' in error) {
      return (error as { toString: () => string }).toString();
    }
    return String(error);
  }

  private logError(errorMessage: string, isStreaming: boolean): void {
    const prefix = isStreaming
      ? 'OpenAI API Streaming Error:'
      : 'OpenAI API Error:';
    debugLogger.error(prefix, errorMessage);
  }

  private isTimeoutError(error: unknown): boolean {
    const errorMessage = this.getErrorMessage(error).toLowerCase();

    // Check for timeout keywords in message
    const timeoutKeywords = [
      'timeout',
      'timed out',
      'deadline exceeded',
      'etimedout',
      'esockettimedout',
    ];

    if (timeoutKeywords.some((keyword) => errorMessage.includes(keyword))) {
      return true;
    }

    // Check for timeout codes
    if (error && typeof error === 'object') {
      const errorObj = error as { code?: string; type?: string };
      if (
        errorObj.code === 'ETIMEDOUT' ||
        errorObj.code === 'ESOCKETTIMEDOUT'
      ) {
        return true;
      }
      if (errorObj.type === 'timeout') {
        return true;
      }
    }

    return false;
  }

  private buildTimeoutErrorMessage(
    errorMessage: string,
    context: RequestContext,
  ): string {
    const durationInSeconds = Math.round(context.duration / 1000);
    const isStreaming = context.isStreaming;

    let message = isStreaming
      ? `Streaming request timeout after ${durationInSeconds}s. Try reducing input length or increasing timeout in config.`
      : `Request timeout after ${durationInSeconds}s. Try reducing input length or increasing timeout in config.`;

    if (isStreaming) {
      message += '\n\nStreaming timeout troubleshooting:\n';
      message += '- Reduce input length or complexity\n';
      message += '- Increase timeout in config: contentGenerator.timeout\n';
      message += '- Check network connectivity\n';
      message += '- Check network stability for streaming connections\n';
      message += '- Consider using non-streaming mode for very long inputs';
    } else {
      message += '\n\nTroubleshooting tips:\n';
      message += '- Reduce input length or complexity\n';
      message += '- Increase timeout in config: contentGenerator.timeout\n';
      message += '- Check network connectivity\n';
      message += '- Consider using streaming mode for long responses';
    }

    return message;
  }
}
