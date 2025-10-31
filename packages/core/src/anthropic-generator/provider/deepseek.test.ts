/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import Anthropic from '@anthropic-ai/sdk';
import { DeepSeekAnthropicCompatibleProvider } from './deepseek.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import type { Config } from '../../config/config.js';

// Mock Anthropic client to avoid real network calls
vi.mock('@anthropic-ai/sdk', () => ({
  default: vi.fn().mockImplementation((config) => ({
    config,
  })),
}));

describe('DeepSeekAnthropicCompatibleProvider', () => {
  let provider: DeepSeekAnthropicCompatibleProvider;
  let mockContentGeneratorConfig: ContentGeneratorConfig;
  let mockCliConfig: Config;

  beforeEach(() => {
    vi.clearAllMocks();

    mockContentGeneratorConfig = {
      apiKey: 'test-api-key',
      baseUrl: 'https://api.deepseek.com/anthropic',
      model: 'deepseek-chat',
    } as ContentGeneratorConfig;

    mockCliConfig = {
      getCliVersion: vi.fn().mockReturnValue('1.0.0'),
    } as unknown as Config;

    provider = new DeepSeekAnthropicCompatibleProvider(
      mockContentGeneratorConfig,
      mockCliConfig,
    );
  });

  describe('isDeepSeekProvider', () => {
    it('returns true when baseUrl includes deepseek', () => {
      const result = DeepSeekAnthropicCompatibleProvider.isDeepSeekProvider(
        mockContentGeneratorConfig,
      );
      expect(result).toBe(true);
    });

    it('returns false for non deepseek baseUrl', () => {
      const config = {
        ...mockContentGeneratorConfig,
        baseUrl: 'https://api.anthropic.com',
      } as ContentGeneratorConfig;

      const result =
        DeepSeekAnthropicCompatibleProvider.isDeepSeekProvider(config);
      expect(result).toBe(false);
    });
  });

  describe('buildHeaders', () => {
    it('includes x-api-key header for DeepSeek authentication', () => {
      const headers = provider.buildHeaders();

      expect(headers['x-api-key']).toBe('test-api-key');
      expect(headers['User-Agent']).toBeDefined();
    });
  });

  describe('buildClient', () => {
    it('creates client with DeepSeek-specific configuration', () => {
      provider.buildClient();

      expect(Anthropic).toHaveBeenCalledWith({
        apiKey: '', // Using headers for authentication
        baseURL: 'https://api.deepseek.com/anthropic',
        timeout: 600000, // DeepSeek recommended timeout
        maxRetries: 3,
        defaultHeaders: expect.objectContaining({
          'x-api-key': 'test-api-key',
        }),
      });
    });
  });

  describe('buildRequest', () => {
    const userPromptId = 'prompt-123';

    it('clamps temperature to DeepSeek range [0.0, 2.0]', () => {
      const originalRequest: Anthropic.Messages.MessageCreateParams = {
        model: 'deepseek-chat',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
        temperature: 2.5, // Above DeepSeek max
      };

      const result = provider.buildRequest(originalRequest, userPromptId);

      expect(result.temperature).toBe(2.0);
    });

    it('handles negative temperature values', () => {
      const originalRequest: Anthropic.Messages.MessageCreateParams = {
        model: 'deepseek-chat',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
        temperature: -1.0, // Below DeepSeek min
      };

      const result = provider.buildRequest(originalRequest, userPromptId);

      expect(result.temperature).toBe(0.0);
    });

    it('preserves valid temperature values', () => {
      const originalRequest: Anthropic.Messages.MessageCreateParams = {
        model: 'deepseek-chat',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
        temperature: 1.5, // Within DeepSeek range
      };

      const result = provider.buildRequest(originalRequest, userPromptId);

      expect(result.temperature).toBe(1.5);
    });
  });
});
