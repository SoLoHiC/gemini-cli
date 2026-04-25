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

    it('maps official Anthropic config names into the request', () => {
      provider = new DeepSeekAnthropicCompatibleProvider(
        {
          ...mockContentGeneratorConfig,
          model: 'deepseek-v4-pro',
          thinking: { type: 'enabled' },
          outputConfig: { effort: 'low' },
        } as ContentGeneratorConfig,
        mockCliConfig,
      );

      const originalRequest: Anthropic.Messages.MessageCreateParams = {
        model: 'deepseek-v4-pro',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
      };

      const result = provider.buildRequest(
        originalRequest,
        userPromptId,
      ) as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;

      expect(result['thinking']).toEqual({ type: 'enabled' });
      expect(result['output_config']).toEqual({ effort: 'low' });
    });

    it('auto-injects thinking and default effort for V4 models', () => {
      const originalRequest: Anthropic.Messages.MessageCreateParams = {
        model: 'deepseek-v4-pro',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
      };

      const result = provider.buildRequest(
        originalRequest,
        userPromptId,
      ) as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;

      expect(result['thinking']).toEqual({ type: 'enabled' });
      expect(result['output_config']).toEqual({ effort: 'high' });
    });

    it('does not inject output_config when thinking is explicitly disabled', () => {
      const originalRequest: Anthropic.Messages.MessageCreateParams &
        Record<string, unknown> = {
        model: 'deepseek-v4-flash',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
        thinking: { type: 'disabled' },
      };

      const result = provider.buildRequest(
        originalRequest,
        userPromptId,
      ) as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;

      expect(result['thinking']).toEqual({ type: 'disabled' });
      expect(result['output_config']).toBeUndefined();
    });

    it('does not auto-inject thinking for non-V4 models', () => {
      const originalRequest: Anthropic.Messages.MessageCreateParams &
        Record<string, unknown> = {
        model: 'deepseek-chat',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
      };

      const result = provider.buildRequest(
        originalRequest,
        userPromptId,
      ) as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;

      expect(result['thinking']).toBeUndefined();
      expect(result['output_config']).toBeUndefined();
    });

    it('preserves user-specified thinking and output_config for V4 models', () => {
      const originalRequest = {
        model: 'deepseek-v4-pro',
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'Hello' }],
        thinking: { type: 'enabled' } as Anthropic.Messages.ThinkingConfigParam,
        output_config: { effort: 'max' },
      } as unknown as Anthropic.Messages.MessageCreateParams &
        Record<string, unknown>;

      const result = provider.buildRequest(
        originalRequest as Anthropic.Messages.MessageCreateParams,
        userPromptId,
      ) as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;

      expect(result['thinking']).toEqual({ type: 'enabled' });
      expect(result['output_config']).toEqual({ effort: 'max' });
    });

    describe('thinking detection', () => {
      it('treats absent thinking config as not disabled and not enabled', () => {
        const originalRequest: Anthropic.Messages.MessageCreateParams = {
          model: 'deepseek-chat',
          max_tokens: 1000,
          messages: [{ role: 'user', content: 'Hello' }],
        };

        // Non-V4: no auto-injection, no thinking modifications
        const result = provider.buildRequest(
          originalRequest,
          userPromptId,
        ) as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;

        expect(result['thinking']).toBeUndefined();
      });

      it('treats thinking type enabled as active thinking', () => {
        const originalRequest = {
          model: 'deepseek-v4-pro',
          max_tokens: 1000,
          messages: [{ role: 'user', content: 'Hello' }],
          thinking: { type: 'enabled' } as Anthropic.Messages.ThinkingConfigParam,
        } as unknown as Anthropic.Messages.MessageCreateParams &
          Record<string, unknown>;

        const result = provider.buildRequest(
          originalRequest as Anthropic.Messages.MessageCreateParams,
          userPromptId,
        ) as Anthropic.Messages.MessageCreateParams & Record<string, unknown>;

        // Config should be preserved (not auto-injected since already set)
        expect(result['thinking']).toEqual({ type: 'enabled' });
        // output_config should still be auto-injected since it's undefined
        expect(result['output_config']).toEqual({ effort: 'high' });
      });
    });
  });
});
