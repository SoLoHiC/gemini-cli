/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type OpenAI from 'openai';
import { DeepSeekOpenAICompatibleProvider } from './deepseek.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import type { Config } from '../../config/config.js';

// Mock OpenAI client to avoid real network calls
vi.mock('openai', () => ({
  default: vi.fn().mockImplementation((config) => ({
    config,
  })),
}));

describe('DeepSeekOpenAICompatibleProvider', () => {
  let provider: DeepSeekOpenAICompatibleProvider;
  let mockContentGeneratorConfig: ContentGeneratorConfig;
  let mockCliConfig: Config;

  beforeEach(() => {
    vi.clearAllMocks();

    mockContentGeneratorConfig = {
      apiKey: 'test-api-key',
      baseUrl: 'https://api.deepseek.com/v1',
      model: 'deepseek-chat',
    } as ContentGeneratorConfig;

    mockCliConfig = {
      getCliVersion: vi.fn().mockReturnValue('1.0.0'),
    } as unknown as Config;

    provider = new DeepSeekOpenAICompatibleProvider(
      mockContentGeneratorConfig,
      mockCliConfig,
    );
  });

  describe('isDeepSeekProvider', () => {
    it('returns true when baseUrl includes deepseek', () => {
      const result = DeepSeekOpenAICompatibleProvider.isDeepSeekProvider(
        mockContentGeneratorConfig,
      );
      expect(result).toBe(true);
    });

    it('returns false for non deepseek baseUrl', () => {
      const config = {
        ...mockContentGeneratorConfig,
        baseUrl: 'https://api.example.com/v1',
      } as ContentGeneratorConfig;

      const result =
        DeepSeekOpenAICompatibleProvider.isDeepSeekProvider(config);
      expect(result).toBe(false);
    });
  });

  describe('buildRequest', () => {
    const userPromptId = 'prompt-123';

    it('converts array content into a string', () => {
      const originalRequest: OpenAI.Chat.ChatCompletionCreateParams = {
        model: 'deepseek-chat',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Hello' },
              { type: 'text', text: ' world' },
            ],
          },
        ],
      };

      const result = provider.buildRequest(originalRequest, userPromptId);

      expect(result.messages).toHaveLength(1);
      expect(result.messages?.[0]).toEqual({
        role: 'user',
        content: 'Hello world',
      });
      expect(originalRequest.messages?.[0].content).toEqual([
        { type: 'text', text: 'Hello' },
        { type: 'text', text: ' world' },
      ]);
    });

    it('leaves string content unchanged', () => {
      const originalRequest: OpenAI.Chat.ChatCompletionCreateParams = {
        model: 'deepseek-chat',
        messages: [
          {
            role: 'user',
            content: 'Hello world',
          },
        ],
      };

      const result = provider.buildRequest(originalRequest, userPromptId);

      expect(result.messages?.[0].content).toBe('Hello world');
    });

    it('throws when encountering non-text multimodal parts', () => {
      const originalRequest: OpenAI.Chat.ChatCompletionCreateParams = {
        model: 'deepseek-chat',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Hello' },
              {
                type: 'image_url',
                image_url: { url: 'https://example.com/image.png' },
              },
            ],
          },
        ],
      };

      expect(() =>
        provider.buildRequest(originalRequest, userPromptId),
      ).toThrow(/only supports text content/i);
    });

    it('clamps max_tokens to the DeepSeek limit', () => {
      const originalRequest: OpenAI.Chat.ChatCompletionCreateParams = {
        model: 'deepseek-chat',
        max_tokens: 16000,
        messages: [
          {
            role: 'user',
            content: 'Hello world',
          },
        ],
      };

      const result = provider.buildRequest(originalRequest, userPromptId);

      expect(result.max_tokens).toBe(8192);
    });

    describe('thinking mode', () => {
      it('auto-injects thinking mode and reasoning_effort for V4 models', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams = {
          model: 'deepseek-v4-pro',
          messages: [{ role: 'user', content: 'Hello' }],
        };

        const result = provider.buildRequest(
          request,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown>;

        expect(result['extra_body']).toEqual({
          thinking: { type: 'enabled' },
        });
        expect(result['reasoning_effort']).toBe('high');
      });

      it('does not override user-specified thinking config for V4 models', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-v4-pro',
          messages: [{ role: 'user', content: 'Hello' }],
          extra_body: { thinking: { type: 'enabled' }, other: 'preserved' },
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown>;

        expect(result['extra_body']).toEqual({
          thinking: { type: 'enabled' },
          other: 'preserved',
        });
      });

      it('does not override user-specified reasoning_effort for V4 models', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-v4-pro',
          messages: [{ role: 'user', content: 'Hello' }],
          reasoning_effort: 'low',
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown>;

        expect(result['reasoning_effort']).toBe('low');
      });

      it('does not auto-inject reasoning_effort when thinking is explicitly disabled for V4 models', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-v4-flash',
          messages: [{ role: 'user', content: 'Hello' }],
          extra_body: { thinking: { type: 'disabled' } },
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown>;

        expect(result['extra_body']).toEqual({
          thinking: { type: 'disabled' },
        });
        expect(result['reasoning_effort']).toBeUndefined();
      });

      it('does not auto-inject thinking for non-V4 models', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-chat',
          messages: [{ role: 'user', content: 'Hello' }],
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown>;

        expect(result['extra_body']).toBeUndefined();
        expect(result['reasoning_effort']).toBeUndefined();
      });

      it('ensures reasoning_content on assistant messages when thinking is enabled', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-v4-pro',
          messages: [
            { role: 'user', content: 'Hello' },
            {
              role: 'assistant',
              content: 'Hi there',
            } as OpenAI.Chat.ChatCompletionAssistantMessageParam &
              Record<string, unknown>,
          ],
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams;

        const assistantMsg = result.messages?.[1] as unknown as Record<string, unknown>;
        expect(assistantMsg['reasoning_content']).toBe('');
      });

      it('preserves existing reasoning_content on assistant messages when thinking is enabled', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-v4-pro',
          messages: [
            { role: 'user', content: 'Hello' },
            {
              role: 'assistant',
              content: 'Hi there',
              reasoning_content: 'existing thought',
            } as OpenAI.Chat.ChatCompletionAssistantMessageParam &
              Record<string, unknown>,
          ],
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams;

        const assistantMsg = result.messages?.[1] as unknown as Record<string, unknown>;
        expect(assistantMsg['reasoning_content']).toBe('existing thought');
      });

      it('strips reasoning_content from assistant messages when thinking is disabled', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-v4-flash',
          messages: [
            { role: 'user', content: 'Hello' },
            {
              role: 'assistant',
              content: 'Hi there',
              reasoning_content: 'previous thought',
            } as OpenAI.Chat.ChatCompletionAssistantMessageParam &
              Record<string, unknown>,
          ],
          extra_body: { thinking: { type: 'disabled' } },
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams;

        const assistantMsg = result.messages?.[1] as unknown as Record<string, unknown>;
        expect(assistantMsg['reasoning_content']).toBeUndefined();
      });

      it('does not modify reasoning_content when thinking is not explicitly configured', () => {
        const request: OpenAI.Chat.ChatCompletionCreateParams &
          Record<string, unknown> = {
          model: 'deepseek-chat',
          messages: [
            { role: 'user', content: 'Hello' },
            {
              role: 'assistant',
              content: 'Hi there',
              reasoning_content: 'some thought',
            } as OpenAI.Chat.ChatCompletionAssistantMessageParam &
              Record<string, unknown>,
          ],
        };

        const result = provider.buildRequest(
          request as OpenAI.Chat.ChatCompletionCreateParams,
          userPromptId,
        ) as OpenAI.Chat.ChatCompletionCreateParams;

        const assistantMsg = result.messages?.[1] as unknown as Record<string, unknown>;
        // For non-thinking models, reasoning_content is left intact
        // (the API will handle it)
        expect(assistantMsg['reasoning_content']).toBe('some thought');
      });
    });
  });
});
