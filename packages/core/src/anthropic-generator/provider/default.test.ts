/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import Anthropic from '@anthropic-ai/sdk';
import { DefaultAnthropicCompatibleProvider } from './default.js';
import type { ContentGeneratorConfig } from '../../core/contentGenerator.js';
import type { Config } from '../../config/config.js';

vi.mock('@anthropic-ai/sdk', () => ({
  default: vi.fn().mockImplementation((config) => ({
    config,
  })),
}));

describe('DefaultAnthropicCompatibleProvider', () => {
  let provider: DefaultAnthropicCompatibleProvider;

  beforeEach(() => {
    provider = new DefaultAnthropicCompatibleProvider(
      {
        apiKey: 'test-key',
        baseUrl: 'https://api.anthropic.com/v1',
        model: 'claude-sonnet-4-6',
      } as ContentGeneratorConfig,
      {
        getCliVersion: vi.fn().mockReturnValue('1.0.0'),
      } as unknown as Config,
    );
  });

  it('builds an Anthropic client', () => {
    provider.buildClient();

    expect(Anthropic).toHaveBeenCalled();
  });

  it('clamps max_tokens to the model output limit', () => {
    const request: Anthropic.Messages.MessageCreateParams = {
      model: 'claude-sonnet-4-6',
      max_tokens: 80_000,
      messages: [{ role: 'user', content: 'Hello' }],
    };

    const result = provider.buildRequest(request, 'prompt-id');

    expect(result.max_tokens).toBe(65_536);
  });
});
