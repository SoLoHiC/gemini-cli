/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import { AuthType } from './contentGenerator.js';
import {
  DEFAULT_INPUT_TOKEN_LIMIT,
  DEFAULT_OUTPUT_TOKEN_LIMIT,
  getModelTokenMetadata,
  normalizeModelId,
} from './tokenMetadata.js';
import type { Config, ModelProvidersConfig } from '../config/config.js';

function createConfig(
  modelProvidersConfig: ModelProvidersConfig,
  authType?: AuthType,
): Config {
  return {
    getModelProvidersConfig: () => modelProvidersConfig,
    getContentGeneratorConfig: () => ({
      authType,
    }),
  } as unknown as Config;
}

describe('normalizeModelId', () => {
  it('normalizes provider prefixes, casing, and version suffixes', () => {
    expect(
      normalizeModelId('  provider/openai|GPT-4O:gpt-4o-2024-05-13-q4  '),
    ).toBe('gpt-4o');
  });

  it('preserves provider-specific suffixes that are part of the model id', () => {
    expect(normalizeModelId('qwen-plus-latest')).toBe('qwen-plus-latest');
    expect(normalizeModelId('kimi-k2-0905')).toBe('kimi-k2-0905');
  });
});

describe('getModelTokenMetadata', () => {
  it('returns built-in metadata for known model families', () => {
    expect(getModelTokenMetadata('gpt-5')).toMatchObject({
      providerFamily: 'openai',
      modelFamily: 'gpt',
      inputTokenLimit: 400_000,
      outputTokenLimit: 131_072,
      source: 'built-in',
    });

    expect(getModelTokenMetadata('claude-sonnet-4-6')).toMatchObject({
      providerFamily: 'anthropic',
      modelFamily: 'claude',
      inputTokenLimit: 200_000,
      outputTokenLimit: 65_536,
      source: 'built-in',
    });

    expect(getModelTokenMetadata('deepseek-chat')).toMatchObject({
      providerFamily: 'deepseek',
      modelFamily: 'deepseek',
      inputTokenLimit: 131_072,
      outputTokenLimit: 8_192,
      source: 'built-in',
    });
  });

  it('covers the built-in multi-vendor registry families', () => {
    expect(getModelTokenMetadata('qwen3-coder-plus')).toMatchObject({
      providerFamily: 'qwen',
      modelFamily: 'qwen',
      inputTokenLimit: 1_000_000,
      outputTokenLimit: 8_192,
    });
    expect(getModelTokenMetadata('glm-5-air')).toMatchObject({
      providerFamily: 'glm',
      modelFamily: 'glm',
      inputTokenLimit: 202_752,
      outputTokenLimit: 16_384,
    });
    expect(getModelTokenMetadata('minimax-m2.5-chat')).toMatchObject({
      providerFamily: 'minimax',
      modelFamily: 'minimax',
      inputTokenLimit: 1_000_000,
      outputTokenLimit: 65_536,
    });
    expect(getModelTokenMetadata('kimi-k2.5-instruct')).toMatchObject({
      providerFamily: 'kimi',
      modelFamily: 'kimi',
      inputTokenLimit: 262_144,
      outputTokenLimit: 32_768,
    });
    expect(getModelTokenMetadata('seed-oss-32b-instruct')).toMatchObject({
      providerFamily: 'seed',
      modelFamily: 'seed',
      inputTokenLimit: 524_288,
      outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
    });
  });

  it('returns inferred defaults for unknown models', () => {
    expect(getModelTokenMetadata('mistral-large-2')).toMatchObject({
      providerFamily: 'unknown',
      modelFamily: 'mistral-large-2',
      inputTokenLimit: DEFAULT_INPUT_TOKEN_LIMIT,
      outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
      source: 'inferred',
    });
  });

  it('applies config tokenLimit overrides before built-in limits', () => {
    const config = createConfig(
      {
        openai: [
          {
            id: 'deepseek-chat',
            tokenLimit: 262_144,
          },
        ],
      },
      AuthType.USE_OPENAI,
    );

    expect(getModelTokenMetadata('deepseek-chat', config)).toMatchObject({
      providerFamily: 'deepseek',
      inputTokenLimit: 262_144,
      outputTokenLimit: 8_192,
      source: 'config override',
    });
  });

  it('uses contextWindowSize when tokenLimit is not explicitly set', () => {
    const config = createConfig(
      {
        anthropic: [
          {
            id: 'custom-model',
            generationConfig: {
              contextWindowSize: 196_608,
            },
          },
        ],
      },
      AuthType.USE_ANTHROPIC,
    );

    expect(getModelTokenMetadata('custom-model', config)).toMatchObject({
      providerFamily: 'anthropic',
      modelFamily: 'custom-model',
      inputTokenLimit: 196_608,
      outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
      source: 'config override',
    });
  });

  it('prefers the override whose auth matches the active provider', () => {
    const config = createConfig(
      {
        openai: [
          {
            id: 'deepseek-chat',
            tokenLimit: 131_072,
          },
        ],
        anthropic: [
          {
            id: 'deepseek-chat',
            tokenLimit: 196_608,
          },
        ],
      },
      AuthType.USE_ANTHROPIC,
    );

    expect(getModelTokenMetadata('deepseek-chat', config)).toMatchObject({
      providerFamily: 'deepseek',
      inputTokenLimit: 196_608,
      outputTokenLimit: 8_192,
      source: 'config override',
    });
  });
});
