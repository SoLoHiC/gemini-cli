/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Config,
  ModelProviderConfig,
  ModelProvidersConfig,
} from '../config/config.js';
import { AuthType } from './contentGenerator.js';

export type TokenLimitType = 'input' | 'output';
export type TokenMetadataSource = 'built-in' | 'config override' | 'inferred';
export type ProviderFamily =
  | 'google'
  | 'openai'
  | 'anthropic'
  | 'qwen'
  | 'deepseek'
  | 'glm'
  | 'minimax'
  | 'kimi'
  | 'seed'
  | 'unknown';

export interface ModelTokenMetadata {
  normalizedModel: string;
  providerFamily: ProviderFamily;
  modelFamily: string;
  inputTokenLimit: number;
  outputTokenLimit: number;
  source: TokenMetadataSource;
}

export const DEFAULT_INPUT_TOKEN_LIMIT = 131_072;
export const DEFAULT_OUTPUT_TOKEN_LIMIT = 8_192;

interface BuiltInTokenMetadataPattern {
  match: RegExp;
  providerFamily: ProviderFamily;
  modelFamily: string;
  inputTokenLimit: number;
  outputTokenLimit: number;
}

const BUILT_IN_PATTERNS: BuiltInTokenMetadataPattern[] = [
  {
    match: /^gemini-3/,
    providerFamily: 'google',
    modelFamily: 'gemini',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 65_536,
  },
  {
    match: /^gemini-/,
    providerFamily: 'google',
    modelFamily: 'gemini',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 8_192,
  },
  {
    match: /^gpt-5/,
    providerFamily: 'openai',
    modelFamily: 'gpt',
    inputTokenLimit: 400_000,
    outputTokenLimit: 131_072,
  },
  {
    match: /^gpt-/,
    providerFamily: 'openai',
    modelFamily: 'gpt',
    inputTokenLimit: 131_072,
    outputTokenLimit: 16_384,
  },
  {
    match: /^o\d/,
    providerFamily: 'openai',
    modelFamily: 'o-series',
    inputTokenLimit: 200_000,
    outputTokenLimit: 131_072,
  },
  {
    match: /^claude-opus-4-6/,
    providerFamily: 'anthropic',
    modelFamily: 'claude',
    inputTokenLimit: 200_000,
    outputTokenLimit: 131_072,
  },
  {
    match: /^claude-sonnet-4-6/,
    providerFamily: 'anthropic',
    modelFamily: 'claude',
    inputTokenLimit: 200_000,
    outputTokenLimit: 65_536,
  },
  {
    match: /^claude-/,
    providerFamily: 'anthropic',
    modelFamily: 'claude',
    inputTokenLimit: 200_000,
    outputTokenLimit: 65_536,
  },
  {
    match: /^qwen3-coder-plus/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 8_192,
  },
  {
    match: /^qwen3-coder-flash/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 8_192,
  },
  {
    match: /^qwen3\.5-plus/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 65_536,
  },
  {
    match: /^coder-model$/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 65_536,
  },
  {
    match: /^qwen3-max/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 262_144,
    outputTokenLimit: 65_536,
  },
  {
    match: /^qwen3-vl-plus/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 262_144,
    outputTokenLimit: 8_192,
  },
  {
    match: /^qwen-vl-max-latest$/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 262_144,
    outputTokenLimit: 8_192,
  },
  {
    match: /^qwen3-coder-/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 262_144,
    outputTokenLimit: 8_192,
  },
  {
    match: /^qwen-(plus|flash)-latest$/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 65_536,
  },
  {
    match: /^qwen/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 262_144,
    outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
  },
  {
    match: /^deepseek-v4-(flash|pro)/,
    providerFamily: 'deepseek',
    modelFamily: 'deepseek',
    inputTokenLimit: 1_048_576,
    outputTokenLimit: 393_216,
  },
  {
    match: /^deepseek-(reasoner|r1)/,
    providerFamily: 'deepseek',
    modelFamily: 'deepseek',
    inputTokenLimit: 1_048_576,
    outputTokenLimit: 393_216,
  },
  {
    match: /^deepseek-chat$/,
    providerFamily: 'deepseek',
    modelFamily: 'deepseek',
    inputTokenLimit: 1_048_576,
    outputTokenLimit: 393_216,
  },
  {
    match: /^deepseek/,
    providerFamily: 'deepseek',
    modelFamily: 'deepseek',
    inputTokenLimit: 1_048_576,
    outputTokenLimit: 393_216,
  },
  {
    match: /^glm-5/,
    providerFamily: 'glm',
    modelFamily: 'glm',
    inputTokenLimit: 202_752,
    outputTokenLimit: 16_384,
  },
  {
    match: /^glm-4\.7/,
    providerFamily: 'glm',
    modelFamily: 'glm',
    inputTokenLimit: 202_752,
    outputTokenLimit: 16_384,
  },
  {
    match: /^glm-/,
    providerFamily: 'glm',
    modelFamily: 'glm',
    inputTokenLimit: 202_752,
    outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
  },
  {
    match: /^minimax-m2\.5/i,
    providerFamily: 'minimax',
    modelFamily: 'minimax',
    inputTokenLimit: 1_000_000,
    outputTokenLimit: 65_536,
  },
  {
    match: /^minimax-/i,
    providerFamily: 'minimax',
    modelFamily: 'minimax',
    inputTokenLimit: 200_000,
    outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
  },
  {
    match: /^kimi-k2\.5/,
    providerFamily: 'kimi',
    modelFamily: 'kimi',
    inputTokenLimit: 262_144,
    outputTokenLimit: 32_768,
  },
  {
    match: /^kimi-/,
    providerFamily: 'kimi',
    modelFamily: 'kimi',
    inputTokenLimit: 262_144,
    outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
  },
  {
    match: /^seed-oss/,
    providerFamily: 'seed',
    modelFamily: 'seed',
    inputTokenLimit: 524_288,
    outputTokenLimit: DEFAULT_OUTPUT_TOKEN_LIMIT,
  },
  {
    match: /^vision-model$/,
    providerFamily: 'qwen',
    modelFamily: 'qwen',
    inputTokenLimit: 262_144,
    outputTokenLimit: 8_192,
  },
];

function authTypeToProviderFamily(
  authType?: AuthType,
): ProviderFamily | undefined {
  switch (authType) {
    case 'gemini-api-key':
    case 'vertex-ai':
    case 'oauth-personal':
    case 'cloud-shell':
    case 'compute-default-credentials':
      return 'google';
    case 'openai':
      return 'openai';
    case 'anthropic':
      return 'anthropic';
    default:
      return undefined;
  }
}

function getActiveAuthType(config?: Config): AuthType | undefined {
  return config?.getContentGeneratorConfig?.()?.authType;
}

function findModelInProviders(
  modelId: string,
  modelProviders: ModelProvidersConfig | undefined,
  authType: AuthType,
): ModelProviderConfig | undefined {
  if (!modelProviders) {
    return undefined;
  }

  // 根据 authType 确定 provider key
  let providerKey: keyof ModelProvidersConfig | undefined;
  if (authType === AuthType.USE_OPENAI) {
    providerKey = 'openai';
  } else if (authType === AuthType.USE_ANTHROPIC) {
    providerKey = 'anthropic';
  }

  if (!providerKey) {
    return undefined;
  }

  const providerEntries = modelProviders[providerKey];
  if (!Array.isArray(providerEntries)) {
    return undefined;
  }

  // 首先尝试精确匹配
  const exactMatch = providerEntries.find((entry) => entry.id === modelId);
  if (exactMatch) {
    return exactMatch;
  }

  // 然后尝试规范化匹配
  const normalizedId = normalizeModelId(modelId);
  return providerEntries.find(
    (entry) => normalizeModelId(entry.id) === normalizedId,
  );
}

function findCompatibleModelOverride(
  model: string,
  config?: Config,
): ModelProviderConfig | undefined {
  const modelProvidersConfig = config?.getModelProvidersConfig?.();
  const activeAuthType = getActiveAuthType(config);

  if (!modelProvidersConfig || !activeAuthType) {
    return undefined;
  }

  return findModelInProviders(model, modelProvidersConfig, activeAuthType);
}

export function normalizeModelId(model: string | undefined): string {
  let normalized = (model ?? '').toLowerCase().trim();

  normalized = normalized.replace(/^.*\//, '');
  normalized = normalized.split('|').pop() ?? normalized;
  normalized = normalized.split(':').pop() ?? normalized;
  normalized = normalized.replace(/\s+/g, '-');
  normalized = normalized.replace(/-preview/g, '');
  normalized = normalized.replace(/-(?:\d{4}-\d{2}-\d{2}|\d{8})$/g, '');
  normalized = normalized.replace(
    /-(?:\d?bit|int[48]|bf16|fp16|q[45]|quantized)$/g,
    '',
  );
  normalized = normalized.replace(/-(?:\d{4}-\d{2}-\d{2}|\d{8})$/g, '');

  if (
    !normalized.match(/^qwen-(?:plus|flash|vl-max)-latest$/) &&
    !normalized.match(/^kimi-k2-\d{4}$/)
  ) {
    normalized = normalized.replace(
      /-(?:\d{4,}|\d+x\d+b|v\d+(?:\.\d+)*|(?<=-[^-]+-)\d+(?:\.\d+)+|latest|exp)$/g,
      '',
    );
  }

  return normalized;
}

export function getModelTokenMetadata(
  model: string | undefined,
  config?: Config,
): ModelTokenMetadata {
  const normalizedModel = normalizeModelId(model);
  const builtInMatch = BUILT_IN_PATTERNS.find((pattern) =>
    pattern.match.test(normalizedModel),
  );
  const override = model
    ? findCompatibleModelOverride(model, config)
    : undefined;

  // 从 config 获取 authType，而不是从 override
  const activeAuthType = config?.getContentGeneratorConfig?.()?.authType;
  const providerFamilyFromAuth = authTypeToProviderFamily(activeAuthType);

  const metadata: ModelTokenMetadata = {
    normalizedModel,
    providerFamily:
      builtInMatch?.providerFamily ?? providerFamilyFromAuth ?? 'unknown',
    modelFamily: builtInMatch?.modelFamily ?? (normalizedModel || 'unknown'),
    inputTokenLimit: builtInMatch?.inputTokenLimit ?? DEFAULT_INPUT_TOKEN_LIMIT,
    outputTokenLimit:
      builtInMatch?.outputTokenLimit ?? DEFAULT_OUTPUT_TOKEN_LIMIT,
    source: builtInMatch ? 'built-in' : 'inferred',
  };

  const overrideInputLimit =
    override?.tokenLimit ?? override?.generationConfig?.contextWindowSize;
  if (overrideInputLimit !== undefined) {
    metadata.inputTokenLimit = overrideInputLimit;
    metadata.source = 'config override';
  }

  return metadata;
}
