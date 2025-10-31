/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Config } from '../config/config.js';
import {
  DEFAULT_INPUT_TOKEN_LIMIT,
  DEFAULT_OUTPUT_TOKEN_LIMIT,
  getModelTokenMetadata,
  normalizeModelId,
  type ModelTokenMetadata,
  type TokenLimitType,
} from './tokenMetadata.js';

type Model = string;
type TokenCount = number;

export const DEFAULT_TOKEN_LIMIT = DEFAULT_INPUT_TOKEN_LIMIT;
export { DEFAULT_OUTPUT_TOKEN_LIMIT, normalizeModelId, getModelTokenMetadata };
export type { ModelTokenMetadata, TokenLimitType };

function parseArgs(
  configOrType?: Config | TokenLimitType,
  maybeType?: TokenLimitType,
): { config: Config | undefined; type: TokenLimitType } {
  if (configOrType === 'input' || configOrType === 'output') {
    return {
      config: undefined,
      type: configOrType,
    };
  }

  return {
    config: configOrType,
    type: maybeType ?? 'input',
  };
}

export function tokenLimit(model: Model, type: TokenLimitType): TokenCount;
export function tokenLimit(model: Model, config?: Config): TokenCount;
export function tokenLimit(
  model: Model,
  config: Config | undefined,
  type: TokenLimitType,
): TokenCount;
export function tokenLimit(
  model: Model,
  configOrType?: Config | TokenLimitType,
  maybeType?: TokenLimitType,
): TokenCount {
  const { config, type } = parseArgs(configOrType, maybeType);
  const metadata = getModelTokenMetadata(model, config);

  return type === 'output'
    ? metadata.outputTokenLimit
    : metadata.inputTokenLimit;
}
