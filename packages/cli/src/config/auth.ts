/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  AuthType,
  isAnthropicAuthType,
  isOpenAIAuthType,
  type ModelProviderConfig,
} from '@google/gemini-cli-core';
import {
  loadEnvironment,
  loadSettings,
  type MergedSettings,
} from './settings.js';

function getProviderConfigForModel(
  settings: Pick<MergedSettings, 'modelProviders' | 'model'>,
  authType: AuthType,
  modelId?: string,
): ModelProviderConfig | undefined {
  const providerKey = isOpenAIAuthType(authType)
    ? 'openai'
    : isAnthropicAuthType(authType)
      ? 'anthropic'
      : undefined;

  if (!providerKey) {
    return undefined;
  }

  const targetModel = modelId ?? settings.model?.name;
  if (!targetModel) {
    return undefined;
  }

  return settings.modelProviders?.[providerKey]?.find(
    (entry) => entry.id === targetModel,
  );
}

function isOfficialProviderBaseUrl(
  authType: AuthType,
  baseUrl: string | undefined,
): boolean {
  if (!baseUrl) {
    return true;
  }

  const normalizedBaseUrl = baseUrl.toLowerCase().replace(/\/+$/, '');
  return isOpenAIAuthType(authType)
    ? normalizedBaseUrl.includes('api.openai.com')
    : normalizedBaseUrl.includes('api.anthropic.com');
}

export function resolveProviderApiKeyForModel(
  settings: Pick<MergedSettings, 'modelProviders' | 'model' | 'security'>,
  authType: AuthType,
  modelId?: string,
): string | undefined {
  if (!isOpenAIAuthType(authType) && !isAnthropicAuthType(authType)) {
    return undefined;
  }

  const providerConfig = getProviderConfigForModel(settings, authType, modelId);
  const defaultEnvKey = isOpenAIAuthType(authType)
    ? 'OPENAI_API_KEY'
    : 'ANTHROPIC_API_KEY';

  if (providerConfig?.envKey && process.env[providerConfig.envKey]) {
    return process.env[providerConfig.envKey];
  }

  if (
    isOfficialProviderBaseUrl(authType, providerConfig?.baseUrl) &&
    process.env[defaultEnvKey]
  ) {
    return process.env[defaultEnvKey];
  }

  return settings.security.auth.apiKey;
}

export function resolveProviderBaseUrlForModel(
  settings: Pick<MergedSettings, 'modelProviders' | 'model'>,
  authType: AuthType,
  modelId?: string,
): string | undefined {
  if (!isOpenAIAuthType(authType) && !isAnthropicAuthType(authType)) {
    return undefined;
  }

  return getProviderConfigForModel(settings, authType, modelId)?.baseUrl;
}

export function parseAuthType(value: string | undefined): AuthType | undefined {
  return Object.values(AuthType).find((authType) => authType === value);
}

export function validateAuthMethod(authMethod: string): string | null {
  const loadedSettings = loadSettings();
  loadEnvironment(loadedSettings.merged, process.cwd());
  const settings = loadedSettings.merged;
  const authMethodType = parseAuthType(authMethod);
  if (
    authMethodType === AuthType.LOGIN_WITH_GOOGLE ||
    authMethodType === AuthType.COMPUTE_ADC
  ) {
    return null;
  }

  if (authMethodType === AuthType.USE_GEMINI) {
    if (!process.env['GEMINI_API_KEY']) {
      return (
        'When using Gemini API, you must specify the GEMINI_API_KEY environment variable.\n' +
        'Update your environment and try again (no reload needed if using .env)!'
      );
    }
    return null;
  }

  if (authMethodType === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!process.env['GOOGLE_CLOUD_PROJECT'] &&
      !!process.env['GOOGLE_CLOUD_LOCATION'];
    const hasGoogleApiKey = !!process.env['GOOGLE_API_KEY'];
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return (
        'When using Vertex AI, you must specify either:\n' +
        '• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.\n' +
        '• GOOGLE_API_KEY environment variable (if using express mode).\n' +
        'Update your environment and try again (no reload needed if using .env)!'
      );
    }
    return null;
  }

  if (
    authMethodType &&
    (isOpenAIAuthType(authMethodType) || isAnthropicAuthType(authMethodType))
  ) {
    const providerKey = isOpenAIAuthType(authMethodType)
      ? 'openai'
      : 'anthropic';
    const modelEnvKey = isOpenAIAuthType(authMethodType)
      ? 'OPENAI_MODEL'
      : 'ANTHROPIC_MODEL';
    const apiKeyEnvKey = isOpenAIAuthType(authMethodType)
      ? 'OPENAI_API_KEY'
      : 'ANTHROPIC_API_KEY';
    const configuredProviderModels =
      settings.modelProviders?.[providerKey] ?? [];
    const selectedModel = settings.model?.name;
    const selectedProviderModel = configuredProviderModels.find(
      (entry) => entry.id === selectedModel,
    );
    const configuredApiKey =
      // 优先使用提供者特定的环境变量（用于第三方提供者如DeepSeek）
      (selectedProviderModel?.envKey
        ? process.env[selectedProviderModel.envKey]
        : undefined) ||
      settings.security.auth.apiKey ||
      process.env[apiKeyEnvKey];
    const configuredModel =
      process.env[modelEnvKey] ||
      settings.model?.name ||
      configuredProviderModels[0]?.id;

    if (!configuredApiKey) {
      return (
        `When using ${providerKey}, you must specify either:\n` +
        `• security.auth.apiKey in settings.\n` +
        `• ${apiKeyEnvKey} in your environment.\n` +
        `• A provider-specific envKey on the selected model provider.\n`
      );
    }

    if (!configuredModel) {
      return `When using ${providerKey}, you must specify a model via --model, ${modelEnvKey}, settings.model.name, or modelProviders.${providerKey}.`;
    }

    return null;
  }

  return 'Invalid auth method selected.';
}
