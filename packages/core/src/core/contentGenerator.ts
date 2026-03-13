/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GoogleGenAI,
  type CountTokensResponse,
  type GenerateContentResponse,
  type GenerateContentParameters,
  type CountTokensParameters,
  type EmbedContentResponse,
  type EmbedContentParameters,
} from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import type { Config } from '../config/config.js';
import { loadApiKey } from './apiKeyCredentialStorage.js';

import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import { LoggingContentGenerator } from './loggingContentGenerator.js';
import { InstallationManager } from '../utils/installationManager.js';
import { FakeContentGenerator } from './fakeContentGenerator.js';
import { parseCustomHeaders } from '../utils/customHeaderUtils.js';
import { RecordingContentGenerator } from './recordingContentGenerator.js';
import { getVersion, resolveModel } from '../../index.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import { createOpenAIContentGenerator } from '../openai-generator/index.js';
import { AnthropicContentGenerator } from '../anthropic-generator/anthropicContentGenerator.js';
import { createAnthropicCompatibleProvider } from '../anthropic-generator/index.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  userTier?: UserTierId;

  userTierName?: string;

  paidTier?: GeminiUserTier;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  LEGACY_CLOUD_SHELL = 'cloud-shell',
  COMPUTE_ADC = 'compute-default-credentials',
  USE_OPENAI = 'openai',
  USE_ANTHROPIC = 'anthropic',
  OPENAI_COMPATIBLE = 'openai-compatible',
  ANTHROPIC_COMPATIBLE = 'anthropic-compatible',
}

export function normalizeAuthType(authType?: AuthType): AuthType | undefined {
  switch (authType) {
    case AuthType.OPENAI_COMPATIBLE:
      return AuthType.USE_OPENAI;
    case AuthType.ANTHROPIC_COMPATIBLE:
      return AuthType.USE_ANTHROPIC;
    default:
      return authType;
  }
}

export function isGoogleAuthType(authType?: AuthType): boolean {
  const normalized = normalizeAuthType(authType);
  return (
    normalized === AuthType.LOGIN_WITH_GOOGLE ||
    normalized === AuthType.USE_GEMINI ||
    normalized === AuthType.USE_VERTEX_AI ||
    normalized === AuthType.COMPUTE_ADC ||
    normalized === AuthType.LEGACY_CLOUD_SHELL
  );
}

export function isOpenAIAuthType(authType?: AuthType): boolean {
  return normalizeAuthType(authType) === AuthType.USE_OPENAI;
}

export function isAnthropicAuthType(authType?: AuthType): boolean {
  return normalizeAuthType(authType) === AuthType.USE_ANTHROPIC;
}

export function isProviderAuthType(authType?: AuthType): boolean {
  return isOpenAIAuthType(authType) || isAnthropicAuthType(authType);
}

/**
 * Detects the best authentication type based on environment variables.
 *
 * Checks in order:
 * 1. GOOGLE_GENAI_USE_GCA=true -> LOGIN_WITH_GOOGLE
 * 2. GOOGLE_GENAI_USE_VERTEXAI=true -> USE_VERTEX_AI
 * 3. GEMINI_API_KEY -> USE_GEMINI
 */
export function getAuthTypeFromEnv(): AuthType | undefined {
  if (process.env['GOOGLE_GENAI_USE_GCA'] === 'true') {
    return AuthType.LOGIN_WITH_GOOGLE;
  }
  if (process.env['GOOGLE_GENAI_USE_VERTEXAI'] === 'true') {
    return AuthType.USE_VERTEX_AI;
  }
  if (process.env['GEMINI_API_KEY']) {
    return AuthType.USE_GEMINI;
  }
  if (process.env['ANTHROPIC_API_KEY']) {
    return AuthType.USE_ANTHROPIC;
  }
  if (process.env['OPENAI_API_KEY']) {
    return AuthType.USE_OPENAI;
  }
  if (
    process.env['CLOUD_SHELL'] === 'true' ||
    process.env['GEMINI_CLI_USE_COMPUTE_ADC'] === 'true'
  ) {
    return AuthType.COMPUTE_ADC;
  }
  return undefined;
}

export type ContentGeneratorConfig = {
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType;
  proxy?: string;
  model?: string;
  baseUrl?: string;
  apiKeyEnvKey?: string;
  timeout?: number;
  maxRetries?: number;
  retryErrorCodes?: number[];
  enableCacheControl?: boolean;
  enableOpenAILogging?: boolean;
  disableCacheControl?: boolean;
  reasoning?: Record<string, unknown>;
  schemaCompliance?: Record<string, unknown>;
  contextWindowSize?: number;
  customHeaders?: Record<string, string>;
  extra_body?: Record<string, unknown>;
  modalities?: string[];
  providerSubtype?: string;
  embeddingModel?: string;
  samplingParams?: {
    top_p?: number;
    top_k?: number;
    repetition_penalty?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    temperature?: number;
    max_tokens?: number;
  };
};

const OPENAI_DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const ANTHROPIC_DEFAULT_BASE_URL = 'https://api.anthropic.com/v1';

type ProviderDefaults = {
  apiKeyEnvKey: string;
  modelEnvKey: string;
  baseUrlEnvKey: string;
  defaultBaseUrl: string;
};

function getProviderDefaults(authType: AuthType): ProviderDefaults {
  if (authType === AuthType.USE_ANTHROPIC) {
    return {
      apiKeyEnvKey: 'ANTHROPIC_API_KEY',
      modelEnvKey: 'ANTHROPIC_MODEL',
      baseUrlEnvKey: 'ANTHROPIC_BASE_URL',
      defaultBaseUrl: ANTHROPIC_DEFAULT_BASE_URL,
    };
  }

  return {
    apiKeyEnvKey: 'OPENAI_API_KEY',
    modelEnvKey: 'OPENAI_MODEL',
    baseUrlEnvKey: 'OPENAI_BASE_URL',
    defaultBaseUrl: OPENAI_DEFAULT_BASE_URL,
  };
}

function resolveProviderSubtype(
  authType: AuthType,
  baseUrl?: string,
): string {
  const normalizedBaseUrl = (baseUrl || '').toLowerCase();

  if (authType === AuthType.USE_OPENAI) {
    if (normalizedBaseUrl.includes('openrouter.ai')) {
      return 'openrouter';
    }
    if (normalizedBaseUrl.includes('api.deepseek.com')) {
      return 'deepseek-openai';
    }
    return 'default-openai';
  }

  if (normalizedBaseUrl.includes('api.deepseek.com')) {
    return 'deepseek-anthropic';
  }

  return 'default-anthropic';
}

export async function createContentGeneratorConfig(
  config: Config,
  authType: AuthType | undefined,
  apiKey?: string,
): Promise<ContentGeneratorConfig> {
  const normalizedAuthType = normalizeAuthType(authType);
  const geminiApiKey =
    apiKey ||
    process.env['GEMINI_API_KEY'] ||
    (await loadApiKey()) ||
    undefined;
  const googleApiKey = process.env['GOOGLE_API_KEY'] || undefined;
  const googleCloudProject =
    process.env['GOOGLE_CLOUD_PROJECT'] ||
    process.env['GOOGLE_CLOUD_PROJECT_ID'] ||
    undefined;
  const googleCloudLocation = process.env['GOOGLE_CLOUD_LOCATION'] || undefined;

  if (normalizedAuthType && isProviderAuthType(normalizedAuthType)) {
    const modelName = config.getModel();
    const compatibleModelConfig = config.getCompatibleModels().find((candidate) => {
      const candidateAuth = normalizeAuthType(candidate.authType);
      if (candidateAuth && candidateAuth !== normalizedAuthType) {
        return false;
      }
      return candidate.model === modelName;
    });
    const defaults = getProviderDefaults(normalizedAuthType);
    const apiKeyEnvKey =
      compatibleModelConfig?.apiKeyEnvKey ?? defaults.apiKeyEnvKey;
    const resolvedApiKey =
      apiKey ||
      config.getProviderApiKey() ||
      (apiKeyEnvKey ? process.env[apiKeyEnvKey] : undefined) ||
      process.env[defaults.apiKeyEnvKey];
    const resolvedBaseUrl =
      config.getProviderBaseUrl() ||
      compatibleModelConfig?.baseUrl ||
      process.env[defaults.baseUrlEnvKey] ||
      defaults.defaultBaseUrl;
    const resolvedModel =
      process.env[defaults.modelEnvKey] || compatibleModelConfig?.model || modelName;

    if (!resolvedModel) {
      throw new Error(
        `No model configured for ${normalizedAuthType}. Set --model, ${defaults.modelEnvKey}, settings.model.name, or modelProviders.${normalizedAuthType === AuthType.USE_OPENAI ? 'openai' : 'anthropic'}.`,
      );
    }

    if (!resolvedApiKey) {
      throw new Error(
        `Missing API key for ${normalizedAuthType}. Set security.auth.apiKey, ${apiKeyEnvKey}, or ${defaults.apiKeyEnvKey}.`,
      );
    }

    return {
      ...compatibleModelConfig,
      authType: normalizedAuthType,
      proxy: config.getProxy(),
      model: resolvedModel,
      apiKey: resolvedApiKey,
      apiKeyEnvKey,
      baseUrl: resolvedBaseUrl,
      providerSubtype: resolveProviderSubtype(
        normalizedAuthType,
        resolvedBaseUrl,
      ),
    };
  }

  const contentGeneratorConfig: ContentGeneratorConfig = {
    authType: normalizedAuthType,
    proxy: config?.getProxy(),
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    normalizedAuthType === AuthType.LOGIN_WITH_GOOGLE ||
    normalizedAuthType === AuthType.COMPUTE_ADC
  ) {
    return contentGeneratorConfig;
  }

  if (normalizedAuthType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;

    return contentGeneratorConfig;
  }

  if (
    normalizedAuthType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;

    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const generator = await (async () => {
    if (gcConfig.fakeResponses) {
      const fakeGenerator = await FakeContentGenerator.fromFile(
        gcConfig.fakeResponses,
      );
      return new LoggingContentGenerator(fakeGenerator, gcConfig);
    }
    const version = await getVersion();
    const normalizedConfigAuthType = normalizeAuthType(config.authType);
    const model = resolveModel(
      gcConfig.getModel(),
      normalizedConfigAuthType === AuthType.USE_GEMINI ||
        normalizedConfigAuthType === AuthType.USE_VERTEX_AI ||
        ((await gcConfig.getGemini31Launched?.()) ?? false),
    );
    const customHeadersEnv =
      process.env['GEMINI_CLI_CUSTOM_HEADERS'] || undefined;
    const userAgent = `GeminiCLI/${version}/${model} (${process.platform}; ${process.arch})`;
    const customHeadersMap = parseCustomHeaders(customHeadersEnv);
    const apiKeyAuthMechanism =
      process.env['GEMINI_API_KEY_AUTH_MECHANISM'] || 'x-goog-api-key';
    const apiVersionEnv = process.env['GOOGLE_GENAI_API_VERSION'];

    const baseHeaders: Record<string, string> = {
      ...customHeadersMap,
      'User-Agent': userAgent,
    };

    if (
      apiKeyAuthMechanism === 'bearer' &&
      (normalizedConfigAuthType === AuthType.USE_GEMINI ||
        normalizedConfigAuthType === AuthType.USE_VERTEX_AI) &&
      config.apiKey
    ) {
      baseHeaders['Authorization'] = `Bearer ${config.apiKey}`;
    }
    if (
      normalizedConfigAuthType === AuthType.LOGIN_WITH_GOOGLE ||
      normalizedConfigAuthType === AuthType.COMPUTE_ADC
    ) {
      const httpOptions = { headers: baseHeaders };
      return new LoggingContentGenerator(
        await createCodeAssistContentGenerator(
          httpOptions,
          normalizedConfigAuthType,
          gcConfig,
          sessionId,
        ),
        gcConfig,
      );
    }

    if (
      normalizedConfigAuthType === AuthType.USE_GEMINI ||
      normalizedConfigAuthType === AuthType.USE_VERTEX_AI
    ) {
      let headers: Record<string, string> = { ...baseHeaders };
      if (gcConfig?.getUsageStatisticsEnabled()) {
        const installationManager = new InstallationManager();
        const installationId = installationManager.getInstallationId();
        headers = {
          ...headers,
          'x-gemini-api-privileged-user-id': `${installationId}`,
        };
      }
      const httpOptions = { headers };

      const googleGenAI = new GoogleGenAI({
        apiKey: config.apiKey === '' ? undefined : config.apiKey,
        vertexai: config.vertexai,
        httpOptions,
        ...(apiVersionEnv && { apiVersion: apiVersionEnv }),
      });
      return new LoggingContentGenerator(googleGenAI.models, gcConfig);
    }

    if (isOpenAIAuthType(normalizedConfigAuthType)) {
      return createOpenAIContentGenerator(config, gcConfig);
    }

    if (isAnthropicAuthType(normalizedConfigAuthType)) {
      const provider = createAnthropicCompatibleProvider(config, gcConfig);
      return new AnthropicContentGenerator(config, gcConfig, provider);
    }

    throw new Error(
      `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
    );
  })();

  if (gcConfig.recordResponses) {
    return new RecordingContentGenerator(generator, gcConfig.recordResponses);
  }

  return generator;
}
