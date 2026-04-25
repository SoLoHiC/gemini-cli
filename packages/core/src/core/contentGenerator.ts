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
import * as os from 'node:os';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { isCloudShell } from '../ide/detect-ide.js';
import type {
  Config,
  ModelProviderConfig,
  ModelProvidersConfig,
} from '../config/config.js';
import { loadApiKey } from './apiKeyCredentialStorage.js';

import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import { LoggingContentGenerator } from './loggingContentGenerator.js';
import { InstallationManager } from '../utils/installationManager.js';
import { FakeContentGenerator } from './fakeContentGenerator.js';
import { parseCustomHeaders } from '../utils/customHeaderUtils.js';
import { determineSurface } from '../utils/surface.js';
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
  GATEWAY = 'gateway',
  USE_OPENAI = 'openai',
  USE_ANTHROPIC = 'anthropic',
}

export function isGoogleAuthType(authType?: AuthType): boolean {
  const normalized = authType;
  return (
    normalized === AuthType.LOGIN_WITH_GOOGLE ||
    normalized === AuthType.USE_GEMINI ||
    normalized === AuthType.USE_VERTEX_AI ||
    normalized === AuthType.COMPUTE_ADC ||
    normalized === AuthType.LEGACY_CLOUD_SHELL
  );
}

export function isOpenAIAuthType(authType?: AuthType): boolean {
  return authType === AuthType.USE_OPENAI;
}

export function isAnthropicAuthType(authType?: AuthType): boolean {
  return authType === AuthType.USE_ANTHROPIC;
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
  baseUrl?: string;
  customHeaders?: Record<string, string>;
  vertexAiRouting?: VertexAiRoutingConfig;
  model?: string;
  apiKeyEnvKey?: string;
  timeout?: number;
  maxRetries?: number;
  retryErrorCodes?: number[];
  enableCacheControl?: boolean;
  enableOpenAILogging?: boolean;
  disableCacheControl?: boolean;
  reasoning?: Record<string, unknown>;
  reasoningEffort?: string;
  thinking?: Record<string, unknown>;
  outputConfig?: Record<string, unknown>;
  schemaCompliance?: Record<string, unknown>;
  contextWindowSize?: number;
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

export type VertexAiRequestType = 'dedicated' | 'shared';
export type VertexAiSharedRequestType = 'priority' | 'flex';

export interface VertexAiRoutingConfig {
  requestType?: VertexAiRequestType;
  sharedRequestType?: VertexAiSharedRequestType;
}

const LOCAL_HOSTNAMES = ['localhost', '127.0.0.1', '[::1]'];
const VERTEX_AI_REQUEST_TYPE_HEADER = 'X-Vertex-AI-LLM-Request-Type';
const VERTEX_AI_SHARED_REQUEST_TYPE_HEADER =
  'X-Vertex-AI-LLM-Shared-Request-Type';

function validateBaseUrl(baseUrl: string): void {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    throw new Error(`Invalid custom base URL: ${baseUrl}`);
  }

  if (url.protocol !== 'https:' && !LOCAL_HOSTNAMES.includes(url.hostname)) {
    throw new Error('Custom base URL must use HTTPS unless it is localhost.');
  }
}

const OPENAI_DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const ANTHROPIC_DEFAULT_BASE_URL = 'https://api.anthropic.com/v1';

type ProviderDefaults = {
  apiKeyEnvKey: string;
  modelEnvKey: string;
  baseUrlEnvKey: string;
  defaultBaseUrl: string;
};

function getProviderKey(
  authType: AuthType,
): keyof ModelProvidersConfig | undefined {
  if (isOpenAIAuthType(authType)) {
    return 'openai';
  }
  if (isAnthropicAuthType(authType)) {
    return 'anthropic';
  }
  return undefined;
}

function findProviderModelConfig(
  modelProviders: ModelProvidersConfig | undefined,
  authType: AuthType,
  modelName: string | undefined,
): ModelProviderConfig | undefined {
  if (!modelName) {
    return undefined;
  }

  const providerKey = getProviderKey(authType);
  if (!providerKey) {
    return undefined;
  }

  const providerEntries = modelProviders?.[providerKey];
  if (!Array.isArray(providerEntries)) {
    return undefined;
  }

  return providerEntries.find((entry) => entry.id === modelName);
}

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

function resolveProviderSubtype(authType: AuthType, baseUrl?: string): string {
  const normalizedBaseUrl = (baseUrl || '').toLowerCase().replace(/\/+$/, '');

  if (authType === AuthType.USE_OPENAI) {
    if (normalizedBaseUrl.includes('openrouter.ai')) {
      return 'openrouter';
    }
    if (
      normalizedBaseUrl.includes('dashscope.aliyuncs.com/compatible-mode/v1') ||
      normalizedBaseUrl.includes(
        'dashscope-intl.aliyuncs.com/compatible-mode/v1',
      )
    ) {
      return 'dashscope-openai';
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
  baseUrl?: string,
  customHeaders?: Record<string, string>,
  vertexAiRouting?: VertexAiRoutingConfig,
): Promise<ContentGeneratorConfig> {
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

  if (authType && isProviderAuthType(authType)) {
    const modelName = config.getModel();
    const defaults = getProviderDefaults(authType);
    const providerModelConfig = findProviderModelConfig(
      config.getModelProvidersConfig?.(),
      authType,
      modelName,
    );
    const providerGenerationConfig = providerModelConfig?.generationConfig;
    const apiKeyEnvKey = providerModelConfig?.envKey ?? defaults.apiKeyEnvKey;
    const providerEnvApiKey = apiKeyEnvKey
      ? process.env[apiKeyEnvKey]
      : undefined;
    const resolvedApiKey =
      apiKey ||
      providerEnvApiKey ||
      config.getProviderApiKey() ||
      process.env[defaults.apiKeyEnvKey];
    const resolvedBaseUrl =
      baseUrl ||
      config.getProviderBaseUrl() ||
      providerModelConfig?.baseUrl ||
      process.env[defaults.baseUrlEnvKey] ||
      defaults.defaultBaseUrl;
    const resolvedModel = process.env[defaults.modelEnvKey] || modelName;

    if (!resolvedModel) {
      throw new Error(
        `No model configured for ${authType}. Set --model, ${defaults.modelEnvKey}, settings.model.name, or modelProviders.${authType === AuthType.USE_OPENAI ? 'openai' : 'anthropic'}.`,
      );
    }

    if (!resolvedApiKey) {
      throw new Error(
        `Missing API key for ${authType}. Set security.auth.apiKey, ${apiKeyEnvKey}, or ${defaults.apiKeyEnvKey}.`,
      );
    }

    return {
      authType,
      proxy: config.getProxy(),
      model: resolvedModel,
      apiKey: resolvedApiKey,
      apiKeyEnvKey,
      baseUrl: resolvedBaseUrl,
      timeout: providerGenerationConfig?.timeout,
      maxRetries: providerGenerationConfig?.maxRetries,
      retryErrorCodes: providerGenerationConfig?.retryErrorCodes,
      enableCacheControl: providerGenerationConfig?.enableCacheControl,
      samplingParams: providerGenerationConfig?.samplingParams,
      reasoning: providerGenerationConfig?.reasoning,
      reasoningEffort: providerGenerationConfig?.reasoningEffort,
      thinking: providerGenerationConfig?.thinking,
      outputConfig: providerGenerationConfig?.outputConfig,
      schemaCompliance: providerGenerationConfig?.schemaCompliance,
      contextWindowSize: providerGenerationConfig?.contextWindowSize,
      customHeaders: {
        ...providerGenerationConfig?.customHeaders,
        ...customHeaders,
      },
      extra_body: providerGenerationConfig?.extra_body,
      modalities: providerGenerationConfig?.modalities,
      embeddingModel: providerGenerationConfig?.embeddingModel,
      providerSubtype: resolveProviderSubtype(authType, resolvedBaseUrl),
      vertexAiRouting,
    };
  }

  const contentGeneratorConfig: ContentGeneratorConfig = {
    authType,
    proxy: config?.getProxy(),
    baseUrl,
    customHeaders,
    vertexAiRouting,
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.COMPUTE_ADC
  ) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;

    return contentGeneratorConfig;
  }

  if (authType === AuthType.GATEWAY) {
    contentGeneratorConfig.apiKey = apiKey || 'gateway-placeholder-key';
    contentGeneratorConfig.vertexai = false;

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
    const model = resolveModel(
      gcConfig.getModel(),
      config.authType === AuthType.USE_GEMINI ||
        config.authType === AuthType.USE_VERTEX_AI ||
        ((await gcConfig.getGemini31Launched?.()) ?? false),
      config.authType === AuthType.USE_GEMINI ||
        config.authType === AuthType.USE_VERTEX_AI ||
        ((await gcConfig.getGemini31FlashLiteLaunched?.()) ?? false),
      false,
      gcConfig.getHasAccessToPreviewModel?.() ?? true,
      gcConfig,
    );
    const customHeadersEnv =
      process.env['GEMINI_CLI_CUSTOM_HEADERS'] || undefined;
    const clientName = gcConfig.getClientName();
    const surface = determineSurface();

    let userAgent: string;
    // Use unified format for VS Code traffic.
    // Note: We don't automatically assume a2a-server is VS Code,
    // as it could be used by other clients unless the surface explicitly says 'vscode'.
    if (clientName === 'acp-vscode' || surface === 'vscode') {
      const osTypeMap: Record<string, string> = {
        darwin: 'macOS',
        win32: 'Windows',
        linux: 'Linux',
      };
      const osType = osTypeMap[process.platform] || process.platform;
      const osVersion = os.release();
      const arch = process.arch;

      const vscodeVersion = process.env['TERM_PROGRAM_VERSION'] || 'unknown';
      let hostPath = `VSCode/${vscodeVersion}`;
      if (isCloudShell()) {
        const cloudShellVersion =
          process.env['CLOUD_SHELL_VERSION'] || 'unknown';
        hostPath += ` > CloudShell/${cloudShellVersion}`;
      }

      userAgent = `CloudCodeVSCode/${version} (aidev_client; os_type=${osType}; os_version=${osVersion}; arch=${arch}; host_path=${hostPath}; proxy_client=geminicli)`;
    } else {
      const userAgentPrefix = clientName
        ? `GeminiCLI-${clientName}`
        : 'GeminiCLI';
      userAgent = `${userAgentPrefix}/${version}/${model} (${process.platform}; ${process.arch}; ${surface})`;
    }

    const customHeadersMap = parseCustomHeaders(customHeadersEnv);
    const apiKeyAuthMechanism =
      process.env['GEMINI_API_KEY_AUTH_MECHANISM'] || 'x-goog-api-key';
    const apiVersionEnv = process.env['GOOGLE_GENAI_API_VERSION'];

    const baseHeaders: Record<string, string> = {
      'User-Agent': userAgent,
      ...customHeadersMap,
    };

    if (
      apiKeyAuthMechanism === 'bearer' &&
      (config.authType === AuthType.USE_GEMINI ||
        config.authType === AuthType.USE_VERTEX_AI) &&
      config.apiKey
    ) {
      baseHeaders['Authorization'] = `Bearer ${config.apiKey}`;
    }
    if (
      config.authType === AuthType.LOGIN_WITH_GOOGLE ||
      config.authType === AuthType.COMPUTE_ADC
    ) {
      const httpOptions = { headers: baseHeaders };
      return new LoggingContentGenerator(
        await createCodeAssistContentGenerator(
          httpOptions,
          config.authType,
          gcConfig,
          sessionId,
        ),
        gcConfig,
      );
    }

    if (
      config.authType === AuthType.USE_GEMINI ||
      config.authType === AuthType.USE_VERTEX_AI ||
      config.authType === AuthType.GATEWAY
    ) {
      let headers: Record<string, string> = { ...baseHeaders };
      if (config.customHeaders) {
        headers = { ...headers, ...config.customHeaders };
      }
      if (
        config.authType === AuthType.USE_VERTEX_AI &&
        config.vertexAiRouting
      ) {
        const { requestType, sharedRequestType } = config.vertexAiRouting;
        headers = {
          ...headers,
          ...(requestType
            ? { [VERTEX_AI_REQUEST_TYPE_HEADER]: requestType }
            : {}),
          ...(sharedRequestType
            ? { [VERTEX_AI_SHARED_REQUEST_TYPE_HEADER]: sharedRequestType }
            : {}),
        };
      }
      if (gcConfig?.getUsageStatisticsEnabled()) {
        const installationManager = new InstallationManager();
        const installationId = installationManager.getInstallationId();
        headers = {
          ...headers,
          'x-gemini-api-privileged-user-id': `${installationId}`,
        };
      }
      let baseUrl = config.baseUrl;
      if (!baseUrl) {
        const envBaseUrl =
          config.authType === AuthType.USE_VERTEX_AI
            ? process.env['GOOGLE_VERTEX_BASE_URL']
            : process.env['GOOGLE_GEMINI_BASE_URL'];
        if (envBaseUrl) {
          validateBaseUrl(envBaseUrl);
          baseUrl = envBaseUrl;
        }
      } else {
        validateBaseUrl(baseUrl);
      }

      const httpOptions: {
        baseUrl?: string;
        headers: Record<string, string>;
      } = { headers };

      if (baseUrl) {
        httpOptions.baseUrl = baseUrl;
      }

      const googleGenAI = new GoogleGenAI({
        apiKey: config.apiKey === '' ? undefined : config.apiKey,
        vertexai: config.vertexai ?? config.authType === AuthType.USE_VERTEX_AI,
        httpOptions,
        ...(apiVersionEnv && { apiVersion: apiVersionEnv }),
      });
      return new LoggingContentGenerator(googleGenAI.models, gcConfig);
    }

    if (isOpenAIAuthType(config.authType)) {
      return createOpenAIContentGenerator(config, gcConfig);
    }

    if (isAnthropicAuthType(config.authType)) {
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
