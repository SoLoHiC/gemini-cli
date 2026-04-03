/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useCallback, useState } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { RadioButtonSelect } from '../components/shared/RadioButtonSelect.js';
import {
  SettingScope,
  type LoadableSettingScope,
  type LoadedSettings,
} from '../../config/settings.js';
import {
  AuthType,
  clearCachedCredentialFile,
  type Config,
  isAnthropicAuthType,
  isOpenAIAuthType,
  isProviderAuthType,
} from '@google/gemini-cli-core';
import { useKeypress } from '../hooks/useKeypress.js';
import { AuthState } from '../types.js';
import { validateAuthMethodWithSettings } from './useAuth.js';
import { relaunchApp } from '../../utils/processUtils.js';
import { parseAuthType } from '../../config/auth.js';

interface AuthDialogProps {
  config: Config;
  settings: LoadedSettings;
  setAuthState: (state: AuthState) => void;
  authError: string | null;
  onAuthError: (error: string | null) => void;
  setAuthContext: (context: { requiresRestart?: boolean }) => void;
}

export function AuthDialog({
  config,
  settings,
  setAuthState,
  authError,
  onAuthError,
  setAuthContext,
}: AuthDialogProps): React.JSX.Element {
  const [exiting, setExiting] = useState(false);
  const ensureProviderModel = useCallback(
    (authType: AuthType): string | null => {
      const providerKey = isOpenAIAuthType(authType)
        ? 'openai'
        : isAnthropicAuthType(authType)
          ? 'anthropic'
          : undefined;
      if (!providerKey) {
        return null;
      }

      const configuredModels =
        settings.merged.modelProviders?.[providerKey] ?? [];
      const currentModel = settings.merged.model?.name;
      const currentModelMatchesProvider = configuredModels.some(
        (entry) => entry.id === currentModel,
      );
      const envModel =
        process.env[
          providerKey === 'openai' ? 'OPENAI_MODEL' : 'ANTHROPIC_MODEL'
        ];

      if (!currentModelMatchesProvider && configuredModels[0]?.id) {
        config.setModel(configuredModels[0].id, false);
      }

      if (
        !currentModelMatchesProvider &&
        !configuredModels[0]?.id &&
        !envModel
      ) {
        return `No model configured for ${providerKey}. Set --model, ${providerKey === 'openai' ? 'OPENAI_MODEL' : 'ANTHROPIC_MODEL'}, settings.model.name, or modelProviders.${providerKey}.`;
      }

      return null;
    },
    [config, settings],
  );
  let items = [
    {
      label: 'Sign in with Google',
      value: AuthType.LOGIN_WITH_GOOGLE,
      key: AuthType.LOGIN_WITH_GOOGLE,
    },
    ...(process.env['CLOUD_SHELL'] === 'true'
      ? [
          {
            label: 'Use Cloud Shell user credentials',
            value: AuthType.COMPUTE_ADC,
            key: AuthType.COMPUTE_ADC,
          },
        ]
      : process.env['GEMINI_CLI_USE_COMPUTE_ADC'] === 'true'
        ? [
            {
              label: 'Use metadata server application default credentials',
              value: AuthType.COMPUTE_ADC,
              key: AuthType.COMPUTE_ADC,
            },
          ]
        : []),
    {
      label: 'Use Gemini API Key',
      value: AuthType.USE_GEMINI,
      key: AuthType.USE_GEMINI,
    },
    {
      label: 'Vertex AI',
      value: AuthType.USE_VERTEX_AI,
      key: AuthType.USE_VERTEX_AI,
    },
    {
      label: 'OpenAI API Key',
      value: AuthType.USE_OPENAI,
      key: AuthType.USE_OPENAI,
    },
    {
      label: 'Anthropic API Key',
      value: AuthType.USE_ANTHROPIC,
      key: AuthType.USE_ANTHROPIC,
    },
  ];

  const enforcedType = parseAuthType(
    settings.merged.security.auth.enforcedType,
  );

  if (enforcedType) {
    items = items.filter((item) => item.value === enforcedType);
  }

  let defaultAuthType = null;
  const defaultAuthTypeEnv = parseAuthType(
    process.env['GEMINI_DEFAULT_AUTH_TYPE'],
  );
  if (defaultAuthTypeEnv) {
    defaultAuthType = defaultAuthTypeEnv;
  }

  let initialAuthIndex = items.findIndex((item) => {
    const selectedType = parseAuthType(
      settings.merged.security.auth.selectedType,
    );
    if (selectedType) {
      return item.value === selectedType;
    }

    if (defaultAuthType) {
      return item.value === defaultAuthType;
    }

    if (process.env['GEMINI_API_KEY']) {
      return item.value === AuthType.USE_GEMINI;
    }
    if (process.env['OPENAI_API_KEY']) {
      return item.value === AuthType.USE_OPENAI;
    }
    if (process.env['ANTHROPIC_API_KEY']) {
      return item.value === AuthType.USE_ANTHROPIC;
    }

    return item.value === AuthType.LOGIN_WITH_GOOGLE;
  });
  if (enforcedType) {
    initialAuthIndex = 0;
  }

  const onSelect = useCallback(
    async (authType: AuthType | undefined, scope: LoadableSettingScope) => {
      if (exiting) {
        return;
      }
      if (authType) {
        const providerModelError = isProviderAuthType(authType)
          ? ensureProviderModel(authType)
          : null;
        if (providerModelError) {
          onAuthError(providerModelError);
          return;
        }
        if (authType === AuthType.LOGIN_WITH_GOOGLE) {
          setAuthContext({ requiresRestart: true });
        } else {
          setAuthContext({});
        }
        await clearCachedCredentialFile();

        settings.setValue(scope, 'security.auth.selectedType', authType);
        if (
          authType === AuthType.LOGIN_WITH_GOOGLE &&
          config.isBrowserLaunchSuppressed()
        ) {
          setExiting(true);
          setTimeout(relaunchApp, 100);
          return;
        }

        if (authType === AuthType.USE_GEMINI) {
          // Always show the API key input dialog so the user can
          // explicitly enter or confirm their key, regardless of
          // whether GEMINI_API_KEY env var or a stored key exists.
          setAuthState(AuthState.AwaitingApiKeyInput);
          return;
        }
        if (isProviderAuthType(authType)) {
          // Match Gemini API key flow: always open the API key dialog so the
          // user can confirm the prefilled key from env/settings or enter a new one.
          setAuthState(AuthState.AwaitingApiKeyInput);
          return;
        }
      }
      setAuthState(AuthState.Unauthenticated);
    },
    [
      settings,
      config,
      setAuthState,
      exiting,
      setAuthContext,
      ensureProviderModel,
      onAuthError,
    ],
  );

  const handleAuthSelect = (authMethod: AuthType) => {
    const error = validateAuthMethodWithSettings(authMethod, settings);
    if (error) {
      onAuthError(error);
    } else {
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      onSelect(authMethod, SettingScope.User);
    }
  };

  useKeypress(
    (key) => {
      if (key.name === 'escape') {
        // Prevent exit if there is an error message.
        // This means they user is not authenticated yet.
        if (authError) {
          return true;
        }
        if (settings.merged.security.auth.selectedType === undefined) {
          // Prevent exiting if no auth method is set
          onAuthError(
            'You must select an auth method to proceed. Press Ctrl+C twice to exit.',
          );
          return true;
        }
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        onSelect(undefined, SettingScope.User);
        return true;
      }
      return false;
    },
    { isActive: true },
  );

  if (exiting) {
    return (
      <Box
        borderStyle="round"
        borderColor={theme.ui.focus}
        flexDirection="row"
        padding={1}
        width="100%"
        alignItems="flex-start"
      >
        <Text color={theme.text.primary}>
          Logging in with Google... Restarting Gemini CLI to continue.
        </Text>
      </Box>
    );
  }

  return (
    <Box
      borderStyle="round"
      borderColor={theme.ui.focus}
      flexDirection="row"
      padding={1}
      width="100%"
      alignItems="flex-start"
    >
      <Text color={theme.text.accent}>? </Text>
      <Box flexDirection="column" flexGrow={1}>
        <Text bold color={theme.text.primary}>
          Get started
        </Text>
        <Box marginTop={1}>
          <Text color={theme.text.primary}>
            How would you like to authenticate for this project?
          </Text>
        </Box>
        <Box marginTop={1}>
          <RadioButtonSelect
            items={items}
            initialIndex={initialAuthIndex}
            onSelect={handleAuthSelect}
            onHighlight={() => {
              onAuthError(null);
            }}
          />
        </Box>
        {authError && (
          <Box marginTop={1}>
            <Text color={theme.status.error}>{authError}</Text>
          </Box>
        )}
        <Box marginTop={1}>
          <Text color={theme.text.secondary}>(Use Enter to select)</Text>
        </Box>
        <Box marginTop={1}>
          <Text color={theme.text.primary}>
            Terms of Services and Privacy Notice for Gemini CLI
          </Text>
        </Box>
        <Box marginTop={1}>
          <Text color={theme.text.link}>
            {'https://geminicli.com/docs/resources/tos-privacy/'}
          </Text>
        </Box>
      </Box>
    </Box>
  );
}
