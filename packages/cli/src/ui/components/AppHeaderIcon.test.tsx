/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { renderWithProviders } from '../../test-utils/render.js';
import { AppHeader } from './AppHeader.js';

describe('AppHeader ASCII Rendering', () => {
  it('renders the ASCII header', async () => {
    const result = await renderWithProviders(<AppHeader version="1.0.0" />);
    await result.waitUntilReady();

    await expect(result).toMatchSvgSnapshot();
  });
});
