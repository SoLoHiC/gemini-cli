/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import type Anthropic from '@anthropic-ai/sdk';
import { AnthropicContentConverter } from './converter.js';

describe('AnthropicContentConverter', () => {
  const converter = new AnthropicContentConverter('claude-test');

  it('keeps assistant thought parts in thinking blocks when converting requests', async () => {
    const request = await converter.convertGeminiRequestToAnthropic({
      model: 'claude-test',
      contents: [
        {
          role: 'model',
          parts: [
            {
              text: 'internal reasoning',
              thought: true,
              thoughtSignature: 'sig-1',
            },
            { text: 'visible answer' },
          ],
        },
      ],
    });

    expect(request.messages).toEqual([
      {
        role: 'assistant',
        content: [
          {
            type: 'thinking',
            thinking: 'internal reasoning',
            signature: 'sig-1',
          },
          {
            type: 'text',
            text: 'visible answer',
          },
        ],
      },
    ]);
  });

  it('keeps assistant thought parts alongside tool use when converting requests', async () => {
    const request = await converter.convertGeminiRequestToAnthropic({
      model: 'claude-test',
      contents: [
        {
          role: 'model',
          parts: [
            {
              text: 'internal reasoning',
              thought: true,
              thoughtSignature: 'sig-1',
            },
            {
              functionCall: {
                id: 'tool_1',
                name: 'lookup_weather',
                args: { city: 'Shanghai' },
              },
            },
          ],
        },
      ],
    });

    expect(request.messages).toEqual([
      {
        role: 'assistant',
        content: [
          {
            type: 'thinking',
            thinking: 'internal reasoning',
            signature: 'sig-1',
          },
          {
            type: 'tool_use',
            id: 'tool_1',
            name: 'lookup_weather',
            input: { city: 'Shanghai' },
          },
        ],
      },
    ]);
  });

  it('converts Anthropic thinking blocks into thought parts', () => {
    const response = converter.convertAnthropicResponseToGemini({
      id: 'msg_1',
      model: 'claude-test',
      role: 'assistant',
      type: 'message',
      stop_reason: 'end_turn',
      stop_sequence: null,
      content: [
        {
          type: 'thinking',
          thinking: 'internal reasoning',
          signature: 'sig-1',
        },
        {
          type: 'text',
          text: 'visible answer',
        },
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
      },
    } as unknown as Anthropic.Messages.Message);

    expect(response.candidates?.[0]?.content?.parts).toEqual([
      {
        text: 'internal reasoning',
        thought: true,
        thoughtSignature: 'sig-1',
      },
      {
        text: 'visible answer',
      },
    ]);
  });

  it('converts thinking deltas into streaming thought parts', () => {
    const response = converter.convertAnthropicChunkToGemini({
      type: 'content_block_delta',
      index: 0,
      delta: {
        type: 'thinking_delta',
        thinking: 'thinking...',
      },
    } as unknown as Anthropic.Messages.MessageStreamEvent);

    expect(response?.candidates?.[0]?.content?.parts).toEqual([
      {
        text: 'thinking...',
        thought: true,
      },
    ]);
  });
});
