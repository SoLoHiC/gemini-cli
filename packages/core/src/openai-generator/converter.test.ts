/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { OpenAIContentConverter } from './converter.js';
import type { StreamingToolCallParser } from './streamingToolCallParser.js';
import type OpenAI from 'openai';

describe('OpenAIContentConverter', () => {
  let converter: OpenAIContentConverter;

  beforeEach(() => {
    converter = new OpenAIContentConverter('test-model');
  });

  describe('resetStreamingToolCalls', () => {
    it('should clear streaming tool calls accumulator', () => {
      // Access private field for testing
      const parser = (
        converter as unknown as {
          streamingToolCallParser: StreamingToolCallParser;
        }
      ).streamingToolCallParser;

      // Add some test data to the parser
      parser.addChunk(0, '{"arg": "value"}', 'test-id', 'test-function');
      parser.addChunk(1, '{"arg2": "value2"}', 'test-id-2', 'test-function-2');

      // Verify data is present
      expect(parser.getBuffer(0)).toBe('{"arg": "value"}');
      expect(parser.getBuffer(1)).toBe('{"arg2": "value2"}');

      // Call reset method
      converter.resetStreamingToolCalls();

      // Verify data is cleared
      expect(parser.getBuffer(0)).toBe('');
      expect(parser.getBuffer(1)).toBe('');
    });

    it('should be safe to call multiple times', () => {
      // Call reset multiple times
      converter.resetStreamingToolCalls();
      converter.resetStreamingToolCalls();
      converter.resetStreamingToolCalls();

      // Should not throw any errors
      const parser = (
        converter as unknown as {
          streamingToolCallParser: StreamingToolCallParser;
        }
      ).streamingToolCallParser;
      expect(parser.getBuffer(0)).toBe('');
    });

    it('should be safe to call on empty accumulator', () => {
      // Call reset on empty accumulator
      converter.resetStreamingToolCalls();

      // Should not throw any errors
      const parser = (
        converter as unknown as {
          streamingToolCallParser: StreamingToolCallParser;
        }
      ).streamingToolCallParser;
      expect(parser.getBuffer(0)).toBe('');
    });
  });

  describe('thinking content', () => {
    it('keeps assistant thought parts in reasoning_content when converting requests', () => {
      const messages = converter.convertGeminiRequestToOpenAI({
        model: 'test-model',
        contents: [
          {
            role: 'model',
            parts: [
              { text: 'internal reasoning', thought: true },
              { text: 'visible answer' },
            ],
          },
        ],
      });

      expect(messages).toHaveLength(1);
      expect(messages[0]).toMatchObject({
        role: 'assistant',
        content: [{ type: 'text', text: 'visible answer' }],
        reasoning_content: 'internal reasoning',
      });
    });

    it('keeps assistant thought parts alongside tool calls when converting requests', () => {
      const messages = converter.convertGeminiRequestToOpenAI({
        model: 'test-model',
        contents: [
          {
            role: 'model',
            parts: [
              { text: 'internal reasoning', thought: true },
              {
                functionCall: {
                  id: 'call_1',
                  name: 'lookup_weather',
                  args: { city: 'Shanghai' },
                },
              },
            ],
          },
          {
            role: 'user',
            parts: [
              {
                functionResponse: {
                  id: 'call_1',
                  name: 'lookup_weather',
                  response: { temperature: '22C' },
                },
              },
            ],
          },
        ],
      });

      expect(messages).toHaveLength(2);
      expect(messages[0]).toMatchObject({
        role: 'assistant',
        content: null,
        reasoning_content: 'internal reasoning',
        tool_calls: [
          {
            id: 'call_1',
            type: 'function',
            function: {
              name: 'lookup_weather',
              arguments: '{"city":"Shanghai"}',
            },
          },
        ],
      });
      expect(messages[1]).toMatchObject({
        role: 'tool',
        tool_call_id: 'call_1',
        content: '{"temperature":"22C"}',
      });
    });

    it('converts non-streaming reasoning_content into a thought part', () => {
      const response = converter.convertOpenAIResponseToGemini({
        id: 'chatcmpl-test',
        object: 'chat.completion',
        created: 123,
        model: 'test-model',
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            logprobs: null,
            message: {
              role: 'assistant',
              content: 'visible answer',
              refusal: null,
              reasoning_content: 'internal reasoning',
            },
          },
        ],
      } as unknown as OpenAI.Chat.ChatCompletion);

      expect(response.candidates?.[0]?.content?.parts).toEqual([
        { text: 'internal reasoning', thought: true },
        { text: 'visible answer' },
      ]);
    });

    it('converts streaming reasoning_content deltas into thought parts', () => {
      const response = converter.convertOpenAIChunkToGemini({
        id: 'chatcmpl-test',
        object: 'chat.completion.chunk',
        created: 123,
        model: 'test-model',
        choices: [
          {
            index: 0,
            finish_reason: null,
            delta: {
              reasoning_content: 'thinking...',
              content: 'visible',
            },
            logprobs: null,
          },
        ],
      } as unknown as OpenAI.Chat.ChatCompletionChunk);

      expect(response.candidates?.[0]?.content?.parts).toEqual([
        { text: 'thinking...', thought: true },
        { text: 'visible' },
      ]);
    });
  });
});
