/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Part, FunctionCall } from '@google/genai';

/**
 * A class that parses streaming tool calls from an Anthropic-compatible API.
 * Anthropic streams tool calls as a series of JSON patches, which need to be aggregated.
 */
export class StreamingToolCallParser {
  private toolCalls: FunctionCall[] = [];
  private currentToolCall: FunctionCall | null = null;
  private currentToolCallIndex = 0;

  /**
   * Processes a chunk of streaming data and updates the tool calls.
   * @param chunk The streaming chunk from the Anthropic API.
   * @returns A `Part` object representing the parsed tool call, or `null` if parsing is incomplete.
   */
  parse(chunk: Record<string, unknown>): Part | null {
    // This is a placeholder implementation. A real implementation would need to handle
    // Anthropic's specific streaming events for tool calls, such as
    // 'content_block_start', 'content_block_delta' with 'input_json_delta', and 'content_block_stop'.

    // For now, we'll simulate a simple parser based on a hypothetical chunk structure.
    if (chunk['type'] === 'tool_use') {
      if (!this.currentToolCall) {
        this.currentToolCall = {
          name: chunk['name'] as string | undefined,
          args: {},
          id: chunk['id'] as string | undefined,
        };
      }
      // A real implementation would aggregate args from 'input_json_delta' events.
      if (this.currentToolCall) {
        this.currentToolCall.args = {
          ...this.currentToolCall.args,
          ...((chunk['input'] as Record<string, unknown>) || {}),
        };
      }
    }

    if (this.currentToolCall) {
      this.toolCalls[this.currentToolCallIndex] = this.currentToolCall;
      return { functionCall: this.currentToolCall };
    }

    return null;
  }

  /**
   * Returns the currently parsed tool calls.
   */
  getToolCalls(): FunctionCall[] {
    return this.toolCalls;
  }

  /**
   * Resets the parser state for a new stream.
   */
  reset(): void {
    this.toolCalls = [];
    this.currentToolCall = null;
    this.currentToolCallIndex = 0;
  }
}
