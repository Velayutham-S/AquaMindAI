import axios from 'axios';
import { apiClient, normalizeError } from './api';
import type { ChatRequest, ChatResponse, NormalizedError } from '../types';

/**
 * The single backend endpoint the frontend is allowed to call.
 * The frontend must NEVER call individual agents.
 */
const CHAT_ENDPOINT = '/api/chat';

/** Thrown by the service layer with a normalized, user-friendly error. */
export class ChatServiceError extends Error {
  readonly normalized: NormalizedError;

  constructor(normalized: NormalizedError) {
    super(normalized.message);
    this.name = 'ChatServiceError';
    this.normalized = normalized;
  }
}

function isChatResponse(value: unknown): value is ChatResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    'response' in value &&
    typeof (value as Record<string, unknown>).response === 'string'
  );
}

/**
 * Send a user message to the backend and return the generated answer text.
 *
 * @param message  The raw user query.
 * @param signal   Optional AbortSignal to cancel the request.
 * @throws ChatServiceError with a normalized error on any failure.
 */
export async function sendMessage(message: string, signal?: AbortSignal): Promise<string> {
  const payload: ChatRequest = { message };

  try {
    const { data } = await apiClient.post<ChatResponse>(CHAT_ENDPOINT, payload, { signal });

    if (!isChatResponse(data)) {
      throw new ChatServiceError({
        kind: 'invalid_response',
        message: 'AquaMind AI returned an unexpected response format. Please try again.',
      });
    }

    const answer = data.response.trim();
    if (!answer) {
      throw new ChatServiceError({
        kind: 'empty_response',
        message: 'AquaMind AI returned an empty response. Please rephrase and try again.',
      });
    }

    return answer;
  } catch (error) {
    if (error instanceof ChatServiceError) {
      throw error;
    }
    // Preserve explicit cancellations so callers can ignore them.
    if (axios.isCancel(error) || (error instanceof DOMException && error.name === 'AbortError')) {
      throw error;
    }
    throw new ChatServiceError(normalizeError(error));
  }
}
