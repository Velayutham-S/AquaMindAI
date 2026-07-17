/**
 * Shared domain types for the AquaMind AI frontend.
 */

export type MessageRole = 'user' | 'assistant';

export type MessageStatus = 'sending' | 'done' | 'error';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  /** Epoch milliseconds when the message was created. */
  timestamp: number;
  status: MessageStatus;
  /** Present only when status === 'error'. */
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

/** Payload sent to the backend. The frontend calls ONLY POST /api/chat. */
export interface ChatRequest {
  message: string;
}

/** Response contract returned by the backend. */
export interface ChatResponse {
  status: string;
  response: string;
}

/** Normalized error categories surfaced to the UI. */
export type ApiErrorKind =
  | 'network'
  | 'timeout'
  | 'server'
  | 'invalid_response'
  | 'empty_response'
  | 'unknown';

/** A user-friendly, normalized error produced by the service layer. */
export interface NormalizedError {
  kind: ApiErrorKind;
  message: string;
  /** HTTP status code when available. */
  statusCode?: number;
}

export type ThemeMode = 'light' | 'dark';
