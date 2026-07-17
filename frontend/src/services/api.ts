import axios, { AxiosError, type AxiosInstance } from 'axios';
import type { NormalizedError } from '../types';

/**
 * Centralized Axios instance for the AquaMind AI backend.
 *
 * All requests are relative to `${VITE_API_BASE_URL}` (falling back to the
 * dev proxy path when unset). The frontend talks ONLY to this backend.
 */

const DEFAULT_TIMEOUT = 60_000;

function resolveTimeout(): number {
  const raw = import.meta.env.VITE_API_TIMEOUT;
  const parsed = raw ? Number.parseInt(raw, 10) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMEOUT;
}

/**
 * Base URL resolution:
 * - If VITE_API_BASE_URL is set, use it directly.
 * - Otherwise use a relative base ('') so requests hit the same origin, which
 *   works with the Vite dev proxy and same-origin production deployments.
 */
function resolveBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');
}

export const apiClient: AxiosInstance = axios.create({
  baseURL: resolveBaseUrl(),
  timeout: resolveTimeout(),
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

/**
 * Convert any thrown error (Axios or otherwise) into a user-friendly,
 * normalized shape the UI can render consistently.
 */
export function normalizeError(error: unknown): NormalizedError {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;

    if (axiosError.code === 'ECONNABORTED' || /timeout/i.test(axiosError.message)) {
      return {
        kind: 'timeout',
        message:
          'The request timed out. The server took too long to respond. Please try again.',
      };
    }

    // No response received => network / backend unavailable.
    if (!axiosError.response) {
      return {
        kind: 'network',
        message:
          'Unable to reach AquaMind AI. Please check your connection and ensure the backend is running.',
      };
    }

    const statusCode = axiosError.response.status;
    if (statusCode >= 500) {
      return {
        kind: 'server',
        message:
          'AquaMind AI encountered a server error while processing your request. Please try again.',
        statusCode,
      };
    }

    return {
      kind: 'server',
      message: `The request failed (HTTP ${statusCode}). Please try again.`,
      statusCode,
    };
  }

  if (error instanceof Error) {
    return { kind: 'unknown', message: error.message };
  }

  return {
    kind: 'unknown',
    message: 'An unexpected error occurred. Please try again.',
  };
}
