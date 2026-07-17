/**
 * Small, dependency-free utilities shared across the app.
 */

/** Generate a reasonably unique id (crypto.randomUUID when available). */
export function generateId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Format an epoch-ms timestamp as a short local time, e.g. "14:05". */
export function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Format an epoch-ms timestamp as a relative day label for the sidebar. */
export function formatRelativeDay(timestamp: number): string {
  const now = new Date();
  const date = new Date(timestamp);
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const dayMs = 86_400_000;

  if (timestamp >= startOfToday) return 'Today';
  if (timestamp >= startOfToday - dayMs) return 'Yesterday';
  if (timestamp >= startOfToday - 7 * dayMs) return 'Previous 7 days';
  return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
}

/** Derive a concise conversation title from the first user message. */
export function deriveTitle(message: string): string {
  const cleaned = message.replace(/\s+/g, ' ').trim();
  if (!cleaned) return 'New conversation';
  const max = 48;
  return cleaned.length > max ? `${cleaned.slice(0, max).trimEnd()}…` : cleaned;
}
