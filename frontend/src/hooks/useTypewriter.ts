import { useEffect, useRef, useState } from 'react';

interface TypewriterOptions {
  /** When false, the full text is shown immediately (no animation). */
  enabled: boolean;
  /** Characters revealed per animation tick. */
  charsPerTick?: number;
  /** Tick interval in milliseconds. */
  intervalMs?: number;
  /** Called on each reveal tick (e.g. to keep the view scrolled). */
  onTick?: () => void;
}

/**
 * Progressively reveal `text` like a typing effect. When disabled (or on
 * re-render of already-seen text), the full text is returned immediately.
 */
export function useTypewriter(text: string, options: TypewriterOptions): { displayed: string; done: boolean } {
  const { enabled, charsPerTick = 3, intervalMs = 16, onTick } = options;
  const [count, setCount] = useState(enabled ? 0 : text.length);
  const onTickRef = useRef(onTick);
  onTickRef.current = onTick;

  useEffect(() => {
    if (!enabled) {
      setCount(text.length);
      return;
    }
    setCount(0);
    if (!text) return;

    let current = 0;
    const timer = window.setInterval(() => {
      current = Math.min(current + charsPerTick, text.length);
      setCount(current);
      onTickRef.current?.();
      if (current >= text.length) window.clearInterval(timer);
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [text, enabled, charsPerTick, intervalMs]);

  return { displayed: text.slice(0, count), done: count >= text.length };
}
