import { useCallback, useEffect, useRef, useState } from 'react';

interface AutoScrollResult<T extends HTMLElement> {
  containerRef: React.RefObject<T>;
  bottomRef: React.RefObject<HTMLDivElement>;
  isPinnedToBottom: boolean;
  scrollToBottom: (behavior?: ScrollBehavior) => void;
}

const NEAR_BOTTOM_THRESHOLD = 80;

/**
 * Keeps a scroll container pinned to the bottom as new content arrives,
 * unless the user has scrolled up (in which case auto-scroll pauses).
 *
 * @param deps  Values that, when changed, should trigger an auto-scroll
 *              (e.g. message count, streaming text length).
 */
export function useAutoScroll<T extends HTMLElement>(deps: unknown[]): AutoScrollResult<T> {
  const containerRef = useRef<T>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [isPinnedToBottom, setIsPinnedToBottom] = useState(true);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    bottomRef.current?.scrollIntoView({ behavior, block: 'end' });
  }, []);

  // Track whether the user is near the bottom.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      setIsPinnedToBottom(distance <= NEAR_BOTTOM_THRESHOLD);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  // Auto-scroll when tracked dependencies change and the user is pinned.
  useEffect(() => {
    if (isPinnedToBottom) scrollToBottom('smooth');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { containerRef, bottomRef, isPinnedToBottom, scrollToBottom };
}
