import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * A typed, SSR-safe localStorage-backed state hook.
 *
 * Reads the initial value once, then persists updates. Safely degrades to
 * in-memory state if storage is unavailable (private mode, quota, etc.).
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, (value: T | ((prev: T) => T)) => void] {
  const readValue = useCallback((): T => {
    if (typeof window === 'undefined') return initialValue;
    try {
      const raw = window.localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initialValue;
    } catch {
      return initialValue;
    }
  }, [key, initialValue]);

  const [storedValue, setStoredValue] = useState<T>(readValue);

  // Keep the latest value in a ref so the setter identity stays stable.
  const valueRef = useRef(storedValue);
  valueRef.current = storedValue;

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      setStoredValue((prev) => {
        const next = value instanceof Function ? (value as (p: T) => T)(prev) : value;
        try {
          if (typeof window !== 'undefined') {
            window.localStorage.setItem(key, JSON.stringify(next));
          }
        } catch {
          // Ignore write failures (e.g. quota exceeded); keep in-memory value.
        }
        return next;
      });
    },
    [key],
  );

  // Sync updates from other tabs/windows.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onStorage = (event: StorageEvent) => {
      if (event.key === key && event.newValue !== null) {
        try {
          setStoredValue(JSON.parse(event.newValue) as T);
        } catch {
          // Ignore malformed cross-tab payloads.
        }
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [key]);

  return [storedValue, setValue];
}
