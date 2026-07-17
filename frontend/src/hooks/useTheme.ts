import { useContext } from 'react';
import { ThemeContext } from '../context/ThemeContext';

/** Access the theme state. Must be used within a <ThemeProvider>. */
export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return ctx;
}
