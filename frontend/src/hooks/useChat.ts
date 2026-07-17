import { useContext } from 'react';
import { ChatContext, type ChatContextValue } from '../context/ChatContext';

/** Access chat state and actions. Must be used within a <ChatProvider>. */
export function useChat(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return ctx;
}
