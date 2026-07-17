import { useEffect, useRef, useState } from 'react';
import type { ChatMessage } from '../types';
import { ChatMessageItem } from './ChatMessage';
import { TypingIndicator } from './TypingIndicator';
import { useAutoScroll } from '../hooks/useAutoScroll';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onRetry: () => void;
}

/** Scrollable transcript with auto-scroll and a typing indicator. */
export function MessageList({ messages, isLoading, onRetry }: MessageListProps) {
  const lastMessage = messages[messages.length - 1];
  const lastContentLength = lastMessage?.content.length ?? 0;

  const { containerRef, bottomRef, scrollToBottom } = useAutoScroll<HTMLDivElement>([
    messages.length,
    lastContentLength,
    isLoading,
  ]);

  // Animate only assistant messages that were received during this session
  // (i.e. transitioned from a 'sending' placeholder to 'done'). History and
  // conversation switches never re-animate.
  const sendingSeenRef = useRef<Set<string>>(new Set());
  const animatedRef = useRef<string | null>(null);
  const [animateId, setAnimateId] = useState<string | null>(null);

  useEffect(() => {
    for (const m of messages) {
      if (m.role === 'assistant' && m.status === 'sending') sendingSeenRef.current.add(m.id);
    }
    const last = messages[messages.length - 1];
    if (
      last &&
      last.role === 'assistant' &&
      last.status === 'done' &&
      last.content &&
      sendingSeenRef.current.has(last.id) &&
      animatedRef.current !== last.id
    ) {
      animatedRef.current = last.id;
      setAnimateId(last.id);
    }
  }, [messages]);

  return (
    <div className="message-list" ref={containerRef}>
      <div className="message-list__inner">
        {messages.map((message) => {
          if (message.role === 'assistant' && message.status === 'sending') {
            return <TypingIndicator key={message.id} />;
          }
          return (
            <ChatMessageItem
              key={message.id}
              message={message}
              animate={message.id === animateId}
              onRetry={onRetry}
              onAnimateTick={() => scrollToBottom('auto')}
            />
          );
        })}
        <div ref={bottomRef} aria-hidden />
      </div>
    </div>
  );
}
