import { memo } from 'react';
import type { ChatMessage as ChatMessageType } from '../types';
import { Avatar } from './Avatar';
import { MarkdownRenderer } from './MarkdownRenderer';
import { CheckIcon, CopyIcon, RetryIcon } from './Icons';
import { useClipboard } from '../hooks/useClipboard';
import { useTypewriter } from '../hooks/useTypewriter';
import { formatTime } from '../utils';

interface ChatMessageProps {
  message: ChatMessageType;
  /** Animate the assistant text with a typewriter effect (newly received only). */
  animate?: boolean;
  onRetry?: () => void;
  onAnimateTick?: () => void;
}

export const ChatMessageItem = memo(function ChatMessageItem({
  message,
  animate = false,
  onRetry,
  onAnimateTick,
}: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isError = message.status === 'error';
  const { copied, copy } = useClipboard();

  const { displayed } = useTypewriter(message.content, {
    enabled: animate && !isUser && message.status === 'done',
    onTick: onAnimateTick,
  });

  const shownContent = isUser ? message.content : displayed;

  return (
    <div className={`message ${isUser ? 'message--user' : 'message--assistant'}`}>
      <Avatar role={message.role} />
      <div className="message__body">
        <div className="message__meta">
          <span className="message__author">{isUser ? 'You' : 'AquaMind AI'}</span>
          <time className="message__time" dateTime={new Date(message.timestamp).toISOString()}>
            {formatTime(message.timestamp)}
          </time>
        </div>

        <div className={`message__bubble${isError ? ' message__bubble--error' : ''}`}>
          {isError ? (
            <p className="message__error-text">{message.error ?? 'Something went wrong.'}</p>
          ) : isUser ? (
            <p className="message__user-text">{shownContent}</p>
          ) : (
            <MarkdownRenderer content={shownContent} />
          )}
        </div>

        {!isUser && (
          <div className="message__actions">
            {message.status === 'done' && message.content && (
              <button
                type="button"
                className="message__action-btn"
                onClick={() => void copy(message.content)}
                aria-label="Copy response"
              >
                {copied ? <CheckIcon width={15} height={15} /> : <CopyIcon width={15} height={15} />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            )}
            {isError && onRetry && (
              <button type="button" className="message__action-btn" onClick={onRetry}>
                <RetryIcon width={15} height={15} />
                Retry
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
});
