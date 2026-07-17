import { Avatar } from './Avatar';

/** Assistant "typing…" bubble shown while awaiting the backend response. */
export function TypingIndicator() {
  return (
    <div className="message message--assistant" aria-live="polite" aria-label="AquaMind AI is typing">
      <Avatar role="assistant" />
      <div className="message__body">
        <div className="message__bubble message__bubble--typing">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
