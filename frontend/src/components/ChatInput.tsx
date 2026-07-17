import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { SendIcon } from './Icons';
import { LoadingSpinner } from './LoadingSpinner';

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  /** Optional initial text (e.g. from a suggested prompt on the landing page). */
  initialValue?: string;
  autoFocus?: boolean;
}

const MAX_HEIGHT = 200;

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Ask about groundwater in Tamil Nadu…',
  initialValue = '',
  autoFocus = false,
}: ChatInputProps) {
  const [value, setValue] = useState(initialValue);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const autoGrow = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, []);

  useEffect(() => {
    autoGrow();
  }, [value, autoGrow]);

  useEffect(() => {
    if (autoFocus) textareaRef.current?.focus();
  }, [autoFocus]);

  const submit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  }, [value, disabled, onSend]);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const canSend = value.trim().length > 0 && !disabled;

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <div className="chat-input__field">
        <textarea
          ref={textareaRef}
          className="chat-input__textarea"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          disabled={disabled}
          aria-label="Message AquaMind AI"
        />
        <button
          type="submit"
          className="chat-input__send"
          disabled={!canSend}
          aria-label="Send message"
        >
          {disabled ? <LoadingSpinner size={18} label="Sending" /> : <SendIcon width={18} height={18} />}
        </button>
      </div>
      <p className="chat-input__hint">
        AquaMind AI answers from official groundwater datasets and documents. Press Enter to send,
        Shift+Enter for a new line.
      </p>
    </form>
  );
}
