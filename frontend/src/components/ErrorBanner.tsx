import { AlertIcon, CloseIcon, RetryIcon } from './Icons';

interface ErrorBannerProps {
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
  retrying?: boolean;
}

/** Dismissible error banner with an optional retry action. */
export function ErrorBanner({ message, onRetry, onDismiss, retrying }: ErrorBannerProps) {
  return (
    <div className="error-banner" role="alert">
      <span className="error-banner__icon">
        <AlertIcon width={18} height={18} />
      </span>
      <span className="error-banner__message">{message}</span>
      <span className="error-banner__actions">
        {onRetry && (
          <button
            type="button"
            className="error-banner__btn"
            onClick={onRetry}
            disabled={retrying}
          >
            <RetryIcon width={16} height={16} />
            {retrying ? 'Retrying…' : 'Retry'}
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            className="error-banner__btn error-banner__btn--ghost"
            onClick={onDismiss}
            aria-label="Dismiss error"
          >
            <CloseIcon width={16} height={16} />
          </button>
        )}
      </span>
    </div>
  );
}
