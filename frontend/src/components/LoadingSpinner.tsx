interface LoadingSpinnerProps {
  size?: number;
  label?: string;
}

/** Accessible circular loading spinner. */
export function LoadingSpinner({ size = 20, label = 'Loading' }: LoadingSpinnerProps) {
  return (
    <span
      className="spinner"
      style={{ width: size, height: size }}
      role="status"
      aria-label={label}
    />
  );
}
