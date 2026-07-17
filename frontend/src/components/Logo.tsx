import logoUrl from '../assets/logo.svg';

interface LogoProps {
  size?: number;
  withWordmark?: boolean;
  className?: string;
}

/** AquaMind AI water-themed logo with optional wordmark. */
export function Logo({ size = 36, withWordmark = true, className }: LogoProps) {
  return (
    <span className={`logo${className ? ` ${className}` : ''}`}>
      <img src={logoUrl} width={size} height={size} alt="AquaMind AI" className="logo__mark" />
      {withWordmark && (
        <span className="logo__wordmark">
          Aqua<span className="logo__wordmark-accent">Mind</span> AI
        </span>
      )}
    </span>
  );
}
