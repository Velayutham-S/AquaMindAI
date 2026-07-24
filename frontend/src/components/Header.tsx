import { useEffect, useRef, useState } from 'react';
import { useChat } from '../hooks/useChat';
import { useTheme } from '../hooks/useTheme';
import { DotsIcon, MenuIcon, MoonIcon, PanelLeftIcon, SunIcon, TrashIcon } from './Icons';
import { Logo } from './Logo';

interface HeaderProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

/** Chat page top bar. Shows the static product brand (never the conversation title). */
export function Header({ collapsed = false, onToggleCollapse }: HeaderProps) {
  const { activeConversation, clearActiveMessages, toggleSidebar } = useChat();
  const { theme, toggleTheme } = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const hasMessages = (activeConversation?.messages.length ?? 0) > 0;

  useEffect(() => {
    if (!menuOpen) return;
    const onPointer = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) setMenuOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMenuOpen(false);
    };
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [menuOpen]);

  const handleClear = () => {
    setMenuOpen(false);
    if (hasMessages && window.confirm('Clear all messages in this conversation?')) {
      clearActiveMessages();
    }
  };

  return (
    <header className="header">
      <div className="header__gradient" aria-hidden />
      <div className="header__content">
        <div className="header__left">
          <button
            type="button"
            className="icon-btn header__menu"
            onClick={() => toggleSidebar()}
            aria-label="Open menu"
          >
            <MenuIcon width={20} height={20} />
          </button>
          <button
            type="button"
            className="icon-btn header__collapse"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <PanelLeftIcon width={20} height={20} />
          </button>
          <span className="header__brand">
            <Logo size={26} withWordmark={false} className="header__logo" />
            <span className="header__brand-text">AquaMind AI</span>
          </span>
        </div>

        <div className="header__actions">
          <button
            type="button"
            className="icon-btn"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <SunIcon width={18} height={18} /> : <MoonIcon width={18} height={18} />}
          </button>

          <div className="menu" ref={menuRef}>
            <button
              type="button"
              className="icon-btn"
              onClick={() => setMenuOpen((open) => !open)}
              aria-label="More options"
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              title="More options"
            >
              <DotsIcon width={18} height={18} />
            </button>
            {menuOpen && (
              <div className="menu__dropdown" role="menu">
                <button
                  type="button"
                  className="menu__item menu__item--danger"
                  role="menuitem"
                  onClick={handleClear}
                  disabled={!hasMessages}
                >
                  <TrashIcon width={16} height={16} />
                  Clear chat
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
