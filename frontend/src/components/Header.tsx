import { useChat } from '../hooks/useChat';
import { useTheme } from '../hooks/useTheme';
import { MenuIcon, MoonIcon, SunIcon, TrashIcon } from './Icons';
import { Logo } from './Logo';

/** Chat page top bar with an animated gradient accent. */
export function Header() {
  const { activeConversation, clearActiveMessages, toggleSidebar } = useChat();
  const { theme, toggleTheme } = useTheme();

  const hasMessages = (activeConversation?.messages.length ?? 0) > 0;
  const title = activeConversation?.title && hasMessages ? activeConversation.title : 'AquaMind AI';

  const handleClear = () => {
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
            aria-label="Toggle sidebar"
          >
            <MenuIcon width={20} height={20} />
          </button>
          <span className="header__title-wrap">
            <Logo size={26} withWordmark={false} className="header__logo" />
            <span className="header__title">{title}</span>
          </span>
        </div>

        <div className="header__actions">
          {hasMessages && (
            <button
              type="button"
              className="icon-btn"
              onClick={handleClear}
              aria-label="Clear chat"
              title="Clear chat"
            >
              <TrashIcon width={18} height={18} />
            </button>
          )}
          <button
            type="button"
            className="icon-btn"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <SunIcon width={18} height={18} /> : <MoonIcon width={18} height={18} />}
          </button>
        </div>
      </div>
    </header>
  );
}
