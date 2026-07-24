import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChat } from '../hooks/useChat';
import type { Conversation } from '../types';
import { Logo } from './Logo';
import { ChevronLeftIcon, ChevronRightIcon, CloseIcon, DotsIcon, PlusIcon, TrashIcon } from './Icons';
import { formatRelativeDay } from '../utils';

interface SidebarProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

interface SidebarGroup {
  label: string;
  items: Conversation[];
}

function groupConversations(conversations: Conversation[]): SidebarGroup[] {
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const groups: SidebarGroup[] = [];
  const indexByLabel = new Map<string, number>();

  for (const conversation of sorted) {
    const label = formatRelativeDay(conversation.updatedAt);
    const existing = indexByLabel.get(label);
    if (existing === undefined) {
      indexByLabel.set(label, groups.length);
      groups.push({ label, items: [conversation] });
    } else {
      groups[existing].items.push(conversation);
    }
  }
  return groups;
}

export function Sidebar({ collapsed = false, onToggleCollapse }: SidebarProps) {
  const navigate = useNavigate();
  const {
    conversations,
    activeConversation,
    sidebarOpen,
    createConversation,
    selectConversation,
    deleteConversation,
    toggleSidebar,
  } = useChat();

  const [menuId, setMenuId] = useState<string | null>(null);
  const listRef = useRef<HTMLElement>(null);

  const groups = useMemo(() => groupConversations(conversations), [conversations]);
  const isMobile = () => typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches;

  useEffect(() => {
    if (!menuId) return;
    const onPointer = (event: MouseEvent) => {
      if (listRef.current && !listRef.current.contains(event.target as Node)) setMenuId(null);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMenuId(null);
    };
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [menuId]);

  const handleNew = () => {
    const id = createConversation();
    navigate(`/chat/${id}`);
    if (isMobile()) toggleSidebar(false);
  };

  const handleSelect = (id: string) => {
    selectConversation(id);
    navigate(`/chat/${id}`);
    if (isMobile()) toggleSidebar(false);
  };

  const handleDelete = (id: string) => {
    setMenuId(null);
    const wasActive = activeConversation?.id === id;
    deleteConversation(id);
    if (wasActive) navigate('/chat');
  };

  return (
    <>
      <div
        className={`sidebar-overlay${sidebarOpen ? ' sidebar-overlay--visible' : ''}`}
        onClick={() => toggleSidebar(false)}
        aria-hidden
      />
      <aside
        className={`sidebar${sidebarOpen ? ' sidebar--open' : ''}${
          collapsed ? ' sidebar--collapsed' : ''
        }`}
        aria-label="Conversation history"
      >
        <div className="sidebar__header">
          <Logo size={30} className="sidebar__logo" />
          <button
            type="button"
            className="sidebar__collapse icon-btn"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? (
              <ChevronRightIcon width={18} height={18} />
            ) : (
              <ChevronLeftIcon width={18} height={18} />
            )}
          </button>
          <button
            type="button"
            className="sidebar__close icon-btn"
            onClick={() => toggleSidebar(false)}
            aria-label="Close sidebar"
          >
            <CloseIcon width={18} height={18} />
          </button>
        </div>

        <button
          type="button"
          className="sidebar__new"
          onClick={handleNew}
          title="New chat"
          aria-label="New chat"
        >
          <PlusIcon width={18} height={18} />
          <span className="sidebar__new-label">New chat</span>
        </button>

        <nav className="sidebar__list" ref={listRef} aria-label="Conversations">
          {conversations.length === 0 ? (
            <p className="sidebar__empty">No conversations yet. Start a new chat to begin.</p>
          ) : (
            groups.map((group) => (
              <div key={group.label} className="sidebar__group">
                <p className="sidebar__group-label">{group.label}</p>
                {group.items.map((conversation) => {
                  const isActive = activeConversation?.id === conversation.id;
                  const isMenuOpen = menuId === conversation.id;
                  return (
                    <div
                      key={conversation.id}
                      className={`sidebar__item${isActive ? ' sidebar__item--active' : ''}${
                        isMenuOpen ? ' sidebar__item--menu-open' : ''
                      }`}
                    >
                      <button
                        type="button"
                        className="sidebar__item-main"
                        onClick={() => handleSelect(conversation.id)}
                        title={conversation.title}
                      >
                        <span className="sidebar__item-title">{conversation.title}</span>
                      </button>
                      <button
                        type="button"
                        className="sidebar__item-menu-btn"
                        aria-label={`Options for "${conversation.title}"`}
                        aria-haspopup="menu"
                        aria-expanded={isMenuOpen}
                        onClick={(event) => {
                          event.stopPropagation();
                          setMenuId(isMenuOpen ? null : conversation.id);
                        }}
                      >
                        <DotsIcon width={16} height={16} />
                      </button>
                      {isMenuOpen && (
                        <div className="menu__dropdown menu__dropdown--item" role="menu">
                          <button
                            type="button"
                            className="menu__item menu__item--danger"
                            role="menuitem"
                            onClick={() => handleDelete(conversation.id)}
                          >
                            <TrashIcon width={16} height={16} />
                            Delete
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </nav>

        <div className="sidebar__footer">
          <span className="sidebar__footer-text">AquaMind AI · Tamil Nadu Groundwater</span>
        </div>
      </aside>
    </>
  );
}
