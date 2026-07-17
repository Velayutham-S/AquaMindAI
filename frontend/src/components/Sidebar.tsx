import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChat } from '../hooks/useChat';
import type { Conversation } from '../types';
import { Logo } from './Logo';
import { CloseIcon, PlusIcon, TrashIcon } from './Icons';
import { formatRelativeDay } from '../utils';

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

export function Sidebar() {
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

  const groups = useMemo(() => groupConversations(conversations), [conversations]);
  const isMobile = () => typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches;

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

  const handleDelete = (event: React.MouseEvent, id: string) => {
    event.stopPropagation();
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
      <aside className={`sidebar${sidebarOpen ? ' sidebar--open' : ''}`} aria-label="Conversation history">
        <div className="sidebar__header">
          <Logo size={30} />
          <button
            type="button"
            className="sidebar__close icon-btn"
            onClick={() => toggleSidebar(false)}
            aria-label="Close sidebar"
          >
            <CloseIcon width={18} height={18} />
          </button>
        </div>

        <button type="button" className="sidebar__new" onClick={handleNew}>
          <PlusIcon width={18} height={18} />
          New chat
        </button>

        <nav className="sidebar__list">
          {conversations.length === 0 ? (
            <p className="sidebar__empty">No conversations yet. Start a new chat to begin.</p>
          ) : (
            groups.map((group) => (
              <div key={group.label} className="sidebar__group">
                <p className="sidebar__group-label">{group.label}</p>
                {group.items.map((conversation) => (
                  <button
                    key={conversation.id}
                    type="button"
                    className={`sidebar__item${
                      activeConversation?.id === conversation.id ? ' sidebar__item--active' : ''
                    }`}
                    onClick={() => handleSelect(conversation.id)}
                    title={conversation.title}
                  >
                    <span className="sidebar__item-title">{conversation.title}</span>
                    <span
                      className="sidebar__item-delete"
                      role="button"
                      tabIndex={0}
                      aria-label={`Delete conversation "${conversation.title}"`}
                      onClick={(e) => handleDelete(e, conversation.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          handleDelete(e as unknown as React.MouseEvent, conversation.id);
                        }
                      }}
                    >
                      <TrashIcon width={15} height={15} />
                    </span>
                  </button>
                ))}
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
