import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useChat } from '../hooks/useChat';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { MessageList } from '../components/MessageList';
import { ChatInput } from '../components/ChatInput';
import { ErrorBanner } from '../components/ErrorBanner';
import { SuggestedPrompts } from '../components/SuggestedPrompts';
import { Logo } from '../components/Logo';

function ChatEmptyState({ onSelect }: { onSelect: (prompt: string) => void }) {
  return (
    <div className="chat-empty">
      <div className="chat-empty__inner">
        <Logo size={56} withWordmark={false} />
        <h2 className="chat-empty__title">How can I help with groundwater today?</h2>
        <p className="chat-empty__subtitle">
          Ask about groundwater levels, rainfall, aquifers, predictions, or management guidance for
          Tamil Nadu.
        </p>
        <SuggestedPrompts onSelect={onSelect} />
      </div>
    </div>
  );
}

export function ChatPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const {
    conversations,
    activeConversation,
    isLoading,
    error,
    selectConversation,
    sendMessage,
    retryLastMessage,
    dismissError,
  } = useChat();

  // Keep the URL and the active conversation in sync.
  useEffect(() => {
    if (id) {
      if (conversations.some((c) => c.id === id)) {
        if (activeConversation?.id !== id) selectConversation(id);
      } else {
        navigate('/chat', { replace: true });
      }
      return;
    }
    if (activeConversation) {
      navigate(`/chat/${activeConversation.id}`, { replace: true });
    }
  }, [id, activeConversation, conversations, selectConversation, navigate]);

  const [collapsed, setCollapsed] = useLocalStorage('aquamind.sidebarCollapsed', false);
  const toggleCollapse = () => setCollapsed((prev) => !prev);

  const messages = activeConversation?.messages ?? [];
  const hasMessages = messages.length > 0;

  return (
    <div className={`chat-layout${collapsed ? ' chat-layout--collapsed' : ''}`}>
      <Sidebar collapsed={collapsed} onToggleCollapse={toggleCollapse} />
      <div className="chat-main">
        <Header collapsed={collapsed} onToggleCollapse={toggleCollapse} />

        {hasMessages ? (
          <MessageList messages={messages} isLoading={isLoading} onRetry={() => void retryLastMessage()} />
        ) : (
          <ChatEmptyState onSelect={(prompt) => void sendMessage(prompt)} />
        )}

        <div className="chat-footer">
          {error && (
            <div className="chat-footer__error">
              <ErrorBanner
                message={error.message}
                onRetry={() => void retryLastMessage()}
                onDismiss={dismissError}
                retrying={isLoading}
              />
            </div>
          )}
          <ChatInput onSend={(text) => void sendMessage(text)} disabled={isLoading} autoFocus />
        </div>
      </div>
    </div>
  );
}
