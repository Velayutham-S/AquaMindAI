import { useNavigate } from 'react-router-dom';
import { useChat } from '../hooks/useChat';
import { useTheme } from '../hooks/useTheme';
import { Logo } from '../components/Logo';
import { SuggestedPrompts } from '../components/SuggestedPrompts';
import { ChatInput } from '../components/ChatInput';
import { ArrowRightIcon, MoonIcon, SparklesIcon, SunIcon } from '../components/Icons';

const FEATURES: ReadonlyArray<{ title: string; description: string }> = [
  {
    title: 'Grounded in official data',
    description:
      'Answers are backed by CGWB and Tamil Nadu groundwater datasets, aquifer reports, and year books.',
  },
  {
    title: 'Structured + document intelligence',
    description:
      'Combines SQL retrieval, semantic search, and machine-learning predictions into one answer.',
  },
  {
    title: 'Evidence-based recommendations',
    description:
      'Get clear, actionable guidance for groundwater management when the situation calls for it.',
  },
];

export function LandingPage() {
  const navigate = useNavigate();
  const { createConversation, sendMessage } = useChat();
  const { theme, toggleTheme } = useTheme();

  const startChat = (prompt?: string) => {
    const id = createConversation();
    navigate(`/chat/${id}`);
    if (prompt) void sendMessage(prompt);
  };

  return (
    <div className="landing">
      <header className="landing__nav">
        <Logo size={36} />
        <div className="landing__nav-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <SunIcon width={18} height={18} /> : <MoonIcon width={18} height={18} />}
          </button>
          <button type="button" className="btn btn--primary" onClick={() => startChat()}>
            Open chat
            <ArrowRightIcon width={16} height={16} />
          </button>
        </div>
      </header>

      <main className="landing__main">
        <section className="landing__hero">
          <span className="landing__badge">
            <SparklesIcon width={15} height={15} />
            Groundwater intelligence for Tamil Nadu
          </span>
          <h1 className="landing__title">
            Ask anything about <span className="landing__title-accent">groundwater</span>.
          </h1>
          <p className="landing__subtitle">
            AquaMind AI turns official groundwater datasets and documents into clear, evidence-based
            answers — levels, rainfall, aquifers, predictions, and management guidance.
          </p>

          <div className="landing__composer">
            <ChatInput onSend={(text) => startChat(text)} autoFocus placeholder="Ask about groundwater levels, rainfall, aquifers…" />
          </div>

          <SuggestedPrompts onSelect={(prompt) => startChat(prompt)} />
        </section>

        <section className="landing__features">
          {FEATURES.map((feature) => (
            <article key={feature.title} className="feature-card">
              <h3 className="feature-card__title">{feature.title}</h3>
              <p className="feature-card__desc">{feature.description}</p>
            </article>
          ))}
        </section>
      </main>

      <footer className="landing__footer">
        <span>AquaMind AI · Agentic RAG groundwater decision-support</span>
      </footer>
    </div>
  );
}
