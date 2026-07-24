import type { ReactElement } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme';
import { Logo } from '../components/Logo';
import {
  ArrowRightIcon,
  BookIcon,
  BrainIcon,
  ChatBubbleIcon,
  CloudRainIcon,
  DatabaseIcon,
  DropletIcon,
  EyeIcon,
  GithubIcon,
  LightbulbIcon,
  MapPinIcon,
  MoonIcon,
  RechargeIcon,
  ShieldCheckIcon,
  SparklesIcon,
  SunIcon,
  TrendingUpIcon,
  type IconProps,
} from '../components/Icons';

type Feature = { title: string; description: string; Icon: (p: IconProps) => ReactElement };

const FEATURES: ReadonlyArray<Feature> = [
  {
    title: 'Groundwater Levels',
    description: 'Latest and historical water-table readings across districts and firkas.',
    Icon: DropletIcon,
  },
  {
    title: 'Rainfall Analysis',
    description: 'Seasonal and annual rainfall trends that shape recharge and availability.',
    Icon: CloudRainIcon,
  },
  {
    title: 'Groundwater Prediction',
    description: 'Machine-learning forecasts of future groundwater levels by year.',
    Icon: TrendingUpIcon,
  },
  {
    title: 'Recharge Recommendations',
    description: 'Evidence-based guidance to recharge and protect local groundwater.',
    Icon: RechargeIcon,
  },
  {
    title: 'Aquifer Knowledge',
    description: 'Concepts, aquifer reports and hydrogeology explained in plain language.',
    Icon: BookIcon,
  },
  {
    title: 'District Intelligence',
    description: 'Safe, critical and over-exploited status for every Tamil Nadu district.',
    Icon: MapPinIcon,
  },
  {
    title: 'Interactive AI Assistant',
    description: 'A conversational assistant that remembers context and explains answers.',
    Icon: ChatBubbleIcon,
  },
  {
    title: 'Official Government Data',
    description: 'Grounded in CGWB, SG&SWRDC and INGRES datasets and year books.',
    Icon: ShieldCheckIcon,
  },
];

const STATS: ReadonlyArray<{ value: string; label: string }> = [
  { value: '1203+', label: 'Villages' },
  { value: '38', label: 'Districts' },
  { value: '5+', label: 'Years of Data' },
  { value: 'Agentic', label: 'AI Powered' },
];

const STEPS: ReadonlyArray<{ title: string; description: string }> = [
  { title: 'You ask a question', description: 'Ask anything about groundwater in natural language.' },
  { title: 'Planner Agent', description: 'A supervisor plans which specialist agents should respond.' },
  { title: 'Specialized AI Agents', description: 'Data, knowledge and prediction agents gather evidence.' },
  { title: 'Groundwater Intelligence', description: 'Evidence is combined and checked for reliability.' },
  { title: 'Final AI Response', description: 'You get a clear, grounded answer with recommendations.' },
];

const REASONS: ReadonlyArray<Feature> = [
  {
    title: 'Official Government Data',
    description: 'Every answer traces back to authoritative CGWB and Tamil Nadu datasets.',
    Icon: DatabaseIcon,
  },
  {
    title: 'Agentic AI',
    description: 'A supervisor orchestrates specialist agents instead of a single black box.',
    Icon: BrainIcon,
  },
  {
    title: 'Predictions',
    description: 'Forecast future groundwater levels to plan ahead with confidence.',
    Icon: TrendingUpIcon,
  },
  {
    title: 'Recommendations',
    description: 'Actionable, situation-aware guidance for recharge and management.',
    Icon: LightbulbIcon,
  },
  {
    title: 'Explainable AI',
    description: 'Answers are grounded in retrieved evidence, not guesses.',
    Icon: EyeIcon,
  },
  {
    title: 'Tamil Nadu Focus',
    description: 'Purpose-built for Tamil Nadu groundwater, districts and firkas.',
    Icon: MapPinIcon,
  },
];

const APP_VERSION = 'v1.0';

export function LandingPage() {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();

  const openChat = () => navigate('/chat');

  return (
    <div className="landing">
      <header className="landing__nav">
        <Logo size={34} />
        <div className="landing__nav-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <SunIcon width={18} height={18} /> : <MoonIcon width={18} height={18} />}
          </button>
          <button type="button" className="btn btn--primary" onClick={openChat}>
            <span>Open Chat</span>
            <ArrowRightIcon width={16} height={16} />
          </button>
        </div>
      </header>

      <main className="landing__main">
        {/* Hero */}
        <section className="hero reveal">
          <span className="hero__badge">
            <SparklesIcon width={15} height={15} />
            Agentic Groundwater Intelligence
          </span>
          <h1 className="hero__title">
            AquaMind AI
            <span className="hero__title-sub">Agentic Groundwater Intelligence for Tamil Nadu</span>
          </h1>
          <p className="hero__subtitle">
            AquaMind AI turns official groundwater datasets and documents into clear, evidence-based
            answers — groundwater insights, predictions, and recommendations powered by AI decision
            support you can trust.
          </p>
          <div className="hero__cta">
            <button type="button" className="btn btn--primary btn--lg" onClick={openChat}>
              <span>Open Chat</span>
              <ArrowRightIcon width={18} height={18} />
            </button>
            <a className="btn btn--secondary btn--lg" href="#features">
              Learn More
            </a>
          </div>

          <div className="stats">
            {STATS.map((stat) => (
              <div key={stat.label} className="stat-card">
                <span className="stat-card__value">{stat.value}</span>
                <span className="stat-card__label">{stat.label}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Features */}
        <section id="features" className="section reveal">
          <div className="section__head">
            <h2 className="section__title">Everything groundwater, in one assistant</h2>
            <p className="section__lead">
              From measured levels to predictions and management guidance — grounded in official data.
            </p>
          </div>
          <div className="feature-grid">
            {FEATURES.map(({ title, description, Icon }) => (
              <article key={title} className="feature-card">
                <span className="feature-card__icon" aria-hidden>
                  <Icon width={22} height={22} />
                </span>
                <h3 className="feature-card__title">{title}</h3>
                <p className="feature-card__desc">{description}</p>
              </article>
            ))}
          </div>
        </section>

        {/* How it works */}
        <section className="section reveal">
          <div className="section__head">
            <h2 className="section__title">How AquaMind AI works</h2>
            <p className="section__lead">An agentic pipeline turns your question into a grounded answer.</p>
          </div>
          <ol className="timeline">
            {STEPS.map((step, index) => (
              <li key={step.title} className="timeline__step">
                <span className="timeline__marker">{index + 1}</span>
                <div className="timeline__body">
                  <h3 className="timeline__title">{step.title}</h3>
                  <p className="timeline__desc">{step.description}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        {/* Why */}
        <section className="section reveal">
          <div className="section__head">
            <h2 className="section__title">Why AquaMind AI</h2>
            <p className="section__lead">Built to be trustworthy, explainable and focused on Tamil Nadu.</p>
          </div>
          <div className="why-grid">
            {REASONS.map(({ title, description, Icon }) => (
              <article key={title} className="why-card">
                <span className="why-card__icon" aria-hidden>
                  <Icon width={20} height={20} />
                </span>
                <div>
                  <h3 className="why-card__title">{title}</h3>
                  <p className="why-card__desc">{description}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* Closing CTA */}
        <section className="cta-band reveal">
          <h2 className="cta-band__title">Ready to explore Tamil Nadu groundwater?</h2>
          <p className="cta-band__lead">Open the assistant and ask your first question.</p>
          <button type="button" className="btn btn--primary btn--lg" onClick={openChat}>
            <span>Open Chat</span>
            <ArrowRightIcon width={18} height={18} />
          </button>
        </section>
      </main>

      <footer className="landing__footer">
        <div className="landing__footer-inner">
          <div className="landing__footer-brand">
            <Logo size={28} />
            <p className="landing__footer-tag">Agentic RAG groundwater decision-support for Tamil Nadu.</p>
          </div>
          <nav className="landing__footer-links" aria-label="Footer">
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer noopener"
              className="landing__footer-link"
            >
              <GithubIcon width={16} height={16} />
              GitHub
            </a>
            <a href="#features" className="landing__footer-link">
              About
            </a>
            <a href="#features" className="landing__footer-link">
              Privacy
            </a>
            <span className="landing__footer-meta">Built with Agentic RAG</span>
            <span className="landing__footer-meta">{APP_VERSION}</span>
          </nav>
        </div>
        <p className="landing__footer-copy">© {new Date().getFullYear()} AquaMind AI</p>
      </footer>
    </div>
  );
}
