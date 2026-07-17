import { memo } from 'react';
import { ArrowRightIcon } from './Icons';

interface SuggestedPromptsProps {
  onSelect: (prompt: string) => void;
}

/** Curated starter prompts spanning the backend's capabilities. */
export const SUGGESTED_PROMPTS: readonly string[] = [
  'What is the groundwater level in Salem?',
  'Explain what an over-exploited groundwater block means.',
  'Predict the groundwater level in Coimbatore for 2030.',
  'How can groundwater be recharged, and what do you suggest for Chennai?',
];

export const SuggestedPrompts = memo(function SuggestedPrompts({ onSelect }: SuggestedPromptsProps) {
  return (
    <div className="suggested">
      <p className="suggested__label">Try asking</p>
      <div className="suggested__grid">
        {SUGGESTED_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="suggested__card"
            onClick={() => onSelect(prompt)}
          >
            <span>{prompt}</span>
            <ArrowRightIcon width={16} height={16} />
          </button>
        ))}
      </div>
    </div>
  );
});
