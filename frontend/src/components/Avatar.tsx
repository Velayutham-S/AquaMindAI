import { memo } from 'react';
import type { MessageRole } from '../types';
import { UserIcon } from './Icons';
import logoUrl from '../assets/logo.svg';

interface AvatarProps {
  role: MessageRole;
}

/** Circular avatar: AquaMind logo for the assistant, a user glyph otherwise. */
export const Avatar = memo(function Avatar({ role }: AvatarProps) {
  if (role === 'assistant') {
    return (
      <span className="avatar avatar--assistant" aria-label="AquaMind AI">
        <img src={logoUrl} width={22} height={22} alt="" />
      </span>
    );
  }
  return (
    <span className="avatar avatar--user" aria-label="You">
      <UserIcon width={18} height={18} />
    </span>
  );
});
