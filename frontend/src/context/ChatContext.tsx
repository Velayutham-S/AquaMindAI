import {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from 'react';
import type { ChatMessage, Conversation, NormalizedError } from '../types';
import { ChatServiceError, sendMessage as sendMessageRequest } from '../services/chatService';
import { deriveTitle, generateId } from '../utils';

// --------------------------------------------------------------------------- //
// State + actions
// --------------------------------------------------------------------------- //

interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  isLoading: boolean;
  error: NormalizedError | null;
  sidebarOpen: boolean;
}

type ChatAction =
  | { type: 'HYDRATE'; payload: { conversations: Conversation[]; activeId: string | null } }
  | { type: 'CREATE_CONVERSATION'; payload: Conversation }
  | { type: 'SELECT_CONVERSATION'; payload: string }
  | { type: 'DELETE_CONVERSATION'; payload: string }
  | { type: 'CLEAR_MESSAGES'; payload: string }
  | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: ChatMessage } }
  | {
      type: 'UPDATE_MESSAGE';
      payload: { conversationId: string; messageId: string; patch: Partial<ChatMessage> };
    }
  | { type: 'REMOVE_MESSAGE'; payload: { conversationId: string; messageId: string } }
  | { type: 'SET_TITLE'; payload: { conversationId: string; title: string } }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: NormalizedError | null }
  | { type: 'SET_SIDEBAR'; payload: boolean }
  | { type: 'TOGGLE_SIDEBAR' };

const STORAGE_KEY = 'aquamind.conversations';
const ACTIVE_KEY = 'aquamind.activeConversation';

const initialState: ChatState = {
  conversations: [],
  activeId: null,
  isLoading: false,
  error: null,
  sidebarOpen: true,
};

function touch(conversation: Conversation): Conversation {
  return { ...conversation, updatedAt: Date.now() };
}

function mapConversation(
  state: ChatState,
  conversationId: string,
  updater: (c: Conversation) => Conversation,
): Conversation[] {
  return state.conversations.map((c) => (c.id === conversationId ? updater(c) : c));
}

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'HYDRATE':
      return { ...state, ...action.payload };

    case 'CREATE_CONVERSATION':
      return {
        ...state,
        conversations: [action.payload, ...state.conversations],
        activeId: action.payload.id,
        error: null,
      };

    case 'SELECT_CONVERSATION':
      return { ...state, activeId: action.payload, error: null };

    case 'DELETE_CONVERSATION': {
      const conversations = state.conversations.filter((c) => c.id !== action.payload);
      const activeId =
        state.activeId === action.payload ? (conversations[0]?.id ?? null) : state.activeId;
      return { ...state, conversations, activeId };
    }

    case 'CLEAR_MESSAGES':
      return {
        ...state,
        error: null,
        conversations: mapConversation(state, action.payload, (c) =>
          touch({ ...c, messages: [] }),
        ),
      };

    case 'ADD_MESSAGE':
      return {
        ...state,
        conversations: mapConversation(state, action.payload.conversationId, (c) =>
          touch({ ...c, messages: [...c.messages, action.payload.message] }),
        ),
      };

    case 'UPDATE_MESSAGE':
      return {
        ...state,
        conversations: mapConversation(state, action.payload.conversationId, (c) =>
          touch({
            ...c,
            messages: c.messages.map((m) =>
              m.id === action.payload.messageId ? { ...m, ...action.payload.patch } : m,
            ),
          }),
        ),
      };

    case 'REMOVE_MESSAGE':
      return {
        ...state,
        conversations: mapConversation(state, action.payload.conversationId, (c) =>
          touch({
            ...c,
            messages: c.messages.filter((m) => m.id !== action.payload.messageId),
          }),
        ),
      };

    case 'SET_TITLE':
      return {
        ...state,
        conversations: mapConversation(state, action.payload.conversationId, (c) => ({
          ...c,
          title: action.payload.title,
        })),
      };

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload };

    case 'SET_SIDEBAR':
      return { ...state, sidebarOpen: action.payload };

    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen };

    default:
      return state;
  }
}

// --------------------------------------------------------------------------- //
// Context value
// --------------------------------------------------------------------------- //

export interface ChatContextValue {
  conversations: Conversation[];
  activeConversation: Conversation | null;
  isLoading: boolean;
  error: NormalizedError | null;
  sidebarOpen: boolean;
  createConversation: () => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  clearActiveMessages: () => void;
  sendMessage: (text: string) => Promise<void>;
  retryLastMessage: () => Promise<void>;
  toggleSidebar: (open?: boolean) => void;
  dismissError: () => void;
}

export const ChatContext = createContext<ChatContextValue | undefined>(undefined);

function readInitialState(): ChatState {
  if (typeof window === 'undefined') return initialState;
  try {
    const rawConversations = window.localStorage.getItem(STORAGE_KEY);
    const conversations = rawConversations
      ? (JSON.parse(rawConversations) as Conversation[])
      : [];
    const activeId = window.localStorage.getItem(ACTIVE_KEY);
    const isMobile = window.matchMedia?.('(max-width: 900px)').matches ?? false;
    return {
      ...initialState,
      sidebarOpen: !isMobile,
      conversations: Array.isArray(conversations) ? conversations : [],
      activeId: activeId && conversations.some((c) => c.id === activeId) ? activeId : null,
    };
  } catch {
    return initialState;
  }
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, undefined, readInitialState);
  const abortRef = useRef<AbortController | null>(null);

  // Persist conversations + active id.
  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations));
    } catch {
      /* ignore quota/availability errors */
    }
  }, [state.conversations]);

  useEffect(() => {
    try {
      if (state.activeId) window.localStorage.setItem(ACTIVE_KEY, state.activeId);
      else window.localStorage.removeItem(ACTIVE_KEY);
    } catch {
      /* ignore */
    }
  }, [state.activeId]);

  // Abort any in-flight request on unmount.
  useEffect(() => () => abortRef.current?.abort(), []);

  const activeConversation = useMemo(
    () => state.conversations.find((c) => c.id === state.activeId) ?? null,
    [state.conversations, state.activeId],
  );

  const createConversation = useCallback((): string => {
    const now = Date.now();
    const conversation: Conversation = {
      id: generateId(),
      title: 'New conversation',
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    dispatch({ type: 'CREATE_CONVERSATION', payload: conversation });
    return conversation.id;
  }, []);

  const selectConversation = useCallback((id: string) => {
    dispatch({ type: 'SELECT_CONVERSATION', payload: id });
  }, []);

  const deleteConversation = useCallback((id: string) => {
    dispatch({ type: 'DELETE_CONVERSATION', payload: id });
  }, []);

  const clearActiveMessages = useCallback(() => {
    if (state.activeId) dispatch({ type: 'CLEAR_MESSAGES', payload: state.activeId });
  }, [state.activeId]);

  const toggleSidebar = useCallback((open?: boolean) => {
    if (typeof open === 'boolean') dispatch({ type: 'SET_SIDEBAR', payload: open });
    else dispatch({ type: 'TOGGLE_SIDEBAR' });
  }, []);

  const dismissError = useCallback(() => dispatch({ type: 'SET_ERROR', payload: null }), []);

  /**
   * Core send flow: ensures a conversation exists, records the user message,
   * streams in the assistant placeholder, calls the backend, and resolves the
   * assistant message (or marks it errored).
   */
  const runSend = useCallback(
    async (conversationId: string, text: string, isFirstMessage: boolean) => {
      if (isFirstMessage) {
        dispatch({
          type: 'SET_TITLE',
          payload: { conversationId, title: deriveTitle(text) },
        });
      }

      const assistantId = generateId();
      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          conversationId,
          message: {
            id: assistantId,
            role: 'assistant',
            content: '',
            timestamp: Date.now(),
            status: 'sending',
          },
        },
      });

      dispatch({ type: 'SET_ERROR', payload: null });
      dispatch({ type: 'SET_LOADING', payload: true });

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const answer = await sendMessageRequest(text, controller.signal);
        dispatch({
          type: 'UPDATE_MESSAGE',
          payload: {
            conversationId,
            messageId: assistantId,
            patch: { content: answer, status: 'done', timestamp: Date.now(), error: undefined },
          },
        });
      } catch (error) {
        // Ignore silent cancellations (e.g. a newer send superseded this one).
        const cancelled =
          (error instanceof DOMException && error.name === 'AbortError') ||
          (typeof error === 'object' && error !== null && (error as { code?: string }).code === 'ERR_CANCELED');
        if (cancelled) return;

        const normalized =
          error instanceof ChatServiceError
            ? error.normalized
            : { kind: 'unknown' as const, message: 'Something went wrong. Please try again.' };

        dispatch({
          type: 'UPDATE_MESSAGE',
          payload: {
            conversationId,
            messageId: assistantId,
            patch: { status: 'error', error: normalized.message },
          },
        });
        dispatch({ type: 'SET_ERROR', payload: normalized });
      } finally {
        if (abortRef.current === controller) {
          dispatch({ type: 'SET_LOADING', payload: false });
        }
      }
    },
    [],
  );

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || state.isLoading) return;

      let conversationId = state.activeId;
      let isFirstMessage = false;

      if (!conversationId) {
        conversationId = createConversation();
        isFirstMessage = true;
      } else {
        const current = state.conversations.find((c) => c.id === conversationId);
        isFirstMessage = !current || current.messages.length === 0;
      }

      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          conversationId,
          message: {
            id: generateId(),
            role: 'user',
            content: trimmed,
            timestamp: Date.now(),
            status: 'done',
          },
        },
      });

      await runSend(conversationId, trimmed, isFirstMessage);
    },
    [state.activeId, state.conversations, state.isLoading, runSend, createConversation],
  );

  const retryLastMessage = useCallback(async () => {
    if (state.isLoading || !activeConversation) return;
    const { messages } = activeConversation;

    // Drop a trailing errored assistant message, if present.
    const last = messages[messages.length - 1];
    if (last && last.role === 'assistant' && last.status === 'error') {
      dispatch({
        type: 'REMOVE_MESSAGE',
        payload: { conversationId: activeConversation.id, messageId: last.id },
      });
    }

    // Find the most recent user message to resend.
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');
    if (!lastUser) return;

    await runSend(activeConversation.id, lastUser.content, false);
  }, [state.isLoading, activeConversation, runSend]);

  const value = useMemo<ChatContextValue>(
    () => ({
      conversations: state.conversations,
      activeConversation,
      isLoading: state.isLoading,
      error: state.error,
      sidebarOpen: state.sidebarOpen,
      createConversation,
      selectConversation,
      deleteConversation,
      clearActiveMessages,
      sendMessage,
      retryLastMessage,
      toggleSidebar,
      dismissError,
    }),
    [
      state.conversations,
      state.isLoading,
      state.error,
      state.sidebarOpen,
      activeConversation,
      createConversation,
      selectConversation,
      deleteConversation,
      clearActiveMessages,
      sendMessage,
      retryLastMessage,
      toggleSidebar,
      dismissError,
    ],
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}
