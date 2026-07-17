import { memo, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CheckIcon, CopyIcon } from './Icons';
import { useClipboard } from '../hooks/useClipboard';

interface MarkdownRendererProps {
  content: string;
}

/** Minimal hast-like node shape (avoids pulling in hast type packages). */
interface HastNode {
  type?: string;
  value?: string;
  tagName?: string;
  properties?: { className?: unknown };
  children?: HastNode[];
}

/** Recursively collect the raw text of a hast node (pre-highlight source). */
function hastToText(node: HastNode | undefined): string {
  if (!node) return '';
  if (node.type === 'text') return node.value ?? '';
  return (node.children ?? []).map(hastToText).join('');
}

function extractLanguage(node: HastNode | undefined): string | null {
  const codeChild = node?.children?.find((c) => c.tagName === 'code');
  const className = codeChild?.properties?.className;
  const classes = Array.isArray(className) ? (className as string[]) : [];
  const lang = classes.find((c) => typeof c === 'string' && c.startsWith('language-'));
  return lang ? lang.replace('language-', '') : null;
}

function CodeBlock({ code, language, children }: { code: string; language: string | null; children: ReactNode }) {
  const { copied, copy } = useClipboard();
  return (
    <div className="code-block">
      <div className="code-block__header">
        <span className="code-block__lang">{language ?? 'code'}</span>
        <button
          type="button"
          className="code-block__copy"
          onClick={() => void copy(code)}
          aria-label="Copy code"
        >
          {copied ? <CheckIcon width={14} height={14} /> : <CopyIcon width={14} height={14} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="code-block__pre">{children}</pre>
    </div>
  );
}

const components: Components = {
  // Wrap fenced code blocks with a header + copy button. `node` carries the
  // original source text even after rehype-highlight adds highlight spans.
  pre({ node, children }) {
    const code = hastToText(node as unknown as HastNode);
    const language = extractLanguage(node as unknown as HastNode);
    return (
      <CodeBlock code={code} language={language}>
        {children}
      </CodeBlock>
    );
  },
  // Open links safely in a new tab.
  a({ children, href }) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  },
  // Prevent nested tables from overflowing on mobile.
  table({ children }) {
    return (
      <div className="markdown-table-wrap">
        <table>{children}</table>
      </div>
    );
  },
};

/** Renders assistant markdown safely with GFM + syntax-highlighted code. */
export const MarkdownRenderer = memo(function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
});
