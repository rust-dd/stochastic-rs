'use client';

import { useState } from 'react';

const BUTTON =
  'inline-flex items-center rounded-md border border-fd-border px-2.5 py-1 text-xs font-medium text-fd-muted-foreground transition hover:bg-fd-accent hover:text-fd-accent-foreground';

const EXTERNAL = { target: '_blank', rel: 'noopener noreferrer nofollow' } as const;

export interface PageActionsProps {
  /** Site-relative Markdown twin of the page, fetched for the copy button. */
  markdownPath: string;
  /** The same URL, absolute, for the assistant prompts. */
  markdownUrl: string;
}

/**
 * Hands the page to a reader's assistant: copy it as Markdown, open the
 * Markdown twin, or start a ChatGPT / Claude conversation primed with its URL.
 */
export function PageActions({ markdownPath, markdownUrl }: PageActionsProps) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const prompt = encodeURIComponent(`Read ${markdownUrl}, I want to ask questions about it.`);

  async function copy() {
    try {
      const response = await fetch(markdownPath);
      await navigator.clipboard.writeText(await response.text());
      setState('copied');
    } catch {
      setState('failed');
    }
    setTimeout(() => setState('idle'), 2000);
  }

  return (
    <div className="not-prose flex flex-wrap items-center gap-2 border-b border-fd-border pb-6">
      <button type="button" onClick={copy} className={BUTTON}>
        {state === 'copied' ? 'Copied' : state === 'failed' ? 'Copy failed' : 'Copy Markdown'}
      </button>
      <a href={markdownPath} className={BUTTON}>
        View as Markdown
      </a>
      <a href={`https://chatgpt.com/?hints=search&q=${prompt}`} className={BUTTON} {...EXTERNAL}>
        Open in ChatGPT
      </a>
      <a href={`https://claude.ai/new?q=${prompt}`} className={BUTTON} {...EXTERNAL}>
        Open in Claude
      </a>
    </div>
  );
}
