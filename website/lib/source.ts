import { docs } from '@/.source/server';
import { loader, type InferPageType } from 'fumadocs-core/source';
import type { Reference } from '@/lib/frontmatter';
import { SITE } from '@/lib/site';

export const source = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
});

export type DocsPage = InferPageType<typeof source>;

/**
 * The page's social card. The trailing `image.png` segment keeps the
 * catch-all route non-empty for the docs index, whose own slug list is empty.
 */
export function pageImage(page: DocsPage): { segments: string[]; url: string } {
  const segments = [...page.slugs, 'image.png'];
  return { segments, url: `/og/docs/${segments.join('/')}` };
}

/** `/docs/processes` → `/docs/processes.md`; `/docs` → `/docs.md`. */
export function markdownUrl(page: DocsPage): string {
  return `${page.url}.md`;
}

export function referenceLink(ref: Reference): { href: string; label: string } | null {
  if (ref.doi) return { href: `https://doi.org/${ref.doi}`, label: `doi:${ref.doi}` };
  if (ref.arxiv) return { href: `https://arxiv.org/abs/${ref.arxiv}`, label: `arXiv:${ref.arxiv}` };
  if (ref.url) return { href: ref.url, label: ref.url };
  return null;
}

function referencesMarkdown(refs: Reference[]): string {
  if (refs.length === 0) return '';
  const items = refs.map((ref) => {
    const link = referenceLink(ref);
    return `- ${ref.author} (${ref.year}). ${ref.title}.${link ? ` ${link.label}` : ''}`;
  });
  return ['', '## References', '', ...items].join('\n');
}

/**
 * A page as plain Markdown for assistants: title, canonical URL, summary, the
 * processed body (Rust examples already inlined, layout wrappers dropped) and
 * the frontmatter references the HTML page renders as its reference list.
 */
export async function llmText(page: DocsPage): Promise<string> {
  const body = await page.data.getText('processed');
  return [
    `# ${page.data.title}`,
    '',
    `URL: ${new URL(page.url, SITE.url)}`,
    '',
    page.data.description ? `> ${page.data.description}\n` : '',
    body.trim(),
    referencesMarkdown(page.data.references),
    '',
  ].join('\n');
}

/** Pages in sidebar order, so the concatenated file reads like the site. */
export function pagesInOrder(): DocsPage[] {
  const order = new Map<string, number>();
  let i = 0;
  const walk = (nodes: (typeof source.pageTree)['children']) => {
    for (const node of nodes) {
      if (node.type === 'page') order.set(node.url, i++);
      else if (node.type === 'folder') {
        if (node.index) order.set(node.index.url, i++);
        walk(node.children);
      }
    }
  };
  walk(source.pageTree.children);
  return source
    .getPages()
    .slice()
    .sort((a, b) => (order.get(a.url) ?? 1e9) - (order.get(b.url) ?? 1e9));
}
