import { source } from '@/lib/source';
import { SITE } from '@/lib/site';

export const dynamic = 'force-static';

/**
 * https://llmstxt.org — a flat index of the documentation for assistants that
 * would otherwise have to scrape the rendered SPA. Points at the `.mdx` source
 * of each page so a model fetching a link gets prose, not markup.
 */
export function GET(): Response {
  const pages = source
    .getPages()
    .slice()
    .sort((a, b) => a.url.localeCompare(b.url));

  const lines = [
    `# ${SITE.name}`,
    '',
    `> ${SITE.description}`,
    '',
    `Source: ${SITE.repository} · Rust API: ${SITE.docsRs} · Python: ${SITE.pypi}`,
    `Full documentation as one file: ${SITE.url}/llms-full.txt`,
    '',
    '## Docs',
    '',
    ...pages.map((page) => {
      const url = new URL(page.url, SITE.url).toString();
      const description = page.data.description;
      return description
        ? `- [${page.data.title}](${url}): ${description}`
        : `- [${page.data.title}](${url})`;
    }),
    '',
  ];

  return new Response(lines.join('\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
