import { FACTS } from '@/lib/facts';
import { markdownUrl, pagesInOrder } from '@/lib/source';
import { SITE } from '@/lib/site';

export const dynamic = 'force-static';

/**
 * https://llmstxt.org — a flat index of the documentation for assistants.
 * Each entry links the page's Markdown twin (`/docs/<page>.md`), which carries
 * the prose, the math and the compiled Rust examples without the HTML around
 * them; the canonical HTML page is the same URL without the `.md`.
 */
export function GET(): Response {
  const entry = (page: ReturnType<typeof pagesInOrder>[number]) => {
    const url = new URL(markdownUrl(page), SITE.url).toString();
    return page.data.description
      ? `- [${page.data.title}](${url}): ${page.data.description}`
      : `- [${page.data.title}](${url})`;
  };

  const pages = pagesInOrder();
  const tutorials = pages.filter((page) => page.slugs[0] === 'tutorials');
  const rest = pages.filter((page) => page.slugs[0] !== 'tutorials');

  const lines = [
    `# ${SITE.name}`,
    '',
    `> ${SITE.definition}`,
    '',
    `Version ${FACTS.version}. ${FACTS.processes} stochastic processes behind one \`ProcessExt\` trait, ${FACTS.distributions} SIMD distribution samplers, ${FACTS.copulas} copulas, ${FACTS.calibrators} calibrators, and ${FACTS.pythonEntries} Python entries (\`pip install stochastic-rs\`, NumPy in and out). Rust: \`cargo add stochastic-rs\`.`,
    '',
    `Source: ${SITE.repository} · Rust API: ${SITE.docsRs} · Python: ${SITE.pypi} · Cite: https://doi.org/${SITE.doi}`,
    `Full documentation as one file: ${SITE.url}/llms-full.txt`,
    '',
    '## Tutorials',
    '',
    ...tutorials.map(entry),
    '',
    '## Docs',
    '',
    ...rest.map(entry),
    '',
  ];

  return new Response(lines.join('\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
