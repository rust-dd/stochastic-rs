import { llmText, pagesInOrder } from '@/lib/source';
import { SITE } from '@/lib/site';

export const dynamic = 'force-static';

/**
 * Every documentation page as Markdown, in sidebar order, so an assistant can
 * ingest the whole library in one fetch. Built from the processed Markdown of
 * each page, so the Rust examples are inlined code, not `<RustExample>` tags.
 */
export async function GET(): Promise<Response> {
  const sections = await Promise.all(pagesInOrder().map(llmText));

  const body = [
    `# ${SITE.name} — full documentation`,
    '',
    `> ${SITE.definition}`,
    '',
    `Source: ${SITE.repository}`,
    '',
    sections.join('\n---\n\n'),
  ].join('\n');

  return new Response(body, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
