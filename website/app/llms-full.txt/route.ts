import { llmText, pagesInOrder } from '@/lib/source';
import { SITE } from '@/lib/site';

export const dynamic = 'force-static';

/** Every docs page as one Markdown file, so an assistant can ingest the library in one fetch. */
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
