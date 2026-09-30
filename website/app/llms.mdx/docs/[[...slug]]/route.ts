import { notFound } from 'next/navigation';
import { llmText, source } from '@/lib/source';

export const dynamic = 'force-static';
export const dynamicParams = false;

/**
 * The Markdown twin of every docs page. `next.config.mjs` rewrites
 * `/docs/<page>.md` and `/docs/<page>.mdx` here, and does the same for a
 * request to `/docs/<page>` whose `Accept` header asks for Markdown.
 */
export async function GET(_req: Request, { params }: { params: Promise<{ slug?: string[] }> }) {
  const { slug } = await params;
  const page = source.getPage(slug);
  if (!page) notFound();

  return new Response(await llmText(page), {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });
}

export function generateStaticParams() {
  return source.generateParams();
}
