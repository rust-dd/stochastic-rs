import { notFound } from 'next/navigation';
import { llmText, source } from '@/lib/source';

export const dynamic = 'force-static';
export const dynamicParams = false;

/** Each page's Markdown twin, reached through the rewrites in `next.config.mjs`. */
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
