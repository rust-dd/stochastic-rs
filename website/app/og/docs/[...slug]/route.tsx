import { notFound } from 'next/navigation';
import { ogCard } from '@/lib/og-card';
import { pageImage, source } from '@/lib/source';

export const dynamic = 'force-static';
export const dynamicParams = false;

/** Eyebrow per sidebar section, so a shared link says where on the site it points. */
function eyebrowFor(slugs: string[]): string {
  switch (slugs[0]) {
    case 'tutorials':
      return 'Tutorial · Rust · Python';
    case 'concepts':
      return 'Concept · stochastic-rs';
    case 'getting-started':
      return 'Getting started · stochastic-rs';
    default:
      return 'Documentation · stochastic-rs';
  }
}

export async function GET(_req: Request, { params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params;
  const page = source.getPage(slug.slice(0, -1));
  if (!page) notFound();

  return ogCard({
    eyebrow: eyebrowFor(page.slugs),
    title: page.data.title,
    subtitle: page.data.description,
  });
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({ slug: pageImage(page).segments }));
}
