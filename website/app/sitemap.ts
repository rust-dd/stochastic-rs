import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';
import { SITE } from '@/lib/site';

/**
 * Ranks the landing page above the docs root, and the docs root above the
 * individual pages, so crawlers spend their budget on the hubs first.
 */
function priorityFor(url: string): number {
  if (url === '/docs') return 0.9;
  if (url.startsWith('/docs/getting-started')) return 0.8;
  return 0.7;
}

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();

  const home = {
    url: SITE.url,
    lastModified,
    changeFrequency: 'weekly' as const,
    priority: 1,
  };

  const pages = source.getPages().map((page) => ({
    url: new URL(page.url, SITE.url).toString(),
    lastModified,
    changeFrequency: 'weekly' as const,
    priority: priorityFor(page.url),
  }));

  return [home, ...pages];
}
