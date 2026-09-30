import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';
import { SITE } from '@/lib/site';

/**
 * Ranks the landing page above the docs root, and the docs root above the
 * individual pages, so crawlers spend their budget on the hubs first.
 */
function priorityFor(url: string): number {
  if (url === '/docs') return 0.9;
  if (url.startsWith('/docs/getting-started') || url.startsWith('/docs/tutorials')) return 0.8;
  return 0.7;
}

/**
 * `lastModified` is the last commit that touched the page's MDX, and is left
 * out when git cannot say (see `lib/git-last-modified.ts`): a date that moves
 * on every build teaches crawlers to ignore the field.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const pages = source.getPages().map((page) => ({
    url: new URL(page.url, SITE.url).toString(),
    ...(page.data.lastModified ? { lastModified: page.data.lastModified } : {}),
    priority: priorityFor(page.url),
  }));

  return [{ url: SITE.url, priority: 1 }, ...pages];
}
