import type { MetadataRoute } from 'next';
import { SITE } from '@/lib/site';

/**
 * AI answer engines are a first-class discovery channel for a library like
 * this — "how do I price a Heston option in Rust" is far more likely to be
 * asked of an assistant than typed into a search box — so their crawlers are
 * named explicitly rather than left to the wildcard.
 */
const AI_CRAWLERS = [
  'GPTBot',
  'OAI-SearchBot',
  'ChatGPT-User',
  'ClaudeBot',
  'Claude-User',
  'Claude-SearchBot',
  'PerplexityBot',
  'Perplexity-User',
  'Google-Extended',
  'Applebot-Extended',
  'CCBot',
  'Bytespider',
  'meta-externalagent',
];

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      { userAgent: '*', allow: '/', disallow: '/api/' },
      ...AI_CRAWLERS.map((userAgent) => ({ userAgent, allow: '/' })),
    ],
    sitemap: `${SITE.url}/sitemap.xml`,
    host: SITE.url,
  };
}
