import { JSON_LD_IDS, SITE } from '@/lib/site';
import { type DocsPage, pageImage, referenceLink, source } from '@/lib/source';

function absolute(path: string): string {
  return new URL(path, SITE.url).toString();
}

/** Documentation → section index (when the folder has one) → page. */
function breadcrumbs(page: DocsPage): { name: string; url: string }[] {
  const crumbs = [{ name: 'Documentation', url: absolute('/docs') }];
  for (let depth = 1; depth < page.slugs.length; depth += 1) {
    const parent = source.getPage(page.slugs.slice(0, depth));
    if (parent) crumbs.push({ name: parent.data.title, url: absolute(parent.url) });
  }
  if (page.slugs.length > 0) crumbs.push({ name: page.data.title, url: absolute(page.url) });
  return crumbs;
}

/**
 * `TechArticle` + `BreadcrumbList` for one docs page. The article points back
 * at the site-wide `SoftwareSourceCode` node by `@id`, and its `citation`
 * carries the frontmatter references, so an engine reading the page sees what
 * it documents and which papers it rests on.
 */
export function docsPageJsonLd(page: DocsPage) {
  const url = absolute(page.url);
  const citation = page.data.references.map((ref) => {
    const link = referenceLink(ref);
    return {
      '@type': 'ScholarlyArticle',
      name: ref.title,
      author: ref.author,
      datePublished: String(ref.year),
      ...(link ? { url: link.href } : {}),
    };
  });

  return {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'TechArticle',
        '@id': `${url}#article`,
        headline: page.data.title,
        description: page.data.description,
        url,
        inLanguage: 'en',
        image: absolute(pageImage(page).url),
        ...(page.data.lastModified ? { dateModified: page.data.lastModified.toISOString() } : {}),
        author: { '@id': JSON_LD_IDS.author },
        publisher: { '@id': JSON_LD_IDS.author },
        isPartOf: { '@id': JSON_LD_IDS.website },
        about: { '@id': JSON_LD_IDS.software },
        ...(citation.length > 0 ? { citation } : {}),
      },
      {
        '@type': 'BreadcrumbList',
        itemListElement: breadcrumbs(page).map((crumb, i) => ({
          '@type': 'ListItem',
          position: i + 1,
          name: crumb.name,
          item: crumb.url,
        })),
      },
    ],
  };
}
