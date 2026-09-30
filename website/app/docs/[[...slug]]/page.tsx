import { markdownUrl, pageImage, source } from '@/lib/source';
import { DocsPage, DocsBody, DocsDescription, DocsTitle } from 'fumadocs-ui/page';
import { notFound } from 'next/navigation';
import { getMDXComponents } from '@/mdx-components';
import { PageActions } from '@/components/PageActions';
import { References } from '@/components/References';
import { SITE } from '@/lib/site';
import { docsPageJsonLd } from '@/lib/structured-data';
import type { Metadata } from 'next';

/**
 * Every docs page is prerendered from `generateStaticParams`, so an unknown
 * slug is answered with the static 404 instead of invoking the function —
 * which in production answered it with a 500.
 */
export const dynamicParams = false;

/** Brand suffix only while the whole title still fits a search result (~60 chars). */
function searchTitle(title: string): string | { absolute: string } {
  const branded = `${title} | ${SITE.name}`;
  return branded.length <= 60 && !title.includes(SITE.name) ? title : { absolute: title };
}

export default async function Page(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const { references } = page.data;
  const toc =
    references.length > 0
      ? [...page.data.toc, { title: 'References', url: '#references', depth: 2 }]
      : page.data.toc;

  return (
    <DocsPage
      toc={toc}
      full={page.data.full}
      lastUpdate={page.data.lastModified}
      editOnGithub={{
        owner: 'rust-dd',
        repo: 'stochastic-rs',
        sha: 'main',
        path: `website/content/docs/${page.path}`,
      }}
    >
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(docsPageJsonLd(page)) }}
      />
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription>{page.data.description}</DocsDescription>
      <PageActions
        markdownPath={markdownUrl(page)}
        markdownUrl={new URL(markdownUrl(page), SITE.url).toString()}
      />
      <DocsBody>
        <MDX components={getMDXComponents()} />
        <References references={references} />
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}): Promise<Metadata> {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const title = page.data.seo_title ?? page.data.title;
  const description = page.data.description ?? SITE.description;
  const image = { url: pageImage(page).url, width: 1200, height: 630, alt: page.data.title };

  return {
    title: searchTitle(title),
    description,
    alternates: {
      canonical: page.url,
      types: { 'text/markdown': markdownUrl(page) },
    },
    openGraph: {
      type: 'article',
      url: page.url,
      siteName: SITE.name,
      title,
      description,
      images: [image],
      ...(page.data.lastModified ? { modifiedTime: page.data.lastModified.toISOString() } : {}),
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [image.url],
    },
  };
}
