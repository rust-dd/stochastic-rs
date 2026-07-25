import { readFile, readdir } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { SITE } from '@/lib/site';

export const dynamic = 'force-static';

const DOCS_DIR = join(process.cwd(), 'content', 'docs');

async function mdxFiles(): Promise<string[]> {
  const entries = await readdir(DOCS_DIR, { recursive: true });
  return entries
    .filter((entry) => entry.endsWith('.mdx'))
    .map((entry) => join(DOCS_DIR, entry))
    .sort();
}

/** `content/docs/concepts/traits.mdx` → `/docs/concepts/traits`. */
function urlFor(file: string): string {
  const slug = relative(DOCS_DIR, file)
    .replace(/\.mdx$/, '')
    .split(sep)
    .filter((part) => part !== 'index');
  return ['', 'docs', ...slug].join('/');
}

/**
 * Every documentation page concatenated as plain Markdown, so an assistant can
 * ingest the whole library in one fetch instead of crawling page by page.
 */
export async function GET(): Promise<Response> {
  const files = await mdxFiles();

  const sections = await Promise.all(
    files.map(async (file) => {
      const raw = await readFile(file, 'utf8');
      const url = new URL(urlFor(file), SITE.url).toString();
      return `<!-- ${url} -->\n\n${raw.trim()}\n`;
    }),
  );

  const body = [
    `# ${SITE.name} — full documentation`,
    '',
    `> ${SITE.description}`,
    '',
    `Source: ${SITE.repository}`,
    '',
    ...sections,
  ].join('\n');

  return new Response(body, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
