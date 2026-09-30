import Link from 'next/link';
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { baseOptions } from '@/app/layout.config';

const LINKS: [string, string][] = [
  ['Documentation', '/docs'],
  ['Tutorials', '/docs/tutorials'],
  ['Stochastic processes', '/docs/processes'],
  ['Quickstart', '/docs/getting-started/quickstart'],
  ['Python bindings', '/docs/python'],
];

/**
 * A 404 that points somewhere. Readers and assistants alike arrive here from
 * guessed URLs (`/docs/processes/heston`), so it names the pages they were
 * most likely after, plus the index of every page.
 */
export default function NotFound() {
  return (
    <HomeLayout {...baseOptions}>
      <main className="flex flex-1 flex-col items-center justify-center px-6 py-24 text-center">
        <p className="font-mono text-sm text-fd-muted-foreground">404</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight">This page does not exist</h1>
        <p className="mt-4 max-w-md text-fd-muted-foreground">
          The page may have moved. These are the likeliest places to look, and{' '}
          <a href="/llms.txt" className="underline underline-offset-4">
            llms.txt
          </a>{' '}
          lists every page on the site.
        </p>
        <ul className="mt-8 flex flex-wrap justify-center gap-3">
          {LINKS.map(([label, href]) => (
            <li key={href}>
              <Link
                href={href}
                className="inline-block rounded-lg border border-fd-border px-4 py-2 text-sm font-medium transition hover:bg-fd-muted"
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </main>
    </HomeLayout>
  );
}
