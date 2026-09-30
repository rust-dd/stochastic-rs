import { execFileSync } from 'node:child_process';

let shallow: boolean | undefined;

function git(args: string[]): string {
  return execFileSync('git', args, { encoding: 'utf8' }).trim();
}

/**
 * The last commit that touched a docs file, or `null` when the answer would be
 * a guess. In a shallow clone (Vercel's default) every file older than the
 * clone boundary looks as if the boundary commit touched it, so there the
 * sitemap and the page footer leave the date out rather than state a wrong
 * one. `VERCEL_DEEP_CLONE=true` on the Vercel project gives real dates.
 */
export async function gitLastModified(file: string): Promise<Date | null> {
  try {
    shallow ??= git(['rev-parse', '--is-shallow-repository']) === 'true';
    if (shallow) return null;

    const iso = git(['log', '-1', '--format=%cI', '--', file]);
    return iso ? new Date(iso) : null;
  } catch {
    return null;
  }
}
