import { execFileSync } from 'node:child_process';

let shallow: boolean | undefined;

function git(args: string[]): string {
  return execFileSync('git', args, { encoding: 'utf8' }).trim();
}

/**
 * `null` in a shallow clone (Vercel's default), where the boundary commit seems to touch every
 * older file; `VERCEL_DEEP_CLONE=true` on the Vercel project gives real dates.
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
