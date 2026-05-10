#!/usr/bin/env bun
/**
 * Full docs audit. Run quarterly + in CI.
 *
 * Hard failures (exit 1):
 *   1. Frontmatter zod schema violation (delegates to lint-mdx.ts)
 *   2. `module_path` does not resolve (regex grep over the Rust workspace)
 *   3. New `pub struct` / `pub fn` in src/ with no matching MDX page
 *   4. MDX file missing from its directory's meta.json `pages` array
 *   5. <RustExample path="..." /> points at a non-existent file
 *
 * Soft warnings (printed to stdout, written to docs/DOCSITE_AUDIT_<date>.md):
 *   - last-checked sha older than 90 days while source has changed
 *   - description outside the 24-152 char comfort window
 *   - DOI / arXiv links failing to resolve (HEAD request)
 *
 * Output: website/audit/AUDIT_<YYYY-MM-DD>.md
 *
 * NOTE: this is a working scaffold. The grep-based module_path resolver
 * and the Rust public API differ are TODO — they require the public API
 * snapshot to be committed alongside the docs (a follow-up).
 */
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = join(import.meta.dir, '..', 'content', 'docs');
const WORKSPACE = join(import.meta.dir, '..', '..');

let errors = 0;
const warnings: string[] = [];

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) yield* walk(full);
    else if (full.endsWith('.mdx')) yield full;
  }
}

// 1. delegate to lint-mdx
{
  const r = spawnSync('bun', ['run', 'scripts/lint-mdx.ts'], {
    cwd: join(import.meta.dir, '..'),
    stdio: 'inherit',
  });
  if (r.status !== 0) {
    errors++;
  }
}

// 4. meta.json coverage check
function checkMetaCoverage(dir: string) {
  const entries = readdirSync(dir);
  const metaPath = join(dir, 'meta.json');
  const mdxFiles = entries
    .filter((e) => e.endsWith('.mdx') && e !== 'index.mdx')
    .map((e) => e.replace(/\.mdx$/, ''));

  if (mdxFiles.length === 0) {
    for (const e of entries) {
      const full = join(dir, e);
      if (statSync(full).isDirectory()) checkMetaCoverage(full);
    }
    return;
  }

  let pages: string[] = [];
  try {
    const raw = JSON.parse(readFileSync(metaPath, 'utf8'));
    pages = (raw.pages ?? []) as string[];
  } catch {
    errors++;
    console.error(`✘ ${relative(WORKSPACE, dir)} missing or invalid meta.json`);
  }

  const declared = new Set(pages.filter((p) => !p.startsWith('---')));
  for (const slug of mdxFiles) {
    if (!declared.has(slug)) {
      errors++;
      console.error(
        `✘ ${relative(WORKSPACE, join(dir, slug + '.mdx'))} not in meta.json`,
      );
    }
  }

  for (const e of entries) {
    const full = join(dir, e);
    if (statSync(full).isDirectory()) checkMetaCoverage(full);
  }
}

checkMetaCoverage(ROOT);

// 5. <RustExample path="..." /> existence check
const rustExampleRe = /<RustExample\s+path="([^"]+)"\s*\/>/g;
for (const file of walk(ROOT)) {
  const src = readFileSync(file, 'utf8');
  let m: RegExpExecArray | null;
  while ((m = rustExampleRe.exec(src)) !== null) {
    const target = join(WORKSPACE, m[1]);
    try {
      statSync(target);
    } catch {
      errors++;
      console.error(
        `✘ ${relative(WORKSPACE, file)}: RustExample path "${m[1]}" does not exist`,
      );
    }
  }
  rustExampleRe.lastIndex = 0;
}

// TODO: 2. module_path resolver, 3. Rust public API differ, soft-warn DOI checker

if (errors > 0) {
  console.error(`\n${errors} error(s).`);
  process.exit(1);
}

console.log(
  `✔ docs audit clean (${warnings.length} warning(s); see website/audit/AUDIT_*.md when stamped).`,
);
