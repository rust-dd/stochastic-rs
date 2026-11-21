#!/usr/bin/env bun
// CI's docs audit: `lint-mdx.ts`, every page listed in its directory's `meta.json`, and every
// `<RustExample>` path present; it reports each miss and exits 1 after the last.
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

{
  const r = spawnSync('bun', ['run', 'scripts/lint-mdx.ts'], {
    cwd: join(import.meta.dir, '..'),
    stdio: 'inherit',
  });
  if (r.status !== 0) {
    errors++;
  }
}

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

// TODO: resolve each page's `module_path`, diff the public API, check the DOI and arXiv links.

if (errors > 0) {
  console.error(`\n${errors} error(s).`);
  process.exit(1);
}

console.log(
  `✔ docs audit clean (${warnings.length} warning(s); see website/audit/AUDIT_*.md when stamped).`,
);
