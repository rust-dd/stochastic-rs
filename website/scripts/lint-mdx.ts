#!/usr/bin/env bun
// `bun run lint:mdx`: each page's frontmatter parsed as YAML, where an unquoted `: ` nests a
// mapping and fails the site build, then checked against `lib/frontmatter.ts`'s schema.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { z } from 'zod';
import { pageFrontmatter } from '../lib/frontmatter';

const ROOT = join(import.meta.dir, '..', 'content', 'docs');
const WORKSPACE = join(import.meta.dir, '..', '..');

const schema = pageFrontmatter
  .extend({
    title: z.string().min(1),
    description: z.string().min(20).max(160),
  })
  .strict();

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) yield* walk(full);
    else if (full.endsWith('.mdx')) yield full;
  }
}

let errors = 0;
let warnings = 0;

for (const file of walk(ROOT)) {
  const rel = relative(WORKSPACE, file);
  const src = readFileSync(file, 'utf8');
  const match = src.match(/^---\n([\s\S]*?)\n---\n?/);

  let fm: unknown = {};
  try {
    fm = match ? (Bun.YAML.parse(match[1]) ?? {}) : {};
  } catch (err) {
    errors++;
    console.error(`✘ ${rel}: frontmatter is not valid YAML — ${(err as Error).message}`);
    continue;
  }

  const result = schema.safeParse(fm);
  if (!result.success) {
    errors++;
    console.error(`✘ ${rel}`);
    for (const issue of result.error.issues) {
      console.error(`    ${issue.path.join('.') || '(root)'}: ${issue.message}`);
    }
    continue;
  }

  const data = result.data;
  if (data.status === 'deprecated' && !data.replaced_by) {
    errors++;
    console.error(`✘ ${rel}: status=deprecated requires replaced_by`);
  }

  const body = match ? src.slice(match[0].length) : src;
  if (/^\s*# /.test(body)) {
    errors++;
    console.error(`✘ ${rel}: body opens with "# …" — the layout already renders the title as <h1>`);
  }

  const dlen = data.description.length;
  if (dlen < 24 || dlen > 152) {
    warnings++;
    console.warn(`⚠ ${rel}: description length=${dlen} (target 24-152)`);
  }
}

if (errors > 0) {
  console.error(`\n${errors} error(s), ${warnings} warning(s).`);
  process.exit(1);
}

console.log(`✔ MDX frontmatter OK (${warnings} warning(s)).`);
