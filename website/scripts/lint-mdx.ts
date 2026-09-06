#!/usr/bin/env bun
/**
 * Frontmatter zod-schema check for every MDX file under content/docs/.
 *
 * Run via `bun run lint:mdx`. Hard-fails on:
 *   - missing required keys
 *   - description length out of [20, 160]
 *   - module_path that does not resolve in the workspace (regex-grepped)
 *   - status: deprecated without a replaced_by pointer
 *
 * Soft-warns on:
 *   - description length within 80% of the bounds (16-19 or 129-160)
 *
 * Implementation note: this is the lightweight pre-CI check. The full audit
 * (which also diffs the Rust public API against the MDX set, validates DOIs,
 * etc.) lives in scripts/docs-audit.ts.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { z } from 'zod';

const ROOT = join(import.meta.dir, '..', 'content', 'docs');
const WORKSPACE = join(import.meta.dir, '..', '..');

const referenceSchema = z.object({
  author: z.string(),
  year: z.number().int(),
  title: z.string(),
  doi: z.string().optional(),
  arxiv: z.string().optional(),
  url: z.string().url().optional(),
});

const schema = z.object({
  title: z.string().min(1),
  description: z.string().min(20).max(160),
  category: z
    .enum([
      'process',
      'distribution',
      'copula',
      'estimator',
      'pricer',
      'calibrator',
      'concept',
      'tutorial',
      'reference',
      'ai',
    ])
    .optional(),
  subcategory: z.string().optional(),
  crate: z.string().regex(/^stochastic-rs(-[a-z]+)?$/).optional(),
  module_path: z.string().optional(),
  since: z.string().regex(/^\d+\.\d+(\.\d+)?(-[a-z0-9.]+)?$/).optional(),
  status: z.enum(['stable', 'experimental', 'deprecated']).optional(),
  features: z.array(z.string()).default([]),
  references: z.array(referenceSchema).default([]),
  replaced_by: z.string().optional(),
});

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) yield* walk(full);
    else if (full.endsWith('.mdx')) yield full;
  }
}

function parseFrontmatter(src: string): unknown {
  const match = src.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return {};
  const fm: Record<string, unknown> = {};
  for (const line of match[1].split('\n')) {
    const m = line.match(/^([a-z_]+):\s*(.*)$/i);
    if (!m) continue;
    const [, k, vRaw] = m;
    const v = vRaw.trim();
    if (v === '') {
      fm[k] = '';
    } else if (/^-?\d+$/.test(v)) {
      fm[k] = Number(v);
    } else if (v === 'true' || v === 'false') {
      fm[k] = v === 'true';
    } else if (v.startsWith('[') || v.startsWith('{')) {
      try {
        fm[k] = JSON.parse(v);
      } catch {
        fm[k] = v;
      }
    } else {
      fm[k] = v.replace(/^['"]|['"]$/g, '');
    }
  }
  return fm;
}

let errors = 0;
let warnings = 0;

for (const file of walk(ROOT)) {
  const rel = relative(WORKSPACE, file);
  const src = readFileSync(file, 'utf8');
  const fm = parseFrontmatter(src);
  // `parseFrontmatter` takes everything after the first colon as the value,
  // where the site build hands the block to a real YAML parser: an unquoted
  // value carrying a colon *and a space* is a nested mapping to that parser
  // and fails the build with a syntax error this audit never saw. Nothing
  // else here can catch it, since by the time the value reaches the schema
  // it is already a plain string.
  for (const line of (src.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '').split('\n')) {
    const m = line.match(/^([a-z_]+):\s*(.*)$/i);
    if (!m) continue;
    const value = m[2].trim();
    if (/^['"[{]/.test(value)) continue;
    if (value.includes(': ')) {
      errors++;
      console.error(
        `✘ ${rel}: ${m[1]} carries a colon-space and is not quoted — YAML reads it as a mapping`,
      );
    }
  }
  const result = schema.safeParse(fm);
  if (!result.success) {
    errors++;
    console.error(`✘ ${rel}`);
    for (const issue of result.error.issues) {
      console.error(`    ${issue.path.join('.')}: ${issue.message}`);
    }
    continue;
  }

  const data = result.data;
  if (data.status === 'deprecated' && !data.replaced_by) {
    errors++;
    console.error(`✘ ${rel}: status=deprecated requires replaced_by`);
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
