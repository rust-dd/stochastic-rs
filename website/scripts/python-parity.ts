#!/usr/bin/env bun
/**
 * Regenerate website/public/python-parity.json — the data backing the
 * <PythonParityTable /> component.
 *
 * Strategy:
 *   1. Walk stochastic-rs-py/src/ for the macro invocations
 *      `py_distribution!`, `py_distribution_int!`, `py_process_1d!`,
 *      `py_process_2x1d!`, `py_process_2d!`, plus hand-written
 *      `#[pyclass]` and `#[pyfunction]` annotations.
 *   2. Emit a flat list:
 *        { rust_path, python_name, kind, status }
 *      where status ∈ {"exposed","partial","planned","rust_only"}.
 *
 * The "rust_only" rows are computed by diffing this set against the
 * authoritative Rust public-type list (TODO: parse rustdoc JSON output).
 *
 * For now, this scaffold only emits the "exposed" rows. The "rust_only"
 * delta is a follow-up tied to release-checklist SKILL.
 */
import { readdirSync, readFileSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const PY_SRC = join(import.meta.dir, '..', '..', 'stochastic-rs-py', 'src');
const OUT = join(import.meta.dir, '..', 'public', 'python-parity.json');

interface Row {
  python_name: string;
  rust_macro?: string;
  kind: 'distribution' | 'process' | 'pricer' | 'calibrator' | 'function' | 'unknown';
  source_file: string;
  status: 'exposed' | 'partial' | 'planned' | 'rust_only';
}

const rows: Row[] = [];

function* walk(dir: string): Generator<string> {
  for (const e of readdirSync(dir)) {
    const full = join(dir, e);
    if (statSync(full).isDirectory()) yield* walk(full);
    else if (full.endsWith('.rs')) yield full;
  }
}

const macroRe = /py_(distribution|distribution_int|process_1d|process_2x1d|process_2d)!\s*\(\s*(Py\w+)/g;
const pyclassRe = /#\[pyclass[^\]]*\][\s\S]{0,200}?(?:struct|enum)\s+(Py\w+)/g;
const pyfnRe = /#\[pyfunction[^\]]*\][\s\S]{0,200}?fn\s+(\w+)/g;

for (const file of walk(PY_SRC)) {
  const src = readFileSync(file, 'utf8');
  let m: RegExpExecArray | null;

  while ((m = macroRe.exec(src)) !== null) {
    const macro = m[1];
    const py = m[2];
    const kind = macro.startsWith('distribution') ? 'distribution' : 'process';
    rows.push({
      python_name: py,
      rust_macro: `py_${macro}!`,
      kind,
      source_file: file,
      status: 'exposed',
    });
  }
  macroRe.lastIndex = 0;

  while ((m = pyclassRe.exec(src)) !== null) {
    const py = m[1];
    if (rows.some((r) => r.python_name === py)) continue;
    rows.push({
      python_name: py,
      kind: 'unknown',
      source_file: file,
      status: 'exposed',
    });
  }
  pyclassRe.lastIndex = 0;

  while ((m = pyfnRe.exec(src)) !== null) {
    rows.push({
      python_name: m[1],
      kind: 'function',
      source_file: file,
      status: 'exposed',
    });
  }
  pyfnRe.lastIndex = 0;
}

const out = {
  generated_at: new Date().toISOString().slice(0, 10),
  count: rows.length,
  rows,
};

mkdirSync(join(import.meta.dir, '..', 'public'), { recursive: true });
writeFileSync(OUT, JSON.stringify(out, null, 2));
console.log(`✔ wrote ${rows.length} rows to ${OUT}`);
