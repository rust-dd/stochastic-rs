#!/usr/bin/env bun
/**
 * Regenerate website/public/python-parity.json — the data backing a future
 * <PythonParityTable /> component (not yet built; this script only emits
 * the JSON).
 *
 * Strategy: parse `stochastic-rs-py/src/lib.rs`'s `#[pymodule]` function
 * directly for `m.add_class::<PyXxx>()` and
 * `m.add_function(pyo3::wrap_pyfunction!(path::to::fn, m))` calls. That
 * function is the single authoritative list of what the compiled
 * `stochastic_rs` Python module actually exposes — a `#[pyclass]` defined
 * in a sub-crate but never registered there is not a real Python entry.
 *
 * This script previously walked `stochastic-rs-py/src/` for macro
 * invocations and bare `#[pyclass]`/`#[pyfunction]` attributes, but every
 * `py_distribution!`/`py_process_*!` call and every hand-written pyclass
 * lives in the *sub-crates* (`stochastic-rs-distributions`,
 * `stochastic-rs-stochastic`, `stochastic-rs-quant`, …) — `stochastic-rs-py`
 * itself holds only this one registration file, so the walk found nothing
 * and silently wrote 0 rows every run.
 *
 * `kind` is inferred from the `use` import that brought each `PyXxx` name
 * into scope (the crate segment of the import path). `openblas_gated` is
 * true when the registration line falls inside a
 * `#[cfg(feature = "openblas")]` block.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const LIB_RS = join(
  import.meta.dir,
  '..',
  '..',
  'stochastic-rs-py',
  'src',
  'lib.rs',
);
const OUT = join(import.meta.dir, '..', 'public', 'python-parity.json');

type Kind = 'distribution' | 'process' | 'pricer' | 'copula' | 'estimator' | 'unknown';

interface Row {
  python_name: string;
  kind: Kind;
  entry_kind: 'class' | 'function';
  openblas_gated: boolean;
}

function kindFromCratePath(cratePath: string): Kind {
  if (cratePath.includes('stochastic_rs_distributions')) return 'distribution';
  if (cratePath.includes('stochastic_rs_stochastic')) return 'process';
  if (cratePath.includes('stochastic_rs_quant')) return 'pricer';
  if (cratePath.includes('stochastic_rs_copulas')) return 'copula';
  if (cratePath.includes('stochastic_rs_stats')) return 'estimator';
  return 'unknown';
}

const lines = readFileSync(LIB_RS, 'utf8').split('\n');

// Map each imported `PyXxx` symbol to the crate it came from, so a later
// `m.add_class::<PyXxx>()` can be classified without re-parsing `use`.
const importCrate = new Map<string, string>();
const useRe = /^\s*use\s+([\w:]+)::(Py\w+);\s*$/;
for (const line of lines) {
  const m = useRe.exec(line);
  if (m) importCrate.set(m[2], m[1]);
}

const rows: Row[] = [];
let braceDepth = 0;
let gateDepth: number | null = null;
let pendingGate = false;

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];

  if (line.includes('#[cfg(feature = "openblas")]')) {
    pendingGate = true;
  }

  const opens = (line.match(/{/g) ?? []).length;
  const closes = (line.match(/}/g) ?? []).length;

  // Capture the depth *before* this line's own opening brace(s), so every
  // line inside the block (braceDepth > gateDepth) reads as gated —
  // including the `{` line itself.
  if (pendingGate && opens > 0) {
    gateDepth = braceDepth;
    pendingGate = false;
  }

  braceDepth += opens - closes;
  const gated = gateDepth !== null;

  const classMatch = /m\.add_class::<(\w+)>\(\)/.exec(line);
  if (classMatch) {
    const py = classMatch[1];
    rows.push({
      python_name: py,
      kind: kindFromCratePath(importCrate.get(py) ?? ''),
      entry_kind: 'class',
      openblas_gated: gated,
    });
  }

  // `m.add_function(pyo3::wrap_pyfunction!(` opens a call whose qualified
  // function path is the next non-blank line; the closing `m)?)?;` follows
  // two lines after that in every occurrence in this file.
  if (/pyo3::wrap_pyfunction!\(\s*$/.test(line)) {
    const pathLine = (lines[i + 1] ?? '').trim().replace(/,\s*$/, '');
    if (/^[\w:]+$/.test(pathLine)) {
      rows.push({
        python_name: pathLine.split('::').pop() ?? pathLine,
        kind: kindFromCratePath(pathLine),
        entry_kind: 'function',
        openblas_gated: gated,
      });
    }
  }

  if (gateDepth !== null && braceDepth <= gateDepth) gateDepth = null;
}

const out = {
  generated_at: new Date().toISOString().slice(0, 10),
  source: 'stochastic-rs-py/src/lib.rs (#[pymodule] registrations)',
  count: rows.length,
  classes: rows.filter((r) => r.entry_kind === 'class').length,
  functions: rows.filter((r) => r.entry_kind === 'function').length,
  openblas_gated: rows.filter((r) => r.openblas_gated).length,
  rows,
};

mkdirSync(join(import.meta.dir, '..', 'public'), { recursive: true });
writeFileSync(OUT, JSON.stringify(out, null, 2));
console.log(
  `✔ wrote ${rows.length} rows to ${OUT} (${out.classes} classes + ${out.functions} functions, ${out.openblas_gated} openblas-gated)`,
);
