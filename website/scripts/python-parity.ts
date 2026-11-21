#!/usr/bin/env bun
// Writes public/python-parity.json from the registrations in `stochastic-rs-py/src/lib.rs`, the
// one list of what the module exposes; its four `ai` rows reach only a source build with `ai`.
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
for (let i = 0; i < lines.length; i++) {
  const line = lines[i];

  // A qualified path (`m.add_class::<stochastic_rs_copulas::python::PyBb1>()`) counts too.
  const classMatch = /m\.add_class::<([\w:]+)>\(\)/.exec(line);
  if (classMatch) {
    const py = classMatch[1].split('::').pop() as string;
    rows.push({
      python_name: py,
      kind: kindFromCratePath(importCrate.get(py) ?? ''),
      entry_kind: 'class',
    });
  }

  // A `wrap_pyfunction!(` that ends its line has the function's qualified path on the next one.
  if (/pyo3::wrap_pyfunction!\(\s*$/.test(line)) {
    const pathLine = (lines[i + 1] ?? '').trim().replace(/,\s*$/, '');
    if (/^[\w:]+$/.test(pathLine)) {
      rows.push({
        python_name: pathLine.split('::').pop() ?? pathLine,
        kind: kindFromCratePath(pathLine),
        entry_kind: 'function',
        });
    }
  }

}

const out = {
  generated_at: new Date().toISOString().slice(0, 10),
  source: 'stochastic-rs-py/src/lib.rs (#[pymodule] registrations)',
  count: rows.length,
  classes: rows.filter((r) => r.entry_kind === 'class').length,
  functions: rows.filter((r) => r.entry_kind === 'function').length,
  rows,
};

mkdirSync(join(import.meta.dir, '..', 'public'), { recursive: true });
writeFileSync(OUT, JSON.stringify(out, null, 2));
console.log(
  `✔ wrote ${rows.length} rows to ${OUT} (${out.classes} classes + ${out.functions} functions)`,
);
