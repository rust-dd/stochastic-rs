import { readFileSync } from 'node:fs';
import { join } from 'node:path';

function workspaceVersion(): string {
  const manifest = readFileSync(join(process.cwd(), '..', 'Cargo.toml'), 'utf8');
  const version = manifest.match(/\[workspace\.package\][^[]*?\nversion\s*=\s*"([^"]+)"/)?.[1];
  if (!version) throw new Error('no [workspace.package] version in the root Cargo.toml');
  return version;
}

/**
 * The library's headline numbers, in one place. Everything outside the MDX
 * content — the landing page, the meta description, the social card,
 * `llms.txt`, the JSON-LD — reads them from here, so a count cannot drift
 * between surfaces the way "120+", "131" and "140+" once did. Each figure
 * names where it is derived; re-derive them when a release adds a type.
 */
export const FACTS = {
  /** `[workspace.package] version` in the root `Cargo.toml`, read at build time. */
  version: workspaceVersion(),
  /** Root `CLAUDE.md`, `ProcessExt`: 142 implementors over 132 processes. */
  processes: 132,
  /** `concepts/gpu-support.mdx`: families of the device engine and the processes they carry. */
  deviceFamilies: 121,
  processesOnEngine: 130,
  /** `Simd*` samplers in stochastic-rs-distributions (37 types minus `ComplexDistribution`). */
  distributions: 36,
  /** 15 bivariate + 8 multivariate `BivariateExt` / `MultivariateExt` implementors. */
  copulas: 23,
  /** `impl Calibrator for` in stochastic-rs-quant. */
  calibrators: 15,
  /** `public/python-parity.json` count minus the four `ai`-feature registrations. */
  pythonEntries: 308,
  /** `#[test]` functions across the workspace, rounded down. */
  testsFloor: 3000,
} as const;

/** "130+" for prose that should stay true between releases. */
export const PROCESSES_ROUNDED = `${Math.floor(FACTS.processes / 10) * 10}+`;
