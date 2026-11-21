import { readFileSync } from 'node:fs';
import { join } from 'node:path';

function workspaceVersion(): string {
  const manifest = readFileSync(join(process.cwd(), '..', 'Cargo.toml'), 'utf8');
  const version = manifest.match(/\[workspace\.package\][^[]*?\nversion\s*=\s*"([^"]+)"/)?.[1];
  if (!version) throw new Error('no [workspace.package] version in the root Cargo.toml');
  return version;
}

/**
 * The headline numbers every surface outside the MDX reads, so a count cannot drift between
 * them; re-derive each from the source it names when a release adds a type.
 */
export const FACTS = {
  /** `[workspace.package] version` in the root `Cargo.toml`, read at build time. */
  version: workspaceVersion(),
  /** Root `CLAUDE.md`, `ProcessExt`: 140 implementors over 130 processes. */
  processes: 130,
  /** `EulerSpec`'s variants, and the processes they carry (`concepts/gpu-support.mdx`). */
  deviceFamilies: 134,
  processesOnEngine: 128,
  /** The `Simd*` samplers of stochastic-rs-distributions; `ComplexDistribution` is not one. */
  distributions: 35,
  /** 15 bivariate + 6 multivariate `BivariateExt` / `MultivariateExt` implementors. */
  copulas: 21,
  /** `impl Calibrator for` in stochastic-rs-quant. */
  calibrators: 16,
  /** `public/python-parity.json` count minus the four `ai`-feature registrations. */
  pythonEntries: 307,
  /** `#[test]` functions across the workspace, rounded down. */
  testsFloor: 3000,
} as const;

/** The process count rounded down to ten, for prose that should stay true between releases. */
export const PROCESSES_ROUNDED = `${Math.floor(FACTS.processes / 10) * 10}+`;
