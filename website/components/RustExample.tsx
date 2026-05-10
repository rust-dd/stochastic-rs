import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export interface RustExampleProps {
  /**
   * Path to the example file, relative to the workspace root
   * (one directory above this `website/` folder).
   * Example: `tests/doctest_ou_quickstart.rs` or `examples/calibration_demo.rs`.
   */
  path: string;
  /** Optional: highlight a specific range (1-based inclusive) */
  highlight?: string;
}

export function RustExample({ path, highlight }: RustExampleProps) {
  const workspaceRoot = join(process.cwd(), '..');
  const filePath = join(workspaceRoot, path);

  let source: string;
  try {
    source = readFileSync(filePath, 'utf8');
  } catch (err) {
    return (
      <pre className="rounded-md border border-red-500/40 bg-red-500/5 p-4 text-sm text-red-600">
        {`<RustExample path="${path}" /> — file not found at ${filePath}`}
      </pre>
    );
  }

  const meta = highlight ? `rust {${highlight}}` : 'rust';
  return (
    <pre>
      <code className={`language-${meta.split(' ')[0]}`}>{source}</code>
    </pre>
  );
}
