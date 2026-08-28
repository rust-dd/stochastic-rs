import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { highlight } from 'fumadocs-core/highlight';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';

export interface RustExampleProps {
  /**
   * Path to the example file, relative to the workspace root
   * (one directory above this `website/` folder).
   * Example: `tests/doctest_quickstart_ou.rs` or `examples/calibration_demo.rs`.
   */
  path: string;
  /** Optional: highlight a specific range (1-based inclusive) */
  highlight?: string;
}

/**
 * Inlines a compiled example from the workspace.
 *
 * The source is highlighted through the same Shiki pipeline Fumadocs uses for
 * fenced ```rust blocks, so an included example renders identically to one
 * written inline. Emitting a bare `<pre><code>` here instead would bypass that
 * pipeline entirely and the block would render as unstyled plain text beside
 * its highlighted Python neighbours.
 */
export async function RustExample({
  path,
  highlight: highlightRange,
}: RustExampleProps) {
  const workspaceRoot = join(process.cwd(), '..');
  const filePath = join(workspaceRoot, path);

  let source: string;
  try {
    source = readFileSync(filePath, 'utf8');
  } catch {
    return (
      <pre className="rounded-md border border-red-500/40 bg-red-500/5 p-4 text-sm text-red-600">
        {`<RustExample path="${path}" /> — file not found at ${filePath}`}
      </pre>
    );
  }

  const rendered = await highlight(source.trimEnd(), {
    lang: 'rust',
    meta: highlightRange ? { __raw: `{${highlightRange}}` } : undefined,
    components: {
      pre: (props) => (
        <CodeBlock {...props} title={path}>
          <Pre>{props.children}</Pre>
        </CodeBlock>
      ),
    },
  });

  return rendered;
}
