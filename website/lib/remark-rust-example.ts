import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Code, Parent, RootContent } from 'mdast';

interface JsxAttribute {
  type: string;
  name?: string;
  value?: unknown;
}

interface JsxElement {
  type: 'mdxJsxFlowElement';
  name: string | null;
  attributes: JsxAttribute[];
}

function attribute(node: JsxElement, name: string): string | undefined {
  const found = node.attributes.find(
    (attr) => attr.type === 'mdxJsxAttribute' && attr.name === name,
  );
  return typeof found?.value === 'string' ? found.value : undefined;
}

function toCode(node: JsxElement, workspace: string): Code {
  const path = attribute(node, 'path');
  if (!path) throw new Error('<RustExample> needs a `path` attribute');

  const range = attribute(node, 'highlight');
  return {
    type: 'code',
    lang: 'rust',
    meta: range ? `title="${path}" {${range}}` : `title="${path}"`,
    value: readFileSync(join(workspace, path), 'utf8').trimEnd(),
  };
}

function inline(parent: Parent, workspace: string) {
  parent.children.forEach((child, i) => {
    const node = child as unknown as JsxElement;
    if (node.type === 'mdxJsxFlowElement' && node.name === 'RustExample') {
      parent.children[i] = toCode(node, workspace) as RootContent;
    } else if ('children' in child) {
      inline(child as Parent, workspace);
    }
  });
}

/**
 * Replaces `<RustExample path="tests/…" />` with a fenced `rust` block holding
 * that file, at compile time. The page highlights it through the same Shiki
 * pipeline as any fenced block, and the Markdown served to assistants
 * (`llms-full.txt`, `/docs/<page>.md`) carries the code itself instead of a tag
 * naming a file the reader cannot open. A missing file fails the build.
 */
export function remarkRustExample() {
  const workspace = join(process.cwd(), '..');
  return (tree: Parent) => inline(tree, workspace);
}
