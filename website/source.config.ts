import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import lastModified from 'fumadocs-mdx/plugins/last-modified';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { pageFrontmatter } from './lib/frontmatter';
import { gitLastModified } from './lib/git-last-modified';
import { remarkRustExample } from './lib/remark-rust-example';

/** Layout-only wrappers whose children are the content an assistant needs. */
const TRANSPARENT = new Set(['Tabs', 'Tab', 'Callout']);

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageFrontmatter,
    postprocess: {
      includeProcessedMarkdown: {
        headingIds: false,
        // `remarkLLMs` replaces any `filterElement` passed here with its own, so
        // the wrappers are unwrapped through `stringify`, which it does forward.
        // Emitting the children as flow content also drops the indentation a
        // JSX parent would put in front of each fenced code block.
        stringify: (node, _parent, state, info) => {
          if (node.type !== 'mdxJsxFlowElement' || !TRANSPARENT.has(node.name ?? '')) return;
          return state.containerFlow(node, info);
        },
      },
    },
  },
});

export default defineConfig({
  plugins: [lastModified({ versionControl: gitLastModified })],
  mdxOptions: {
    remarkPlugins: [remarkMath, remarkRustExample],
    rehypePlugins: (v) => [rehypeKatex, ...v],
  },
});
