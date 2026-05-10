import { defineConfig, defineDocs, frontmatterSchema } from 'fumadocs-mdx/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { z } from 'zod';

const referenceSchema = z.object({
  author: z.string(),
  year: z.number().int(),
  title: z.string(),
  doi: z.string().optional(),
  arxiv: z.string().optional(),
  url: z.string().url().optional(),
});

const stochasticRsFrontmatter = frontmatterSchema.extend({
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
  crate: z
    .string()
    .regex(/^stochastic-rs(-[a-z]+)?$/)
    .optional(),
  module_path: z.string().optional(),
  since: z
    .string()
    .regex(/^\d+\.\d+(\.\d+)?(-[a-z0-9.]+)?$/)
    .optional(),
  status: z.enum(['stable', 'experimental', 'deprecated']).optional(),
  features: z.array(z.string()).default([]),
  references: z.array(referenceSchema).default([]),
  replaced_by: z.string().optional(),
});

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: stochasticRsFrontmatter,
  },
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    rehypePlugins: (v) => [rehypeKatex, ...v],
  },
});
