import { z } from 'zod';

export const referenceSchema = z.object({
  author: z.string(),
  year: z.number().int(),
  title: z.string(),
  doi: z.string().optional(),
  arxiv: z.string().optional(),
  url: z.string().url().optional(),
});

export type Reference = z.infer<typeof referenceSchema>;

export const categories = [
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
] as const;

/** Keys this site adds to the fields fumadocs itself reads from a page. */
export const extraFrontmatter = {
  /**
   * The `<title>` a search result shows, when the sidebar-sized `title` is too
   * terse to say what the page is ("Copulas" vs "Copulas in Rust and Python").
   */
  seo_title: z.string().max(60).optional(),
  category: z.enum(categories).optional(),
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
};

/**
 * The whole page schema: fumadocs' own `pageSchema` fields (`title`,
 * `description`, `icon`, `full`) plus the keys above. It is written with this
 * site's zod rather than as `frontmatterSchema.extend(…)`, because fumadocs
 * ships its own zod 4 and extending it with these schemas erases their types.
 * The build (`source.config.ts`) and the pre-CI check (`scripts/lint-mdx.ts`)
 * both use it, so the two cannot disagree about what a page may declare.
 */
export const pageFrontmatter = z.object({
  title: z.string(),
  description: z.string().optional(),
  icon: z.string().optional(),
  full: z.boolean().optional(),
  ...extraFrontmatter,
});
