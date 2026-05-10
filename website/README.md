# stochastic-rs documentation site

Fumadocs + Next.js + MDX. Deployed to Vercel.

## Prerequisites

- [Bun](https://bun.sh) ≥ 1.3 (this project uses Bun, not pnpm/npm)
- Node 20+ (Bun bundles its own runtime, but `next` still talks to Node)

## Develop

```bash
cd website
bun install
bun run dev          # http://localhost:3000
```

## Production build

```bash
bun run build        # validates internal links, frontmatter schema, MDX
bun run start        # serve the production build locally
```

## Lint / audit

```bash
bun run typecheck    # tsc --noEmit
bun run lint         # eslint via next
bun run lint:mdx     # frontmatter zod schema (scripts/lint-mdx.ts)
bun run audit        # full docs audit (see .claude/skills/docs-writing)
bun run python:parity   # regenerate public/python-parity.json
```

## Layout

```
website/
├── app/                   # Next App Router (RSC)
│   ├── (home)/            # landing page route group
│   ├── docs/              # /docs/* (Fumadocs pages)
│   ├── api/search/        # search route handler
│   ├── layout.tsx         # root provider
│   ├── layout.config.tsx  # nav links, repo URL
│   └── global.css         # Tailwind v4 + KaTeX CSS
├── content/docs/          # MDX content (the docs themselves)
├── components/            # custom MDX components (RustExample, PaperRef, ...)
├── lib/source.ts          # Fumadocs source loader
├── scripts/               # docs-audit, lint-mdx, python-parity
├── source.config.ts       # Fumadocs MDX config (KaTeX wired here)
├── next.config.mjs
└── tsconfig.json
```

## Adding a page

See `.claude/skills/docs-writing/SKILL.md` for the per-page recipe and the
eight templates (process / distribution / copula / pricer / calibrator /
estimator / AI surrogate / concept).

## Deployment

Vercel project root directory: `website/`. Framework preset: Next.js.
Install command: `bun install --frozen-lockfile`. Build command: `bun run build`.
