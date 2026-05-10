---
name: docs-writing
description: Conventions for writing and maintaining stochastic-rs documentation pages under website/content/docs/. Eight page templates (process / distribution / copula / pricer / calibrator / estimator / AI surrogate / concept), frontmatter schema, KaTeX gotchas, meta.json sidebar wiring, doctest-backed examples, and the audit-script contract. Invoke whenever a new public type ships and needs a docs page, or when fixing rot in existing pages.
---

# Docs writing — stochastic-rs

The docs site lives under `website/` (Fumadocs + Next.js + MDX). Sections
with a single page live as a flat `website/content/docs/<section>.mdx`;
sections with multiple pages live as `website/content/docs/<section>/<name>.mdx`
plus a `meta.json` sidebar manifest. Currently only `getting-started/`
and `concepts/` are folder-form; everything else is flat.

This SKILL is the **per-page authoring contract**. The audit script
(`website/scripts/docs-audit.ts`) enforces every rule under §6 and §10.

## 1. Trigger map — which template to use

```
Adding pub struct  →  Page goes under              →  Use template
─────────────────     ──────────────────────────       ──────────────
process            →  expand processes.mdx (or split  §3.1 Process
                       into processes/<subcat>/ if it
                       outgrows a single page)
distribution       →  expand distributions.mdx        §3.2 Distribution
copula             →  expand copulas.mdx              §3.6 Copula
option pricer      →  expand quant.mdx (or quant/<…>) §3.3 Pricer
calibrator         →  expand quant.mdx                §3.4 Calibrator
stats estimator    →  expand stats.mdx                §3.5 Estimator
NN surrogate       →  expand ai.mdx                   §3.7 AI surrogate
trait / cross-cut  →  concepts/                       §3.8 Concept
end-to-end use case →  expand tutorials.mdx            §3.9 Tutorial
```

When a section page outgrows a single MDX file (≈ 600 lines), promote
it to folder-form: create `<section>/index.mdx` (move the overview
content here) plus per-feature `<section>/<name>.mdx` pages, add a
`<section>/meta.json` sidebar manifest, and remove `<section>.mdx`.

If a new type does not fit any of these slots, **stop and ask** — adding
a new top-level section is a sidebar decision, not a per-page one.

## 2. Frontmatter schema (mandatory)

Every page begins with frontmatter validated by zod in
`website/source.config.ts`. Required keys:

```yaml
---
title: <human-readable name>
description: <one sentence, 20-160 chars — used for OG meta + search snippet>
category: process | distribution | copula | estimator | pricer | calibrator | concept | tutorial | reference | ai
subcategory: <free string — e.g. diffusion, jump, fourier> # optional
crate: stochastic-rs-<sub>                     # which sub-crate owns this
module_path: stochastic_rs::stochastic::diffusion::ou
since: 2.0.0                                   # first version that shipped this
status: stable | experimental | deprecated
features: []                                   # required Cargo features (cuda, openblas, ai, ...)
references:
  - author: "Heston, S."
    year: 1993
    title: "A closed-form solution for options with stochastic volatility..."
    doi: "10.1093/rfs/6.2.327"
    arxiv: ""                                   # use either doi or arxiv, not both
    url: ""                                     # only if neither doi nor arxiv applies
---
```

**Lints (audit script will fail the build):**

- `description` length 20–160 characters.
- `module_path` must resolve in the workspace (the audit greps the Rust
  source for the corresponding `pub struct` / `pub fn`).
- `status: deprecated` requires a `replaced_by:` key pointing at the
  successor page slug.
- `references` empty array is allowed only on `concept` pages. Every
  numerical / model page must cite at least one paper.

## 3. The page templates

Each section below is a copy-paste-ready skeleton. Replace the angle-bracketed
fields, keep the section ordering, do not invent extra top-level headers.

### 3.1 Process page (≈80 pages — by far the most common)

```mdx
---
title: <Process name (acronym)>
description: <one-sentence description, 20-160 chars>
category: process
subcategory: <diffusion | jump | volatility | interest | rough | noise>
crate: stochastic-rs-stochastic
module_path: stochastic_rs::stochastic::<subcategory>::<snake_case_name>
since: 2.0.0
status: stable
features: []
references:
  - { author: "...", year: ..., title: "...", doi: "..." }
---

# <Full process name>

> <Two-sentence summary. State the modelling intent + the mathematical
> family in one breath.>

## SDE

$$ dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t,\quad X_0 = x_0 $$

<Optional: closed-form transition density / characteristic function /
moment generating function — only if it exists.>

## Constructor

| Parameter | Type        | Description                                  |
|-----------|-------------|----------------------------------------------|
| `<arg>`   | `T`         | <one-line meaning + units / range>           |
| `n`       | `usize`     | Number of steps                              |
| `x0`      | `Option<T>` | Initial value (defaults to <doc default>)    |
| `t`       | `Option<T>` | Horizon (defaults to <doc default>)          |

## Examples

<Tabs items={['Rust', 'Python']}>
<Tab value="Rust">
<RustExample path="tests/doctest_<name>_quickstart.rs" />
</Tab>
<Tab value="Python">
```python
from stochastic_rs import <PyName>
p = <PyName>(<keyword args>)
path = p.sample()  # numpy.ndarray, shape (n,)
```
</Tab>
</Tabs>

## Properties

- **Stationary**: <yes / no — and conditions>
- **Markov**: <yes / no>
- **Closed-form transition**: <yes / no>
- **Mean / variance** (when applicable): $\mathbb{E}[X_t] = \dots$, $\text{Var}(X_t) = \dots$
- **Acceleration**: <CPU SIMD / CUDA / Metal / none>

## Calibration / estimation

<Cross-link to relevant pages — never inline calibration code here.>

## See also

- [<related process>](/docs/processes/<...>) — <one-line why related>

## References

<Either rendered automatically from frontmatter `references:`, or
hand-written `<PaperRef />` blocks for the rare paper without DOI.>
```

### 3.2 Distribution page (19 pages)

```mdx
---
title: <Distribution name>
description: <one-sentence>
category: distribution
crate: stochastic-rs-distributions
module_path: stochastic_rs::distributions::<snake>::Simd<Name>
since: 2.0.0
status: stable
features: []
references: [...]
---

# <Distribution name>

> <Two-sentence summary.>

## Density

$$ f(x; \theta) = \dots,\quad x \in <\text{support}> $$

## CDF

$$ F(x; \theta) = \dots $$

## Characteristic function

$$ \varphi(t) = \dots $$

## Moments (closed-form via `DistributionExt`)

| Moment        | Formula                                           |
|---------------|---------------------------------------------------|
| Mean          | $\dots$                                           |
| Variance      | $\dots$                                           |
| Skewness      | $\dots$ (or "**not implemented** — see notes")    |
| Excess kurtosis | $\dots$                                          |

> **DistributionExt status note**: per `memory/project_distribution_ext_status.md`,
> 18/19 distributions ship closed-form `DistributionExt`. If a specific moment
> falls into the 5 "no closed-form" cases, mark it explicitly here as
> `unimplemented!` and link to the issue tracking the fix. Never leave it as
> a silent zero.

## Examples

<Tabs items={['Rust', 'Python']}>
<Tab value="Rust"><RustExample path="tests/doctest_<dist>_quickstart.rs" /></Tab>
<Tab value="Python">
```python
from stochastic_rs import <PyName>
import numpy as np
d = <PyName>(<keyword args>, seed=42)
xs = d.sample(10_000)
```
</Tab>
</Tabs>

## Sampling strategy

<Pick one and explain in 2-3 sentences:>
- Transformation (closed-form $F^{-1}(U)$)
- Ziggurat
- Rejection / inversion
- Composition

## KS-test reference

<Cite the integration test file under `stochastic-rs-distributions/tests/` that
validates the sampler against the analytic CDF.>

## References
```

### 3.3 Pricer page (~30 pages)

```mdx
---
title: <Pricer name>
description: <one-sentence>
category: pricer
subcategory: <closed-form | fourier | monte-carlo | finite-difference | lattice | lsm | malliavin>
crate: stochastic-rs-quant
module_path: stochastic_rs::quant::pricing::<snake>::<PricerName>
since: 2.0.0
status: stable
features: []
references: [...]
---

# <Pricer name>

> <Model + payoff + method, in one sentence.>

## Model

<SDE / characteristic function / payoff. Use $\LaTeX$.>

## Method

<Closed-form / Carr-Madan / Lewis / MC / Crank-Nicolson / Bermudan-LSM / etc.>

## Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| ...       | ...  | ...         |

## Examples

<Tabs items={['Rust', 'Python']}>
<Tab value="Rust"><RustExample path="tests/doctest_<pricer>_quickstart.rs" /></Tab>
<Tab value="Python">...</Tab>
</Tabs>

## Greeks

<Either:>
- ✅ All first- and second-order Greeks via `GreeksExt` (delta, gamma, vega,
  theta, rho, vanna, charm, volga, veta) — single-pass override available
  for MC pricers per `greeks-pattern` SKILL.
- ⚠️ Partial — only <list>.
- ❌ Not implemented — track at <issue link>.

## Variance reduction (MC pricers only)

- Antithetic: ✅ / ❌
- Control variate: ✅ / ❌
- Quasi-MC (Sobol / Halton): ✅ / ❌
- MLMC: ✅ / ❌

## Calibration

<Link to the matching calibrator page if applicable.>

## Complexity

$O(\text{steps} \times \text{paths})$ for MC, $O(N \log N)$ for FFT, etc.

## References
```

### 3.4 Calibrator page (~12 pages)

```mdx
---
title: <Calibrator name>
description: <one-sentence>
category: calibrator
crate: stochastic-rs-quant
module_path: stochastic_rs::quant::calibration::<snake>::<Name>
since: 2.0.0
status: stable
features: []
references: [...]
---

# <Calibrator name>

> <Model + market data + optimiser, in one sentence.>

## Model

<Brief. Cross-link to the pricer page that defines the model fully.>

## Market data

<What is required: strike grid, IV grid, OTM-only, full smile, ATM only, etc.>

## Optimiser

<Levenberg-Marquardt (argmin), Differential Evolution, NMLE, NMLE-CEKF, Cui
analytic Jacobian, etc. — one paragraph on why.>

## `CalibrationResult` fields

| Field          | Type        | Meaning                       |
|----------------|-------------|-------------------------------|
| `params`       | `<T>::Params` | Calibrated parameter struct |
| `rmse`         | `T`         | Final fit error               |
| `iterations`   | `usize`     | Optimiser iteration count     |
| `success`      | `bool`      | Convergence flag              |

## Example

<Rust example only — Python calibration example goes in
`tutorials/<model>-calibration.mdx`.>

## Typical RMSE

<Order-of-magnitude expectation on synthetic data, plus a note on real-data
caveats.>

## See also

- [<pricer>](/docs/quant/pricing/<...>) — the matching pricer

## References
```

### 3.5 Estimator page (~30 pages)

```mdx
---
title: <Estimator name>
description: <one-sentence>
category: estimator
subcategory: <mle | realized | normality | stationarity | econometrics | filtering | hurst | tail | spectral>
crate: stochastic-rs-stats
module_path: stochastic_rs::stats::<snake>::<Name>
since: 2.0.0
status: stable
features: []
references: [...]                 # **mandatory** — papers must be cited verbatim
---

# <Estimator name>

> <Estimand + method, in one sentence.>

## Estimand

$$ \widehat{\theta} = \dots $$

## Method

<Brief description. Per `feedback_implementation.md`, the implementation
follows the cited paper exactly — link to the file:line where the
formulas are anchored.>

## Result struct

| Field         | Type    | Meaning                                      |
|---------------|---------|----------------------------------------------|
| `point`       | `T`     | Point estimate                               |
| `se`          | `T`     | Standard error                               |
| `ci_lower`    | `T`     | 95% confidence interval lower bound          |
| `ci_upper`    | `T`     | 95% confidence interval upper bound          |
| `p_value`     | `T`     | <parametric / bootstrap — call it out>       |

## Asymptotic distribution

<Either: closed-form (e.g. $\sqrt{n}(\hat\theta-\theta) \to \mathcal N$),
or "bootstrap with `B = ...` resamples".>

## Example

<Rust + Python tabs.>

## Validation

<Cross-link to the integration test that compares against the paper's
numerical examples (e.g. Fukasawa intraday Table 1).>

## References
```

### 3.6 Copula page (10 pages)

```mdx
---
title: <Copula name>
description: <one-sentence>
category: copula
subcategory: <bivariate | multivariate | empirical>
crate: stochastic-rs-copulas
module_path: stochastic_rs::copulas::<snake>::<Name>
since: 2.0.0
status: stable
features: []
references: [...]
---

# <Copula name>

> <Family + dependence intuition, in one sentence.>

## Definition

$$ C(u, v; \theta) = \dots $$

## Parameter range

$\theta \in <\text{interval}>$, with limits:

- $\theta \to <\text{lower}>$ ⇒ <independence / countermonotonic / ...>
- $\theta \to <\text{upper}>$ ⇒ <comonotonic>

## Dependence measures

| Measure         | Formula                |
|-----------------|------------------------|
| Kendall's $\tau$| $\dots$                |
| Spearman's $\rho$| $\dots$               |
| Tail dependence | $\lambda_L = \dots,\ \lambda_U = \dots$ |

## Sampling algorithm

<E.g. Marshall-Olkin, conditional inversion. Cite the algorithm.>

## Example

<Rust + Python tabs.>

## References
```

### 3.7 AI surrogate page (3 pages)

Per `vol-surrogate-nn` SKILL. Required sections:

```
1. Model spec     (StochVolModelSpec — input dims, output dims)
2. Scaler         (BoundedScaler / StandardScaler — pre/post norm)
3. Training set   (gzip-npy file path, generator script, sample count)
4. Architecture   (layers, activation, hidden width)
5. Training       (optimiser, loss, epochs, batch size)
6. Inference      (predict_surface integration with ImpliedVolSurface::from_flat_iv_grid)
7. Round-trip test (train_save_load_<model>)
8. Benchmark      (vs Fourier / closed-form baseline; speed + accuracy)
9. References
```

### 3.8 Concept page (~8 pages)

Free-form. Required ingredients:

- One-paragraph elevator pitch
- Why it exists (the pain it solves)
- Worked example with `<RustExample>`
- Decision-table for "when to use which alternative" (e.g. `f32` vs `f64`,
  `ProcessExt::sample()` vs `sample_par()`)
- Cross-links to the SKILLs that operationalise the concept

### 3.9 Tutorial page (8–10 pages)

Long-form, end-to-end, narrative. Required structure:

```
1. What you'll build (screenshot or static IV-surface plot)
2. Prerequisites (which sub-crates, which features)
3. Setup
4. Step 1 ... Step N (each ≤ 30 lines of code, with prose)
5. Result (numerical output, plot)
6. Where to go next (3 cross-links)
```

## 4. Cross-linking conventions

- **Internal links**: always relative, no leading `/docs/` prefix duplication.
  Use `[OU](/docs/processes/diffusion/ou)`. Fumadocs MDX validates these at
  build time — broken links fail CI.
- **docs.rs links**: don't hand-write them. Frontmatter `module_path` is
  rendered by `<DocsRsLink />` automatically in the page header.
- **Issue / PR links**: use full URL once, then `(see #123)` for repeats
  in the same page.
- **Paper links**: prefer DOI. Fall back to arXiv. Avoid raw publisher URLs
  (paywalls rot).

## 5. KaTeX gotchas

Fumadocs uses `remark-math` + `rehype-katex`. The two recurring traps:

1. **Inline vs block**. Inline `$E[X]$`, block `$$ \dots $$` on its own line
   with blank lines either side. Otherwise MDX may treat the `$` as
   prose punctuation.
2. **Underscores in `$$ ... $$`**. KaTeX treats `_` correctly inside math
   blocks, but **MDX** can mistake `_x_` for italics. Fix: wrap the block
   with the `<Math>` component, or escape (`\_`).
3. **Multi-line aligned environments** need `&` for column separators and
   `\\` for row breaks — `align*` works:

```
$$
\begin{aligned}
dX_t &= \theta(\mu - X_t)\,dt + \sigma\,dW_t \\
X_0  &= x_0
\end{aligned}
$$
```

4. **No `\mathbb{...}` collisions**. KaTeX ships `\mathbb{R, N, Z, Q, C}`
   only; `\mathbb{P}` works post-v0.16. If using older KaTeX, fall back
   to `\Pr`.

## 6. `meta.json` (sidebar)

Each section directory has a `meta.json` declaring sidebar order. Example
from `processes/diffusion/meta.json`:

```json
{
  "title": "Diffusion processes",
  "pages": [
    "overview",
    "---Single-factor---",
    "ou", "gbm", "gbm-log", "cir", "cev", "ckls", "ait-sahalia",
    "---Bounded support---",
    "jacobi", "fjacobi", "logistic", "verhulst", "kimura", "pearson",
    "---Special---",
    "regime-switching", "fouque", "feller", "feller-root", "three-half"
  ]
}
```

`---Section---` strings render as non-clickable group headers. The audit
script ensures **every** mdx file in the directory appears in `pages` —
forgetting a new file is a hard error.

## 7. The `<RustExample>` component (doctest-backed examples)

For hero pages (top-50 traffic — landing, OU, GBM, Heston, BSM, ...), the
Rust block is **not** inline. It is `<RustExample path="..." />`, which
inlines a file from the workspace. The contract:

1. The referenced file must exist under `tests/doctest_*.rs` (or
   `examples/`) and pass `cargo test --workspace`.
2. The file's first comment line is `// docs: <slug>` — the audit script
   uses this to verify the page-to-test mapping.
3. The file ships only **runnable** code — no `unimplemented!`, no `todo!`,
   no `unwrap()` on user input.

For long-tail pages, inline fenced `rust` blocks are fine. Annotate each
with `// last-checked: <commit-sha>`. The quarterly audit flags blocks
where the corresponding source has changed since the pinned sha.

## 8. References block (auto-rendered)

The frontmatter `references:` array is rendered into a `## References`
section by the page layout. Hand-written `## References` headings are
ignored unless the page is a `concept` (which legitimately has none in
frontmatter).

Citation format:

- DOI present → `Author (year). Title. doi:10.xxxx/yyy`
- arXiv present → `Author (year). Title. arXiv:NNNN.NNNNN`
- url only → `Author (year). Title. <url>`

The audit pings DOIs / arXiv IDs quarterly. Broken links produce a row
in `website/audit/AUDIT_<YYYY-MM-DD>.md`.

## 9. Adding a new page — checklist

```
[ ] Pick the right template from §3
[ ] Create the file at website/content/docs/<section>/<name>.mdx
[ ] Fill frontmatter — schema is enforced (§2)
[ ] Math: prefer block $$ … $$, escape $ in prose
[ ] Examples: <Tabs> with Rust + Python (§3.x)
[ ] Cross-links: §4 conventions
[ ] Add the slug to the parent meta.json (§6)
[ ] Optional: add a doctest under tests/doctest_<slug>.rs (§7)
[ ] Run `bun lint:mdx` (frontmatter schema check)
[ ] Run `bun build` (Fumadocs link check)
[ ] If new doctest: `cargo test --test 'doctest_<slug>'`
```

## 10. Audit script — what it enforces

`website/scripts/docs-audit.ts` runs in CI on every PR and as a quarterly
cron. Hard failures (block CI):

1. Frontmatter zod schema violation
2. `module_path` does not resolve in the workspace
3. Public type added in `src/` with no matching MDX page
4. MDX file missing from its directory's `meta.json`
5. `<RustExample path="..." />` points at a non-existent file
6. Broken internal Markdown link

Soft warnings (report-only):

- `last-checked` sha older than 90 days while source changed
- DOI / arXiv URL no longer resolves
- Description outside 20–160 chars (warn at 80% of bound)

Output: `website/audit/AUDIT_<YYYY-MM-DD>.md`.

## 11. Bun cheatsheet (the workspace uses Bun, not pnpm/npm)

```bash
cd website

bun install                    # install / update deps
bun run dev                    # local dev server (http://localhost:3000)
bun run build                  # production build (validates internal links)
bun run lint                   # eslint + frontmatter schema (scripts/lint-mdx.ts)
bun run lint:mdx               # frontmatter schema only
bun run audit                  # full docs audit (scripts/docs-audit.ts)
bun run python:parity          # regenerate public/python-parity.json
bun run bench:publish          # regenerate public/bench/*.json from criterion
bun add <pkg>                  # add a runtime dep
bun add -d <pkg>               # add a dev dep
bunx <bin> ...                 # one-off binary (no global install)
```

If a script bypasses Bun (e.g. CI calls `node scripts/foo.ts`), prefer
`bun run scripts/foo.ts` — Bun's TS runtime is part of the workspace
contract.

## 12. Long-tail rules (small but load-bearing)

- **No emoji in body text** (frontmatter `status` icons rendered by the
  layout are fine). Per project convention.
- **No `// ---` separator banners** (`feedback_no_section_separators.md`).
- **No statrs in code examples** (`feedback_no_statrs_distributions.md`).
- **Convert relative dates to absolute** (`Today` → `2026-05-10`) the same
  way the memory system does. Future-readers thank you.
- **Capitalisation**: titles use sentence case, not Title Case.
  ("Heston model", not "Heston Model".) Acronyms stay capitalised
  (`OU`, `GBM`, `CIR`, `SABR`).
- **First-person plural is fine** ("we use the Cui Jacobian"); first-person
  singular is not.
- **Hungarian comments**: never. The site is English-only.
