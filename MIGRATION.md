# Migration Guide

Breaking changes are recorded here as they land, grouped by release. Entries
under `## Unreleased` describe changes on `main` that have not shipped yet.

## Unreleased

### stochastic-rs-distributions: one seeded stream, honest signatures

- `fill_slice(rng, out)` → `fill_slice(out)`. Every `Simd*` distribution's
  bulk-fill method dropped its `Rng` argument. The argument was ignored for
  24 of 27 types already (they always drew from the internal SIMD stream
  seeded at construction); `SimdBinomial`, `SimdHypergeometric`, and
  `SimdPoisson` used to honor it — meaning `sample_n`, which handed those
  three types a fresh globally-seeded `SimdRng`, silently ignored the
  `Deterministic` seed passed to their constructors. All three now draw
  from their own internal stream like every other type. Callers passing an
  explicit `Rng` (e.g. `dist.fill_slice(&mut rng, &mut out)`) drop that
  first argument; there is no other change needed since the internal
  stream was always the one actually driving output for every other type.
- `rand_distr::Distribution::sample(&self, rng)` is unchanged in shape but
  the `rng` argument is now uniformly unused across all 27 `Simd*` types
  (previously true for 24/27; `SimdBinomial`, `SimdHypergeometric`, and
  `SimdPoisson` now match). Each impl documents this; construct with
  `Deterministic::new(seed)` for reproducible output regardless of what
  `Rng` is passed to `.sample()`.
- `DistributionSampler::sample_matrix`'s parallel fan-out is now
  reproducible under `Deterministic` seeding, including across repeated
  calls on the same object. Previously each rayon worker received a
  `Clone` of the sampler, and every `Simd*` `Clone` impl re-seeds from
  `Unseeded` by design (`Clone` means "give me an independent stream") —
  so a `Deterministic`-seeded sampler silently lost reproducibility the
  moment `sample_matrix` went multi-threaded. Workers now come from a new
  `#[doc(hidden)] DistributionSampler::fork(stream_idx)` that derives each
  worker's seed from a basis value drawn fresh off the sampler's own live
  state — an interior-mutable cell distinct from the stream driving real
  samples — on every call that takes the parallel path, combined with
  `stream_idx` via `splitmix64(basis ^ stream_idx)`. Two
  identically-`Deterministic`-seeded samplers now produce bit-identical
  `sample_matrix` output call-for-call (first call matches first, second
  matches second, ...) regardless of thread count; repeated calls on the
  *same* sampler never replay, for `Deterministic`- and
  `Unseeded`-constructed samplers alike; and a serial call (below the
  parallel threshold) never touches the fork basis, so interleaving
  serial and parallel calls stays deterministic across two
  identically-seeded samplers. No API signature changed; this is a
  behavior fix.
- The Python bindings' `sample_par(m, n)` inherits this fix directly:
  seeded (`seed=...`) callers previously always executed the serial path
  under the hood (a workaround for the same-call-replay behavior above —
  going parallel for a reproducible sampler wasn't safe yet); they now
  take the same parallel path as unseeded callers, reproducible
  call-for-call via the per-call fork basis described above.
- Integer-count distributions (`SimdBinomial`, `SimdGeometric`,
  `SimdHypergeometric`, `SimdPoisson`) no longer silently emit `0` when a
  draw overflows the requested output integer type (e.g. sampling
  `Binomial(n=300, ..)` into a `u8` buffer). Overflowing draws now saturate
  to the type's `MAX` and trip a `debug_assert!` in debug builds. Code that
  relied on silent-zero overflow (there should be none — it was a
  correctness bug) will now see saturated values instead; size output
  buffers to fit the distribution's support.
- `stochastic_rs_core::simd_rng::SeedExt` gained a new required method,
  `seed_value(&self) -> u64`. This only affects code implementing
  `SeedExt` directly (no in-tree implementors besides `Unseeded` and
  `Deterministic`); consumers of `Unseeded`/`Deterministic` are
  unaffected.
- `SimdGed` and `SimdGev` now implement `DistributionSampler<T>` (were
  previously missing from the trait's coverage despite having the same
  internal-stream shape as every other float distribution) — additive,
  not breaking.

### stochastic-rs-copulas: seedable sampling everywhere

- `MultivariateExt` gained a new required method,
  `sample_with_seed(n, seed) -> Result<Array2<f64>, Box<dyn Error>>`,
  mirroring `BivariateExt::sample_with_seed`. Any external implementor of
  `MultivariateExt` (the trait is feature-gated behind `openblas`) must add
  this method.
- `GaussianMultivariate`, `TMultivariate`, `VineMultivariate`,
  `TreeMultivariate` previously had **no reproducible-sampling path** —
  `sample` always drew from `Unseeded`. They now implement
  `sample_with_seed`, routing the same internal Cholesky/χ² machinery
  through a `Deterministic::new(seed)` source instead.
- `RVine::sample_with_seed` forwards to the wrapped `DVine`/`CVine`
  variant.
- `EmpiricalCopula2D::sample_seeded` → `sample_with_seed` (renamed for
  naming consistency with `BivariateExt::sample_with_seed`; not part of
  any trait).
- `CVine::sample_seeded`, `DVine::sample_seeded`,
  `NestedArchimedean::sample_seeded` — all renamed to `sample_with_seed`
  and moved from a bespoke inherent method into the `MultivariateExt`
  trait implementation itself (same call syntax `x.sample_with_seed(n,
  seed)`, but the method now requires `use
  stochastic_rs_copulas::traits::MultivariateExt;` in scope, exactly like
  the existing `.sample(n)`).
- Fixed a latent determinism bug in `NestedArchimedean`: the Clayton
  family's root frailty draw was hardcoded to `Unseeded` regardless of the
  caller's seed (every row re-drew from a fresh unseeded `Gamma`), so
  `sample_with_seed` was silently non-reproducible for any Clayton-family
  nested Archimedean copula. The root draw now derives its seed from the
  caller's own RNG stream.

### stochastic-rs-copulas: unify trait signatures and error behavior

- `MultivariateExt::{pdf, cdf, log_pdf}` now take `&Array2<f64>` instead
  of `Array2<f64>` by value — every implementation already re-borrowed
  internally, so callers previously paid for a clone (or a move) at call
  sites for no benefit. Update call sites from `.pdf(x)` to `.pdf(&x)`
  (drop any `.clone()` that existed only to satisfy the by-value
  signature).
- `BivariateExt::{sample, sample_with_seed}` now take `&self` instead of
  `&mut self` — no implementation mutates during sampling.
  `Independence::sample`'s override was updated to match; callers no
  longer need a `mut` binding just to draw a sample.
- `MarshallOlkin::{pdf, cdf, partial_derivative}` on an unfit copula
  (neither `theta` nor `(alpha, beta)` set) now return `Err("Fit the
  copula first")` instead of panicking through
  `resolve_params().expect(..)` — matching every sibling bivariate
  copula's `check_fit`-gated contract. `tail_dependence`'s existing
  panic-on-invalid-theta contract is unchanged.
- `TCopula.nu` field is now private. Read it via the new `TCopula::nu()`
  getter; write it via the new `TCopula::set_nu(nu) -> Result<(),
  Box<dyn Error>>`, which validates `nu > 0` (mirrors
  `TMultivariate::set_degrees_of_freedom`). `TCopula::with_nu` still
  panics on invalid input but now routes through `set_nu` instead of
  duplicating the check.

### stochastic-rs-stats: no silent fallback without a signal

- `MleResult` gains two fields, `converged: bool` and `iterations: usize`
  (additive to the struct's data — existing field *reads* are unaffected,
  but any code building an `MleResult` via a full struct literal, e.g. in
  tests, must now supply the two new fields). `fit_mle`'s optimizer-failure
  path — previously a silent `Err(_) => init` that returned the untouched
  initial guess indistinguishable from a converged fit — now sets
  `converged = false`. The same signal also covers every other non-success
  termination (`MaxItersReached`, an internal line-search `SolverExit`,
  `Timeout`, `Interrupt`): only `SolverConverged` / `TargetCostReached` set
  `converged = true`. Always check `converged` before trusting `params`.
- `hurst::from_prices::estimate_hurst` and
  `hurst::from_prices::hurst_from_signal` now return `Result<f64,
  HurstError>` instead of a bare `f64` (breaking). They previously
  swallowed degenerate or too-short input into a magic `0.1` default with
  no signal that the estimate was unreliable. Callers who want the pre-2.7
  clamp write `.unwrap_or(0.1)` explicitly at the call site.
- `leverage::estimate_leverage_rho` now returns `Result<f64, HurstError>`
  instead of a bare `f64` (breaking) — reusing `HurstError` rather than a
  bespoke type, since its failure modes (too few observations, a
  degenerate/zero-variance return series) are the same shape as the Hurst
  estimators' over the same close-price-series input. It previously
  swallowed insufficient or degenerate data into a magic `-0.5` default
  with no signal. Callers who want the pre-2.7 clamp write
  `.unwrap_or(-0.5)` explicitly. The Python `Leverage` constructor now
  raises `ValueError` instead of silently returning `-0.5`.

### stochastic-rs-stochastic: sub-Feller Cir/Fcir/Heston2D paths are accepted, not rejected

- `Cir::new`, `Fcir::new`, and `Heston2D::new` no longer hard-`assert!` the
  Feller condition (`2·kappa·theta ≥ sigma²` per factor) — this is a
  behavioral change, not a signature change (breaking only in the sense
  that code relying on the previous panic must update, e.g. a
  `#[should_panic]` test). All three already carry a `use_sym` flag whose
  entire purpose is handling paths that touch the zero boundary
  (reflecting when `true`, flooring at zero otherwise), so the Feller
  precondition was rejecting parameter sets the sampler already handles
  correctly — and was already absent on `Heston`'s (also-CIR) variance
  factor, which is the inconsistency this closes. Sub-Feller parameters
  are now accepted unconditionally; in debug builds, constructing with a
  Feller violation and `use_sym` not set to `Some(true)` prints a one-line
  diagnostic to stderr (never panics); release builds pay nothing for the
  check either way.
- `stochastic-rs-stochastic::volatility::heston2d`'s
  `rejects_matlab_feller_violation` test (asserted the old panic) is now
  `accepts_matlab_feller_violation_with_use_sym`, which builds with
  `use_sym = Some(true)` and asserts the sampled path is finite instead.
