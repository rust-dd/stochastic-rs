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
- `fill_slice_fast(out)` → `fill_slice(out)` (rename, same behavior). This
  was the pre-wave recommended name for the same amortised-SIMD bulk fill;
  the rng-ignoring `fill_slice(rng, out)` above was a second, discouraged
  entry point into the same code. Both collapse into the one
  `fill_slice(out)` method now — update any call site still spelled
  `.fill_slice_fast(...)`.
- `SimdNormal::fill_16(rng, out)` → `fill_16(out)`. Same
  drop-the-ignored-`Rng`-argument change as `fill_slice`, applied to
  `SimdNormal`'s 16-lane fixed-size hot-path fill.
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
  `#[doc(hidden)] DistributionSampler::fork(stream_idx)`, called once per
  worker, sequentially on the caller thread, before any worker starts
  filling; each call reads *and advances* the sampler's own live state —
  an interior-mutable cell distinct from the stream driving real samples —
  so every worker draws its own fresh basis value: one fresh basis per
  worker, never one basis drawn once per call and fanned out across
  workers by index, combined with its `stream_idx` via
  `derive_fork_seed(basis, stream_idx)`. No API signature changed; this is
  a behavior fix.
- **`sample_matrix`'s worker count no longer depends on the ambient
  thread pool, and pinned-seed output may change again as a result.** The
  fork mechanism above originally picked its worker count as
  `min(rayon::current_num_threads(), size-derived cap)`, so two
  identically-`Deterministic`-seeded samplers only agreed under a matching
  thread-pool size — comparing them under a different
  `rayon::current_num_threads()` changed how many times `fork` was called
  and broke the correspondence. Worker count is now
  `total.div_ceil(16 * 1024).max(1).min(total)` where `total = m * n` — a
  pure function of the matrix size alone, never of
  `rayon::current_num_threads()`, mirroring
  `stochastic-rs-stochastic`'s `sample_par`/`sample_map` fix below. Two
  identically-`Deterministic`-seeded samplers now produce bit-identical
  `sample_matrix` output call-for-call on any machine and under any rayon
  thread-pool size; repeated calls on the *same* sampler still never
  replay, for `Deterministic`- and `Unseeded`-constructed samplers alike;
  and a serial call (below the parallel threshold) still never touches
  the fork basis. Any caller who pinned expected `sample_matrix` values
  while running under a thread-pool size other than what this size-only
  rule now picks will see different output — this is a second,
  independent output-changing fix layered on top of the fork mechanism
  above, not a continuation of the same values.
- `DistributionSampler` gained a new required method,
  `#[doc(hidden)] fn fork(&self, stream_idx: u64) -> Self`, and
  `sample_matrix`'s own bound changed from `Self: Clone + Send` to
  `Self: Sized + Send` — `fork` replaced `Clone` as the fan-out mechanism,
  so `Clone` is no longer required. Together with `fill_slice` losing its
  `Rng` argument (above), this is everything an external implementor of
  `DistributionSampler` (there are none in-tree besides the 27 `Simd*`
  types) must update.
- The Python bindings' `sample_par(m, n)` inherits this fix directly:
  seeded (`seed=...`) callers previously always executed the serial path
  under the hood (a workaround for the same-call-replay behavior above —
  going parallel for a reproducible sampler wasn't safe yet); they now
  take the same parallel path as unseeded callers, reproducible
  call-for-call via the per-call fork basis described above. The same
  fixed-thread-pool-size caveat applies here too.
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
  trait implementation itself. This is *not* a pure rename: the old
  inherent method returned a bare `Array2<f64>`; the new trait method
  returns `Result<Array2<f64>, Box<dyn Error>>` (matching every other
  `MultivariateExt`/`BivariateExt` sampler), so callers add `?` (or
  `.unwrap()`) — `let paths = x.sample_seeded(n, seed);` becomes `let
  paths = x.sample_with_seed(n, seed)?;`. The method also now requires
  `use stochastic_rs_copulas::traits::MultivariateExt;` in scope, exactly
  like the existing `.sample(n)`.
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
  `TMultivariate::set_nu`). `TCopula::with_nu` still panics on invalid
  input but now routes through `set_nu` instead of duplicating the check.

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

### stochastic-rs-stats: split CusumResult and normalize acronym casing

- `econometrics::changepoint::CusumResult` → `ChangepointCusumResult`. Two
  unrelated types shared the name `CusumResult` (this control-chart shape
  with `upper`/`lower`/`alarms`, versus the `stationarity::cusum::CusumResult`
  hypothesis-test shape implementing `HypothesisTest`), so glob-importing
  both `econometrics::changepoint::*` and `stationarity::cusum::*` in the
  same scope was ambiguous (`E0659`). `stationarity::cusum::CusumResult`
  keeps its name as the more canonical "CUSUM test" meaning.
- Acronym casing normalized to Rust RFC 430 (an acronym is one word in
  UpperCamelCase, matching already-correct siblings like `GaussianHmm` and
  `HestonCekfFilterResult` in the same modules): `HestonNMLECEKFConfig` →
  `HestonNmleCekfConfig`, `HestonNMLECEKFResult` → `HestonNmleCekfResult`,
  `HestonNMLECEKFParams` → `HestonNmleCekfParams`, `GaussianKDE` →
  `GaussianKde`, `ADFConfig`/`ADFResult` → `AdfConfig`/`AdfResult`,
  `KPSSConfig`/`KPSSResult`/`KPSSCriticalValues`/`KPSSTrend` →
  `KpssConfig`/`KpssResult`/`KpssCriticalValues`/`KpssTrend`,
  `ERSConfig`/`ERSResult`/`ERSTrend` → `ErsConfig`/`ErsResult`/`ErsTrend`,
  `PPTestType` → `PpTestType`, `LMTrend` → `LmTrend`, and
  `FOUParameterEstimationV3` → `FouParameterEstimationV3`. Free functions
  (`adf_test`, `kpss_test`, `ers_dfgls_test`, `cusum_test`, …) were already
  snake_case and are unaffected. Python-visible class names are unchanged —
  every affected type is wrapped by a `#[pyclass(name = "...")]` binding
  with an explicit name already pinned independently of the Rust type name
  (e.g. `ADFTest`, `KPSSTest`, `HestonNMLECEKF`, `GaussianKDE` keep their
  Python spelling).

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
  are now accepted unconditionally; constructing with a Feller violation
  and `use_sym` not set to `Some(true)` unconditionally prints a one-line
  diagnostic to stderr — in release builds too, since that's where real
  Monte Carlo / calibration runs happen — but never panics.
- `stochastic-rs-stochastic::volatility::heston2d`'s
  `rejects_matlab_feller_violation` test (asserted the old panic) is now
  `accepts_matlab_feller_violation_with_use_sym`, which builds with
  `use_sym = Some(true)` and asserts the sampled path is finite instead.

### stochastic-rs-stochastic: Sabr/MultifactorSabr field names now match their own documented SDE

- `Sabr`'s module doc has always stated the literature SDE `dF_t = α_t
  F_t^β dW_t^1`, `dα_t = ν α_t dW_t^2` (α_t is the stochastic-volatility
  *state*, ν is the constant vol-of-vol), but the struct stored ν in a
  field named `alpha` and α's initial value in a field named `v0` —
  contradicting the doc it sits under. Both are renamed to match:
  `Sabr::alpha` → `Sabr::nu` (vol-of-vol) and `Sabr::v0` → `Sabr::alpha0`
  (initial volatility state α₀). `Sabr::new`'s parameter list keeps the
  exact same positional order, just with the two parameters renamed to
  match — `Sabr::new(nu, beta, rho, n, f0, alpha0, t, seed)` — so any
  call site using positional arguments (the only way to call it in Rust)
  needs no change; only code reading `.alpha`/`.v0` on a constructed
  `Sabr` needs to switch to `.nu`/`.alpha0`. The simulation math is
  unchanged — this is a pure rename, verified bit-identical for a fixed
  seed before and after.
- `MultifactorSabr::v0` → `MultifactorSabr::alpha0` for the same reason:
  the field's own doc comment already called it "$\alpha_0$" while the
  field itself was named `v0`. `MultifactorSabr` already named its
  vol-of-vol term-structure field `nu` correctly, so only this one field
  changes. `MultifactorSabr::new`'s positional parameter order is
  unchanged.
- The Python surface is unaffected: `PySabr`'s constructor keeps the
  keyword names `alpha`/`v0` (its `py_process_2x1d!` `sig:`/`params:`
  identifiers are independent local bindings that forward positionally
  into `Sabr::new`, so they never had to match the Rust struct's field
  names). `MultifactorSabr` has no Python binding. For bilingual callers:
  Python's `alpha=` sets Rust's `nu` (vol-of-vol) and Python's `v0=` sets
  Rust's `alpha0` (initial α-state).

### stochastic-rs-stochastic: rough-vol and Hull-White 2F field names now match their own documented SDE

- `RlFOU::sigma` → `RlFOU::nu` and `RlHeston::sigma` → `RlHeston::nu`. Both
  modules' own SDE (`dX_t=\kappa(\mu-X_t)dt+\nu dW^H_t` for `RlFOU`; the
  Volterra-Cir variance diffusion `g(V)=\nu\sqrt{V^+}` for `RlHeston`) and
  each field's own doc comment already called this quantity ν ("Diffusion
  scale $\nu$" / "Volatility of variance $\nu$") — only the field's
  identifier was wrong. Verified against the sampler arithmetic
  (`self.sigma * dfbm` in `RlFOU`; `sigma * vv.max(0).sqrt()` in
  `RlHeston`'s variance diffusion): both multiply exactly where ν belongs.
  Neither file had a competing `nu` field, so this is a plain rename, not a
  cross-swap. `RlHeston::theta` (long-run variance) was already correctly
  named and is untouched. `RlFOU::new`/`RlHeston::new`'s positional
  parameter order is unchanged; only code reading `.sigma` on a constructed
  `RlFOU`/`RlHeston` needs to switch to `.nu`. Neither type has a Python
  binding (`grep -rn "RlFOU\|RlHeston" stochastic-rs-py/` → no hits), so
  there is no Python surface to preserve. Pure rename, verified
  bit-identical for a fixed seed before and after.
- `HullWhite2F::theta` → `HullWhite2F::a`: the field is used
  multiplicatively (`- self.theta * x[i-1]`), i.e. it is the
  mean-reversion **speed** that this module's own doc calls `a` in
  `dx_t=-a x_t dt+\sigma_1 dW_t^1` — not an additive level. Keeping the
  name `theta` for a speed also contradicted the sibling `hull_white.rs`,
  whose `theta: Fn1D<T>` correctly holds the additive θ(t) drift term.
- `HullWhite2F::k` → `HullWhite2F::theta`: verified `k` is in fact that
  same additive, time-dependent calibration term — it is added directly
  into the drift (`self.k.call(t) + u[i-1] - self.theta*x[i-1]`), exactly
  the structural role `hull_white.rs`'s `theta: Fn1D<T>` plays
  (`self.theta.call(t) - self.alpha*prev`). So the two Hull-White files
  now agree on what `theta` means. `HullWhite2F::new`'s parameter list
  keeps the exact same positional order, just with the two parameters
  renamed to match — `HullWhite2F::new(theta, a, sigma1, sigma2, rho, b,
  x0, t, n, seed)` — so any call site using positional arguments (the
  only way to call it in Rust) needs no change; only code reading
  `.theta`/`.k` on a constructed `HullWhite2F` needs to switch to
  `.a`/`.theta`. `HullWhite2F::b` already matched the module doc's own
  `b` symbol and is untouched. The simulation math is unchanged — this is
  a pure rename, verified bit-identical for a fixed seed before and
  after.
- The Python surface is unaffected: `PyHullWhite2F::new`'s own function
  parameters (and its `#[pyo3(signature = (k, theta, ...))]` keyword
  names) are independent local bindings that forward positionally into
  `HullWhite2F::new`, so the Python-visible keyword names stay `k=`/
  `theta=` exactly as before (`tests/python_bindings_smoke.py` calls
  `sr.PyHullWhite2F(k=..., theta=..., ...)` unchanged and still passes).
  This is the sharper mapping to learn, precisely because the names don't
  change: Python's `theta=` is a **float speed** (Rust's `a`), while
  Rust's own `.theta` is a drift **function** — the value Python's `k=`
  supplies.

### stochastic-rs-copulas: unify quantile and degrees-of-freedom naming

- `TMultivariate::degrees_of_freedom()` / `set_degrees_of_freedom()` →
  `nu()` / `set_nu()` (breaking rename), matching `TCopula::nu()` /
  `TCopula::set_nu()` — ν is the standard symbol for degrees of freedom,
  so the two Student-t copula types now share one spelling instead of two
  for the same quantity. The descriptive phrase moved into the `///` doc;
  the error string (`"Degrees of freedom must be positive"`) is
  byte-identical to before. Neither method is Python-visible (`TMultivariate`
  is not wrapped in `python.rs`), so no Python-side change is needed.
- `GaussianUnivariate` gains `percent_point(p)` as the canonical
  quantile-function name, matching `BivariateExt::percent_point` and every
  bivariate family. `ppf(p)` remains available as a documented
  SciPy-compatible alias that delegates to `percent_point` — same
  behavior, callers of `.ppf(...)` need no change.
- `BivariateExt::ppf` (already present, previously undocumented) is now
  explicitly documented as a SciPy-compatible alias for
  `BivariateExt::percent_point`, the canonical name used internally by the
  trait's default sampler and by all 13 bivariate families. No signature
  or behavior change on either trait method.

### stochastic-rs-stochastic: BlackKarasinski/CirPlusPlus constructor order now matches the crate convention

- `BlackKarasinski::new`'s parameter order changes from `(theta, a, sigma,
  r0, n, t, seed)` to `(theta, a, sigma, n, r0, t, seed)` — `n` moves
  before `r0`, matching every other single-factor short-rate model in the
  crate (`Vasicek`, `Cir`, `HullWhite`: `(..., n, x0, t, seed)`). Both
  types are unreleased (added on `main`, never published), so there is no
  deprecated shim; update positional call sites by swapping the `r0` and
  `n` arguments. The Python binding's keyword names (`theta=`, `a=`,
  `sigma=`, `n=`, `r0=`, `t=`, `seed=`) are unchanged — its
  `#[pyo3(signature = ...)]` already listed `n` before `r0` (pyo3 requires
  required arguments before optional ones), so this reorder makes the Rust
  constructor agree with the Python signature instead of contradicting it.
- `CirPlusPlus::new`'s parameter order changes from `(kappa, theta, sigma,
  x0, phi, n, t, use_sym, seed)` to `(kappa, theta, sigma, phi, n, x0, t,
  use_sym, seed)` — `x0` moves after `phi`/`n`, matching `Cir::new`'s own
  `(theta, mu, sigma, n, x0, t, use_sym, seed)` tail. Same rationale, and
  the same already-agreeing Python keyword signature, as `BlackKarasinski`
  above.

### stochastic-rs-stochastic: `ProcessExt::sample_par` / `sample_map` are now reproducible across thread counts

- **Output values under a pinned seed will change — twice.** A
  `Deterministic`-seeded process's `sample_par(m)` / `sample_map(m, f)`
  previously returned different paths from run to run — they were not
  reproducible at all, not even under a fixed thread-pool size. A first
  fix (chunk count `m.div_ceil(8).max(1).min(m)`, one sampler built per
  chunk sequentially before rayon) genuinely repaired this for processes
  whose `sampler()` itself advances the seed at construction (`Gbm`,
  `Ou`, most of the crate), but a follow-up review found it **actively
  regressed** processes whose `sampler()` instead *clones* the seed
  (`Sabr`, `DoubleHeston`, `Bergomi`, and 26 more — a non-advancing
  snapshot per `SeedExt`'s design, so every chunk cloned the identical,
  unchanged state and `sample_par(m)` degenerated to only as many distinct
  paths as one chunk's own length under the `m.div_ceil(8)` rule then in
  effect (8, for `m` a multiple of 8 — a path-count, not a chunk-count;
  the `MAX_CHUNKS` bound described later in this entry did not exist at
  this point in the fix's history), each repeated across every chunk,
  worse than the original scheduler-dependent bug it replaced). This entry
  now describes the corrected fix
  for that class (a separate follow-up fixes a second class, "lazy"
  processes such as `Heston`, for which the first fix was merely a no-op —
  see the next entry below); any code that hard-coded expected output
  from the first fix will see different values again. Neither fix's
  values were ever a supported contract.
- Root cause of the original bug: both default methods ran
  `(0..m).into_par_iter().map_init(|| self.sampler(), ...)`. Rayon decides
  how many times `map_init`'s init closure fires — and how the `m` items
  are grouped across those calls — based on work-stealing at run time, not
  on `m`. Every `self.sampler()` call advances a `Deterministic` process's
  shared atomic seed state (`SeedExt::seed_value`), so the number and
  order of those advances, and therefore the output, depended on
  scheduling. A reviewer measured three runs of one pinned config giving
  means 4.01900 / 4.02577 / 4.00938.
- Fix, corrected shape: `m` paths split into `chunks = m.min(64)`
  contiguous groups — a pure function of `m` alone, never of
  `rayon::current_num_threads()`, and bounded (unlike the first fix's
  `m.div_ceil(8)`, which made the *sequential* chunk-building prologue
  grow with `m` — `sample_par(1000)` built 125 chunks; for a process whose
  `sampler()` is itself expensive, e.g. `Cir2F`, which evaluates a
  `Fn1D` — a Python callback for the `Fn1D::Py` variant — once per grid
  point, that is 125 sequential, potentially GIL-round-tripping
  constructions where the old scheduler-driven code built roughly one per
  core). `ProcessExt` gained a new `#[doc(hidden)] fn advance_chunk_seed(&self)`
  (default: no-op), called once per chunk immediately before that chunk's
  `sampler()`, in the same sequential, pre-rayon loop. A process whose
  `sampler()` clones the seed overrides it (`self.seed.seed_value()`) so
  each chunk's clone snapshots a distinct, advancing state instead of the
  same one; a process whose `sampler()` already advances the seed itself
  uses the no-op default unchanged. `ProcessExt::sample()`'s default also
  now calls `advance_chunk_seed()` once, *after* sampling (not before, or
  a clone-based sampler's first call would skip a state it never
  consumed) — this is what keeps repeated top-level `sample()` calls
  advancing for clone-based processes too, fixing a second, previously
  unnoticed bug where e.g. `Sabr::sample()` called twice in a row (outside
  `sample_par` entirely) silently returned the identical path both times.
- No signature change on either method. `Fgn::sample_par` and
  `Fbm::sample_par` override the default with their own batched-backend
  implementations, so they do not go through this fix (they never used
  `map_init` in the first place) — every other process in the crate does.
  Their own `Cpu`/`Accelerate` batch backends have an independent,
  still-open instance of the same class of defect (each of the `m` batch
  items derives its seed from the same shared atomic *inside* the parallel
  region rather than before it), confirmed empirically but not addressed
  here; tracked as follow-up work, not part of this fix.

### stochastic-rs-stochastic: `Heston` and 9 other "lazy" processes' `sample_par` / `sample_map` are now reproducible too

- **Output values under a pinned seed change again for these 10 types**
  (`Heston`, `Hjm`, `Adg`, `FVasicek`, `Cfgns`, and the fractional family
  `Fou`/`Fgbm`/`Fcir`/`FJacobi`/`Cfou`): the chunk-sequencing fix in the
  entry above (advancing the seed once per chunk before that chunk's
  `sampler()` runs) is a no-op for a process whose `sampler()` reads
  `&self.seed` *lazily*, per path, from inside the returned sampler,
  rather than once at construction — every chunk's sampler shares live
  access to the same atomic regardless of how carefully the chunks
  themselves were sequenced, so concurrent chunks still raced on it during
  the parallel region itself. These 10 are rewritten to capture an owned,
  cloned seed in their sampler at construction instead; their existing
  per-path code already derived from *a* seed each call, so it now derives
  from that owned clone rather than the shared field — a clone, not a
  fresh derive, so the value on the very first path is unchanged from
  before this whole wave (both fixes combined reproduce the pre-`map_init`
  -era stream bit-for-bit there), but the second and later paths, and
  every chunk boundary, now land on values the scheduler-dependent
  original code could never reliably reproduce.
- Two processes named by the same review, `Bates1996` and `RoughHeston`,
  **cannot** be fixed this way and are not: neither's sampled randomness
  derives from `self.seed` **at all**, by pre-existing design predating
  this requirement. `Bates1996`'s diffusion hard-wires an `Unseeded`
  correlated-Gaussian source (`Cgns::new(rho, n - 1, t, Unseeded)` in its
  constructor) and its jump component reads its own `CompoundPoisson`
  driver's seed field directly through `sample_grid_relative_increments`,
  bypassing `ProcessExt` entirely; `RoughHeston`'s correlated-Gaussian
  source is documented in its own `sampler()` as ignoring `self.seed`
  outright. Their `sample`/`sample_par`/`sample_map` were never seed-
  reproducible at all — not even serially, not even at `m == 1` — so
  neither this fix nor the one in the entry above changes that.
- `ProcessExt::sample()`'s default now ticks
  `#[doc(hidden)] fn advance_chunk_seed(&self)` once, *after* sampling —
  see the entry above for the mechanism; it applies unchanged to this
  entry's rewritten processes too, since owning a cloned seed converts
  them into instances of the same "clone-based sampler" shape `Sabr` and
  its 28 siblings already were.

### stochastic-rs-stochastic: `sample_par` / `sample_map` chunk bases are now derived, not cloned — adjacent chunks were still correlated

- **Output values under a pinned seed change a third time**, for every
  process touched by the two entries above plus `Cgns`. Both prior fixes
  were individually verified — thread-count independence held, and
  `Sabr::sample()` called twice no longer repeated its first path — but a
  further review measured what they actually left behind: **adjacent
  chunks were still correlated with each other.** Both the clone-snapshot
  shape (`self.seed.clone()` plus the `advance_chunk_seed` override) and
  the "lazy-rewritten" shape from the entry above (also a clone, per that
  entry's own description) copy `self.seed`'s *raw, unmixed* counter into
  the new sampler; whatever then advances that counter — `advance_chunk_seed`
  between chunks, or the per-path `derive()` inside the sampler — does so by
  the same γ stride used everywhere else in `SeedExt`. So chunk `i` path `j`
  and chunk `i+1` path `j-1` sat one stride apart on the same line, not on
  independent hash outputs. Measured at `m = 1000`: `Sabr` produced only 78
  of 1000 paths actually distinct; `Heston`/`Fou` at `m = 256` produced only
  67 of 256. At realistic Monte-Carlo sample sizes this silently cuts
  effective sample size by an order of magnitude and re-weights the
  estimator with duplicated paths — while every thread-count-independence
  test kept passing, since that property never required chunks to be
  mutually *uncorrelated*, only insensitive to how rayon interleaves them.
- Fix: `sampler()` now always captures its basis with `self.seed.derive()` —
  never `.clone()` — and any per-path code that used to call `.derive()`
  *again* on that basis now consumes it directly instead. `derive()` (unlike
  `clone()`) hash-mixes the counter before handing it to the new owner, so
  chunk `i`'s basis and chunk `i+1`'s basis are uncorrelated hash outputs
  regardless of how many further times the owning sampler ticks its own copy
  afterward. One shape now covers both previously-distinct classes —
  clone-snapshot (`Sabr`, `DoubleHeston`, `Bergomi`, ...) and lazy-rewritten
  (`Heston`, `Fou`, `Hjm`, ...) — around 40 types in total. A sampler whose
  per-path code needs its own owned `S` rather than a borrowed `&S` (e.g.
  `MultifactorSabr`, building two fresh `Gn` generators per path) still
  calls `.derive()` there, but only on this already-derived, already-
  decorrelated basis, which stays safe under any further ticking.
  `#[doc(hidden)] fn advance_chunk_seed` becomes a no-op again for every one
  of these types — the mechanism it was introduced for (advancing a *cloned*
  snapshot) no longer applies to them. `CirPlusPlus` keeps its override: its
  clone feeds a persistent Xoshiro engine built once per chunk and reused
  across every path in that chunk via the engine's own advancement, never
  re-consulting the `Deterministic`-level seed per path, so cloning is safe
  there specifically.
- `Cgns` (the correlated-Gaussian generator well over a dozen other
  processes build on) had the same defect in a form a text search for
  `self.seed.clone()` could not find: `sampler()` cloned the *entire* `Cgns`
  struct (`CgnsSampler { cgns: self.clone() }`), not a `seed` field.
  Measured: `Cgns::sample_par(64)` produced 1 distinct path of 64;
  `sample()` called three times in a row on one object produced 1 distinct
  result of 3 — the same "repeated call replays the first path" bug the
  entry above fixed everywhere else via `advance_chunk_seed`, still live
  here. Fixed with the same derive-not-clone shape. `Cgns` is the one type
  in the crate where this change is user-visible even at `m == 1`: unlike
  `Heston` and the other rewritten types (whose legacy `sampler()` always
  had exactly one `derive()` hop before this whole wave), `Cgns`'s own
  pre-existing behavior had *zero* mixing hops between `self.seed` and its
  first draw, so its very first `sample()` now differs from a bare
  `sample_impl(&seed)` call by the one hash-mixing hop `derive()` adds — no
  golden test pinned the old zero-hop value.
- `CompoundPoisson` needed its own golden re-pin (`cum`/`jumps` in
  `golden_compound_poisson_streams`, not `times`) for a third, distinct
  reason: `Poisson::sample_impl` consumes *two* ticks per call
  (`SimdExp::new` then `.rng()`), not one. Moving a `.derive()` from
  per-path code to `sampler()` reproduces a *single*-tick consumer's value
  exactly (that is the whole mechanism this entry relies on for `Heston`/
  `Sabr`/`Fou`/the ~40 other types above), but for a two-tick consumer that
  something *else* still reads the seed after (`cum`/`jumps` are drawn from
  the same seed right after `times`), the clone-based legacy shape hid the
  consumer's second tick inside a disposable derived temporary, invisible
  to anything downstream; deriving the whole basis once, up front, exposes
  both ticks onto the one live counter everything else shares, shifting
  every value downstream of the first two-tick consumer. `times` itself is
  unaffected, since nothing runs before it. No other in-tree type both (a)
  wraps a two-or-more-tick consumer and (b) has a golden test pinning
  values read after it, so this is not expected to recur, but the same
  silent shift applies unobserved to any multi-tick-consumer type without
  golden coverage (e.g. `DoubleHeston`'s second `Cgns::sample_impl` call,
  `Hkde`'s `Cgns` call followed by its own jump draws) — harmless for the
  properties this fix actually guarantees (reproducibility, cross-chunk
  independence), since neither depends on matching a specific historical
  value.
- **`Bates1996`, `RoughHeston` and `JumpFou` turned out not to be the only
  processes with no randomness reachable from `self.seed` at all** — see
  "the process-level reproducibility exception list is now empty" further
  down this file for the corrected, final account instead of restating an
  intermediate exception list here that later turned out wrong twice more.
- New tests at `m = 256` (`> MAX_CHUNKS = 64`, so multiple paths share a
  chunk) cover the regime the earlier `m = 64`/`m = 16` distinctness tests
  could not reach — at `m <= MAX_CHUNKS` every chunk holds exactly one path,
  which cannot expose cross-chunk correlation at all.

### stochastic-rs-stochastic: `Svcgmy`, `Cgmy`, `KoBoL`, `Cts`, `Rdts` are now seed-reproducible

- **Output values under a pinned seed change for these five types** — not a
  chunking defect, a plain missed wire: each one's `fill_path`/`fill_paths`
  built a `Poisson` arrival-time series (reused as Γ_j, the tempered-stable
  Rosiński series' jump-count process) via `Poisson::new(T::one(), Some(size),
  None, Unseeded)`, hard-wiring that one component away from `self.seed`
  entirely, inside the per-path method rather than at `sampler()` construction
  where every other random source in these types is built. Neither the
  `self.seed.clone()` grep behind the first fix in this file nor a sweep of
  `sampler()` bodies for the cross-chunk-correlation fix two entries above
  could find this: `sampler()` itself was already correct (direct,
  `self.seed`-derived `SimdUniform`/`SimdExp` sources built once per chunk,
  the "eager" shape that never needed either fix), and the broken line lives
  entirely inside the per-path method those sweeps did not inspect.
  Measured before the fix: `Svcgmy`, `Cgmy` and `KoBoL` were fully
  non-reproducible — two identically-`Deterministic`-seeded objects
  disagreed on a single `.sample()` call, `sample_par`/`sample_map` not even
  involved; `Cts` was `sample()`-reproducible but not thread-count
  independent; `Rdts` happened to pass in one reviewer configuration but
  carried the identical line. `Svcgmy` is the sharpest case: the
  cross-chunk-correlation entry above had already converted its `sampler()`
  from `clone()` to `derive()`, so it was covered by that entry's "Guarantee,
  corrected" claim while still not being seed-reproducible at all, via this
  one line.
- Fix: `Svcgmy` already retained an owned `seed: S` field on its sampler
  (added by the cross-chunk-correlation fix); `Cgmy`, `KoBoL`, `Cts` and
  `Rdts` did not (their samplers were not generic over `S` at all, holding
  only the `SimdUniform`/`SimdExp` engines built from `self.seed` at
  construction), so each gained one — populated via `self.seed.derive()`
  in `sampler()`, matching the shape the cross-chunk-correlation fix
  established elsewhere. The per-path `Poisson::new(..., Unseeded)` call in
  all five now reads `Poisson::new(T::one(), Some(size), None,
  self.seed.derive())` instead: deriving once per fill from this owned,
  already chunk-decorrelated basis keeps Γ_j reproducible under a
  `Deterministic` seed and distinct path-to-path, safely, for the same
  reason repeated further derives are safe anywhere else in the crate a
  sampler's own basis is already chunk-unique.
- A dedicated sweep for the same hard-wired-`Unseeded`-in-per-path-code
  pattern, run after this fix, found no further instances: every other
  `Unseeded` literal in the crate is either a struct-level default generic,
  a Python-binding's correct behavior when no seed is given, test code, a
  type-constrained convenience constructor that can only ever produce an
  `Unseeded` value (`Volterra::fbm`), or a field whose own `Unseeded` is
  provably inert because every consumer of it is externally re-seeded
  (`Cgns`/`Fgn`/`RlFBm` fields across the volatility/interest/rough
  families) — except the already-documented exceptions
  (`Bates1996`/`RoughHeston`/`JumpFou`/`JumpFOUCustom`'s diffusion) and the
  already-tracked `Fgn`/`Fbm` `sample_par` batched-backend gap, both
  unchanged by this entry.

### stochastic-rs-stochastic: `Fgn`/`Fbm` `sample_par` are now thread-count independent

- **Output values under a pinned seed change for `Fgn::sample_par` and
  `Fbm::sample_par`** (the CPU and `accelerate`-feature backends only — see
  below): these were the crate's only two in-tree `sample_par` overrides,
  bypassing `ProcessExt`'s default `chunked_samplers` mechanism entirely to
  reach the batched backend path, so neither Task 1 nor the later chunk-
  derivation fixes touched them — this entry closes that tracked gap.
  `Backend::generate_batch`'s `Cpu` impl used to do
  `(0..m).into_par_iter().map(|_| fgn.sample_cpu())`, and `sample_cpu` reads
  `&self.seed` — a shared `Deterministic` atomic — fresh, from *inside* the
  parallel region, once per path; which of the `m` parallel iterations
  claimed which tick depended on rayon's scheduling, hence on thread-pool
  size. Fix: `generate_batch` now takes an explicit `seed: &S2` parameter;
  on `Cpu`, it derives one basis **per path** (`(0..m).map(|_|
  seed.derive())`, sequentially, on the calling thread) before handing the
  `m` (basis, path) pairs to rayon, so which physical thread ends up
  computing path `i` no longer changes which basis path `i` consumes.
  Same seed + same `m` ⇒ bit-identical output on any machine, under any
  rayon thread-pool size, for the `Cpu` backend; `Unseeded` still draws
  fresh randomness every call. (`Accelerate` gets the same seed-consumption
  fix but a weaker overall guarantee — see below.)
- **Rejected intermediate design, kept here as a warning:** the first
  implementation reused this wave's `ProcessExt::chunk_count`/`chunk_lens`
  verbatim — capping at `MAX_CHUNKS = 64` chunks, one `SimdNormal` built per
  chunk and reused sequentially across that chunk's paths, exactly
  `ProcessExt::chunked_samplers`'s own shape. It was correct (thread-count
  independent, bit-identical) but ~2× *slower* at `m = 1000`
  (`FGN_sample_par/sample_par/1000`: 7.34 ms → 13.4–18.4 ms measured).
  Root cause: `Fgn::fill_cpu` calls `ndrustfft::ndfft_inplace_par`, which is
  *itself* a nested rayon parallel region (`ndarray::parallel`'s
  `Zip::par_for_each` over the array's rows). The old, unchunked code ran
  `m` independent outer rayon leaf tasks, each making exactly one such
  nested call — cheap, since a single-row `Zip` has nothing left to split
  and returns immediately. Chunking collapsed that to `MAX_CHUNKS` outer
  tasks each firing the same nested-rayon entry point repeatedly, back to
  back, from one worker thread — measurably more expensive than spreading
  the identical nested calls across independent outer tasks. Shipped fix:
  one rayon leaf task per **path**, uncapped (not `chunk_count`-limited),
  preserving the original fine-grained parallelism while still deriving
  every basis sequentially up front. The extra `SimdNormal` construction
  cost (one per path instead of one per `MAX_CHUNKS`-capped chunk) is
  negligible next to an FFT. This is *not* a case for reusing `chunk_count`
  everywhere the wave's shape applies — a nested-parallel hot loop is a
  real exception, and any future batched-FFT-style override should
  benchmark before assuming the same chunking constant is safe.
- **A second, deeper bug in `Fbm::sample_par` specifically — not merely
  thread-count dependence, but seed-blindness:** `Fbm::sample_par` drove the
  batch through `self.fgn.noise_batch(m)`, where `self.fgn: Fgn<T, Unseeded,
  B>` is *always* `Unseeded` by construction (see `Fbm`'s own doc — the
  embedded `fgn` exists only for its FFT/eigenvalue cache and was never
  meant to carry randomness). Since `generate_batch` read `fgn`'s own seed
  field, `Fbm::sample_par` never consulted `Fbm`'s real `self.seed` at all —
  a `Deterministic`-seeded `Fbm::sample_par` drew fresh, non-reproducible
  randomness on *every single call*, seeded or not, regardless of thread
  count. Fixed by threading `self.seed` (the outer, real seed) into
  `noise_batch`/`generate_batch` explicitly, instead of relying on `fgn`'s
  dead field. `Fgn::sample_par` did not have this second bug — its `self`
  already was the real, correctly-seeded object — only the race.
- **Correction — `Accelerate` does NOT carry `Cpu`'s bit-identity
  guarantee; this was asserted without evidence and is wrong.** The
  original version of this bullet claimed "gets the identical guarantee...
  deliberately," reasoning only about the *seed-consumption* mechanism
  (which is genuinely fixed the same way — see below) and never measuring
  the actual output. External review measured it directly: two
  identically-`Deterministic`-seeded `Accelerate` calls, same process, same
  seed, same `m`, nothing else touched, disagreed in 207 of 1024 elements
  (max relative difference `1.29e-5`), repeatably. Independently reproduced
  while fixing this entry (Apple M4 Max, 10 P-cores + 4 E-cores): 400
  repeated calls across 35 `(n, m)` combinations on an otherwise-idle
  system showed **zero** divergence, but the identical sweep run with all
  14 cores saturated by unrelated floating-point work showed **21 of 400**
  configurations diverge, worst observed relative difference `2.08e-3`;
  `Cpu`, run under the identical induced load and sweep, stayed bit-exact
  in all 400 — isolating the effect to `Accelerate`/vDSP specifically, not
  the measurement method. This matches the reviewer's hypothesis: Apple
  Silicon's heterogeneous P-core/E-core scheduler can dispatch
  `vDSP_fft_zip` to different core types across calls, and the vectorized
  FFT code path is not guaranteed to produce bit-identical results across
  core types. **What the seed-consumption fix actually gives `Accelerate`:**
  `sample_accelerate_impl` now takes an external seed generic instead of
  reading `self.seed`, and `Backend::generate_batch`'s `Accelerate` impl
  uses `ProcessExt::chunk_count`/`chunk_lens` (capped at `MAX_CHUNKS = 64`,
  unlike `Cpu` — see the rejected-design note above for why `Cpu` cannot do
  the same; `vDSP_fft_zip` has no internal rayon parallelism to contend
  with, so grouping is free here) before handing each chunk to rayon as one
  `sample_accelerate_impl(len, ..)` vDSP batch call. This makes *which
  derived basis feeds which path* thread-count independent, exactly like
  `Cpu` — a real, meaningful fix, just not a sufficient one for bit
  identity, since vDSP's own arithmetic sits on top of it. Corrected
  guarantee: `Accelerate` is seed-consumption-deterministic (thread-count
  independent) but **not** bit-identical — reproducible-effort-only, the
  same tier as the GPU backends below, not `Cpu`'s tier. See `device.rs`'s
  `Backend` trait doc for the corrected table and
  `tests/deterministic_parallelism_accelerate.rs` for the measurement this
  correction is based on.
- **The GPU backends (`CudaNative`, `CubeCl`/`gpu`, `MetalNative`) are
  explicitly OUT of this reproducibility guarantee, documented rather than
  fixed:** each already draws one `u32`/`u64` value from `self.seed.rng()`
  per batch call and feeds it to the on-device kernel's own Philox/PCG-style
  RNG, so output is a function of the pinned seed and not of host
  thread-pool size (there is no host-side rayon fan-out inside their
  `generate_batch` — it is a single kernel-launch call), but cross-run
  bit-identity across GPU driver versions, vendors, or repeated runs on the
  same device is untested and not promised. `Backend::generate`/
  `generate_batch`/`generate_pair`'s new `seed: &S2` parameter is ignored by
  all three, exactly as `generate`'s host-side seed parameter already was.
  See `device.rs`'s `Backend` trait doc for the full per-backend table.
- **Correction — the GPU row's "output is a function of the pinned seed"
  claim is false for `Fbm` specifically.** `Fbm` reaches a GPU backend via
  `backend_switch!(… via fgn)`, which re-types only the embedded `fgn:
  Fgn<T, Unseeded, B>` field — `Fbm::sample_par` still passes `&self.seed`
  to `noise_batch`, but the GPU backends ignore that parameter and read
  `fgn.seed` instead (see the trait doc table), which for `Fbm` is *always*
  `Unseeded`, never the real outer seed. So a `Deterministic`-seeded `Fbm`
  on `MetalNative`/`CudaNative`/`CubeCl` draws fresh randomness on every
  call, exactly as if it were `Unseeded` — not merely "untested cross-run
  stability" like bare `Fgn` on the same backends, but zero dependence on
  the pinned seed at all. This wave's `Fbm` seed-blindness fix (see above)
  covers the `Cpu`/`Accelerate` backends only; GPU backends were never in
  scope for either fix and remain seed-blind for `Fbm` specifically.
  Documented on `Fbm::sample_par`'s own doc rather than restated in the
  trait-level table, since it is `Fbm`-specific, not backend-specific.
- Perf, shipped design vs. the pre-existing (buggy) code — `cargo bench
  --bench fgn_fbm -- "FGN_sample_par|FBM_sample_par"` (Apple Silicon,
  `n = 4096`, mean of 100 samples): `FGN_sample_par/sample_par/100` 806 µs
  → 796 µs; `/1000` 7.34 ms → 7.05 ms; `FBM_sample_par/sample_par/10`
  228 µs → 223 µs; `/100` 1.04 ms → 1.01 ms; `/1000` 8.82 ms → 9.00 ms (+2%,
  within this benchmark's run-to-run noise band — the `sample_sequential`
  control at the same `m`, untouched by this fix, moved +1 to +6% between
  the same two runs). No regression at any sampled `m`; the small
  improvements at most `m` are consistent with removing the old code's
  atomic contention (up to `m` threads calling `Deterministic::next_u64()`
  on the *same* shared atomic concurrently) in favor of sequential,
  uncontended `derive()` calls before the parallel region starts.

### stochastic-rs-stochastic: the process-level reproducibility exception list is now empty

- **This entry replaces three earlier ones in this file** ("`Heston` and 9
  other 'lazy' processes'...", "`sample_par`/`sample_map` chunk bases are
  now derived, not cloned...", and the corrections layered onto both) that
  each asserted a guarantee "for every process except a specific,
  shrinking list," revised on nearly every subsequent commit as review
  found the list itself wrong or incomplete. Restating a corrected verdict
  as a new live exception list is exactly how this file drifted out of
  sync with the code repeatedly; this entry states the settled history and
  the current guarantee once, instead of layering a fourth version on top.
- `Bates1996` and `RoughHeston` were twice listed as full exceptions ("no
  randomness reachable from `self.seed` at all") on the theory that their
  correlated-Gaussian source (`Cgns`, always built with a hard-wired
  `Unseeded`) could never be redirected to an external seed. Wrong:
  `Cgns::sample_impl<S2: SeedExt>(&self, seed: &S2)` accepts an *external*
  seed, exactly how every sibling `cgns`-holding type (`DuffieKan`,
  `DuffieKanJumpExp`, `BatesSvj`, `DoubleHeston`, `Hkde`) already drove it —
  nobody had wired these two the same way. Fixed by deriving an owned
  `seed: S` in `sampler()` and calling `sample_impl` on it instead of the
  bare `.sample()`. `RoughHeston` has no jump component, so this alone made
  it fully reproducible; `Bates1996`'s separate `cpoisson` defect (below)
  kept it partially exceptional until Task 2.
- `JumpFou` was also listed as a full exception, on the theory that both
  its `fgn` diffusion field and its `cpoisson` jump field were structural,
  public-field-shaped pins. Wrong about `fgn`: `jump_fou.rs`'s
  `fgn: Fgn<T, Unseeded, B>` is **private** — the identical shape
  `JumpFOUCustom`'s field has, fixed the same non-breaking way (next
  bullet). Its `cpoisson` genuinely was public and pinned, so `JumpFou`
  stayed partially exceptional until Task 2.
- `JumpFOUCustom`'s private `fgn` field's own `sampler()` read `fgn`'s own
  dead `Unseeded` seed instead of the outer `self.seed`. Fixed
  non-breakingly (the field is private, unlike `Merton`'s public
  `cpoisson`): `sampler()` now builds the Gaussian source from
  `self.seed.derive()` directly, borrowing `fgn` only for its `Arc`-shared
  FFT plan/eigenvalues. It has no `CompoundPoisson` field, so this alone
  made it fully reproducible.
- `Merton`, `Kou`, `LevyDiffusion`, `Bates1996` and `JumpFou` shared one
  remaining, genuinely breaking defect: a `pub cpoisson:
  CompoundPoisson<T, D>` field structurally pinned to `Unseeded`
  (`CompoundPoisson`'s third type parameter defaults to `Unseeded`, and
  none of the five types' field declarations named `S`), so no caller could
  ever supply a `Deterministic`-seeded jump driver through it regardless of
  the outer process's own seed. Neither was named in any type doc, the
  `ProcessExt` trait doc, or this file before being found — an omission,
  not a design decision. The zero-exception-reproducibility wave's Task 1
  fixed `Merton`/`Kou`/`LevyDiffusion`; Task 2 fixed `Bates1996`/`JumpFou`
  — see the two dedicated entries below for the breaking constructor change
  and before/after call sites.
- **Current, final state: zero exceptions of any kind.** Every process in
  `stochastic-rs-stochastic` derives all of its sampled randomness from
  `self.seed`, for both its diffusion and (where applicable) jump
  component. Same seed + same `m` ⇒ bit-identical `sample`/`sample_par`/
  `sample_map` output, on any machine and under any rayon thread-pool size.
  This is no longer an assertion in this file alone:
  `tests/reproducibility_all_processes.rs` enumerates every concrete
  `ProcessExt` implementor (124 as of this wave) and asserts it directly,
  so a regression on any one type fails that test instead of waiting for
  another ad-hoc review to notice — the exact failure mode that produced
  the corrections this entry replaces. Backend-level exceptions are a
  separate, unaffected axis: `Accelerate` is seed-consumption-deterministic
  but not bit-identical (vDSP's own arithmetic is not bit-stable — see the
  `Fgn`/`Fbm` entry above and `device.rs`'s `Backend` trait doc), and GPU
  backends are excluded from the guarantee entirely — both by design, not
  regression.

### stochastic-rs-stochastic: `Merton`, `Kou`, `LevyDiffusion` absorb the jump-driver construction — fully seed-reproducible

- **Breaking constructor change, deliberate — Task 1 of the
  zero-exception-reproducibility wave.** All three types' `cpoisson:
  CompoundPoisson<T, D>` field was structurally pinned to `Unseeded`:
  `CompoundPoisson<T, D, S: SeedExt = Unseeded>`'s third parameter defaults
  to `Unseeded`, and the field's declared type never named `S`, so no
  caller could ever supply a `Deterministic`-seeded jump driver through it
  regardless of the outer process's own seed. The field is now `cpoisson:
  CompoundPoisson<T, D, S>` (`S` matching the process's own), and `new()`
  absorbs the compound-Poisson construction: it takes the jump-size
  distribution and the intensity directly and builds the internal
  `Poisson`/`CompoundPoisson` pair itself, seeded from the constructor's
  own `seed: S` parameter. This also collapses the three-step, three-seed
  construction chain (`Poisson::new(…, seed)` → `CompoundPoisson::new(dist,
  poisson, seed)` → `Merton::new(…, cpoisson, …, seed)`) the 2026-08-11 API
  review flagged as a footgun into one call with one seed.
- Before/after call sites:

  ```rust
  // Before
  let cpoisson = CompoundPoisson::new(
    ScalarNormal::new(0.0, 0.1),
    Poisson::new(1.0, Some(252), Some(1.0), Unseeded),
    Unseeded,
  );
  let m = Merton::new(
    0.03, 0.2, 1.0, 0.0, 252, Some(0.0), Some(1.0), cpoisson, Deterministic::new(42),
  );

  // After
  let m = Merton::new(
    0.03, 0.2, 1.0, 0.0, ScalarNormal::new(0.0, 0.1), 252, Some(0.0), Some(1.0),
    Deterministic::new(42),
  );
  ```

  ```rust
  // Before
  let cpoisson = CompoundPoisson::new(
    ScalarNormal::new(0.0, 0.12),
    Poisson::new(1.0, Some(252), Some(1.0), Unseeded),
    Unseeded,
  );
  let k = Kou::new(
    0.03, 0.2, 1.0, 0.0, 252, Some(0.0), Some(1.0), cpoisson, Deterministic::new(42),
  );

  // After
  let k = Kou::new(
    0.03, 0.2, 1.0, 0.0, ScalarNormal::new(0.0, 0.12), 252, Some(0.0), Some(1.0),
    Deterministic::new(42),
  );
  ```

  ```rust
  // Before
  let cpoisson = CompoundPoisson::new(
    ScalarNormal::new(0.0, 0.08),
    Poisson::new(1.0, Some(252), Some(1.0), Unseeded),
    Unseeded,
  );
  let l = LevyDiffusion::new(
    0.01, 0.2, 252, Some(0.0), Some(1.0), cpoisson, Deterministic::new(42),
  );

  // After
  let l = LevyDiffusion::new(
    0.01, 0.2, 1.0, ScalarNormal::new(0.0, 0.08), 252, Some(0.0), Some(1.0),
    Deterministic::new(42),
  );
  ```

  `LevyDiffusion::new` gains an explicit `lambda: T` parameter it did not
  have before (previously only reachable inside the pre-built `cpoisson`'s
  own `Poisson`); `Merton`/`Kou` already had `lambda` as a top-level
  parameter and keep its position, inserting `jump_dist: D` where
  `cpoisson` used to sit, ahead of the `…, n, x0, t, seed` tail.
- The `cpoisson` field stays `pub` on all three (now correctly typed
  `CompoundPoisson<T, D, S>`), for two reasons. First, `cpoisson` is a
  `CompoundPoisson` in its own right: calling `.sample()` on it directly
  (bypassing the outer type entirely) drives it through
  `Poisson::sample_impl`, which genuinely branches on `.n`/`.t_max` (fixed
  count vs. horizon mode) and consults `.seed` — none of that is dead in
  that usage, only on *this type's own* `sample_grid_increments`-driven
  sampling path, where only `.distribution` and `.lambda` are ever read.
  Second, a caller can still replace the whole jump driver via
  `Merton::with_cpoisson` (re-typed, not renamed) — see the correction
  below for what that does and does not preserve. `Merton`'s field doc no
  longer claims the jumps are non-reproducible.
- **Correction, found by this task's own review: the paragraph above
  overclaimed 100% capability preservation via `with_cpoisson`.** `sampler()`
  reads the jump-arrival intensity off `self.lambda` directly, not off
  `cpoisson.poisson.lambda` — so `with_cpoisson`, as first shipped, replaced
  the jump-*size* distribution but silently left sampling at the *old*
  `self.lambda`, ignoring the swapped-in driver's own intensity entirely.
  Measured: `with_cpoisson(lambda=0)` on a `lambda=80` `Merton` produced a
  path matching a fresh `lambda=80` construction, not `lambda=0`. Pre-fix,
  `with_cpoisson` was the *only* way to set intensity, and it worked; this
  regressed it to half-working. Fixed (separate commit, `fix: make the jump
  intensity single sourced`) by making `self.lambda` the single source of
  truth end to end: `with_cpoisson` now adopts the incoming driver's
  `cpoisson.poisson.lambda` into `self.lambda`, and every setter that
  changes `lambda`/`n`/`t` (`with_lambda`, `with_steps`, `with_horizon`)
  re-syncs the otherwise-cosmetic `cpoisson.poisson` mirror via a new
  private `resync_cpoisson_poisson` helper, so it never goes stale for a
  caller inspecting it directly. `LevyDiffusion` gained its own top-level
  `pub lambda: T` field (it had none before — `lambda` lived only inside
  `cpoisson.poisson.lambda`) so all three types agree on where λ lives;
  `Kou`/`LevyDiffusion` have no `with_*` setters, so `new()` establishing
  the invariant is sufficient for them. See
  `Merton::{lambda,cpoisson,with_lambda,with_cpoisson,with_steps,with_horizon}`'s
  doc comments and the new regression tests in `with_setters_merton.rs`
  (`merton_with_cpoisson_changes_sampled_intensity`,
  `merton_with_lambda_syncs_cpoisson_and_changes_sampled_path`) and
  `reproducibility_jump_family.rs`
  (`jump_family_lambda_is_single_sourced_at_construction`).
- `sampler()` for all three now derives a fresh, chunk-local jump seed
  (`self.cpoisson.seed.derive()`) once per chunk, mirroring how the
  diffusion component already derived its own per-chunk basis — never a
  borrowed `&self.cpoisson` shared across chunks, which would let
  concurrent chunks race on the same shared atomic during the parallel
  region (see `ProcessExt`'s trait-level reproducibility requirement).
  `new()` seeds `cpoisson` via `seed.clone().derive()`, not a bare
  `seed.derive()`, specifically so deriving the jump child does not itself
  advance the value stored into `self.seed` — the diffusion component's
  bit-exact stream under a given seed is unchanged by this fix.
- Found and fixed along the way: `Merton::with_seed` previously replaced
  only the top-level `self.seed`, leaving `cpoisson`'s own (now-meaningful)
  seed keyed to whatever it was at construction — silently *not* matching a
  fresh construction with the new seed, contradicted by
  `merton_with_seed_matches_fresh_construction`'s own name. `with_seed` now
  re-derives `cpoisson.seed` the same way `new()` does.
- All three are now **fully** seed-reproducible — no exception to
  `ProcessExt`'s reproducibility guarantee. At the time this task shipped,
  `Bates1996` and `JumpFou` still carried the identical `cpoisson` defect
  and remained partial exceptions; Task 2 of this wave fixed both the same
  way (see that entry below), and a later commit in this wave replaced
  `traits/process.rs`'s exception list with the unconditional guarantee
  stated as prose — see the "Current, final state" summary above.
- New test: `stochastic-rs-stochastic/tests/reproducibility_jump_family.rs`
  — for each type, bit-identical `.sample()` between two
  identically-`Deterministic`-seeded objects under `lambda = 50` (jumps
  dominate), `sample_par` bit-identical across 1/3/8-thread pools at both
  `m = 64` and `m = 256`, and `m = 256` producing 256 distinct paths.
  `defaults.rs`'s `clone_preserves_deterministic_path` now covers `Merton`/
  `Kou` too (previously excluded, with a dedicated test pinning their
  non-reproducibility as expected behavior — deleted, since the premise no
  longer holds).
- `tests/sampler_v3_golden.rs`'s header claimed "`Merton` is intentionally
  absent: it hard-wires its inner `CompoundPoisson<T, D>` to `Unseeded`, so
  its jump chain is not bit-reproducible" — true when written, false as of
  this fix. Corrected (separate commit, `docs: correct the merton golden
  justification`), and a `golden_merton_streams` test added: the first
  golden pin covering a jump chain (`lambda = 3.0` at `N = 8` so the
  8-point stream exercises at least one nonzero jump increment, not just an
  all-zero `sample_grid_increments` short-circuit).

### stochastic-rs-stochastic: `Bates1996` and `JumpFou` absorb the jump-driver construction — zero exceptions left

- **Breaking constructor change, deliberate — Task 2 of the
  zero-exception-reproducibility wave**, applying the identical fix Task 1
  made to `Merton`/`Kou`/`LevyDiffusion` to the two remaining
  partial-exception types. Both types' `cpoisson: CompoundPoisson<T, D>`
  field was structurally pinned to `Unseeded` for the same reason: the
  field's declared type never named `S`, so no caller could ever supply a
  `Deterministic`-seeded jump driver through it regardless of the outer
  process's own seed. The field is now `cpoisson: CompoundPoisson<T, D, S>`
  on both, and `new()` absorbs the compound-Poisson construction: it takes
  the jump-size distribution directly and builds the internal
  `Poisson`/`CompoundPoisson` pair itself, seeded from the constructor's own
  `seed: S` parameter via `seed.clone().derive()` (cloned first so deriving
  the jump child does not itself advance the value stored into `self.seed`
  — identical rationale to Task 1's fix).
- Before/after call sites:

  ```rust
  // Before
  let cpoisson = CompoundPoisson::new(
    ScalarNormal::new(0.0, 0.05),
    Poisson::new(2.0, Some(128), Some(1.0), Unseeded),
    Unseeded,
  );
  let b = Bates1996::new(
    Some(0.05), None, None, None, 2.0, 0.0, 0.04, 1.5, 0.3, -0.6,
    128, Some(100.0), Some(0.04), Some(1.0), Some(false),
    cpoisson, Deterministic::new(42),
  );

  // After
  let b = Bates1996::new(
    Some(0.05), None, None, None, 2.0, 0.0, 0.04, 1.5, 0.3, -0.6,
    ScalarNormal::new(0.0, 0.05),
    128, Some(100.0), Some(0.04), Some(1.0), Some(false),
    Deterministic::new(42),
  );
  ```

  ```rust
  // Before
  let cpoisson = CompoundPoisson::new(
    ScalarNormal::new(0.0, 0.08),
    Poisson::new(1.0, Some(252), Some(1.0), Unseeded),
    Unseeded,
  );
  let j = JumpFou::new(
    0.7, 1.5, 0.03, 0.2, 252, Some(0.0), Some(1.0), cpoisson, Deterministic::new(42),
  );

  // After
  let j = JumpFou::new(
    0.7, 1.5, 0.03, 0.2, 1.0, ScalarNormal::new(0.0, 0.08),
    252, Some(0.0), Some(1.0), Deterministic::new(42),
  );
  ```

  `jump_dist: D` is inserted right before the `n, x0/s0(+v0), t, seed` tail
  in both — the same slot Task 1 used, not `cpoisson`'s old slot immediately
  before `seed` — so it sits alongside the other model parameters rather
  than at the very end. `JumpFou::new` gains an explicit `lambda: T`
  parameter it did not have before (previously only reachable inside the
  pre-built `cpoisson`'s own `Poisson`), inserted right after `sigma`;
  `Bates1996` already had `lambda` as a top-level parameter (used for the
  drift's `-lambda*k` compensator term) and keeps its position unchanged.
- `cpoisson` stays `pub` on both, for the same two reasons Task 1's entry
  gives for `Merton`/`Kou`/`LevyDiffusion`: it is a `CompoundPoisson` in its
  own right (`.sample()` on it directly drives `Poisson::sample_impl`, which
  genuinely branches on `.n`/`.t_max` and consults `.seed`), and a caller can
  still replace it wholesale — via `Bates1996::with_cpoisson` (`JumpFou` has
  no `with_*` setters at all, so `new()` establishing the invariant is
  sufficient for it, the same as `Kou`/`LevyDiffusion` in Task 1).
- **A live instance of Task 1's `with_cpoisson`/intensity bug, found on
  `Bates1996` before any fix was applied here (not introduced by this
  task).** `sampler()` reads the jump-arrival intensity off `self.lambda`
  directly, not off `cpoisson.poisson.lambda` — for `Bates1996` specifically
  this was already live pre-Task-2, because `with_lambda` (`bates.rs:181`,
  pre-fix) wrote only `self.lambda` while the sampler's jump term
  (`self.cpoisson.sample_grid_relative_increments(...)`) read
  `cpoisson.poisson.lambda`, a value `with_lambda` never touched. Measured
  before the fix, on a `lambda = 0` base with `k = 0` (isolating the jump
  half — `k = 0` also neutralizes the drift's `-lambda*k` term, and
  `cpoisson.poisson.lambda = 0` makes `sample_grid_relative_increments`
  short-circuit to an all-zero, RNG-free array, giving a luck-independent
  comparison despite `cpoisson.seed` being `Unseeded` at the time):

      after with_lambda(80):  self.lambda = 80  but  cpoisson.poisson.lambda = 0
      with_lambda(80) path == fresh lambda=80 ? false
      with_lambda(80) path == fresh lambda=0  ? true

  Fixed the same way as `Merton`: `self.lambda` is the single source of
  truth end to end. `with_lambda` and `with_cpoisson` (the latter now
  adopting the incoming driver's `cpoisson.poisson.lambda` into
  `self.lambda`, exactly like `Merton::with_cpoisson`) both call a new
  private `resync_cpoisson_poisson` helper that rebuilds
  `cpoisson.poisson` from `self.{lambda, n, t}`, so the mirror never goes
  stale for a caller inspecting `cpoisson` directly; `with_steps`/
  `with_horizon` call it too (dead on the sampling path, matching `n`/`t`'s
  own inertness there, but kept in sync for the same reason). See
  `Bates1996::{lambda,cpoisson,with_lambda,with_cpoisson,with_steps,with_horizon}`'s
  doc comments and the regression tests in `with_setters_jump_correlation.rs`
  (`bates_with_cpoisson_changes_sampled_intensity`,
  `bates_with_lambda_syncs_cpoisson_and_changes_sampled_path`) and
  `reproducibility_bates_jump_fou.rs`
  (`lambda_is_single_sourced_at_construction`). `JumpFou` has no `with_*`
  setters and no top-level `lambda` before this fix, so it was not exposed
  to this bug — `new()` establishing the invariant is sufficient, matching
  `Kou`/`LevyDiffusion` in Task 1.
- `sampler()` for both now derives a fresh, chunk-local jump seed
  (`self.cpoisson.seed.derive()`) once per chunk, mirroring the diffusion
  component's own per-chunk basis — never a borrowed `&self.cpoisson` shared
  across chunks. This was a *live* risk for both, not merely a style
  preference: `Bates1996`'s old `BatesSampler` held
  `cpoisson: &'a CompoundPoisson<T, D>` (a shared borrow reused across every
  chunk in `sample_par`), and `JumpFou`'s old `JumpFouSampler` held the
  analogous `cpoisson: &'a CompoundPoisson<T, D>`; once `cpoisson`'s own `S`
  can be `Deterministic` (this fix), a shared borrow would let concurrent
  chunks race on the same shared atomic during the parallel region. Both
  samplers now own `jump_distribution: &'a D` (borrowed — read-only) plus an
  owned, derived `jump_seed: S`, the same split `Merton`/`Kou`/`LevyDiffusion`
  use. `Bates1996`'s multiplicative jump term needed a new free function,
  `grid_relative_increments` (parallel to the additive `grid_increments`
  Task 1 already exposed), extracted from
  `CompoundPoisson::sample_grid_relative_increments`'s existing body with no
  behavior change — that method is now a one-line delegator to it, exactly
  mirroring `sample_grid_increments`/`grid_increments`'s existing
  relationship.
- **`JumpFou`'s private `fgn: Fgn<T, Unseeded, B>` diffusion field did not
  need fixing here — it was already fixed by the predecessor wave's final
  round** (see "`Bates1996` and `RoughHeston`'s 'unfixable' verdict was
  wrong" above, the `JumpFou` re-examination bullet, and its own correction
  two bullets later). Verified directly before touching this file:
  `jump_fou.rs`'s `sampler()` already built
  `normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed.derive())`
  (not `self.fgn.sampler()`, which would read `fgn`'s own dead `Unseeded`
  field) and the type's own doc comment already documented the diffusion
  half as fixed; `deterministic_parallelism_jump_fou.rs`'s
  `jump_fou_diffusion_is_seed_reproducible_with_zero_jump_intensity` and its
  thread-count-independence/distinctness siblings already existed and
  already passed before this task's changes. Only the field's doc comment
  and the two test files' now-stale "partial exception" framing needed
  updating to reflect that `cpoisson` — the one remaining broken half — is
  fixed too.
- Both types are now **fully** seed-reproducible — no exception to
  `ProcessExt`'s reproducibility guarantee, and no exception of any kind
  remains anywhere in the crate. A later commit in this wave (`docs: state
  the reproducibility guarantee without exceptions`) replaced
  `traits/process.rs`'s partial-exception list entirely, stating the
  guarantee as prose instead of an enumerated list — see the summary entry
  further up this file for the corresponding update.
- New test file: `stochastic-rs-stochastic/tests/reproducibility_bates_jump_fou.rs`
  — for each type, bit-identical `.sample()` between two
  identically-`Deterministic`-seeded objects under `lambda = 50` (jumps
  dominate), a same-file `lambda = 0` counterfactual proving the bit-identity
  test is not a diffusion-only pin, `sample_par` bit-identical across
  1/3/8-thread pools at both `m = 64` and `m = 256`, and `m = 256` producing
  256 distinct paths. `deterministic_parallelism_bates_rough_heston.rs`'s
  `bates_price_path_jump_component_still_diverges` (pinned the old, broken
  behavior via `assert_ne!`) is replaced by
  `bates_price_path_is_seed_reproducible`; `deterministic_parallelism_jump_fou.rs`'s
  analogous `jump_fou_jump_component_still_diverges` is removed outright
  (its zero-intensity diffusion tests are unaffected and kept).
- `tests/sampler_v3_golden.rs` gains `golden_bates_streams`: the second
  golden pin covering a jump chain (after `golden_merton_streams`), the
  first covering a *multiplicative* jump term and a two-array `[s, v]`
  output. `lambda = 3.0` at `N = 8`, same reasoning as the `Merton` golden;
  `k = 0.0` isolates the jump term from the drift compensator so a same-file
  `lambda = 0` counterfactual (same `k`) diverging on the price path proves
  the pin is not diffusion-only.

### stochastic-rs-copulas: default the generator method

- `BivariateExt::generator` is no longer a required method. It now has a
  default body — `Err("{r#type():?} is not Archimedean — generator not
  defined")` — so the 7 non-Archimedean families (FGM, Gaussian,
  Hüsler-Reiss, Galambos, Plackett, Student-t, Marshall-Olkin) no longer
  hand-write an identical stub; each deleted its own copy and now falls
  through to the default. This is additive for external implementors (a
  required method became optional), but the exact `Err` message text
  changes for 5 of the 7 families whose old hand-written string didn't
  match their `CopulaType`'s `Debug` label verbatim: `"FGM …"` → `"Fgm
  …"`, `"Gaussian copula …"` → `"Gaussian …"`, `"Hüsler-Reiss …"` →
  `"HuslerReiss …"`, `"t-copula …"` → `"TCopula …"`, `"Marshall-Olkin …"`
  → `"MarshallOlkin …"` (Galambos and Plackett were already identical to
  their Debug label and are unchanged). Code matching on the old exact
  strings should match on the `"is not Archimedean — generator not
  defined"` suffix instead.

### stochastic-rs-copulas: hide internal bivariate trait methods

- `BivariateExt::{_compute_theta, check_marginal, partial_derivative_scalar}`
  are now `#[doc(hidden)]`, matching `sample_with_uniform` (hidden in an
  earlier wave). All three remain fully callable — `#[doc(hidden)]` only
  removes them from rendered docs, it is not an access modifier — this
  just signals they are internal plumbing riding the public trait vtable,
  not part of the supported contract. No code changes needed at any call
  site.
