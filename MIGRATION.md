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
- Guarantee, complete: for every process in the crate **except**
  `Bates1996` and `RoughHeston`, a `Deterministic` seed and the same `m`
  now produce bit-identical `sample_par`/`sample_map` output on any
  machine and under any rayon thread-pool size — the same guarantee
  `stochastic-rs-distributions`'s `DistributionSampler::sample_matrix` fix
  elsewhere in this file provides for its own `(m, n)` pair. `Unseeded`
  processes still draw fresh randomness on every call, exactly as before.
  This supersedes the still-accurate-but-incomplete guarantee in the
  entry above, which predates this fix for the lazy class.
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
- **Scope correction — the exception list above was incomplete.**
  `Bates1996` and `RoughHeston` are correctly identified as not
  `self.seed`-reproducible at all, but they are not the only two: `JumpFou`
  has the identical property — its `Fgn<T, Unseeded, B>` diffusion field and
  its `CompoundPoisson<T, D>` jump field (default `S = Unseeded`) are both
  hard-wired away from `self.seed`, so `self.seed` is a dead field there and
  two identically-`Deterministic`-seeded `JumpFou`s produce different
  output, confirmed empirically. `JumpFOUCustom` and `Merton` are a
  narrower, pre-existing case, but not the *same* narrower case: only half
  of each type's randomness is reproducible, and it is not the same half.
  `Merton` hard-wires its `CompoundPoisson<T, D>` jump field to `Unseeded`
  while its diffusion component correctly consults `self.seed` — noted
  already in `tests/sampler_v3_golden.rs`'s header. `JumpFOUCustom` has no
  `CompoundPoisson` field at all — it is the other way around: its jump
  timing/size draws (`rng: self.seed.rng()`, built in `sampler()`) are the
  reproducible half, and its own diffusion driver (`fgn: Fgn<T, Unseeded,
  B>`) is hard-wired away from `self.seed` instead; this was not previously
  documented anywhere. Neither is changed here: `Merton` would need its
  `CompoundPoisson<T, D>` field to become `CompoundPoisson<T, D, S>`, a
  breaking API change to a `pub` field, out of scope for a reproducibility
  bugfix; `JumpFOUCustom` has no such field to widen in the first place —
  its private `fgn` could be threaded more cheaply in isolation, but that is
  a different, narrower change than the one `Merton` would need. Left as
  documented exceptions instead; see the doc comments on each type and
  `ProcessExt`'s trait-level reproducibility section.
- Guarantee, corrected: for every process in the crate **except**
  `Bates1996`, `RoughHeston`, and `JumpFou` (no randomness reachable from
  `self.seed` at all), **except the jump component of** `Merton` (diffusion
  is reproducible, jump arrivals/sizes are not), and **except the diffusion
  component of** `JumpFOUCustom` (jump arrivals/sizes are reproducible,
  diffusion is not), a `Deterministic` seed and the same `m` now produce
  bit-identical `sample_par`/`sample_map` output on any machine, under any
  rayon thread-pool size, and — new in this entry — regardless of how many
  paths land in the same chunk (`m` need not be `<= MAX_CHUNKS` for chunks
  to stay mutually independent). `Fgn`'s and `Fbm`'s own `sample_par`
  overrides remain a separate, still-open exception from the entry above
  (not superseded by this one, since neither entry touches their batched
  backends): each of the `m` items in their `Cpu`/`Accelerate` batches
  derives its seed from the same shared atomic *inside* the parallel region
  rather than before it, so both types are seed-reproducible per call but
  not thread-count independent under `sample_par` specifically (`sample_map`
  on both types goes through this entry's ordinary `chunked_samplers`
  mechanism and is unaffected). This supersedes the "guarantee, complete"
  claim in the entry above, which was accurate about thread-count
  independence but silent on cross-chunk correlation and incomplete about
  the exception list.
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
