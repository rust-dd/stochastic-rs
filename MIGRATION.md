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
  so every worker draws its own fresh basis value (not one basis shared
  across the call and fanned out by index), combined with its
  `stream_idx` via `splitmix64(basis ^ stream_idx)`. Two
  identically-`Deterministic`-seeded samplers now produce bit-identical
  `sample_matrix` output call-for-call (first call matches first, second
  matches second, ...) **for a fixed thread-pool size** — the worker
  count is `min(rayon::current_num_threads(), size-derived cap)`, so the
  same two samplers compared under a different
  `rayon::current_num_threads()` are not guaranteed to agree, since that
  changes how many times `fork` is called; repeated calls on the *same*
  sampler never replay, for `Deterministic`- and `Unseeded`-constructed
  samplers alike; and a serial call (below the parallel threshold) never
  touches the fork basis, so interleaving serial and parallel calls stays
  deterministic across two identically-seeded samplers. No API signature
  changed; this is a behavior fix.
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
  names). `MultifactorSabr` has no Python binding.

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
