# Migration Guide

Breaking changes are recorded here as they land, grouped by release. Entries
under `## Unreleased` describe changes on `main` that have not shipped yet.

## Unreleased

### stochastic-rs-copulas: `Frank` now accepts `θ = 0`, its own independence limit

`Frank::new`/`Frank::default` previously listed `0.0` in `invalid_thetas`,
so a fitted or explicitly-set `θ = 0` — Frank's own independence limit,
`C(u,v) = uv` — failed `check_fit`/`check_theta`, even though `pdf`,
`percent_point` and `partial_derivative` each already special-cased
`θ == 0.0` internally. `θ = 0` is now accepted, matching how
`Clayton`/`Gumbel` already treat their own independence limits (`θ = 0`
/ `θ = 1` respectively, both in-bounds and un-excluded).

`percent_point` and `partial_derivative`'s existing `θ == 0.0` branches
were also fixed to match (both returned the wrong operand), and `cdf`
gained the `θ == 0.0` branch it never had:

```rust
// Before: errors out entirely (θ = 0 rejected by check_theta).
let f = Frank::new(Some(0.0), None);
f.cdf(&x).unwrap_err(); // "Theta must be in the interval..."

// After: behaves as the independence copula, C(u,v) = uv.
let f = Frank::new(Some(0.0), None);
f.cdf(&x).unwrap();               // u * v (was: no branch existed -> NaN once unblocked)
f.percent_point(&y, &v).unwrap(); // y, the fresh uniform (was: v -> comonotonic, not independent)
f.partial_derivative(&x).unwrap(); // u (was: v)
```

If you were relying on `Frank` rejecting `θ = 0` to catch near-independent
data reaching this family, check your fitted `θ`/`τ` directly instead.

### stochastic-rs-copulas: `TCopula` now rejects `ρ = ±1`, matching `GaussianCopula`

`TCopula`'s `invalid_thetas` was empty, so `ρ = ±1` passed `check_theta`
and `pdf` (and, downstream, `log_pdf`) silently returned `NaN` — verified
directly, on- and off-diagonal, across `ν ∈ {1, 4, 30}` — while
`partial_derivative`/`percent_point` silently returned a finite-but-wrong
value instead of erroring. `ρ = ±1` is reachable from `fit()` alone, not
just a raw `set_theta`: `compute_theta`'s `sin(πτ/2).clamp(-1.0, 1.0)` —
identical to `GaussianCopula`'s own — lands exactly on `1.0` for
perfectly rank-correlated input data. `TCopula::default` now lists
`invalid_thetas: vec![-1.0, 1.0]`, exactly like `GaussianCopula`, so the
same inputs now fail `check_fit`/`check_theta` cleanly instead.

```rust
// Before: silently NaN.
let mut c = TCopula::with_nu(4.0);
c.set_theta(1.0);
c.pdf(&x).unwrap()[0]; // NaN

// After: a clean, catchable error.
let mut c = TCopula::with_nu(4.0);
c.set_theta(1.0);
c.pdf(&x).unwrap_err(); // "Theta must be in the interval [-1, 1] and not in [-1.0, 1.0]"
```

If you were relying on `TCopula` accepting `ρ = ±1`, clamp your input the
way this crate's own SABR calibrators already clamp correlation away from
`±1` (e.g. to `±0.99`) before calling `set_theta`/`fit`.

### stochastic-rs-quant: `hagan_implied_vol` validates its parameters instead of returning `0.0`

`hagan_implied_vol` previously returned `0.0` for `k <= 0`, `f <= 0` or
`alpha <= 0`, and `NaN` for `rho == 1`. Both were silent. A zero implied vol is
a plausible-looking number — `bs_price_fx` turns it into an intrinsic-value
price with no signal that the vol computation failed — so it now panics with a
message naming the parameter.

```rust
// Before: silently 0.0, and the caller could not tell.
let v = hagan_implied_vol(0.0, 100.0, 1.0, 0.2, 1.0, 0.5, -0.3);
assert_eq!(v, 0.0);

// After: panics with "strike k must be strictly positive (got 0)".
```

No in-tree caller is affected: both calibrators clamp `rho` to `±0.99`, and the
smile plot floors its strike grid at `1e-6`. If you call it directly with a
strike grid that can reach zero, floor it the same way.

The approximation itself is unchanged, and is now covered by a test comparing
against an independent 40-decimal-digit reimplementation of Hagan Eq. A.69a
rather than against values this crate produced itself.

### stochastic-rs-stochastic: a kernel-generic Volterra SDE engine

The Markov-lift machinery that previously lived inside `rough/` as a
rough-volatility internal is now a general stochastic Volterra equation engine
under `stochastic_rs_stochastic::volterra`. Nothing existing was removed beyond
the two `MarkovLift` fields documented separately above; this section is what
became newly available.

- **`VolterraKernel<T>`** — a trait with an exponential-sum contract
  (`nodes`, `weights`, `degree`, `evaluate`, `integral_from_zero`), implemented
  by `RlKernel` (Riemann–Liouville), `ExponentialKernel` (exact at one mode),
  `GammaKernel` (exponentially damped fractional) and `SumOfExponentials`
  (externally calibrated fits).
- **`VolterraSde`** — the general equation
  `X_t = X_0 + ∫ K(t-s) b(s,X_s) ds + ∫ K(t-s) σ(s,X_s) dW_s`, solved by the
  lift at `O(n N')`. Previously the crate could only produce the Gaussian case
  at `O(n^2)`.
- **`VolterraSquareRoot`** — the Volterra Heston variance leg, nonnegative by
  construction under full truncation.
- **`GaussianPolynomialVolatility`** — volatility as a polynomial of a Gaussian
  Volterra process, including the quintic parameterisation.
- **`fit_l1` / `l1_error`** — refit a kernel's weights to minimise the `L^1`
  error, which is what bounds the weak (pricing) error.
- **`reference_path`** — the direct `O(n^2)` convolution, kept permanently as
  the cross-implementation oracle the lift is tested against.

```rust
// Before: only the Gaussian convolution, O(n^2), one kernel family.
use stochastic_rs_stochastic::process::volterra::{Volterra, VolterraKernelSpec};
let gaussian = Volterra::<f64>::new(VolterraKernelSpec::FractionalBM { h: 0.3 }, 256, Some(1.0), Unseeded);

// After: a genuine SDE with state-dependent coefficients, O(n N').
use stochastic_rs_stochastic::volterra::{VolterraSde, ExponentialKernel};
fn drift(_t: f64, x: f64) -> f64 { 0.3 * (0.5 - x) }
fn diffusion(_t: f64, _x: f64) -> f64 { 0.2 }
let sde = VolterraSde::new(
  ExponentialKernel::new(0.7, 1.0),
  drift as fn(f64, f64) -> f64,
  diffusion as fn(f64, f64) -> f64,
  256, Some(0.1), Some(1.0), Unseeded,
);
```

The reproducibility guard grew from 124 to 127 types accordingly.

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
  doc comments and the regression tests in `with_setters_bates.rs`
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

### stochastic-rs-stochastic: `Cir2F`'s outer seed is now authoritative

- **`Cir2F` was the last instance of the pre-wave `Merton(cpoisson)` shape,
  found by this wave's own closing review, one module outside the `jump/`
  sweep that found the other five.** `Cir2F::new(x, y, phi, seed)` built its
  sampler as `self.x.sampler()`/`self.y.sampler()` and never read
  `self.seed` at all — the two pre-built `Cir` sub-processes' own seeds
  drove every sampled value, and the outer `seed` argument (and
  `c.seed.reseed(k)`) was dead. Measured before the fix:
  `Cir2F::new(Cir(seed=7), Cir(seed=8), phi, Deterministic::new(42))` and
  the identical construction with `Deterministic::new(999_999)` produced
  bit-identical output; keeping the outer seed at `42` but changing the
  sub-seeds to `12345`/`12346` changed the output.
- Fix: `new` now overwrites `x.seed`/`y.seed` with two independent children
  derived from the outer `seed` (`seed.derive()`, called twice in sequence
  — never `clone()`, which is a non-advancing snapshot and would leave both
  factors replaying the identical stream), discarding whatever seed the
  caller built the two `Cir`s with.
- **`new` keeps taking pre-built `Cir<T, S>` factors rather than absorbing
  their constructor parameters the way Task 1 absorbed `CompoundPoisson`'s.**
  Each factor's κ/θ/σ/`x0`/`use_sym` is independently meaningful and worth
  keeping addressable as a standalone `Cir` — unlike a jump driver, which is
  pure plumbing with no standalone identity worth preserving — and
  flattening both factors' seven non-seed fields into `Cir2F::new` would
  roughly double `Cir::new`'s own arity for no readability benefit. Only
  seeding is taken over; parameterization stays delegated.
- Before/after call site — behavior-breaking, not signature-breaking (the
  call still compiles unchanged; only which seed controls the output
  changes, so a `#[should_panic]`-style migration note does not apply —
  this is a silent behavior change any pinned-output caller must re-check):

  ```rust
  // Before — the *sub*-Cir seeds actually drove output; Cir2F::new's own
  // `seed` argument was never read.
  let x = Cir::new(1.0, 0.03, 0.1, 252, Some(0.03), Some(1.0), Some(false), Deterministic::new(7));
  let y = Cir::new(1.2, 0.02, 0.1, 252, Some(0.02), Some(1.0), Some(false), Deterministic::new(8));
  let r = Cir2F::new(x, y, phi as fn(f64) -> f64, Deterministic::new(42)); // 42 was never read

  // After — Cir2F::new's own `seed` is authoritative; the sub-Cirs' own
  // constructor seed value is discarded and replaced with an independent
  // derived child (any value works there — `Deterministic::new(0)` below
  // is a placeholder, not a driver of anything).
  let x = Cir::new(1.0, 0.03, 0.1, 252, Some(0.03), Some(1.0), Some(false), Deterministic::new(0));
  let y = Cir::new(1.2, 0.02, 0.1, 252, Some(0.02), Some(1.0), Some(false), Deterministic::new(0));
  let r = Cir2F::new(x, y, phi as fn(f64) -> f64, Deterministic::new(42)); // now drives both factors
  ```

- No golden test pinned `Cir2F`'s stream before this fix (confirmed by
  grepping `tests/sampler_v3_golden.rs` and every `stochastic-rs-stochastic`
  test file for `Cir2F`), so no re-pin was needed.
- Two new tests in `interest/cir_2f.rs`, each written to fail if the fix
  were reverted: `outer_seed_is_authoritative_over_sub_seeds` (two `Cir2F`s
  differing only in outer seed must differ; two differing only in the
  sub-`Cir`s' seeds must not) and `factors_are_independent_streams` (`x`
  and `y`, built with identical parameters, must sample different paths —
  proof the two derived children are genuinely independent streams, not one
  stream reused twice).
- `tests/reproducibility_all_processes/interest.rs`'s `cir_2f` guard case
  built both sub-`Cir`s from `s.clone()` — a non-advancing snapshot of the
  same outer seed passed to both, so the two factors consumed the identical
  Gaussian shock sequence: correlated noise, not the module doc's own
  independent `W^1`/`W^2`, i.e. a degenerate stand-in for a real two-factor
  CIR model. Changed to `s.derive()` per factor so the guarded instance
  matches the documented model. This does not change whether the guard's
  own assertions pass (fresh-vs-fresh bit-identity and thread-count
  independence are relative properties, unaffected either way, and no
  golden test pins this guard's specific output) — it only makes the
  guarded construction itself correct.

### stochastic-rs-stochastic: the reproducibility guard now asserts that a different seed changes the output

- **The guard's two assertions until this commit — same-seed bit-identity
  and thread-count invariance — cannot distinguish a correctly seeded type
  from one whose `seed` field never reaches its output.** Two fresh
  instances built from the *same* `Deterministic` seed agree bit-for-bit
  whether or not the constructor actually reads that seed, and thread-count
  invariance is a property of whatever stream *is* used, seeded or not. This
  is exactly why the `Cir2F` defect (previous entry) was found by this
  wave's own closing review and not by
  `tests/reproducibility_all_processes.rs`: reverting that fix and
  re-running `cargo test -p stochastic-rs-stochastic --test
  reproducibility_all_processes cir_2f` still reported `ok`. The "so a
  regression on any one type fails that test instead of waiting for another
  ad-hoc review to notice" claim two entries above was true of the
  coverage (all 124 types), not yet of this defect class.
- Fix: `tests/reproducibility_all_processes/common.rs`'s `check()` gained a
  third assertion, applied to all 124 types from this one shared function —
  a fresh instance built from a second fixed seed (`OTHER_SEED = 43`, beside
  the existing `SEED = 42`) must produce `.sample()` output different from
  `SEED`'s. The failure message names both candidate causes directly: a
  dead `seed` field, or a guard configuration where noise cannot reach the
  output.
- **`interest.rs`'s `cir_2f` guard case needed a second fix beyond the
  `s.clone()` → `s.derive()` change the previous entry made.** That earlier
  fix derived the two sub-`Cir`s' own seeds from the same `s` the guard
  varies between `check()`'s two construction calls — so even with
  `Cir2F::new` reverted to the pre-fix, seed-ignoring shape, changing `s`
  still changed the sub-`Cir`s' own seeds (derived in the guard's closure,
  before `Cir2F::new` ever runs) and therefore still changed the output,
  letting the new discrimination assertion pass by accident. Changed the
  two sub-`Cir`s to fixed, hardcoded seeds (`7`/`8`, matching
  `cir_2f.rs`'s own `outer_seed_is_authoritative_over_sub_seeds` unit test)
  so the only thing that can vary between the guard's two calls is what
  `Cir2F::new` itself derives from the outer seed — the thing under test.
  Verified by a destructive check: with this closure fix in place and
  `Cir2F::new` reverted to the buggy shape (the two `x.seed = seed.derive()`
  / `y.seed = seed.derive()` lines removed), `cir_2f` now fails with
  "sample() was bit-identical under seed 42 and seed 43 — this type's seed
  is not reaching its sampled output"; restored, 124/124 pass again, tree
  clean.
- **Two of the 124 types failed the new assertion on first run, both
  degenerate guard configurations, not type-level bugs:**
  - `Kimura` was guarded with `x0 = Some(0.0)`, exactly the Wright–Fisher
    diffusion's absorbing boundary: `sigma * sqrt(x0 * (1 - x0))` is `0`
    there, so the discretized path stays at `0` forever regardless of any
    Gaussian draw — correct model behavior (0 and 1 are absorbing for this
    SDE), not a sampler defect. Changed to `x0 = Some(0.5)`, an interior
    point where the diffusion term is live.
  - `TemperedStableSubordinator` was guarded with `c = 1.0, mu = 1.0`,
    giving a large-jump arrival rate of ~0.123/step and a minimum-jump
    acceptance probability of ~0.61 (`exp(-mu * epsilon)`) — low enough
    that a fully candidate-free 23-step path has ~5.9% probability, and
    seeds `42` and `43` both landed on one. Measured: the "failing" output
    was bit-identical to the pure deterministic small-jump-drift term `i *
    small_jump_drift`, with no jump contribution in either run. Changed to
    `c = 5.0, mu = 0.3` (arrival rate ~0.615/step, minimum-jump acceptance
    ~0.86), which drops the candidate-free-path probability to ~0.0001%.
- No type needed a principled exception — all 124 pass the new assertion
  with real, non-degenerate parameters. The "zero exceptions of any kind"
  claim two entries above still holds; its "a regression on any one type
  fails that test" clause is, from this commit, backed for this defect
  class too.
- Guard wall-clock: 13.8-14.0s, unchanged within noise from before this
  change — one extra `.sample()` call per type is negligible next to the
  existing four `sample_par` runs each type already made.

### stochastic-rs-stochastic: seeded Python `sample_par` no longer serializes into `m` sequential `sample()` calls

- **Behavior change for `m > 64`, same seed — Python-visible only.** Every
  `py_process_1d!`/`py_process_2x1d!`/`py_process_2d!`-generated class's
  `sample_par(m)` special-cased seeded instances: instead of calling
  `ProcessExt::sample_par`, it called `.sample()` `m` times in a loop on one
  instance (`(0..m).map(|_| inner.sample()).collect()`). That dates from a
  genuine race in the default `sample_par` on shared `Deterministic` state
  (`2026-05-08 fix: par seed`) which no longer exists: `sample_par`'s chunk
  count has been a pure function of `m` alone since, and every chunk's
  sampler is built sequentially, before any chunk reaches rayon (see this
  file's reproducibility entries above and the 124-type guard in
  `tests/reproducibility_all_processes.rs`). `sample_par(m)` now always
  calls `ProcessExt::sample_par` — the same call the unseeded path already
  used, and the same call `PyMerton`/`PyKou`/`PyLevyDiffusion::sample_par`
  (hand-written, `src/jump/{merton,kou,levy_diffusion}.rs`) already made
  unconditionally; those three were never part of this serialization and
  needed no change.
- **Only visible once `m` exceeds `MAX_CHUNKS` (64, `traits/process.rs`).**
  `chunk_count(m) = m.min(64)`, so at `m <= 64` every path already got its
  own freshly derived sampler under the *old* code too — the loop-of-
  `sample()` and the chunked path were already identical there, and stay
  identical now. The two diverge only once a chunk holds more than one path
  (`m > 64`): that chunk's one sampler is then asked for several draws in a
  row from its own continuing stream, not a freshly derived basis per draw.
  Measured directly (`PyGbm(0.05, 0.2, 8, x0=100.0, t=1.0, seed=42)`,
  comparing the old loop-of-`sample()` recipe against current
  `sample_par`): identical at `m` = 1, 8, 63, 64; at `m = 65`, row 0 still
  matches (both give `[100.0, 108.778910, 102.730042, 90.836069, 96.806925,
  95.123516, 105.653593, 112.420187]`) but row 1 onward differ (old:
  `[100.0, 102.988805, 113.671610, 120.576411, 110.666630, 119.211345,
  114.536010, 116.051663]`; new: `[100.0, 89.167261, 71.715847, 70.949297,
  72.410013, 69.849579, 73.360293, 68.515575]` — 64 of the 65 rows differ in
  total). The mechanism (`chunk_count`'s purity and the `MAX_CHUNKS`
  threshold) is shared by every `py_process_*!`-generated class, not just
  `PyGbm` — this is one representative measurement, not an isolated case.
- How to tell if you are affected: you called `sample_par(m)` on a
  `seed=`-constructed process with `m > 64`. Unseeded instances are
  unaffected (they always used `sample_par`); seeded instances with
  `m <= 64` are unaffected (identical either way, see above).
- How to reproduce the pre-fix sequence, if you depended on its exact
  values — it was always just `m` sequential `.sample()` calls on one
  instance, and still is:

  ```python
  # Before: sample_par(m) under a seed, for m > 64, drew this sequence.
  # Still available directly, unchanged:
  g = sr.PyGbm(0.05, 0.2, 8, x0=100.0, t=1.0, seed=42)
  legacy = np.stack([g.sample() for _ in range(m)])

  # After: sample_par(m) itself now draws through the same chunked/rayon
  # path as every unseeded call, and every other ProcessExt::sample_par
  # caller in the crate.
  new = sr.PyGbm(0.05, 0.2, 8, x0=100.0, t=1.0, seed=42).sample_par(m)
  ```
- **Why this is the right direction, not just a cleanup.** The new
  behavior is bit-identical across any rayon thread-pool size for a given
  seed and `m` — the same guarantee every other `ProcessExt::sample_par`
  caller in this crate already has (see "the process-level reproducibility
  exception list is now empty" above) — where the old serialized loop's
  only virtue was determinism within one arbitrary, undocumented
  convention that Python users had no way to discover. Verified across the
  PyO3 boundary specifically — not just in Rust — by
  `stochastic-rs-py/tests/test_sample_par_thread_count.py`: it runs the
  same seeded `sample_par` call in subprocesses under different
  `RAYON_NUM_THREADS` values and asserts the arrays are identical, at both
  `m = 64` (one path per chunk) and `m = 256` (several paths per chunk).
- Nothing pins the old numeric values. `stochastic-rs-py/tests/test_stochastic.py`'s
  `test_gbm_sample_par_determinism` compares two independent same-seed
  calls to each other (`np.allclose(a, b)`), not to a hardcoded array, and
  passes under either implementation (`m = 8`, below the 64 threshold
  above, is unaffected either way); no Python test file has a hardcoded
  `sample_par` array. `tests/sampler_v3_golden.rs`'s
  `golden_merton_streams`/`golden_bates_streams` pin `.sample()`, never
  `sample_par`, so they are untouched. No re-pin was needed anywhere.

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

### stochastic-rs-stochastic: `MarkovLift` is now a specialisation of the kernel-generic `VolterraLift`

- **`MarkovLift<T>`'s two public fields are gone.** `pub kernel: RlKernel<T>`
  and `pub dt: T` are replaced by a single private `inner:
  VolterraLift<T, RlKernel<T>>`. Nothing in this crate read either field
  (verified by a repo-wide grep before making the change), but they were
  public, crates.io-visible API. `MarkovLift::new`/`simulate`/
  `simulate_batch`/`simulate_batch_par` keep their exact signatures —
  `Fn(T) -> T` coefficients, same positional arguments — so every
  constructor call site (`RlFBm`, `RlBlackScholes`, `RlFOU`, `RlHeston`,
  all unmodified) needs no change. Only code that read `.kernel`/`.dt`
  directly on a constructed `MarkovLift` needs to hold onto its own
  `RlKernel`/`dt` values instead — which any such caller already has,
  since those are exactly the two arguments it passed to `MarkovLift::new`
  moments earlier:

  ```rust
  // Before
  let kernel = RlKernel::new(0.1, 23);
  let dt = 1.0 / 23.0;
  let markov = MarkovLift::new(kernel.clone(), dt);
  let hurst = markov.kernel.hurst; // read back off the stepper
  let step = markov.dt;

  // After
  let kernel = RlKernel::new(0.1, 23);
  let dt = 1.0 / 23.0;
  let markov = MarkovLift::new(kernel.clone(), dt);
  let hurst = kernel.hurst; // already owned by the caller
  let step = dt;
  ```

- **`RlFBm`/`RlBlackScholes`/`RlFOU`/`RlHeston` output shifts by up to
  ~1e-13 relative (measured max ≈ 512 ULP), because the arithmetic is now
  reassociated, not because a normalising factor moved.** `MarkovLift`
  used to compute its two boundary weights (the drift term's $\delta
  t^{H+1/2}/\Gamma(H+3/2)$ and the diffusion term's $\delta
  t^{H-1/2}/\Gamma(H+1/2)$) via a hand-written multiply-by-reciprocal with
  a cached $\Gamma(H+1/2)$, reusing the algebraic identity $\Gamma(H+3/2) =
  (H+\tfrac12)\Gamma(H+\tfrac12)$; it now calls
  [`VolterraKernel::integral_from_zero`]/[`VolterraKernel::evaluate`]
  through the generic [`VolterraLift`] stepper, which divide directly and
  evaluate $\Gamma(H+3/2)$ independently. The per-mode history-sum weights
  are similarly reassociated: normalised once per node at
  `RlKernel::new()` time and summed directly, instead of summed
  unnormalised and scaled once at the end. Both orderings were checked
  against a 60-decimal-digit (`mpmath`) reference for a spread of
  `(H, dt)` pairs and a representative history-sum snapshot: both sit
  within single-digit ULPs of the reference, on either side of it with no
  consistent direction — the signature of reassociation, not a defect in
  either path (a double-normalisation bug would be off by a whole factor
  of $\Gamma(H+\tfrac12)$, roughly 33% at $H=0.1$, not ~1e-13). Example,
  `RlFBm::new(0.1, 24, Some(1.0), None, Deterministic::new(42)).sample()[3]`:

  ```
  // Before: 0.029421993826289912  (0x3f9e20cc95117012)
  // After:  0.029421993826289898  (0x3f9e20cc9511700e)
  // relative difference: 4.7e-16
  ```

  Anyone who pinned exact `Rl*` output (bit-for-bit or beyond ~11 decimal
  digits) will see this shift once. It does not compound release over
  release — it is a one-time consequence of this refactor, not an ongoing
  source of drift. `tests/volterra_lift_reproducibility.rs` pins all four
  `Rl*` processes' `sample`/`sample_batch` output within a `1e-11` relative
  tolerance (justified there against the measured maximum) as the
  permanent guard against anything larger than this reassociation.

### stochastic-rs-stochastic: `process::volterra::VolterraKernel` renamed to `VolterraKernelSpec`; `Volterra::fbm` now solves `H < 1/2` via the Markov lift

- **The `VolterraKernel` enum is now `VolterraKernelSpec`.** The name was
  freed for [`crate::volterra::VolterraKernel`], the exponential-sum trait
  [`VolterraSde`](crate::volterra::sve::VolterraSde) is built on — the two
  are unrelated types (a closed enum of kernel *shapes* vs. an open trait
  of exponential-sum *fits*) that happened to share a name. Every variant
  (`FractionalBM`, `PowerLaw`, `Exponential`) and every field is unchanged;
  only the type name at the call site changes:

  ```rust
  // Before
  use stochastic_rs_stochastic::process::volterra::VolterraKernel;
  let v = Volterra::new(VolterraKernel::FractionalBM { h: 0.7 }, n, Some(1.0), seed);

  // After
  use stochastic_rs_stochastic::process::volterra::VolterraKernelSpec;
  let v = Volterra::new(VolterraKernelSpec::FractionalBM { h: 0.7 }, n, Some(1.0), seed);
  ```

- **`Volterra::fbm`/`Volterra::new` with `VolterraKernelSpec::FractionalBM {
  h }` for `h` in `(0, 1/2)` now delegates to
  [`VolterraSde`](crate::volterra::sve::VolterraSde) with an internally
  built [`RlKernel`](crate::rough::kernel::RlKernel), solved at $O(nN')$ by
  the Markov lift instead of the previous $O(n^2)$ direct convolution —
  sampled output for this range changes as a result (a real numeric shift
  from a different discretisation, not a bug: the lift integrates the
  kernel exactly over each sub-interval and draws its Brownian increments
  through a different generator than the old direct-convolution sampler
  did). This is the same class of change, for the same reason, as
  `MarkovLift`'s own generalisation above; nothing here pins exact
  pre-change `Volterra::fbm(h < 0.5, ...)` values, so no test needed
  re-pinning.

- **Every other `Volterra` kernel — `FractionalBM` with `h >= 1/2` (the
  124-type reproducibility guard's own case, `h = 0.7`), `PowerLaw` at any
  exponent, and `Exponential` at any rate — is bit-identical to before.**
  These now route through [`reference_path`](crate::volterra::reference::reference_path)
  internally (the same $O(n^2)$ convolution this type always ran, now
  shared with [`VolterraSde`]'s own cross-implementation oracle instead of
  duplicated), but draw exactly the same Gaussian increments in exactly the
  same order and combine them with exactly the same accumulation order, so
  output is unchanged — verified directly, not just by construction:
  `Deterministic::new(42)`-seeded `Volterra::new(kernel, 40, Some(1.0),
  seed).sample()` produces byte-for-byte identical `f64::to_bits()` output
  before and after this change, for `Exponential { beta: 1.3 }`, `PowerLaw
  { gamma: -0.2 }`, and `FractionalBM { h: 0.7 }` alike.

### stochastic-rs-distributions: `SimdNonCentralChiSquared::sample_ncp` now handles `0 < df < 1` correctly

`SimdNonCentralChiSquared::new`/`sample_ncp` previously treated any `df` in
`(0, 1)` the same as `df ≈ 1`: the Gaussian-shift decomposition it uses for
`df ≥ 1` doesn't exist below 1 (it needs a nonnegative central χ²_{df−1}
degrees of freedom), so the struct silently dropped that term and sampled
`(Z + sqrt(ncp))^2` regardless of how far below 1 `df` actually was. The
free function `non_central_chi_squared::sample` already had a correct
Poisson-mixture branch for this range (`Gamma(df/2 + J, 2)`, `J ~
Poisson(ncp/2)`); the struct now delegates to that same branch (reseeded
per draw from an internal fork cursor) instead of keeping a second, wrong
copy.

```rust
// Before: silently sampled as if df were 1 — wrong mean/variance for df != 1.
let s = SimdNonCentralChiSquared::<f64>::new(0.3, &Deterministic::new(1));
let x = s.sample_ncp(2.0); // mean came out ~= 1+ncp = 3.0, not df+ncp = 2.3

// After: matches the free function's Poisson-mixture branch.
let x = s.sample_ncp(2.0); // mean ~= df+ncp = 2.3, variance ~= 2*(df+2*ncp) = 8.6
```

No signature changed; this is a behavior-only fix, and no in-tree test
pinned the old (wrong) output. `df` here is `4*kappa*eta/zeta^2` in
`stochastic-rs-stochastic`'s `Svcgmy` Cir-exact variance step, which can
fall below 1 for sufficiently sub-Feller parameter combinations (already
accepted, not rejected, per the sub-Feller entry above) — so the bug was
reachable from real inputs, not only a theoretical corner.

### stochastic-rs-distributions: `SimdGeometric::new` now validates `p`

`SimdGeometric::new` accepted any `p`, silently sampling garbage for `p`
outside its own documented domain `(0, 1]` (contrast `SimdBinomial::new`,
which already asserted `p ∈ [0, 1]`). It now asserts `p ∈ (0, 1]`, matching
`SimdBinomial`'s wording.

```rust
// Before: silently accepted and sampled garbage.
let g = SimdGeometric::<u64>::new(0.0, &Unseeded);

// After: panics with "p must be in (0, 1]".
```

`p = 1.0` remains valid — it did before and still does; only genuinely
out-of-range `p` (`<= 0.0` or `> 1.0`) now panics. No in-tree caller passes
an out-of-range `p`.

### stochastic-rs-distributions: `SimdGeometric`'s sampler now matches its own `{1, 2, ...}` analytics

`SimdGeometric::fill_slice` sampled the `{0, 1, 2, ...}` "failures before
the first success" convention (`floor(ln(U) / ln(1-p))`, the textbook
inversion for that support), while `pdf`, `cdf`, `mean`, `median`, `mode`,
`skewness`, `kurtosis`, `characteristic_function`,
`moment_generating_function` and the module's own header all describe the
shifted `{1, 2, ...}` "trials up to and including the first success"
convention instead (`P(X=k) = (1-p)^{k-1} p`, `mean() = 1/p`) — the same
convention `scipy.stats.geom` uses (contrast `scipy.stats.nbinom`, which is
`{0, 1, ...}`). Sampler and analytics silently described two different
distributions: the empirical mean of sampled draws tracked `(1-p)/p`, not
`mean()`'s `1/p`, and every draw at `p = 1.0` came out `0`, never the `1`
the documented, degenerate always-succeeds-on-the-first-trial case
requires.

`fill_slice` now shifts its inversion by `+ 1`, landing on the same
`{1, 2, ...}` support the rest of the type already committed to — the same
shift `SimdBinomial`'s own internal geometric waiting-time loop already
applies to its inner gaps.

```rust
// Before: sampler and analytics disagreed.
let g = SimdGeometric::<u64>::new(1.0, &Deterministic::new(1));
let mut buf = [0u64; 1000];
g.fill_slice(&mut buf);
assert!(buf.iter().all(|&x| x == 0)); // every draw 0, but mean() said 1.0

// After: sampler matches mean()/variance()/pdf()/cdf().
g.fill_slice(&mut buf);
assert!(buf.iter().all(|&x| x == 1)); // every draw 1, matching mean() == 1.0
```

If you depended on the old `{0, 1, ...}` sampled values (e.g. treating a
draw as a zero-based failure count), subtract `1` from every value coming
out of `sample`/`sample_fast`/`fill_slice`/`sample_n`/`sample_matrix` to
recover them. `pdf`/`cdf`/`mean`/`variance`/`inv_cdf` and the rest of
`DistributionExt` are unchanged — they already used the `{1, 2, ...}`
convention the sampler now agrees with.

### stochastic-rs-copulas: `Frank::pdf`/`Frank::partial_derivative` now use the correct denominator

Both methods built their denominator as `g(u) + g(v) + g(1)` (a sum,
`g(z) = e^{-θz}-1`) for every `θ ≠ 0`, when the closed form (obtained by
differentiating this family's own `cdf`, and matching Nelsen (2006),
*An Introduction to Copulas*, 2nd ed., Example 4.23 / Table 4.1) needs
`g(1) + g(u)·g(v)`. Measured against a finite difference of `cdf`, `pdf`
was 83%-268% off across `θ ∈ {0.5, 2, 5, -3}` — every practical non-zero
θ. `sample`/`percent_point` root-find through `partial_derivative`, so
sampled output changes for every non-zero θ as well. `θ = 0`
(independence) is unaffected — a disjoint, separately special-cased path.

```rust
// Before: pdf silently wrong for every θ ≠ 0 (and sampling with it).
let f = Frank::new(Some(5.0), None);
f.pdf(&array![[0.4, 0.6]]).unwrap()[0]; // ~0.0042 (268x too small)

// After: matches a finite-difference probe of this family's own cdf.
let f = Frank::new(Some(5.0), None);
f.pdf(&array![[0.4, 0.6]]).unwrap()[0]; // ~1.136
```

If you fitted or sampled a `Frank` copula at any non-zero θ before this
fix, both the density and any samples drawn from it need to be
recomputed — there is no way to recover the intended values from the old
output.

### stochastic-rs-copulas: `Clayton::percent_point` now returns the correct value at its θ=0 independence limit

The `θ = 0` branch returned `V.clone()` (the conditioning value), making
every sampled pair exactly comonotonic (`U = V`, Kendall's `τ ≈ 1`)
instead of independent — the opposite of what `θ = 0` (Clayton's own
independence limit, reachable from `.fit()` on near-independent data) is
supposed to produce. It now returns `y.clone()` (the fresh uniform
draw), matching `Frank::percent_point`'s own `θ = 0` fix above and the
general relation `∂_v C(u,v) = u` at independence (`C(u,v) = uv`).

```rust
// Before: comonotonic samples (τ ≈ 1) at Clayton's own independence limit.
let mut c = Clayton::new();
c.set_theta(0.0);
c.percent_point(&y, &v).unwrap(); // == v

// After: independent samples (τ ≈ 0), as θ = 0 is supposed to mean.
let mut c = Clayton::new();
c.set_theta(0.0);
c.percent_point(&y, &v).unwrap(); // == y
```

`Clayton::pdf`/`cdf`/`partial_derivative` do not special-case `θ = 0` at
all and are unchanged by this fix — see `bivariate.rs`'s module doc for
that separate, still-open gap.

### stochastic-rs-copulas: `Gumbel::partial_derivative` now returns the correct value at its θ=1 independence limit

The `θ = 1` branch returned `V.to_owned()` where independence
(`C(u,v) = uv`) requires `∂_v C(u,v) = u`. It now returns `U.to_owned()`.
`Gumbel::percent_point` already had its own correct `θ = 1` branch
(returning the fresh uniform directly, bypassing `partial_derivative`
entirely), so `Gumbel::sample` was never affected by this bug — only a
direct `partial_derivative` call at `θ = 1` (e.g. computing a conditional
CDF / h-function value) was wrong.

```rust
// Before: wrong conditional CDF value at Gumbel's own independence limit.
let g = Gumbel::new(Some(1.0), None);
g.partial_derivative(&array![[0.3, 0.6]]).unwrap()[0]; // 0.6 (== v)

// After: matches C(u,v) = uv's own ∂_v C = u.
let g = Gumbel::new(Some(1.0), None);
g.partial_derivative(&array![[0.3, 0.6]]).unwrap()[0]; // 0.3 (== u)
```

### stochastic-rs-copulas: `Amh::partial_derivative` now differentiates the correct argument

It computed $\partial_u C(u,v)$ — a correct derivative, but of the wrong
argument. Every other family in this crate computes $\partial_v C(u,v)$
(the derivative w.r.t. the *second* argument, at fixed conditioning
value), which is what `BivariateExt`'s finite-difference default and
`percent_point_numerical`'s root-finder both assume. Because `Amh` does
not override `percent_point`, it fell through to that shared root-finder,
which inverted the wrong function — so `Amh::percent_point`/`Amh::sample`
silently solved the wrong equation for `u` given `v`, and **previously
sampled `Amh` data came from the transposed copula**, not the one
requested.

This was not a harmless relabeling: at a representative `θ=0.6`, the
wrong partial's achievable range (for a fixed conditioning `v`) does not
cover most sampled quantiles, so `percent_point_numerical`'s Brent search
routinely failed to bracket a root and its `unwrap_or(f64::EPSILON)`
fallback clamped roughly three quarters of `U` draws to ~0. Old `Amh`
samples are not simply column-swapped; they are not distributed as any
`Amh` copula and must be redrawn.

```rust
// Before: wrong conditional CDF value (∂_u C, not ∂_v C).
let mut c = Amh::new();
c.set_theta(0.6);
c.partial_derivative(&array![[0.3, 0.6]]).unwrap()[0]; // 0.6587463017751479

// After: matches a finite-difference probe of this family's own cdf.
let mut c = Amh::new();
c.set_theta(0.6);
c.partial_derivative(&array![[0.3, 0.6]]).unwrap()[0]; // 0.25136372041420124
```

If you fitted or sampled an `Amh` copula before this fix, re-fit and
re-sample — there is no way to recover the intended values from the old
output.

### stochastic-rs-copulas: `Clayton::pdf`/`cdf`/`partial_derivative` now special-case `θ=0`

None of the three had a `θ = 0` branch, unlike `percent_point` above. Naive
substitution at exactly `θ = 0` hits a removable singularity that does not
resolve to the independence copula: `cdf(u,v)` evaluated to the constant
`1.0` for every `u,v > 0` (not `uv`); `pdf(u,v)` evaluated to `(uv)^{-1}`
(not `1.0`); `partial_derivative(u,v)` evaluated to `v^{-1}` (not `u`).
`Clayton::sample`, which routes through the already-fixed `percent_point`,
was unaffected — only direct `pdf`/`cdf`/`partial_derivative` calls at
exactly `θ = 0` were wrong.

```rust
// Before: silently wrong at Clayton's own independence limit.
let c = Clayton { theta: Some(0.0), ..Clayton::default() };
c.cdf(&array![[0.3, 0.7]]).unwrap()[0];               // 1.0 (should be 0.21)
c.pdf(&array![[0.3, 0.7]]).unwrap()[0];                // 4.761904761904762 (should be 1.0)
c.partial_derivative(&array![[0.3, 0.7]]).unwrap()[0]; // 1.4285714285714286 (should be 0.3)

// After: matches the independence copula C(u,v) = uv.
let c = Clayton { theta: Some(0.0), ..Clayton::default() };
c.cdf(&array![[0.3, 0.7]]).unwrap()[0];               // 0.21
c.pdf(&array![[0.3, 0.7]]).unwrap()[0];                // 1.0
c.partial_derivative(&array![[0.3, 0.7]]).unwrap()[0]; // 0.3
```

Gumbel and Amh were checked for the same gap at their own independence
limits (`θ = 1`, `θ = 0` respectively). Gumbel's `pdf`/`cdf` already
special-cased `θ = 1` (only `partial_derivative` needed a fix, already
recorded in this file's own `Gumbel::partial_derivative` entry above).
`Amh` needs no such branch at all: its denominator `1-θ(1-u)(1-v)` never
vanishes at `θ = 0`, so `pdf`/`cdf` were already correct there — see the
`Amh::partial_derivative` entry above for `Amh`'s actual (unrelated) bug.
