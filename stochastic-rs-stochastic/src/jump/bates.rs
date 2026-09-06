//! # Bates
//!
//! $$
//! \begin{aligned}dS_t&=(r-r_f-\lambda k)S_tdt+\sqrt{v_t}S_t dW_t^S+(Y-1)S_{t^-}dN_t\\dv_t&=\kappa(\theta-v_t)dt+\sigma\sqrt{v_t}dW_t^v\end{aligned}
//! $$
//!
//! Reference: Bates D. S. (1996) — *Jumps and Stochastic Volatility:
//! Exchange Rate Processes Implicit in Deutsche Mark Options*, Review of
//! Financial Studies 9(1), 69–107, DOI: 10.1093/rfs/9.1.69.
//!

use std::any::Any;

use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::process::cpoisson::CompoundPoisson;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[inline]
fn validate_drift_args<T: FloatExt>(
  mu: Option<T>,
  b: Option<T>,
  r: Option<T>,
  r_f: Option<T>,
  type_name: &'static str,
) {
  let has_r_pair = r.is_some() && r_f.is_some();
  if !(has_r_pair || b.is_some() || mu.is_some()) {
    panic!("{type_name}: one of (r and r_f), b, or mu must be provided");
  }
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Bates1996::new(..).with_lambda(0.8).with_rho(-0.4)`.
pub struct Bates1996<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Direct drift rate μ of the asset — one of three mutually-exclusive
  /// drift specifications (`mu` xor `b` xor the `(r, r_f)` pair); exactly
  /// one must be `Some`.
  pub mu: Option<T>,
  /// Cost-of-carry rate b, an alternative drift specification to `mu`.
  pub b: Option<T>,
  /// Domestic risk-free rate; paired with `r_f` as a third drift
  /// specification via `r - r_f`.
  pub r: Option<T>,
  /// Foreign risk-free rate / dividend yield, paired with `r`.
  pub r_f: Option<T>,
  /// Jump (Poisson) intensity λ — arrival rate of the log-price jumps.
  /// Single source of truth: `sampler()` reads this field directly (not
  /// `cpoisson.poisson.lambda`) for both the jump-arrival rate and the
  /// drift's `-lambda*k` compensator term. Every setter that can change it
  /// (`with_lambda`, `with_cpoisson`) keeps `cpoisson.poisson.lambda`
  /// synced to match — see those methods' docs and
  /// `resync_cpoisson_poisson`.
  pub lambda: T,
  /// Jump-size compensator κ (E[Y−1]-like term, matching the module
  /// header's own λκ_J), subtracted from the drift scaled by `lambda`.
  /// Unrelated to mean-reversion speed despite the letter k.
  pub k: T,
  /// Variance-drift intercept (κθ combined) in the reparametrized variance
  /// recursion `dv = (alpha − beta·v)dt + ...`, equivalent to `κ(θ−v)`
  /// with `alpha = κθ`.
  pub alpha: T,
  /// Variance-drift slope (mean-reversion speed κ) in the same
  /// reparametrized recursion.
  pub beta: T,
  /// Vol-of-vol σ scaling the variance factor's own diffusion.
  pub sigma: T,
  /// Instantaneous correlation ρ between the asset's and variance's
  /// driving Brownian motions.
  pub rho: T,
  /// Number of points sampled along the Bates path.
  pub n: usize,
  /// Initial asset price S₀.
  pub s0: Option<T>,
  /// Initial variance level v₀.
  pub v0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Reflect (true) instead of floor-at-zero (false/None) negative
  /// variance proposals.
  pub use_sym: Option<bool>,
  /// Correlated-Gaussian generator driving the price/variance diffusion.
  /// Constructed once (and rebuilt by `with_rho`/`with_steps`/`with_horizon`)
  /// with a `Cgns<T>` (`S = Unseeded`) that itself is never consulted — the
  /// sampler drives it via `cgns.sample_impl(&self.seed)` instead, so this
  /// field's own dead `Unseeded` is irrelevant to reproducibility. Private,
  /// so this indirection is an implementation detail.
  cgns: Cgns<T>,
  /// Compound-Poisson jump driver added to the asset's log-return.
  /// Fully seed-reproducible: [`new`](Self::new) builds it internally from
  /// `seed` (`seed.clone().derive()` — a hash-mixed child, decorrelated
  /// from but a deterministic function of the same `seed` the diffusion
  /// component (`cgns`) consults directly), and `sampler()` derives a
  /// fresh, chunk-local basis off `self.cpoisson.seed` for every chunk,
  /// mirroring the diffusion component's own per-chunk `self.seed`-derived
  /// basis.
  ///
  /// `sampler()` reads only `cpoisson.distribution` (the jump-size law)
  /// and `self.lambda` — **not** `cpoisson.poisson.lambda` — from this
  /// field on the sampling path; `cpoisson.poisson.{n,t_max,seed}` are
  /// inert there (`grid_relative_increments` never consults them). That
  /// inertness is scoped to *this type's own* sampling, though: `cpoisson`
  /// is a `CompoundPoisson` in its own right, and calling `.sample()` on it
  /// directly (bypassing `Bates1996` entirely) drives it through
  /// `Poisson::sample_impl`, which *does* branch on `.n`/`.t_max` (fixed
  /// count vs. horizon mode) and *does* consult `.seed` — genuinely live
  /// there. Left `pub` for both reasons: a caller can inspect or directly
  /// `.sample()` the embedded compound-Poisson process as its own
  /// standalone `ProcessExt`, and can replace it wholesale via
  /// [`with_cpoisson`](Self::with_cpoisson) (which keeps `self.lambda` in
  /// sync with the replacement — see that method's doc) or direct field
  /// assignment (which does not; assign through `with_cpoisson` unless you
  /// separately update `self.lambda` to match).
  pub cpoisson: CompoundPoisson<T, D, S>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`). Consulted
  /// directly by the diffusion component (via `cgns.sample_impl`);
  /// `cpoisson`'s own seed (set at construction from this same value — see
  /// `cpoisson`'s doc above) drives the jump component.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T, D, S: SeedExt> Bates1996<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Builds the compound-Poisson jump driver internally from `jump_dist`
  /// and `lambda`, seeded from `seed` (see `cpoisson`'s field doc) — the
  /// caller supplies the jump-size distribution and intensity directly
  /// instead of pre-building a `Poisson`/`CompoundPoisson` pair and
  /// threading a third, independent seed through it by hand.
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    lambda: T,
    k: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    jump_dist: D,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }
    validate_drift_args(mu, b, r, r_f, "Bates1996");

    let cpoisson = CompoundPoisson::new(
      jump_dist,
      Poisson::new(lambda, Some(n), t, Unseeded),
      seed.clone().derive(),
    );

    Self {
      backend: Cpu,
      mu,
      b,
      r,
      r_f,
      lambda,
      k,
      alpha,
      beta,
      sigma,
      rho,
      n,
      s0,
      v0,
      t,
      use_sym,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      cpoisson,
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> Bates1996<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Replace `mu`; re-validates that a drift specification still exists.
  pub fn with_mu(mut self, mu: Option<T>) -> Self {
    self.mu = mu;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `b`; re-validates that a drift specification still exists.
  pub fn with_b(mut self, b: Option<T>) -> Self {
    self.b = b;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `r`; re-validates that a drift specification still exists.
  pub fn with_r(mut self, r: Option<T>) -> Self {
    self.r = r;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `r_f`; re-validates that a drift specification still exists.
  pub fn with_r_f(mut self, r_f: Option<T>) -> Self {
    self.r_f = r_f;
    validate_drift_args(self.mu, self.b, self.r, self.r_f, "Bates1996");
    self
  }

  /// Replace `lambda`, all else unchanged. `sampler()` reads `self.lambda`
  /// directly for both the jump-arrival intensity and the drift's
  /// `-lambda*k` compensator term (see `cpoisson`'s field doc), so this
  /// alone already changes both; it also re-syncs the otherwise-cosmetic
  /// mirror `cpoisson.poisson.lambda` (see `resync_cpoisson_poisson`) so a
  /// caller inspecting it does not see a stale value.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    self.lambda = lambda;
    self.resync_cpoisson_poisson();
    self
  }

  /// Replace `k`, all else unchanged.
  pub fn with_k(mut self, k: T) -> Self {
    self.k = k;
    self
  }

  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    self.alpha = alpha;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    self.beta = beta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    if let Some(v) = v0 {
      assert!(v >= T::zero(), "v0 must be non-negative");
    }
    self.v0 = v0;
    self
  }

  /// Replace `use_sym`, all else unchanged.
  pub fn with_use_sym(mut self, use_sym: Option<bool>) -> Self {
    self.use_sym = use_sym;
    self
  }

  /// Replace the compound-Poisson jump driver wholesale, adopting its
  /// intensity as the new `self.lambda` — `sampler()` reads `self.lambda`,
  /// not `cpoisson.poisson.lambda`, for the jump-arrival rate and the
  /// drift's `-lambda*k` compensator term (see `cpoisson`'s field doc), so
  /// without this adoption the incoming driver's own intensity would be
  /// silently ignored and the *old* `self.lambda` would keep driving both
  /// while only the distribution changed. `cpoisson.poisson.{n,t_max}` are
  /// left exactly as the caller supplied them (not normalized to
  /// `self.{n,t}`) since, unlike `lambda`, they carry no live weight on
  /// this type's sampling path either way.
  pub fn with_cpoisson(mut self, cpoisson: CompoundPoisson<T, D, S>) -> Self {
    self.lambda = cpoisson.poisson.lambda;
    self.cpoisson = cpoisson;
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the cached
  /// correlated-Gaussian generator, whose length and step size derive
  /// from `n`, and re-syncs `cpoisson.poisson.n` (see
  /// `resync_cpoisson_poisson`) — dead on this type's own sampling path,
  /// but kept from silently going stale for a caller inspecting `cpoisson`
  /// directly.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self.resync_cpoisson_poisson();
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`,
  /// and re-syncs `cpoisson.poisson.t_max` (see `resync_cpoisson_poisson`)
  /// — dead on this type's own sampling path, but kept from silently going
  /// stale for a caller inspecting `cpoisson` directly.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self.resync_cpoisson_poisson();
    self
  }

  /// Replace the seed strategy's value, all else unchanged — including
  /// re-deriving `cpoisson`'s own seed from the new value exactly as
  /// [`new`](Self::new) does (`cpoisson`'s distribution and lambda are
  /// untouched), so the result matches a fresh construction with this seed
  /// rather than leaving the jump component keyed to the old one.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.cpoisson.seed = seed.clone().derive();
    self.seed = seed;
    self
  }

  /// Rebuilds `cpoisson.poisson` from `self.{lambda, n, t}` so a caller
  /// reading `cpoisson.poisson` directly never sees it disagree with the
  /// outer struct's own record of the same three values — most
  /// load-bearing for `lambda`, which `sampler()` actually reads off
  /// `self` (not off this mirror) for the jump-arrival rate and the
  /// drift's `-lambda*k` compensator term, but applied uniformly to
  /// `n`/`t_max` too even though those two are inert on the sampling path
  /// either way (see `cpoisson`'s field doc). Called from every setter
  /// that changes `lambda`, `n`, or `t`.
  fn resync_cpoisson_poisson(&mut self) {
    self.cpoisson.poisson = Poisson::new(self.lambda, Some(self.n), self.t, Unseeded);
  }
}

impl<T, D, S: SeedExt, B> Bates1996<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  #[inline]
  fn effective_drift(&self) -> T {
    match (self.r, self.r_f, self.b, self.mu) {
      (Some(r), Some(r_f), _, _) => r - r_f,
      (_, _, Some(b), _) => b,
      (_, _, _, Some(mu)) => mu,
      _ => unreachable!("validate_drift_args ensures at least one of (r+r_f), b, mu is set"),
    }
  }
}

impl<T, D, S: SeedExt, B> Bates1996<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.cpoisson.distribution)
      .and_then(crate::euler::JumpSizes::product)
  }

  /// Whether a device can run this process: it has no jumps, or its sizes
  /// follow a law the kernels draw. Anything else samples on the host.
  fn device_ready(&self) -> bool {
    self.lambda <= T::zero() || self.device_jump_sizes().is_some()
  }
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Bates1996<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The spot's drift arrives compensated by `lambda k`, and the variance
  /// keeps whichever treatment of the boundary the process was built with.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let drift_c = self.effective_drift() - self.lambda * self.k;
    if self.use_sym.unwrap_or(false) {
      crate::euler::EulerSpec::Bates1996Reflected {
        drift_c,
        alpha: self.alpha,
        beta: self.beta,
        sigma: self.sigma,
        rho: self.rho,
      }
    } else {
      crate::euler::EulerSpec::Bates1996 {
        drift_c,
        alpha: self.alpha,
        beta: self.beta,
        sigma: self.sigma,
        rho: self.rho,
      }
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.s0.unwrap_or(T::zero()),
      self.v0.unwrap_or(T::zero()).max(T::zero()),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  /// The jump intensity per unit time; `None` when the process carries none,
  /// which is what makes a zero-intensity build skip the draw entirely.
  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// The size law as the kernels draw it. A law they do not carry never
  /// reaches a launch — [`ProcessExt::sample`] keeps such a process on the
  /// host — so asking for one here is a caller bypassing that guard.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    if self.lambda <= T::zero() {
      return None;
    }
    Some(self.device_jump_sizes().expect(
      "Bates1996: the jump-size distribution is not a law the Euler engine compounds multiplicatively on a device; \
       sample through `ProcessExt`, which keeps it on the host",
    ))
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T, D, S: SeedExt] Bates1996<T, D, S> { mu, b, r, r_f, lambda, k, alpha, beta, sigma, rho, n, s0, v0, t, use_sym, cgns, cpoisson, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bates1996<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BatesSampler<'s, T, D, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` and `self.cpoisson.seed`, independently,
  /// into the returned sampler — the same shape `Cgns`'s own `sampler()`
  /// uses — so the correlated-Gaussian diffusion driver (`cgns`, otherwise
  /// permanently `Unseeded`; see the type doc) is driven via
  /// `sample_impl(&self.seed)` instead of a bare `.sample()` that only ever
  /// reads `cgns`'s own dead `Unseeded` field, and the jump driver is
  /// likewise driven from an owned, chunk-local basis rather than a
  /// borrowed `&self.cpoisson` shared across chunks (which would let
  /// concurrent chunks race on the same shared atomic during the parallel
  /// region — see `ProcessExt`'s trait-level reproducibility requirement).
  /// Adjacent chunks land on hash-scrambled, mutually independent bases for
  /// the same reason every other `derive()`-based sampler in this crate
  /// does.
  fn sampler(&self) -> BatesSampler<'_, T, D, S> {
    BatesSampler {
      n: self.n,
      s0: self.s0.unwrap_or(T::zero()),
      v0: self.v0.unwrap_or(T::zero()).max(T::zero()),
      lambda: self.lambda,
      k: self.k,
      alpha: self.alpha,
      beta: self.beta,
      sigma: self.sigma,
      drift: self.effective_drift(),
      use_sym: self.use_sym.unwrap_or(false),
      dt: self.cgns.dt(),
      cgns: self.cgns,
      jump_distribution: &self.cpoisson.distribution,
      jump_seed: self.cpoisson.seed.derive(),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the jump-size law is one the device
  /// kernels draw; any other law keeps the process on the host, chunked
  /// exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.device_ready() {
      self.backend.system_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.system_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    if self.device_ready() {
      self.backend.system_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Reusable [`Bates1996`] sampling state: owns the correlated-Gaussian
/// generator and an owned, already-derived seed to drive it, borrows the
/// jump-size distribution, and owns a separate, already-derived seed to
/// drive the jump arrivals — mirroring
/// [`MertonSampler`](crate::jump::merton::MertonSampler)'s shape — so a
/// Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
pub struct BatesSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  s0: T,
  v0: T,
  lambda: T,
  k: T,
  alpha: T,
  beta: T,
  sigma: T,
  drift: T,
  use_sym: bool,
  dt: T,
  cgns: Cgns<T>,
  jump_distribution: &'a D,
  jump_seed: S,
  seed: S,
}

impl<T, D, S: SeedExt> BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);
    let jump_increments = crate::process::cpoisson::grid_relative_increments(
      self.jump_distribution,
      self.lambda,
      &self.jump_seed,
      self.n,
      dt,
    );

    s[0] = self.s0;
    v[0] = self.v0;

    for i in 1..self.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = s[i - 1]
        + (self.drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v_prev.sqrt() * cgn1[i - 1]
        + s[i - 1] * jump_increments[i];

      let dv = (self.alpha - self.beta * v_prev) * dt + self.sigma * v_prev.sqrt() * cgn2[i - 1];

      v[i] = match self.use_sym {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(T::zero()),
      }
    }
  }
}

impl<T, D, S: SeedExt> PathSampler<T> for BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Bates output must be contiguous"),
      v.as_slice_mut().expect("Bates output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [s, v]
  }
}

// Both submodules split out to keep this file under the project's 600-line
// cap (this type now carries a full set of `with_*` builder setters on top
// of the model itself, plus the Python bindings). Same pattern as
// `volatility/bates_svj.rs` uses for its own test split.
#[cfg(test)]
#[path = "bates_tests.rs"]
mod tests;

#[cfg(feature = "python")]
#[path = "bates_python.rs"]
mod python;
#[cfg(feature = "python")]
pub use python::PyBates;
