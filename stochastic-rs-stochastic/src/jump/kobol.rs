//! The general tempered stable (KoBoL) family; the side weights `p`, `q` need not sum to one.
//! [`Cgmy`](crate::jump::cgmy::Cgmy) is the case `D = C`, `p = q = 1`, `λ₊ = M`, `λ₋ = G`.
//!
//! $$
//! \nu(dx)=D\left(p\,e^{-\lambda_+ x}x^{-1-\alpha}\mathbf 1_{x>0}
//! \;+\;q\,e^{-\lambda_-|x|}|x|^{-1-\alpha}\mathbf 1_{x<0}\right)dx,\quad
//! D>0,\ p,q>0,\ \lambda_\pm>0,\ \alpha\in(0,2)
//! $$
//!
//! Reference: Koponen (1995), Phys. Rev. E 52(1), 1197–1199, DOI 10.1103/PhysRevE.52.1197
//! Reference: Boyarchenko & Levendorskiĭ (2000), IJTAF 3(3), 549–552, DOI 10.1142/S0219024900000541
//! Reference: Rosiński (2007), Stoch. Proc. Appl. 117(6), 677–707, DOI 10.1016/j.spa.2006.10.003
//! Reference: Asmussen & Rosiński (2001), J. Appl. Prob. 38(2), 482–493, DOI 10.1239/jap/996986757

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::jump::series_remainder::SeriesRemainder;
use crate::params::Horizon;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Kobol<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Overall scale D > 0
  pub d: T,
  /// Positive-side weight p > 0
  pub p: T,
  /// Negative-side weight q > 0
  pub q: T,
  /// Positive tempering λ_+ > 0
  pub lambda_plus: T,
  /// Negative tempering λ_- > 0
  pub lambda_minus: T,
  /// Activity α in (0,2)
  pub alpha: T,
  /// Number of time steps
  pub n: usize,
  /// Series terms; the terms past them enter as a Brownian motion of their drift and variance.
  pub j: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Kobol<T, S> {
  /// Panics unless every parameter is finite, `d`, `p`, `q`, `lambda_plus`, `lambda_minus` are
  /// positive, `0 < alpha < 2`, `alpha != 1` when `p != q`, `n, j >= 2` and a given `t` is positive.
  pub fn new(
    d: T,
    p: T,
    q: T,
    lambda_plus: T,
    lambda_minus: T,
    alpha: T,
    n: usize,
    j: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      d.is_finite(),
      "d must satisfy `d.is_finite()`, got d = {d:?}"
    );
    assert!(
      p.is_finite(),
      "p must satisfy `p.is_finite()`, got p = {p:?}"
    );
    assert!(
      q.is_finite(),
      "q must satisfy `q.is_finite()`, got q = {q:?}"
    );
    assert!(
      lambda_plus.is_finite(),
      "lambda_plus must satisfy `lambda_plus.is_finite()`, got lambda_plus = {lambda_plus:?}"
    );
    assert!(
      lambda_minus.is_finite(),
      "lambda_minus must satisfy `lambda_minus.is_finite()`, got lambda_minus = {lambda_minus:?}"
    );
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      d > T::zero(),
      "d must satisfy `d > T::zero()`, got d = {d:?}"
    );
    assert!(
      p > T::zero(),
      "p must satisfy `p > T::zero()`, got p = {p:?}"
    );
    assert!(
      q > T::zero(),
      "q must satisfy `q > T::zero()`, got q = {q:?}"
    );
    assert!(
      lambda_plus > T::zero(),
      "lambda_plus must satisfy `lambda_plus > T::zero()`, got lambda_plus = {lambda_plus:?}"
    );
    assert!(
      lambda_minus > T::zero(),
      "lambda_minus must satisfy `lambda_minus > T::zero()`, got lambda_minus = {lambda_minus:?}"
    );
    assert!(
      alpha > T::zero() && alpha < T::from_usize_(2),
      "alpha must satisfy `alpha > T::zero() && alpha < T::from_usize_(2)`, got alpha = {alpha:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(j >= 2, "j must satisfy `j >= 2`, got j = {j:?}");

    assert!(
      alpha != T::one() || p == q,
      "alpha must satisfy `alpha != T::one() || p == q`, got alpha = {alpha:?}"
    );
    Self {
      backend: Cpu,
      d,
      p,
      q,
      lambda_plus,
      lambda_minus,
      alpha,
      n,
      j,
      x0,
      t,
      seed,
    }
  }

  /// The `D` of unit variance at `t = 1`, from `Var(X_1) = D Γ(2-α) (p λ₊^{α-2} + q λ₋^{α-2})`,
  /// which exists for `α < 2`.
  pub fn d_for_unit_variance(p: T, q: T, lambda_plus: T, lambda_minus: T, alpha: T) -> T {
    let g2 = gamma(2.0 - alpha.to_f64().unwrap());
    (T::from_f64_fast(g2)
      * (p * lambda_plus.powf(alpha - T::from_usize_(2))
        + q * lambda_minus.powf(alpha - T::from_usize_(2))))
    .powi(-1)
  }
}

impl<T: FloatExt, S: SeedExt, B> Kobol<T, S, B> {
  /// The horizon, one when omitted.
  fn t_max(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The grid spacing.
  fn dt(&self) -> T {
    self.t_max() / T::from_usize_(self.n - 1)
  }

  /// The probability a term falls on the positive side, `p / (p + q)`.
  fn positive_share(&self) -> T {
    self.p / (self.p + self.q)
  }

  /// The total coefficient `D (p + q)` the arrival bound divides by.
  fn total_coefficient(&self) -> T {
    self.d * self.p + self.d * self.q
  }

  /// The mean-compensating drift `-Γ(1 - α) (D p λ₊^{α-1} - D q λ₋^{α-1})`, applied only below
  /// `α = 1`; above it the mean is the exponent's own, `Γ(1 - α) D (p λ₊^{α-1} - q λ₋^{α-1})`.
  fn drift_rate(&self) -> T {
    if self.alpha < T::one() {
      let g1 = gamma(1.0 - self.alpha.to_f64().unwrap());
      -T::from_f64_fast(g1)
        * (self.d * self.p * self.lambda_plus.powf(self.alpha - T::one())
          - self.d * self.q * self.lambda_minus.powf(self.alpha - T::one()))
    } else {
      T::zero()
    }
  }

  /// What the series drops past its `j`-th term, from the constants its device declaration takes.
  fn remainder(&self) -> SeriesRemainder {
    let f = |v: T| v.to_f64().unwrap();
    let (alpha, t) = (f(self.alpha), f(self.t_max()));
    let rate = alpha / (f(self.total_coefficient()) * t);
    let (plus, minus) = (f(self.lambda_plus), f(self.lambda_minus));
    SeriesRemainder::new(
      alpha,
      rate,
      t,
      f(self.positive_share()),
      plus,
      minus,
      1.0,
      1.0,
    )
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Kobol<T, S, B>
{
  /// The series constants, folded once: the arrival rate `α / (D (p + q) T)`, the exponential
  /// draw entering plainly, and the positive side taken with probability `p / (p + q)`.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let (drift, variance) = self.remainder().expected_rates(self.j);
    crate::euler::EulerSpec::TemperedStableSeries {
      b_t: self.drift_rate() + T::from_f64_fast(drift),
      rem_sd: T::from_f64_fast(variance.sqrt()),
      rate: self.alpha / (self.total_coefficient() * self.t_max()),
      inv_alpha: T::one() / self.alpha,
      e_scale: T::one(),
      e_pow: T::one(),
      w_plus: self.positive_share(),
      lambda_plus: self.lambda_plus,
      lambda_minus: self.lambda_minus,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t_max()
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  /// One term per series index, as many as the host draws.
  fn series_terms(&self) -> Option<u32> {
    Some(self.j as u32)
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Kobol<T, S> { d, p, q, lambda_plus, lambda_minus, alpha, n, j, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Kobol<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Kobol<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = KobolSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> KobolSampler<T, S> {
    // Uniform, then Exp(1) sources are derived from `self.seed`, an order the pinned streams depend
    // on; reuse advances both. The side probability, total coefficient and drift do not read it.
    let t_max = self.t_max();
    let dt = self.dt();
    let w_plus = self.positive_share();
    let c_total = self.total_coefficient();
    let b_t = self.drift_rate();

    KobolSampler {
      n: self.n,
      j: self.j,
      lambda_plus: self.lambda_plus,
      lambda_minus: self.lambda_minus,
      alpha: self.alpha,
      x0: self.x0.unwrap_or(T::zero()),
      t_max,
      dt,
      w_plus,
      c_total,
      b_t,
      uniform: SimdUniform::<T>::new(T::zero(), T::one()).seeded(&self.seed),
      exp: SimdExp::<T>::new(T::one()).seeded(&self.seed),
      seed: self.seed.derive(),
      remainder: self.remainder(),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine when the grid fits the kernels' per-path series cells; a longer
  /// grid stays on the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: the grid fits the kernels'
  /// per-path series cells. A longer grid samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.n <= crate::euler::SERIES_SLOTS))
      .then_some("a grid longer than the kernels' per-path series cells")
  }
}

/// Reusable [`Kobol`] sampling state: the uniform and exponential sources of the Rosiński series,
/// and a seed derived once more per fill for the Gamma arrival times `P`.
#[doc(hidden)]
#[derive(Debug)]
pub struct KobolSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  j: usize,
  lambda_plus: T,
  lambda_minus: T,
  alpha: T,
  x0: T,
  t_max: T,
  dt: T,
  w_plus: T,
  c_total: T,
  b_t: T,
  uniform: Seeded<SimdUniform<T>>,
  exp: Seeded<SimdExp<T>>,
  seed: S,
  remainder: SeriesRemainder,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt, S: SeedExt> KobolSampler<T, S> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let t_max = self.t_max;
    let dt = self.dt;
    let w_plus = self.w_plus;
    let c_total = self.c_total;
    let b_t = self.b_t;

    let n_terms = self.j;
    let size = n_terms + 1; // index 0 reserved (Γ0=0)

    let mut u = Array1::<T>::zeros(size);
    self.uniform.fill_slice(u.as_slice_mut().unwrap());
    let e = Array1::from_shape_fn(size, |_| self.exp.sample());
    let arrivals = Poisson::new(T::one(), Horizon::Events(n_terms), self.seed.derive()).sample();
    let (drift, variance) = self.remainder.rates(arrivals[n_terms].to_f64().unwrap());
    let drift = b_t + T::from_f64_fast(drift);
    let spread = T::from_f64_fast(variance.sqrt());
    let mut tau_raw = Array1::<T>::zeros(size);
    self.uniform.fill_slice(tau_raw.as_slice_mut().unwrap());
    let tau = tau_raw * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    for j in 1..size {
      // HERE IS THE KoBoL DIFFERENCE:
      // probability of choosing + side is p/(p+q) instead of fixed 0.5
      let v_j = if self.uniform.sample() < w_plus {
        self.lambda_plus
      } else {
        -self.lambda_minus
      };

      // HERE IS THE KoBoL DIFFERENCE:
      // divisor uses total coefficient D(p+q) instead of 2C
      let divisor = c_total * t_max; // <-- (2*C*t_max)
      let numerator = self.alpha * arrivals[j];

      let term1 = (numerator / divisor).powf(-T::one() / self.alpha);
      let term2 = e[j] * u[j].powf(T::one() / self.alpha) / v_j.abs();

      jump_size[j] = term1.min(term2) * (v_j / v_j.abs());
    }

    let mut idx = (1..size).collect::<Vec<usize>>();
    idx.sort_by(|&a, &b| {
      tau[a]
        .to_f64()
        .unwrap()
        .partial_cmp(&tau[b].to_f64().unwrap())
        .unwrap()
    });

    out[0] = self.x0;

    let mut k: usize = 0;
    let mut cum_jumps = T::zero();
    let mut gauss = T::zero();

    for i in 1..out.len() {
      let t_i = T::from_usize_(i) * dt;

      while k < idx.len() && tau[idx[k]] <= t_i {
        cum_jumps += jump_size[idx[k]];
        k += 1;
      }

      gauss += self.normal.sample();
      out[i] = self.x0 + cum_jumps + drift * t_i + spread * gauss;
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for KobolSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for KobolSampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Kobol output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyKoBoL, Kobol,
  sig: (d, p, q, lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, seed=None, dtype=None),
  params: (d: f64, p: f64, q: f64, lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>),
  device
);
