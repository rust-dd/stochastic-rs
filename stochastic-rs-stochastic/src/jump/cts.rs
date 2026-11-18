//! The standard classical tempered stable process of Kim, Rachev, Bianchi & Fabozzi: one scale
//! `C`, set for unit variance per unit of time. The series on `[0, T]` stops at `J` terms.
//!
//! $$
//! \nu(dx)=C\left(e^{-\lambda_+ x}x^{-1-\alpha}\mathbf 1_{x>0}
//! +e^{-\lambda_-|x|}|x|^{-1-\alpha}\mathbf 1_{x<0}\right)dx,\quad
//! C=\left(\Gamma(2-\alpha)(\lambda_+^{\alpha-2}+\lambda_-^{\alpha-2})\right)^{-1}
//! $$
//!
//! $$
//! X_t=\sum_{j=1}^{J}\Big(\Big(\frac{\alpha\Gamma_j}{2CT}\Big)^{-1/\alpha}\wedge
//! E_j\,U_j^{1/\alpha}\,|V_j|^{-1}\Big)\frac{V_j}{|V_j|}\mathbf 1_{\{\tau_j\le t\}}+b_T\,t,
//! $$
//!
//! $$
//! \Gamma_j\ \text{unit Poisson arrivals},\ E_j\sim\mathrm{Exp}(1),\ U_j,\tau_j/T\sim U(0,1),\
//! P(V_j=\lambda_+)=P(V_j=-\lambda_-)=\tfrac12,\ 2CT=(C+C)T\ \text{the Lévy mass over the horizon}
//! $$
//!
//! Reference: Rosiński (2007), Thm 5.1, Stoch. Proc. Appl. 117(6), 677–707
//! Reference: Kim, Rachev, Bianchi & Fabozzi (2010), J. Banking & Finance 34(9), 2096–2109
//! Reference: Bianchi et al. (2010), Tempered stable distributions and processes in finance
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::params::Horizon;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Cts process (Classical Tempered Stable process)
/// <https://sci-hub.se/https://doi.org/10.1016/j.jbankfin.2010.01.015>
///
#[derive(Debug, Clone)]
pub struct Cts<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Positive jump rate lambda_plus (corresponds to G)
  pub lambda_plus: T, // G
  /// Negative jump rate lambda_minus (corresponds to M)
  pub lambda_minus: T, // M
  /// Jump activity parameter alpha (corresponds to Y), with 0 < alpha < 2
  pub alpha: T,
  /// Number of time steps
  pub n: usize,
  /// Jumps
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

impl<T: FloatExt, S: SeedExt> Cts<T, S> {
  /// Panics unless `lambda_plus > 0`, `lambda_minus > 0`, `0 < alpha < 2`, `alpha != 1` (a pole of
  /// `Γ(1 − α)`), `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
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
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
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
    assert!(
      alpha != T::one(),
      "alpha must satisfy `alpha != T::one()`, got alpha = {alpha:?}"
    );

    Self {
      backend: Cpu,
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
}

impl<T: FloatExt, S: SeedExt, B> Cts<T, S, B> {
  /// The horizon, one when omitted.
  fn t_max(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The grid spacing.
  fn dt(&self) -> T {
    self.t_max() / T::from_usize_(self.n - 1)
  }

  /// The scale `C = (Γ(2 - α) (λ₊^{α-2} + λ₋^{α-2}))^{-1}` that normalises
  /// the law's variance to one.
  fn tempering_constant(&self) -> T {
    let g = gamma(2.0 - self.alpha.to_f64().unwrap());
    (T::from_f64_fast(g)
      * (self.lambda_plus.powf(self.alpha - T::from_usize_(2))
        + self.lambda_minus.powf(self.alpha - T::from_usize_(2))))
    .powi(-1)
  }

  /// The mean-compensating drift `-C Γ(1 - α) (λ₊^{α-1} - λ₋^{α-1})`.
  fn drift_rate(&self) -> T {
    let g = gamma(1.0 - self.alpha.to_f64().unwrap());
    -self.tempering_constant()
      * T::from_f64_fast(g)
      * (self.lambda_plus.powf(self.alpha - T::one())
        - self.lambda_minus.powf(self.alpha - T::one()))
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Cts<T, S, B>
{
  /// The series constants, folded once: the arrival rate `α / (2 C T)`, as the host divides by the
  /// scale alone, the exponential draw entering plainly, and the two sides equally likely.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::TemperedStableSeries {
      b_t: self.drift_rate(),
      rate: self.alpha / (T::from_usize_(2) * self.tempering_constant() * self.t_max()),
      inv_alpha: T::one() / self.alpha,
      e_scale: T::one(),
      e_pow: T::one(),
      w_plus: T::from_f64_fast(0.5),
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

backend_switch!([T: FloatExt, S: SeedExt] Cts<T, S> { lambda_plus, lambda_minus, alpha, n, j, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Cts<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Cts<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CtsSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> CtsSampler<T, S> {
    // Uniform, then Exp(1) sources are derived from `self.seed`, an order the pinned streams depend
    // on; reuse advances both. The scale `C` and the drift `b_t` do not read the seed.
    let t_max = self.t_max();
    let dt = self.dt();
    let c = self.tempering_constant();
    let b_t = self.drift_rate();

    CtsSampler {
      n: self.n,
      j: self.j,
      lambda_plus: self.lambda_plus,
      lambda_minus: self.lambda_minus,
      alpha: self.alpha,
      x0: self.x0.unwrap_or(T::zero()),
      t_max,
      dt,
      c,
      b_t,
      uniform: SimdUniform::<T>::new(T::zero(), T::one()).seeded(&self.seed),
      exp: SimdExp::<T>::new(T::one()).seeded(&self.seed),
      seed: self.seed.derive(),
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

/// Reusable [`Cts`] sampling state: the uniform and exponential sources of the Rosiński series,
/// and a seed derived once more per fill for the Gamma arrival times `P`.
#[doc(hidden)]
#[derive(Debug)]
pub struct CtsSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  j: usize,
  lambda_plus: T,
  lambda_minus: T,
  alpha: T,
  x0: T,
  t_max: T,
  dt: T,
  c: T,
  b_t: T,
  uniform: Seeded<SimdUniform<T>>,
  exp: Seeded<SimdExp<T>>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> CtsSampler<T, S> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let t_max = self.t_max;
    let dt = self.dt;
    let c = self.c;
    let b_t = self.b_t;

    let n_terms = self.j;
    let size = n_terms + 1; // index 0 is reserved (Γ0=0)

    let mut u = Array1::<T>::zeros(size);
    self.uniform.fill_slice(u.as_slice_mut().unwrap());
    let e = Array1::from_shape_fn(size, |_| self.exp.sample());
    let arrivals = Poisson::new(T::one(), Horizon::Events(n_terms), self.seed.derive()).sample();
    let mut tau_raw = Array1::<T>::zeros(size);
    self.uniform.fill_slice(tau_raw.as_slice_mut().unwrap());
    let tau = tau_raw * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    for j in 1..size {
      let v_j = if self.uniform.sample() < T::from_f64_fast(0.5) {
        self.lambda_plus
      } else {
        -self.lambda_minus
      };

      // The arrival bound over the total mass of the measure on [0, T]: both
      // sides carry `C`, so `2 C T` (Rosiński 2007, Thm 5.1).
      let divisor = T::from_usize_(2) * c * t_max;
      let numerator = self.alpha * arrivals[j];
      let term1 = (numerator / divisor).powf(-T::one() / self.alpha);
      let term2 = e[j] * u[j].powf(T::one() / self.alpha) / v_j.abs();
      jump_size[j] = term1.min(term2) * (v_j / v_j.abs());
    }

    let mut idx = (1..size).collect::<Vec<_>>();
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

    for i in 1..out.len() {
      let t_i = T::from_usize_(i) * dt;

      while k < idx.len() && tau[idx[k]] <= t_i {
        cum_jumps += jump_size[idx[k]];
        k += 1;
      }

      // direct formula: x(t_i) = x0 + Σ_{τ_j <= t_i} jump_j + b_t * t_i
      out[i] = self.x0 + cum_jumps + b_t * t_i;
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for CtsSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for CtsSampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Cts output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyCts, Cts,
  sig: (lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, seed=None, dtype=None),
  params: (lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>),
  device
);
