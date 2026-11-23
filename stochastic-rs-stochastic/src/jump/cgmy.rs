//! CGMY by `J` terms of Rosiński's series and their Gaussian remainder `R_J`; `C` is an input.
//!
//! $$
//! \nu(dx)=C\left(e^{-Mx}x^{-1-Y}\mathbf 1_{x>0}+e^{-G|x|}|x|^{-1-Y}\mathbf 1_{x<0}\right)dx
//! $$
//!
//! The drift `b_T` carries the mean compensator, with its `Γ(1 − Y)`, only for `Y < 1`; else 0.
//!
//! $$
//! X(t)=\sum_{j=1}^{J}\Big(\Big(\frac{Y\Gamma_j}{2CT}\Big)^{-1/Y}\wedge
//! E_j\,U_j^{1/Y}\,|V_j|^{-1}\Big)\frac{V_j}{|V_j|}\mathbf 1_{\{\tau_j\le t\}}+b_T\,t+R_J(t),\quad
//! V_j\in\{+M,-G\},\ P(V_j=+M)=P(V_j=-G)=\tfrac12
//! $$
//!
//! Reference: Carr, Geman, Madan & Yor (2002), J. Business 75(2), 305–332, DOI 10.1086/338705
//! Reference: Rosiński (2007), Stoch. Proc. Appl. 117(6), 677–707, DOI 10.1016/j.spa.2006.10.003
//! Reference: Asmussen & Rosiński (2001), J. Appl. Probab. 38(2), 482–493, DOI 10.1239/jap/996986757

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
pub struct Cgmy<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Overall jump intensity scale C > 0
  pub c: T,
  /// Tempering rate of the positive jumps, CGMY's M > 0.
  pub lambda_plus: T,
  /// Tempering rate of the negative jumps, CGMY's G > 0.
  pub lambda_minus: T,
  /// Activity parameter Y in (0, 2)
  pub alpha: T, // Y
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

impl<T: FloatExt, S: SeedExt> Cgmy<T, S> {
  /// Panics unless a given `t` is positive and finite, `c > 0`, `lambda_plus > 0`,
  /// `lambda_minus > 0`, `0 < alpha < 2`, `n >= 2`, `j >= 2`, and every parameter is finite.
  pub fn new(
    c: T,
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
      c.is_finite(),
      "c must satisfy `c.is_finite()`, got c = {c:?}"
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
      c > T::zero(),
      "c must satisfy `c > T::zero()`, got c = {c:?}"
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

    Self {
      backend: Cpu,
      c,
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

  /// Optional helper if you still want your old "unit variance at t=1" normalization:
  /// Var(X_1) = C Γ(2-Y) (G^{Y-2} + M^{Y-2})
  pub fn c_for_unit_variance(lambda_plus: T, lambda_minus: T, alpha: T) -> T {
    let g2 = gamma(2.0 - alpha.to_f64().unwrap());
    (T::from_f64_fast(g2)
      * (lambda_plus.powf(alpha - T::from_usize_(2))
        + lambda_minus.powf(alpha - T::from_usize_(2))))
    .powi(-1)
  }
}

impl<T: FloatExt, S: SeedExt, B> Cgmy<T, S, B> {
  /// The horizon, one when omitted.
  fn t_max(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The grid spacing.
  fn dt(&self) -> T {
    self.t_max() / T::from_usize_(self.n - 1)
  }

  /// The mean-compensating drift `-C Γ(1 - Y) (G^{Y-1} - M^{Y-1})`, finite —
  /// and applied — only below `Y = 1`.
  fn drift_rate(&self) -> T {
    if self.alpha < T::one() {
      let g1 = gamma(1.0 - self.alpha.to_f64().unwrap());
      -self.c
        * T::from_f64_fast(g1)
        * (self.lambda_plus.powf(self.alpha - T::one())
          - self.lambda_minus.powf(self.alpha - T::one()))
    } else {
      T::zero()
    }
  }

  /// What the series drops past its `j`-th term, from the constants its device declaration takes.
  fn remainder(&self) -> SeriesRemainder {
    let f = |v: T| v.to_f64().unwrap();
    let (alpha, t) = (f(self.alpha), f(self.t_max()));
    let rate = alpha / (2.0 * f(self.c) * t);
    SeriesRemainder::new(
      alpha,
      rate,
      t,
      0.5,
      f(self.lambda_plus),
      f(self.lambda_minus),
      1.0,
      1.0,
    )
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Cgmy<T, S, B>
{
  /// The series constants folded once: the arrival bound's rate `Y / (2 C T)`,
  /// the exponential draw entering plainly, and the two sides equally likely.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let (drift, variance) = self.remainder().expected_rates(self.j);
    crate::euler::EulerSpec::TemperedStableSeries {
      b_t: self.drift_rate() + T::from_f64_fast(drift),
      rem_sd: T::from_f64_fast(variance.sqrt()),
      rate: self.alpha / (T::from_usize_(2) * self.c * self.t_max()),
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

backend_switch!([T: FloatExt, S: SeedExt] Cgmy<T, S> { c, lambda_plus, lambda_minus, alpha, n, j, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Cgmy<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Cgmy<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CgmySampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> CgmySampler<T, S> {
    // Uniform, then Exp(1) sources are derived from `self.seed`, an order the pinned streams depend
    // on; reuse advances both. The drift `b_t` does not read the seed and is computed here.
    let t_max = self.t_max();
    let dt = self.dt();
    let b_t = self.drift_rate();

    CgmySampler {
      n: self.n,
      j: self.j,
      c: self.c,
      lambda_plus: self.lambda_plus,
      lambda_minus: self.lambda_minus,
      alpha: self.alpha,
      x0: self.x0.unwrap_or(T::zero()),
      t_max,
      dt,
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

/// Reusable [`Cgmy`] sampling state: the uniform and exponential sources of the Rosiński series,
/// and a seed derived once more per fill for the Gamma arrival times `P`.
#[doc(hidden)]
#[derive(Debug)]
pub struct CgmySampler<T: FloatExt, S: SeedExt> {
  n: usize,
  j: usize,
  c: T,
  lambda_plus: T,
  lambda_minus: T,
  alpha: T,
  x0: T,
  t_max: T,
  dt: T,
  b_t: T,
  uniform: Seeded<SimdUniform<T>>,
  exp: Seeded<SimdExp<T>>,
  seed: S,
  remainder: SeriesRemainder,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt, S: SeedExt> CgmySampler<T, S> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let t_max = self.t_max;
    let dt = self.dt;
    let c = self.c;
    let b_t = self.b_t;

    let n_terms = self.j;
    let size = n_terms + 1; // index 0 reserved (Γ0=0)

    // U_j ~ Unif(0,1)
    let mut u = Array1::<T>::zeros(size);
    self.uniform.fill_slice(u.as_slice_mut().unwrap());
    // E_j ~ Exp(1)
    let e = Array1::from_shape_fn(size, |_| self.exp.sample());

    // P_j = Γ_j (PPP/Gamma arrival times), P[0]=0, P[1]=Γ_1, ...
    let arrivals = Poisson::new(T::one(), Horizon::Events(n_terms), self.seed.derive()).sample();
    let (drift, variance) = self.remainder.rates(arrivals[n_terms].to_f64().unwrap());
    let drift = b_t + T::from_f64_fast(drift);
    let spread = T::from_f64_fast(variance.sqrt());

    // τ_j ~ Unif(0,T)
    let mut tau_raw = Array1::<T>::zeros(size);
    self.uniform.fill_slice(tau_raw.as_slice_mut().unwrap());
    let tau = tau_raw * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    // NOTE: Here V_j is +G or -M with 0.5-0.5 probability
    for j in 1..size {
      let v_j = if self.uniform.sample() < T::from_f64_fast(0.5) {
        self.lambda_plus
      } else {
        -self.lambda_minus
      };

      let divisor = T::from_usize_(2) * c * t_max; // <-- 2C appears here
      let numerator = self.alpha * arrivals[j];

      let term1 = (numerator / divisor).powf(-T::one() / self.alpha);
      let term2 = e[j] * u[j].powf(T::one() / self.alpha) / v_j.abs();

      jump_size[j] = term1.min(term2) * (v_j / v_j.abs());
    }

    // sort indices by tau
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

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for CgmySampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for CgmySampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Cgmy output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyCgmy, Cgmy,
  sig: (c, lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, seed=None, dtype=None),
  params: (c: f64, lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>),
  device
);
