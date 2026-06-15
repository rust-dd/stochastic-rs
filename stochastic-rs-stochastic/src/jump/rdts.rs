//! # Rdts
//!
//! $$
//! \nu(dx)\propto e^{-\lambda |x|^\rho}|x|^{-1-\alpha}dx\quad(\text{rapidly decaying tempered stable})
//! $$
//!
use ndarray::Array1;
use scilib::math::basic::gamma;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::buffer::array1_from_fill;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Rdts process (Rapidly Decreasing Tempered Stable process)
/// <https://sci-hub.se/https://doi.org/10.1016/j.jbankfin.2010.01.015>
pub struct Rdts<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Initial value
  pub x0: Option<T>,
  /// Total time horizon
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Rdts<T, S> {
  /// Create a new Rdts process
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
    assert!(lambda_plus > T::zero(), "lambda_plus must be positive");
    assert!(lambda_minus > T::zero(), "lambda_minus must be positive");
    assert!(
      alpha > T::zero() && alpha < T::from_usize_(2),
      "alpha must be in (0, 2)"
    );

    Self {
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

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Rdts<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = RdtsSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> RdtsSampler<T> {
    // Uniform and Exp(1) sources are derived from `self.seed` in the same
    // order as the legacy `sample()`, so the first fill reproduces it
    // bit-for-bit; both owned sources advance on reuse. The seed-independent
    // scale `C` and drift `b_t` are precomputed here.
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let g = gamma(2.0 - self.alpha.to_f64().unwrap());
    let C = (T::from_f64_fast(g)
      * (self.lambda_plus.powf(self.alpha - T::from_usize_(2))
        + self.lambda_minus.powf(self.alpha - T::from_usize_(2))))
    .powi(-1);

    let g = gamma((1.0 - self.alpha.to_f64().unwrap()) / 2.0);
    let b_t = -C
      * (T::from_f64_fast(g) / T::from_usize_(2).powf((self.alpha + T::one()) / T::from_usize_(2)))
      * (self.lambda_plus.powf(self.alpha - T::one())
        - self.lambda_minus.powf(self.alpha - T::one()));

    RdtsSampler {
      n: self.n,
      j: self.j,
      lambda_plus: self.lambda_plus,
      lambda_minus: self.lambda_minus,
      alpha: self.alpha,
      x0: self.x0.unwrap_or(T::zero()),
      t_max,
      dt,
      c: C,
      b_t,
      uniform: SimdUniform::<T>::new(T::zero(), T::one(), &self.seed),
      exp: SimdExp::<T>::new(T::one(), &self.seed),
    }
  }
}

/// Reusable [`Rdts`] sampling state: owns the uniform and exponential sources
/// driving the truncated Rosiński series so a Monte-Carlo loop pays their
/// setup once. The Gamma arrival series, jump sizes and ordering are rebuilt
/// per path inside `fill_path`.
#[doc(hidden)]
pub struct RdtsSampler<T: FloatExt> {
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
  uniform: SimdUniform<T>,
  exp: SimdExp<T>,
}

impl<T: FloatExt> RdtsSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let t_max = self.t_max;
    let dt = self.dt;
    let C = self.c;
    let b_t = self.b_t;

    let J = self.j;
    let size = J + 1; // index 0 is reserved (Γ0=0)

    let mut U = Array1::<T>::zeros(size);
    self.uniform.fill_slice_fast(U.as_slice_mut().unwrap());
    let E = Array1::from_shape_fn(size, |_| self.exp.sample_fast());
    let P = Poisson::new(T::one(), Some(size), None, Unseeded).sample();
    let mut tau_raw = Array1::<T>::zeros(size);
    self
      .uniform
      .fill_slice_fast(tau_raw.as_slice_mut().unwrap());
    let tau = tau_raw * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    for j in 1..size {
      let v_j = if self.uniform.sample_fast() < T::from_f64_fast(0.5) {
        self.lambda_plus
      } else {
        -self.lambda_minus
      };

      let divisor = T::from_usize_(2) * C * t_max;
      let numerator = self.alpha * P[j];
      let term1 = (numerator / divisor).powf(-T::one() / self.alpha);

      // Rdts: term2 = 0.5 * sqrt(E_j) * U_j^{1/alpha} / |V_j|
      let term2 =
        T::from_f64_fast(0.5) * E[j].powf(T::from_f64_fast(0.5)) * U[j].powf(T::one() / self.alpha)
          / v_j.abs();

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

      // direkt formula: x(t_i) = x0 + sum_{tau_j<=t_i} jump_j + b_t * t_i
      out[i] = self.x0 + cum_jumps + b_t * t_i;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for RdtsSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Rdts output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyRdts, Rdts,
  sig: (lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, seed=None, dtype=None),
  params: (lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>)
);
