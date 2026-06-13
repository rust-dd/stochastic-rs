//! # KoBoL / Cts (general tempered stable family)
//!
//! Lévy measure (one common KoBoL/Cts parametrization):
//! $$
//! \nu(dx)=D\left(p\,e^{-\lambda_+ x}x^{-1-\alpha}\mathbf 1_{x>0}
//! \;+\;q\,e^{-\lambda_-|x|}|x|^{-1-\alpha}\mathbf 1_{x<0}\right)dx
//! $$
//!
//! - D > 0 overall scale
//! - p, q > 0 side weights (not necessarily normalized)
//! - λ_+ , λ_- > 0 tempering rates
//! - α in (0,2) activity
//!
//! Differences vs your Cgmy code:
//! - sign selection: P(+)=p/(p+q) instead of fixed 1/2
//! - series constant uses D(p+q) instead of 2C
//! - drift uses D*p and D*q (if α<1; else set 0)
//!
//! Cgmy is a special case of this if you set:
//! - D = C
//! - p = q = 1
//! - λ_+ = G, λ_- = M
//!
//! then D(p+q)=2C and P(+)=1/2.

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

pub struct KoBoL<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Truncation terms (J)
  pub j: usize,
  /// Initial value
  pub x0: Option<T>,
  /// Total horizon T
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> KoBoL<T, S> {
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
    assert!(d > T::zero(), "d (D) must be positive");
    assert!(p > T::zero(), "p must be positive");
    assert!(q > T::zero(), "q must be positive");
    assert!(lambda_plus > T::zero(), "lambda_plus must be positive");
    assert!(lambda_minus > T::zero(), "lambda_minus must be positive");
    assert!(
      alpha > T::zero() && alpha < T::from_usize_(2),
      "alpha must be in (0, 2)"
    );
    assert!(n >= 2, "n must be >= 2");
    assert!(j >= 2, "j must be >= 2 (because we index from 1..j)");

    Self {
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

  /// Var(X_1) exists for α<2:
  /// Var(X_1) = D Γ(2-α) ( p λ_+^{α-2} + q λ_-^{α-2} )
  /// If you want unit variance at t=1, choose D = 1 / [ Γ(2-α) (...) ].
  pub fn d_for_unit_variance(p: T, q: T, lambda_plus: T, lambda_minus: T, alpha: T) -> T {
    let g2 = gamma(2.0 - alpha.to_f64().unwrap());
    (T::from_f64_fast(g2)
      * (p * lambda_plus.powf(alpha - T::from_usize_(2))
        + q * lambda_minus.powf(alpha - T::from_usize_(2))))
    .powi(-1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for KoBoL<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = KoBoLSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> KoBoLSampler<T> {
    // Uniform and Exp(1) sources are derived from `self.seed` in the same
    // order as the legacy `sample()`, so the first fill reproduces it
    // bit-for-bit; both owned sources advance on reuse. The seed-independent
    // side probability, total coefficient and drift are precomputed here.
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let pq = self.p + self.q;
    let w_plus = self.p / pq; // probability of positive side

    // side coefficients:
    let c_plus = self.d * self.p;
    let c_minus = self.d * self.q;
    let c_total = c_plus + c_minus; // = D(p+q)

    // Mean-compensator drift (finite only if alpha < 1 in the classical sense)
    let b_t = if self.alpha < T::one() {
      // b_T = -Γ(1-α) * ( c_plus * λ_+^{α-1} - c_minus * λ_-^{α-1} )
      let g1 = gamma(1.0 - self.alpha.to_f64().unwrap());
      -T::from_f64_fast(g1)
        * (c_plus * self.lambda_plus.powf(self.alpha - T::one())
          - c_minus * self.lambda_minus.powf(self.alpha - T::one()))
    } else {
      T::zero()
    };

    KoBoLSampler {
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
      uniform: SimdUniform::<T>::new(T::zero(), T::one(), &self.seed),
      exp: SimdExp::<T>::new(T::one(), &self.seed),
    }
  }
}

/// Reusable [`KoBoL`] sampling state: owns the uniform and exponential sources
/// driving the truncated Rosiński series so a Monte-Carlo loop pays their
/// setup once. The Gamma arrival series, jump sizes and ordering are rebuilt
/// per path inside `fill_path`.
#[doc(hidden)]
pub struct KoBoLSampler<T: FloatExt> {
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
  uniform: SimdUniform<T>,
  exp: SimdExp<T>,
}

impl<T: FloatExt> KoBoLSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let t_max = self.t_max;
    let dt = self.dt;
    let w_plus = self.w_plus;
    let c_total = self.c_total;
    let b_t = self.b_t;

    let J = self.j;
    let size = J + 1; // index 0 reserved (Γ0=0)

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
      // HERE IS THE KoBoL DIFFERENCE:
      // probability of choosing + side is p/(p+q) instead of fixed 0.5
      let v_j = if self.uniform.sample_fast() < w_plus {
        self.lambda_plus
      } else {
        -self.lambda_minus
      };

      // HERE IS THE KoBoL DIFFERENCE:
      // divisor uses total coefficient D(p+q) instead of 2C
      let divisor = c_total * t_max; // <-- (2*C*t_max)
      let numerator = self.alpha * P[j];

      let term1 = (numerator / divisor).powf(-T::one() / self.alpha);
      let term2 = E[j] * U[j].powf(T::one() / self.alpha) / v_j.abs();

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

    for i in 1..out.len() {
      let t_i = T::from_usize_(i) * dt;

      while k < idx.len() && tau[idx[k]] <= t_i {
        cum_jumps += jump_size[idx[k]];
        k += 1;
      }

      out[i] = self.x0 + cum_jumps + b_t * t_i;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for KoBoLSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("KoBoL output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyKoBoL, KoBoL,
  sig: (d, p, q, lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, seed=None, dtype=None),
  params: (d: f64, p: f64, q: f64, lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>)
);
