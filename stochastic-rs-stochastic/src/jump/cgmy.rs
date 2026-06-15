//! # Cgmy (Carr–Geman–Madan–Yor)
//!
//! Lévy measure:
//! $$
//! \nu(dx)=C\left(e^{-Gx}x^{-1-Y}\mathbf 1_{x>0}+e^{-M|x|}|x|^{-1-Y}\mathbf 1_{x<0}\right)dx
//! $$
//!
//! This implementation uses the same (truncated) Rosiński series style representation you used,
//! but here **C is an input parameter** (NOT implied / variance-normalized).
//!
//! Notes on drift:
//! - The usual mean-compensator term involves Γ(1-Y), which is finite only for Y<1 (finite mean).
//! - For Y>=1, the classical mean does not exist; here we set drift correction to 0 to avoid NaN/inf.
//!
//! Series (truncated at J terms):
//! $$
//! X(t)=\sum_{j=1}^{J}\Big(\Big(\frac{Y\Gamma_j}{2CT}\Big)^{-1/Y}\wedge
//! E_j\,U_j^{1/Y}\,|V_j|^{-1}\Big)\frac{V_j}{|V_j|}\mathbf 1_{\{\tau_j\le t\}}+b_T\,t
//! $$
//! with V_j ∈ {+G, -M}, P(V_j=+G)=P(V_j=-M)=1/2.

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

pub struct Cgmy<T: FloatExt, S: SeedExt = Unseeded> {
  /// Overall jump intensity scale C > 0
  pub c: T,
  /// Positive tempering parameter G > 0
  pub lambda_plus: T, // G
  /// Negative tempering parameter M > 0
  pub lambda_minus: T, // M
  /// Activity parameter Y in (0, 2)
  pub alpha: T, // Y
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

impl<T: FloatExt, S: SeedExt> Cgmy<T, S> {
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
    assert!(c > T::zero(), "c (C) must be positive");
    assert!(lambda_plus > T::zero(), "lambda_plus (G) must be positive");
    assert!(
      lambda_minus > T::zero(),
      "lambda_minus (M) must be positive"
    );
    assert!(
      alpha > T::zero() && alpha < T::from_usize_(2),
      "alpha (Y) must be in (0, 2)"
    );
    assert!(n >= 2, "n must be >= 2");
    assert!(j >= 2, "j must be >= 2 (because we index from 1..j)");

    Self {
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

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cgmy<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CgmySampler<T>
  where
    Self: 's;

  fn sampler(&self) -> CgmySampler<T> {
    // Uniform and Exp(1) sources are derived from `self.seed` in the same
    // order as the legacy `sample()`, so the first fill reproduces it
    // bit-for-bit; both owned sources advance on reuse for independent paths.
    // The seed-independent drift `b_t` is precomputed here.
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let C = self.c;

    // Mean-compensator drift (finite only if alpha < 1 in the classical sense)
    let b_t = if self.alpha < T::one() {
      // b_T = -C Γ(1-Y) (G^{Y-1} - M^{Y-1})
      let g1 = gamma(1.0 - self.alpha.to_f64().unwrap());
      -C * T::from_f64_fast(g1)
        * (self.lambda_plus.powf(self.alpha - T::one())
          - self.lambda_minus.powf(self.alpha - T::one()))
    } else {
      T::zero()
    };

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
      uniform: SimdUniform::<T>::new(T::zero(), T::one(), &self.seed),
      exp: SimdExp::<T>::new(T::one(), &self.seed),
    }
  }
}

/// Reusable [`Cgmy`] sampling state: owns the uniform and exponential sources
/// driving the truncated Rosiński series so a Monte-Carlo loop pays their
/// setup once. The Gamma arrival series, jump sizes and ordering are rebuilt
/// per path inside `fill_path`.
#[doc(hidden)]
pub struct CgmySampler<T: FloatExt> {
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
  uniform: SimdUniform<T>,
  exp: SimdExp<T>,
}

impl<T: FloatExt> CgmySampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let t_max = self.t_max;
    let dt = self.dt;
    let C = self.c;
    let b_t = self.b_t;

    let J = self.j;
    let size = J + 1; // index 0 reserved (Γ0=0)

    // U_j ~ Unif(0,1)
    let mut U = Array1::<T>::zeros(size);
    self.uniform.fill_slice_fast(U.as_slice_mut().unwrap());
    // E_j ~ Exp(1)
    let E = Array1::from_shape_fn(size, |_| self.exp.sample_fast());

    // P_j = Γ_j (PPP/Gamma arrival times), P[0]=0, P[1]=Γ_1, ...
    let P = Poisson::new(T::one(), Some(size), None, Unseeded).sample();

    // τ_j ~ Unif(0,T)
    let mut tau_raw = Array1::<T>::zeros(size);
    self
      .uniform
      .fill_slice_fast(tau_raw.as_slice_mut().unwrap());
    let tau = tau_raw * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    // NOTE: Here V_j is +G or -M with 0.5-0.5 probability
    for j in 1..size {
      let v_j = if self.uniform.sample_fast() < T::from_f64_fast(0.5) {
        self.lambda_plus
      } else {
        -self.lambda_minus
      };

      let divisor = T::from_usize_(2) * C * t_max; // <-- 2C appears here
      let numerator = self.alpha * P[j];

      let term1 = (numerator / divisor).powf(-T::one() / self.alpha);
      let term2 = E[j] * U[j].powf(T::one() / self.alpha) / v_j.abs();

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

impl<T: FloatExt> PathSampler<T> for CgmySampler<T> {
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
  params: (c: f64, lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>)
);
