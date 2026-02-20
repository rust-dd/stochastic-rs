//! # CGMY (Carr–Geman–Madan–Yor)
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
use ndarray_rand::RandomExt;
use rand::Rng;
use scilib::math::basic::gamma;

use crate::distributions::exp::SimdExp;
use crate::distributions::uniform::SimdUniform;
use crate::stochastic::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CGMY<T: FloatExt> {
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
}

impl<T: FloatExt> CGMY<T> {
  pub fn new(
    c: T,
    lambda_plus: T,
    lambda_minus: T,
    alpha: T,
    n: usize,
    j: usize,
    x0: Option<T>,
    t: Option<T>,
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

impl<T: FloatExt> ProcessExt<T> for CGMY<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

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

    let J = self.j;
    let size = J + 1; // index 0 reserved (Γ0=0)

    let uniform = SimdUniform::new(T::zero(), T::one());
    let exp = SimdExp::new(T::one());

    // U_j ~ Unif(0,1)
    let U = Array1::<T>::random(size, &uniform);
    // E_j ~ Exp(1)
    let E = Array1::<T>::random(size, exp);

    // P_j = Γ_j (PPP/Gamma arrival times), P[0]=0, P[1]=Γ_1, ...
    let P = Poisson::new(T::one(), Some(size), None).sample();

    // τ_j ~ Unif(0,T)
    let tau = Array1::<T>::random(size, &uniform) * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    // NOTE: Here V_j is +G or -M with 0.5-0.5 probability
    for j in 1..size {
      let v_j = if rng.random_bool(0.5) {
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

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    let mut k: usize = 0;
    let mut cum_jumps = T::zero();

    for i in 1..self.n {
      let t_i = T::from_usize_(i) * dt;

      while k < idx.len() && tau[idx[k]] <= t_i {
        cum_jumps += jump_size[idx[k]];
        k += 1;
      }

      x[i] = x[0] + cum_jumps + b_t * t_i;
    }

    x
  }
}

py_process_1d!(PyCGMY, CGMY,
  sig: (c, lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, dtype=None),
  params: (c: f64, lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>)
);
