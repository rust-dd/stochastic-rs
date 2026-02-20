//! # KoBoL (CGMY alias)
//!
//! Lévy measure:
//! $$
//! \nu(dx)=C\left(e^{-Gx}x^{-1-Y}\mathbf 1_{x>0}+e^{-M|x|}|x|^{-1-Y}\mathbf 1_{x<0}\right)dx
//! $$
//!
//! Series representation (truncated at J terms, same notation as your CGMY):
//!
//! - **U**:  (U_j) i.i.d. Uniform(0,1)  — appears as U_j^{1/Y}
//! - **E**:  (E_j) i.i.d. Exp(1)        — appears in the min term
//! - **tau**:(τ_j) i.i.d. Uniform(0,T)  — assigns each term to a time in (0,T]
//! - **P**:  (P_j) PPP arrival times    — P[0]=0, P[j]=Γ_j where Γ_j = Σ_{k=1..j} Exp(1)
//! - **V_j**: takes +G or -M with prob 1/2 each (sign + tempering side)
//!
//! $$
//! X(t)=\sum_{j=1}^{J}\Big(\Big(\frac{Y\Gamma_j}{2CT}\Big)^{-1/Y}\wedge
//! E_j\,U_j^{1/Y}\,|V_j|^{-1}\Big)\frac{V_j}{|V_j|}\mathbf 1_{\{\tau_j\in(t_{i-1},t_i]\}}+b_T\,\Delta t
//! $$

use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use scilib::math::basic::gamma;

use crate::distributions::exp::SimdExp;
use crate::distributions::uniform::SimdUniform;
use crate::stochastic::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct KoBoL<T: FloatExt> {
  /// Positive tempering parameter (G) > 0
  pub lambda_plus: T, // G
  /// Negative tempering parameter (M) > 0
  pub lambda_minus: T, // M
  /// Activity parameter (Y) in (0, 2)
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

impl<T: FloatExt> KoBoL<T> {
  pub fn new(
    lambda_plus: T,
    lambda_minus: T,
    alpha: T,
    n: usize,
    j: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(lambda_plus > T::zero(), "lambda_plus must be positive");
    assert!(lambda_minus > T::zero(), "lambda_minus must be positive");
    assert!(
      alpha > T::zero() && alpha < T::from_usize_(2),
      "alpha must be in (0, 2)"
    );
    assert!(n >= 2, "n must be >= 2");
    assert!(j >= 2, "j must be >= 2 (because we index from 1..j)");

    Self {
      lambda_plus,
      lambda_minus,
      alpha,
      n,
      j,
      x0,
      t,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for KoBoL<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    // --- Standardization constants as in your CGMY code (same paper-style):
    // C = [ Γ(2-Y) (G^{Y-2} + M^{Y-2}) ]^{-1}
    let g2 = gamma(2.0 - self.alpha.to_f64().unwrap());
    let C = (T::from_f64_fast(g2)
      * (self.lambda_plus.powf(self.alpha - T::from_usize_(2))
        + self.lambda_minus.powf(self.alpha - T::from_usize_(2))))
    .powi(-1);

    // b_T = -C Γ(1-Y) (G^{Y-1} - M^{Y-1})
    let g1 = gamma(1.0 - self.alpha.to_f64().unwrap());
    let b_t = -C
      * T::from_f64_fast(g1)
      * (self.lambda_plus.powf(self.alpha - T::one())
        - self.lambda_minus.powf(self.alpha - T::one()));

    let J = self.j;
    let size = J + 1; // index 0 is reserved (Γ0=0)

    let uniform = SimdUniform::new(T::zero(), T::one());
    let exp = SimdExp::new(T::one());

    // U_j ~ Unif(0,1)
    let U = Array1::<T>::random(size, &uniform);
    // E_j ~ Exp(1)
    let E = Array1::<T>::random(size, exp);
    // P_j = Γ_j (PPP arrival times), with P[0]=0, P[1]=Γ_1, ...
    let P = Poisson::new(T::one(), Some(size), None).sample();
    // τ_j ~ Unif(0,T)
    let tau = Array1::<T>::random(size, &uniform) * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    for j in 1..size {
      let v_j = if rng.random_bool(0.5) {
        self.lambda_plus
      } else {
        -self.lambda_minus
      };

      let divisor = T::from_usize_(2) * C * t_max;
      let numerator = self.alpha * P[j];
      let term1 = (numerator / divisor).powf(-T::one() / self.alpha);

      let term2 = E[j] * U[j].powf(T::one() / self.alpha) / v_j.abs();
      jump_size[j] = term1.min(term2) * (v_j / v_j.abs());
    }

    let mut idx = (1..size).collect::<Vec<usize>>(); // 1.. because tau[0] exists, but you use 1..j
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

      // sweep in all jumps with tau <= t_i
      while k < idx.len() && tau[idx[k]] <= t_i {
        cum_jumps += jump_size[idx[k]];
        k += 1;
      }

      // direct formula: x(t_i) = x0 + sum_{tau_j<=t_i} jump_j + b_t * t_i
      x[i] = x[0] + cum_jumps + b_t * t_i;
    }

    x
  }
}

py_process_1d!(PyKoBoL, KoBoL,
  sig: (lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, dtype=None),
  params: (lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>)
);
