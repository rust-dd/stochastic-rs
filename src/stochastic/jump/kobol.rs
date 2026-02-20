//! # KoBoL / CTS (general tempered stable family)
//!
//! Lévy measure (one common KoBoL/CTS parametrization):
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
//! Differences vs your CGMY code:
//! - sign selection: P(+)=p/(p+q) instead of fixed 1/2
//! - series constant uses D(p+q) instead of 2C
//! - drift uses D*p and D*q (if α<1; else set 0)
//!
//! CGMY is a special case of this if you set:
//! - D = C
//! - p = q = 1
//! - λ_+ = G, λ_- = M
//! then D(p+q)=2C and P(+)=1/2.

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
}

impl<T: FloatExt> KoBoL<T> {
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

impl<T: FloatExt> ProcessExt<T> for KoBoL<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let pq = self.p + self.q;
    let w_plus = (self.p / pq).to_f64().unwrap(); // probability of positive side

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

    let J = self.j;
    let size = J + 1; // index 0 reserved (Γ0=0)

    let uniform = SimdUniform::new(T::zero(), T::one());
    let exp = SimdExp::new(T::one());

    let U = Array1::<T>::random(size, &uniform);
    let E = Array1::<T>::random(size, exp);
    let P = Poisson::new(T::one(), Some(size), None).sample();
    let tau = Array1::<T>::random(size, &uniform) * t_max;

    let mut jump_size = Array1::<T>::zeros(size);

    for j in 1..size {
      // HERE IS THE KoBoL DIFFERENCE:
      // probability of choosing + side is p/(p+q) instead of fixed 0.5
      let v_j = if rng.random_bool(w_plus) {
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

py_process_1d!(PyKoBoL, KoBoL,
  sig: (d, p, q, lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, dtype=None),
  params: (d: f64, p: f64, q: f64, lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>)
);
