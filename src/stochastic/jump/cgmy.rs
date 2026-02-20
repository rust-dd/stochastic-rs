//! # CGMY
//!
//! $$
//! \nu(dx)=C\left(e^{-Gx}x^{-1-Y}\mathbf 1_{x>0}+e^{-M|x|}|x|^{-1-Y}\mathbf 1_{x<0}\right)dx
//! $$
//!
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use scilib::math::basic::gamma;

use crate::distributions::exp::SimdExp;
use crate::distributions::uniform::SimdUniform;
use crate::stochastic::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// CGMY process
///
/// The CGMY process is a pure jump Lévy process used in financial modeling to capture the dynamics
/// of asset returns with jumps and heavy tails. It is characterized by four parameters:
/// `C`, `G` (lambda_plus), `M` (lambda_minus), and `Y` (alpha).
///
/// The process is defined by the Lévy measure:
///
/// \[ \nu(x) = C \frac{e^{-G x}}{x^{1 + Y}} 1_{x > 0} + C \frac{e^{M x}}{|x|^{1 + Y}} 1_{x < 0} \]
///
/// where:
/// - `c` (C) > 0 controls the overall intensity of the jumps.
/// - `lambda_plus` (G) > 0 is the rate of exponential decay of positive jumps.
/// - `lambda_minus` (M) > 0 is the rate of exponential decay of negative jumps.
/// - `alfa` (Y), with 0 < `alfa` < 2, controls the jump activity (number of small jumps).
///
/// Series representation of the CGMY process:
/// \[ X(t) = \sum_{i=1}^{\infty} ((alpha * Gamma_j / 2C)^(-1/alpha) \land E_j * U_j^(1/alpha) * abs(V_j)^-1)) * V_j / |V_j| 1_[0, t] + b_t * t \]
///
/// This implementation simulates the CGMY process using a discrete approximation over a grid of time points.
/// At each time step, we generate a Poisson random number of jumps, and for each jump, we generate the jump size
/// according to the CGMY process. The process also includes a drift component computed from the parameters.
///
/// # References
///
/// - Cont, R., & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman and Hall/CRC.
/// - Madan, D. B., Carr, P., & Chang, E. C. (1998). The Variance Gamma Process and Option Pricing. *European Finance Review*, 2(1), 79-105.
///   <https://www.econstor.eu/bitstream/10419/239493/1/175133161X.pdf>
///
pub struct CGMY<T: FloatExt> {
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
}

impl<T: FloatExt> CGMY<T> {
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

    CGMY {
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

impl<T: FloatExt> ProcessExt<T> for CGMY<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut rng = rand::rng();

    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let g = gamma(2.0 - self.alpha.to_f64().unwrap());
    let C = (T::from_f64_fast(g)
      * (self.lambda_plus.powf(self.alpha - T::from_usize_(2))
        + self.lambda_minus.powf(self.alpha - T::from_usize_(2))))
    .powi(-1);

    let g = gamma(1.0 - self.alpha.to_f64().unwrap());
    let b_t = -C
      * T::from_f64_fast(g)
      * (self.lambda_plus.powf(self.alpha - T::one())
        - self.lambda_minus.powf(self.alpha - T::one()));

    let J = self.j;
    let size = J + 1; // index 0 is reserved (Γ0=0)

    let uniform = SimdUniform::new(T::zero(), T::one());
    let exp = SimdExp::new(T::one());

    let U = Array1::<T>::random(size, &uniform);
    let E = Array1::<T>::random(size, exp);
    let P = Poisson::new(T::one(), Some(size), None);
    let P = P.sample();
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

    let mut k = 0;
    let mut cum_jumps = T::zero(); // sum_{tau_j <= current t} jump_size[j]

    for i in 1..self.n {
      let t_i = T::from_usize_(i) * dt;

      while k < idx.len() && tau[idx[k]] <= t_i {
        cum_jumps += jump_size[idx[k]];
        k += 1;
      }

      // Equivalent to your interval-summing but faster:
      x[i] = x[0] + cum_jumps + b_t * t_i;
    }

    x
  }
}

py_process_1d!(PyCGMY, CGMY,
  sig: (lambda_plus, lambda_minus, alpha, n, j, x0=None, t=None, dtype=None),
  params: (lambda_plus: f64, lambda_minus: f64, alpha: f64, n: usize, j: usize, x0: Option<f64>, t: Option<f64>)
);
