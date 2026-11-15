//! # Stochastic Correlation Processes
//!
//! Bounded stochastic correlation models on (−1, 1) following
//! Teng, Ehrhardt & Günther (2016) "Modelling stochastic correlation",
//! *Journal of Mathematics in Industry* 6:2.
//!
//! ## Models
//!
//! - [`TransformedOU`]: ρ_t = f(X_t) where dX = κ(μ−X)dt + σ dW
//!   (Section 2.1; supports [`Transformation::Tanh`] and [`Transformation::Arctan`])
//! - [`VanEmmerich`]: dρ = κ(μ−ρ)dt + σ√(1−ρ²) dW  (Eq. 15)
//! - [`TengSCP`]: Modified Ou in X-space, ρ = tanh(X)  (Eq. 19/20)
//!

pub mod heston_stoch_corr;
pub mod teng;
pub mod transformed_ou;
pub mod van_emmerich;

pub use heston_stoch_corr::HestonStochCorr;
use ndarray::Array1;
pub use teng::TengSCP;
pub use transformed_ou::Transformation;
pub use transformed_ou::TransformedOU;
pub use van_emmerich::VanEmmerich;

use crate::traits::FloatExt;

/// `[dW₁, dW₂]`, each scaled by `√dt`, with `Corr(dW₁[i], dW₂[i]) = rho_path[i]` (Teng et al. 2016,
/// Eq. 43); panics if an entry lies outside `[-1, 1]`. A constant ρ samples faster through `Cgns`.
pub fn stochastic_correlated_bm(rho_path: &Array1<f64>, t: Option<f64>) -> [Array1<f64>; 2] {
  let n = rho_path.len();
  if n == 0 {
    return [Array1::zeros(0), Array1::zeros(0)];
  }

  let horizon = t.unwrap_or(1.0);
  let sqrt_dt = (horizon / n as f64).sqrt();

  let mut dw2 = Array1::<f64>::zeros(n);
  let mut dw3 = Array1::<f64>::zeros(n);
  f64::fill_standard_normal_slice(dw2.as_slice_mut().expect("contiguous"));
  f64::fill_standard_normal_slice(dw3.as_slice_mut().expect("contiguous"));

  let mut dw1 = Array1::zeros(n);
  for (i, &r) in rho_path.iter().enumerate() {
    assert!(
      (-1.0..=1.0).contains(&r),
      "rho_path[{i}] must satisfy `(-1.0..=1.0).contains(&rho_path[i])`, got rho_path[{i}] = {r:?}"
    );
    let c = (1.0 - r * r).sqrt();
    dw1[i] = (r * dw2[i] + c * dw3[i]) * sqrt_dt;
    dw2[i] *= sqrt_dt;
  }

  [dw1, dw2]
}
