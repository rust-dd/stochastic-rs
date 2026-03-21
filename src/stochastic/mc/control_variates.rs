//! # Control Variates
//!
//! $$
//! W = Y + c^*\bigl(V - \mathbb{E}[V]\bigr),\quad
//! c^* = -\frac{\operatorname{Cov}(Y,V)}{\operatorname{Var}(V)}
//! $$
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*, §4.1.
//! DOI: 10.1007/978-0-387-21617-1

use ndarray::Array1;

use super::McEstimate;
use crate::traits::FloatExt;

/// Single control-variate MC estimate.
///
/// `payoff(z)` is the quantity to estimate. `control(z)` is a correlated
/// quantity whose expectation `control_mean` is known analytically.
/// The optimal coefficient `c*` is estimated from the sample.
pub fn estimate<T, F, G>(
  n_paths: usize,
  dim: usize,
  payoff: F,
  control: G,
  control_mean: T,
) -> McEstimate<T>
where
  T: FloatExt,
  F: Fn(&Array1<T>) -> T,
  G: Fn(&Array1<T>) -> T,
{
  let mut ys = Array1::<T>::zeros(n_paths);
  let mut vs = Array1::<T>::zeros(n_paths);

  for i in 0..n_paths {
    let z = T::normal_array(dim, T::zero(), T::one());
    ys[i] = payoff(&z);
    vs[i] = control(&z);
  }

  let n = T::from_usize_(n_paths);
  let y_mean = ys.sum() / n;
  let v_mean = vs.sum() / n;

  // c* = −Cov(Y,V) / Var(V)
  let mut cov = T::zero();
  let mut var_v = T::zero();
  for i in 0..n_paths {
    let dy = ys[i] - y_mean;
    let dv = vs[i] - v_mean;
    cov += dy * dv;
    var_v += dv * dv;
  }
  cov = cov / n;
  var_v = var_v / n;

  let c_star = if var_v > T::from_f64_fast(1e-15) {
    -cov / var_v
  } else {
    T::zero()
  };

  // Adjusted estimator: W_i = Y_i + c*(V_i − E[V])
  let adjusted = &ys + &(&vs - control_mean) * c_star;
  let adj_mean = adjusted.sum() / n;
  let adj_var = adjusted.mapv(|x| (x - adj_mean) * (x - adj_mean)).sum() / n;
  let std_err = (adj_var / n).sqrt();

  McEstimate {
    mean: adj_mean,
    std_err,
    n_samples: n_paths,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Use Z² (known mean = 1) as control variate for E[exp(Z)] = e^{1/2}.
  #[test]
  fn control_variate_reduces_variance() {
    let n = 50_000;
    let dim = 1;
    let payoff = |z: &Array1<f64>| z[0].exp();
    let control = |z: &Array1<f64>| z[0] * z[0];
    let control_mean = 1.0;

    let cv = estimate(n, dim, payoff, control, control_mean);
    let expected = (0.5_f64).exp();

    assert!(
      (cv.mean - expected).abs() < 3.0 * cv.std_err + 0.02,
      "CV mean {:.4} far from expected {expected:.4}",
      cv.mean
    );
  }
}
