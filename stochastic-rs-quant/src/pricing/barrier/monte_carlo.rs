//! Monte Carlo barrier pricing on Gbm paths monitored at every step.

use rayon::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::mc::McEstimate;

use super::BarrierType;
use crate::OptionType;
use crate::pricing::exotic::BarrierOption;
use crate::pricing::exotic::assert_sigma;
use crate::pricing::mc_stats::std_err_from_sums;

/// Monte Carlo price of the [`BarrierOption`] the closed form prices, rebate included, on
/// `n_paths` Gbm paths monitored at each of `n_steps` steps and at the spot.
#[derive(Debug, Clone, Copy)]
pub struct McBarrierPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
  pub n_paths: usize,
  pub n_steps: usize,
}

impl McBarrierPricer {
  /// Panics unless `sigma` is finite and `>= 0`, `n_paths >= 1` and `n_steps >= 1`.
  pub fn new(sigma: f64, n_paths: usize, n_steps: usize) -> Self {
    assert_sigma(sigma);
    assert!(
      n_paths >= 1,
      "n_paths must satisfy `n_paths >= 1`, got n_paths = {n_paths:?}"
    );
    assert!(
      n_steps >= 1,
      "n_steps must satisfy `n_steps >= 1`, got n_steps = {n_steps:?}"
    );
    Self {
      sigma,
      n_paths,
      n_steps,
    }
  }

  /// Discounted estimate under the log drift `(r − q − σ²/2) dt`, from a fresh base seed; NaN for
  /// a NaN query or term, `tau < 0` or a spot not positive before expiry, as the closed form.
  pub fn price(&self, option: &BarrierOption, s: f64, r: f64, q: f64, tau: f64) -> McEstimate<f64> {
    self.price_from(option, s, r, q, tau, Unseeded.next_seed())
  }

  /// [`price`](Self::price) from the base `seed`, each path's stream derived from it and its index.
  pub fn price_seeded(
    &self,
    option: &BarrierOption,
    s: f64,
    r: f64,
    q: f64,
    tau: f64,
    seed: u64,
  ) -> McEstimate<f64> {
    self.price_from(option, s, r, q, tau, seed)
  }

  fn price_from(
    &self,
    option: &BarrierOption,
    s: f64,
    r: f64,
    q: f64,
    tau: f64,
    base_seed: u64,
  ) -> McEstimate<f64> {
    let BarrierOption {
      strike: k,
      barrier: h,
      rebate,
      barrier_type,
      option_type,
    } = *option;
    let undefined = [s, r, q, tau, k, h, rebate].iter().any(|x| x.is_nan());
    if undefined || tau < 0.0 || (s <= 0.0 && tau > 0.0) {
      return McEstimate {
        mean: f64::NAN,
        std_err: f64::NAN,
        n_samples: 0,
      };
    }
    let dt = tau / self.n_steps as f64;
    let drift = (r - q - 0.5 * self.sigma * self.sigma) * dt;
    let vol = self.sigma * dt.sqrt();
    let discount = (-r * tau).exp();
    let up = matches!(barrier_type, BarrierType::UpAndIn | BarrierType::UpAndOut);
    let knock_in = matches!(barrier_type, BarrierType::UpAndIn | BarrierType::DownAndIn);
    let breached = |x: f64| if up { x >= h } else { x <= h };

    let payoff_of = |path: usize| -> f64 {
      // Golden-ratio stride keeps consecutive path seeds far apart in the
      // splitmix state space, so neighbouring paths do not share a stream.
      let seed = base_seed.wrapping_add((path as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
      let mut normals = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
      let mut z_buf = vec![0.0_f64; self.n_steps];
      normals.fill_slice(&mut z_buf);

      let mut s_curr = s;
      let mut hit = breached(s).then_some(0);
      for (step, &z) in z_buf.iter().enumerate() {
        s_curr *= (drift + vol * z).exp();
        if hit.is_none() && breached(s_curr) {
          hit = Some(step + 1);
        }
      }

      let payoff = match option_type {
        OptionType::Call => (s_curr - k).max(0.0),
        OptionType::Put => (k - s_curr).max(0.0),
      };
      match (knock_in, hit) {
        (true, Some(_)) | (false, None) => payoff,
        (true, None) => rebate,
        // Paid when hit, so it carries the interest from then to the expiry `discount` removes.
        (false, Some(step)) => rebate * (r * (tau - step as f64 * dt)).exp(),
      }
    };
    let sum: f64 = (0..self.n_paths).into_par_iter().map(&payoff_of).sum();
    let sum_sq: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|path| {
        let y = payoff_of(path);
        y * y
      })
      .sum();

    McEstimate {
      mean: discount * sum / self.n_paths as f64,
      std_err: discount * std_err_from_sums(sum, sum_sq, self.n_paths),
      n_samples: self.n_paths,
    }
  }
}
