//! Monte Carlo barrier pricing on Gbm paths monitored at every step.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::mc::McEstimate;

use super::BarrierType;
use crate::OptionType;
use crate::pricing::digital::not_positive;
use crate::pricing::exotic::BarrierOption;
use crate::pricing::exotic::assert_sigma;
use crate::pricing::mc_stats::par_estimate;

/// Monte Carlo price of the [`BarrierOption`] the closed form prices, rebate included, on
/// `n_paths` Gbm paths monitored at each of `n_steps` steps and at the spot.
#[derive(Debug, Clone, Copy)]
pub struct McBarrierPricer<S: SeedExt = Unseeded> {
  /// Annualised Black–Scholes volatility.
  sigma: f64,
  n_paths: usize,
  n_steps: usize,
  /// Each call reads a clone's next seed as its base, and each path derives its stream from that
  /// base and its index, so a `Deterministic` seed replays whatever rayon's schedule.
  pub seed: S,
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
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> McBarrierPricer<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> McBarrierPricer<S2> {
    McBarrierPricer {
      sigma: self.sigma,
      n_paths: self.n_paths,
      n_steps: self.n_steps,
      seed,
    }
  }

  /// Discounted estimate under the drift `(r − q − σ²/2) dt`, a knocked-out rebate at any strike;
  /// NaN for a NaN term, a barrier not finite, `tau < 0`, or a spot or strike `≤ 0` before expiry.
  pub fn price(&self, option: &BarrierOption, s: f64, r: f64, q: f64, tau: f64) -> McEstimate<f64> {
    let BarrierOption {
      strike: k,
      barrier: h,
      rebate,
      barrier_type,
      option_type,
    } = *option;
    let up = matches!(barrier_type, BarrierType::UpAndIn | BarrierType::UpAndOut);
    let knock_in = matches!(barrier_type, BarrierType::UpAndIn | BarrierType::DownAndIn);
    let breached = |x: f64| if up { x >= h } else { x <= h };
    let undefined = [s, r, q, tau, k, rebate].iter().any(|x| x.is_nan()) || !h.is_finite();
    let unread = tau > 0.0 && (s <= 0.0 || (not_positive(&[k]) && (knock_in || !breached(s))));
    if undefined || tau < 0.0 || unread {
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
    let base_seed = self.seed.clone().next_seed();

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
    par_estimate(self.n_paths, discount, payoff_of)
  }
}
