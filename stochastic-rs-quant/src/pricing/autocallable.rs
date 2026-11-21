//! # Autocallable
//!
//! Auto-callable / phoenix / athena structured note. Observation grid
//! $t_1 < t_2 < \cdots < t_M = T$. Each observation date checks an upper
//! "autocall" barrier; if breached the note redeems early at par plus the
//! accrued coupon. Otherwise a coupon may be paid (phoenix) or skipped, and
//! at maturity the principal is at risk via a knock-in barrier on a worst-of
//! basket (here single underlying).
//!
//! Source:
//! - Bouzoubaa, M. & Osseiran, A. (2010), "Exotic Options and Hybrids", §11
//! - Kuklinski, J., Papaioannou, P. & Tyloo, K. (2016), "Pricing Weakly
//!   Model Dependent Barrier Products", arXiv:1608.00280
//! - Cibrario, F. et al. (2025), "Autocallable Options Pricing with
//!   Integration-Based Exponential Amplitude Loading", arXiv:2507.19039
//!
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::pricing::exotic::AutocallableNote;
use crate::pricing::exotic::assert_sigma;
use crate::pricing::mc_stats::par_estimate;

/// Barrier observation style for the knock-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KnockInStyle {
  /// Knock-in barrier checked only at maturity.
  EuropeanAtMaturity,
  /// Knock-in barrier checked on each observation date.
  Discrete,
  /// Knock-in barrier monitored continuously (approximated by a fine grid).
  Continuous,
}

/// Monte Carlo price of an [`AutocallableNote`] on Gbm paths; continuous knock-in monitoring
/// takes `steps_per_period` steps between observation dates, the other styles one.
#[derive(Debug, Clone, Copy)]
pub struct AutocallablePricer<S: SeedExt = Unseeded> {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
  pub n_paths: usize,
  pub steps_per_period: usize,
  /// Each call draws from a clone, so a `Deterministic` seed replays.
  pub seed: S,
}

impl AutocallablePricer {
  /// Panics unless `sigma` is finite and `>= 0`, `n_paths >= 1` and `steps_per_period >= 1`.
  pub fn new(sigma: f64, n_paths: usize, steps_per_period: usize) -> Self {
    assert_sigma(sigma);
    assert!(
      n_paths >= 1,
      "n_paths must satisfy `n_paths >= 1`, got n_paths = {n_paths:?}"
    );
    assert!(
      steps_per_period >= 1,
      "steps_per_period must satisfy `steps_per_period >= 1`, got steps_per_period = \
       {steps_per_period:?}"
    );
    Self {
      sigma,
      n_paths,
      steps_per_period,
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> AutocallablePricer<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> AutocallablePricer<S2> {
    AutocallablePricer {
      sigma: self.sigma,
      n_paths: self.n_paths,
      steps_per_period: self.steps_per_period,
      seed,
    }
  }

  /// Present value at spot `s`, the level the note's barriers are fractions of; its last
  /// observation date is the maturity. Panics on a note with no observation date.
  pub fn price(&self, note: &AutocallableNote, s: f64, r: f64, q: f64) -> McEstimate<f64> {
    let n_obs = note.observation_times.len();
    assert!(
      n_obs > 0,
      "observation_times must satisfy `observation_times.len() > 0`, got observation_times.len() \
       = 0"
    );
    let auto_b = note.autocall_barrier * s;
    let coup_b = note.coupon_barrier * s;
    let ki_b = note.knock_in_barrier * s;
    let times = &note.observation_times;
    let style = note.knock_in_style;
    let memory = note.memory;
    let coupon = note.coupon;
    let s0 = s;
    let notional = note.notional;
    let n_paths = self.n_paths;
    let steps_per_period = self.steps_per_period;

    // Worst-case normals required per path: every observation period uses
    // either one big step or `steps_per_period` substeps (continuous KI).
    let max_normals_per_path = match style {
      KnockInStyle::Continuous => n_obs * steps_per_period,
      _ => n_obs,
    };
    let mut all_z = vec![0.0_f64; n_paths * max_normals_per_path];
    let seed = self.seed.clone();
    SimdNormal::<f64>::new(0.0, 1.0)
      .seeded(&seed)
      .fill_slice(&mut all_z);

    let payoff_of = |p: usize| -> f64 {
      {
        let z_path = &all_z[p * max_normals_per_path..(p + 1) * max_normals_per_path];
        let mut z_idx = 0_usize;
        let mut s_prev = s0;
        let mut t_prev = 0.0;
        let mut pv: f64 = 0.0;
        let mut missed_coupons: f64 = 0.0;
        let mut autocalled = false;
        let mut breached_ki = false;
        let mut s_at_t: f64 = 0.0;

        for i in 0..n_obs {
          let t_i = times[i];
          let dt = t_i - t_prev;
          // sub-grid for continuous KI monitoring; otherwise one big step
          let sub_steps = match style {
            KnockInStyle::Continuous => steps_per_period,
            _ => 1,
          };
          let dt_sub = dt / sub_steps as f64;
          let drift = (r - q - 0.5 * self.sigma * self.sigma) * dt_sub;
          let vol = self.sigma * dt_sub.sqrt();
          let mut s_curr = s_prev;
          for _ in 0..sub_steps {
            let z = z_path[z_idx];
            z_idx += 1;
            s_curr *= (drift + vol * z).exp();
            if matches!(style, KnockInStyle::Continuous) && s_curr <= ki_b {
              breached_ki = true;
            }
          }
          if matches!(style, KnockInStyle::Discrete) && s_curr <= ki_b {
            breached_ki = true;
          }
          s_at_t = s_curr;
          // Autocall check
          if s_curr >= auto_b {
            let cf = notional * (1.0 + coupon + missed_coupons);
            pv += cf * (-r * t_i).exp();
            autocalled = true;
            break;
          }
          // Coupon barrier
          if s_curr >= coup_b {
            let cf = notional * (coupon + if memory { missed_coupons } else { 0.0 });
            pv += cf * (-r * t_i).exp();
            missed_coupons = 0.0;
          } else if memory {
            missed_coupons += coupon;
          }
          s_prev = s_curr;
          t_prev = t_i;
        }
        if !autocalled {
          if matches!(style, KnockInStyle::EuropeanAtMaturity) && s_at_t <= ki_b {
            breached_ki = true;
          }
          let t_mat = times[n_obs - 1];
          let principal = if breached_ki {
            notional * (s_at_t / s0)
          } else {
            notional
          };
          pv += principal * (-r * t_mat).exp();
        }
        pv
      }
    };
    par_estimate(n_paths, 1.0, payoff_of)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::array;

  use super::*;

  fn quarterly_obs(years: f64, n: usize) -> Array1<f64> {
    let dt = years / n as f64;
    Array1::from_iter((1..=n).map(|i| i as f64 * dt))
  }

  fn note(observation_times: Array1<f64>, autocall_barrier: f64, coupon: f64) -> AutocallableNote {
    AutocallableNote {
      notional: 100.0,
      observation_times,
      autocall_barrier,
      coupon_barrier: 0.7,
      coupon,
      knock_in_barrier: 0.6,
      knock_in_style: KnockInStyle::EuropeanAtMaturity,
      memory: false,
    }
  }

  /// An autocall barrier far below the spot redeems at the first date, `N (1 + c) e^{-r t_1}`.
  #[test]
  fn deep_in_autocall_redeems_first_period() {
    let note = note(array![0.25, 0.5, 0.75, 1.0], 0.5, 0.04);
    let price = AutocallablePricer::new(0.20, 20_000, 4)
      .price(&note, 100.0, 0.03, 0.0)
      .mean;
    let expected = 100.0 * 1.04 * (-0.03_f64 * 0.25).exp();
    let rel = (price - expected).abs() / expected;
    assert!(rel < 0.01, "price={price}, expected={expected}, rel={rel}");
  }

  /// Memory (athena) version is at least as expensive as phoenix
  /// (no-memory).
  #[test]
  fn memory_premium_over_phoenix() {
    let phoenix = AutocallableNote {
      coupon_barrier: 0.75,
      knock_in_barrier: 0.65,
      ..note(quarterly_obs(2.0, 8), 1.0, 0.025)
    };
    let athena = AutocallableNote {
      memory: true,
      ..phoenix.clone()
    };
    let pricer = AutocallablePricer::new(0.30, 30_000, 4);
    let price = |n: &AutocallableNote| pricer.price(n, 100.0, 0.03, 0.01).mean;
    assert!(price(&athena) >= price(&phoenix) - 1e-3);
  }

  /// Continuous knock-in monitoring should give a price below or equal to
  /// the European-at-maturity knock-in (more chances to breach).
  #[test]
  fn continuous_ki_below_european_ki() {
    let euro = AutocallableNote {
      knock_in_barrier: 0.65,
      ..note(quarterly_obs(1.0, 4), 1.05, 0.03)
    };
    let cont = AutocallableNote {
      knock_in_style: KnockInStyle::Continuous,
      ..euro.clone()
    };
    let pricer = AutocallablePricer::new(0.30, 50_000, 8);
    let price = |n: &AutocallableNote| pricer.price(n, 100.0, 0.03, 0.0).mean;
    assert!(price(&cont) <= price(&euro) + 1e-2);
  }

  /// At near-zero volatility, with spot strictly above the autocall barrier,
  /// all paths must redeem at the first observation date.
  #[test]
  fn zero_vol_spot_above_autocall() {
    let note = AutocallableNote {
      coupon_barrier: 0.5,
      knock_in_barrier: 0.4,
      ..note(array![0.5, 1.0], 0.95, 0.05)
    };
    let price = AutocallablePricer::new(1e-6, 5_000, 1)
      .price(&note, 120.0, 0.0, 0.0)
      .mean;
    let expected = 100.0 * 1.05;
    let rel = (price - expected).abs() / expected;
    assert!(rel < 0.005, "price={price}");
  }
}
