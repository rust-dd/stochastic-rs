//! Monte Carlo pricing under a pre-calibrated leverage surface.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

use super::HestonSlvParams;
use super::LeverageSurface;
use crate::mc::McEstimate;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Largest absolute deviation between a queried rate and the rate the
/// leverage surface was calibrated at that still counts as a match. Rates
/// live on the order of $10^{-2}$, so this is "equal up to float round-trip
/// noise" rather than a tolerance band.
const RATE_MATCH_TOL: f64 = 1e-12;

/// Heston-SLV Monte Carlo pricer with a pre-calibrated leverage surface.
///
/// The leverage function is not rate-agnostic: [`calibrate_leverage`] evolves
/// its particle cloud under the drift $r - q$, and a Dupire-sourced target
/// surface carries $(r, q)$ in its own numerator. A surface calibrated at one
/// rate does **not** reproduce the market at another, so [`ModelPricer`] queries
/// are checked against the calibration rates rather than silently substituted.
///
/// Two constructors, two capabilities:
///
/// - [`HestonSlvPricer::new`] — the surface came out of a calibration at a
///   known `(r, q)`. Pricing at any other rate panics naming both pairs.
/// - [`HestonSlvPricer::unanchored`] — the surface is a hand-supplied
///   $L(S,t)$ with no rate provenance (a pure-Heston $L \equiv 1$, say). The
///   SDE is then fully specified by the query, so `r` and `q` drive drift and
///   discounting directly and every rate is accepted.
///
/// # Spot and maturity are bounded by the surface, not by the rate anchor
///
/// The `s` and `tau` of a query are *honoured* — they drive the simulation —
/// but [`LeverageSurface`] is indexed by absolute spot out to a finite
/// horizon, and past either edge
/// [`interpolate`](LeverageSurface::interpolate) holds the boundary value
/// flat. Far enough out every path spends its whole life on that hold, and
/// the price stops depending on the calibrated interior at all: two surfaces
/// differing by a factor of three in the middle and agreeing on their edges
/// return **bit-identical** prices there, while at the calibration spot they
/// disagree by 95 %. The clamped answer is not marked in any way — it is
/// finite, positive and inside the no-arbitrage band — so it is `NaN`-gated
/// instead, on [`LeverageSurface::covers`].
///
/// This is a bound and not a rate-style equality check because a strike
/// ladder around the calibration spot is what the type is *for*. Inside the
/// grid the same flat hold only touches the wings of the terminal
/// distribution, which no finite grid avoids and which
/// [`calibrate_leverage`] applies to its own particles for the same reason.
///
/// Unlike the rate anchor this bound applies to **both** constructors: the
/// extent belongs to the surface, whereas `(r, q)` provenance is something
/// only a calibration can supply.
///
/// [`calibrate_leverage`]: super::calibrate_leverage
#[derive(Debug, Clone)]
pub struct HestonSlvPricer {
  /// Model parameters.
  pub params: HestonSlvParams,
  /// Calibrated leverage surface.
  pub leverage: LeverageSurface,
  /// The `(r, q)` pair [`leverage`](Self::leverage) was calibrated against,
  /// or `None` when the surface carries no rate provenance. Not a pricing
  /// input — the rates that price come from the [`ModelPricer`] query; this
  /// is the guard those queries are checked against.
  pub calibration_rates: Option<(f64, f64)>,
  /// Number of MC paths.
  pub n_paths: usize,
  /// Time-discretization steps per year.
  pub steps_per_year: usize,
  /// RNG seed.
  pub seed: u64,
}

impl HestonSlvPricer {
  /// Pricer for a `leverage` surface calibrated at `(r, q)` — the pair passed
  /// to [`calibrate_leverage`](super::calibrate_leverage), or `dupire.r` /
  /// `dupire.q` for [`calibrate_from_dupire`](super::calibrate_from_dupire).
  /// Pricing at any other rate panics.
  pub fn new(params: HestonSlvParams, leverage: LeverageSurface, r: f64, q: f64) -> Self {
    Self {
      params,
      leverage,
      calibration_rates: Some((r, q)),
      n_paths: 100_000,
      steps_per_year: 200,
      seed: 42,
    }
  }

  /// Pricer for a `leverage` surface with no rate provenance: the query's
  /// `r` and `q` drive drift and discounting, and no rate is rejected.
  ///
  /// Use this only for a surface that was *not* produced by calibrating
  /// against market data at a particular rate. A calibrated surface priced
  /// this way returns a confident number that no longer reprices the surface
  /// it was fitted to.
  pub fn unanchored(params: HestonSlvParams, leverage: LeverageSurface) -> Self {
    Self {
      params,
      leverage,
      calibration_rates: None,
      n_paths: 100_000,
      steps_per_year: 200,
      seed: 42,
    }
  }

  pub fn with_paths(mut self, n: usize) -> Self {
    self.n_paths = n;
    self
  }

  pub fn with_steps_per_year(mut self, n: usize) -> Self {
    self.steps_per_year = n;
    self
  }

  pub fn with_seed(mut self, seed: u64) -> Self {
    self.seed = seed;
    self
  }

  /// Panic unless `(r, q)` is the pair the leverage surface was calibrated
  /// at. A no-op for an [`unanchored`](Self::unanchored) pricer.
  fn check_rates(&self, r: f64, q: f64) {
    let Some((r0, q0)) = self.calibration_rates else {
      return;
    };
    if (r - r0).abs() > RATE_MATCH_TOL || (q - q0).abs() > RATE_MATCH_TOL {
      panic!(
        "HestonSlvPricer: leverage surface calibrated at r={r0}, q={q0} but queried at r={r}, q={q}. \
         L(S,t) is rate-dependent, so pricing at another rate would silently stop reproducing the \
         calibrated surface. Recalibrate at the query rates, or build the pricer with \
         HestonSlvPricer::unanchored if the surface has no rate provenance."
      );
    }
  }

  /// Call-price estimate with its Monte Carlo error bar, under the same
  /// rate-anchoring panic and leverage-extent `NaN` rules as
  /// [`ModelPricer::price_call`]; a query outside the surface's extent has
  /// no paths to estimate from, so every field of the estimate is poisoned.
  pub fn price_call_estimate(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> McEstimate<f64> {
    self.check_rates(r, q);
    if !self.leverage.covers(s, tau) {
      return McEstimate {
        mean: f64::NAN,
        std_err: f64::NAN,
        n_samples: 0,
      };
    }
    let n_steps = ((tau * self.steps_per_year as f64).round() as usize).max(1);
    let dt = tau / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let sigma_mixed = self.params.sigma_mixed();
    let rho_bar = (1.0 - self.params.rho * self.params.rho).sqrt();

    let normals = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(self.seed));
    let mut payoff_sum = 0.0;
    let mut payoff_sq_sum = 0.0;

    for _ in 0..self.n_paths {
      let mut x = s.ln();
      let mut v = self.params.v0;

      for step in 0..n_steps {
        let t = (step as f64 + 1.0) * dt;
        let dw_v = normals.sample_fast() * sqrt_dt;
        let dw_ind = normals.sample_fast() * sqrt_dt;
        let dw_x = self.params.rho * dw_v + rho_bar * dw_ind;

        let v_pos = v.max(0.0);
        let sqrt_v = v_pos.sqrt();
        let s_curr = x.exp();

        let l = self.leverage.interpolate(s_curr, t);

        v =
          (v + self.params.kappa * (self.params.theta - v_pos) * dt + sigma_mixed * sqrt_v * dw_v)
            .max(0.0);

        let drift = (r - q) - 0.5 * l * l * v_pos;
        x += drift * dt + l * sqrt_v * dw_x;
      }

      let s_t = x.exp();
      let payoff = (s_t - k).max(0.0);
      payoff_sum += payoff;
      payoff_sq_sum += payoff * payoff;
    }

    let discount = (-r * tau).exp();
    McEstimate {
      mean: discount * payoff_sum / self.n_paths as f64,
      std_err: discount * std_err_from_sums(payoff_sum, payoff_sq_sum, self.n_paths),
      n_samples: self.n_paths,
    }
  }
}

impl ModelPricer for HestonSlvPricer {
  /// Returns [`f64::NAN`] when `(s, tau)` falls outside the leverage
  /// surface's own extent — case 2 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails), the same one
  /// `CarrMadanPricer::price_call` reports for a strike outside its FFT grid.
  /// Test with [`LeverageSurface::covers`] before pricing, or recalibrate
  /// over a grid that spans the query. See [`HestonSlvPricer`] for what the
  /// clamped answer would otherwise have been.
  ///
  /// The rate check runs first, so a query that is wrong on both counts
  /// panics rather than returning `NaN`: a rate the pricer never agreed to
  /// price at is a wiring error, and it should not be reported as a surface
  /// edge.
  ///
  /// # Panics
  ///
  /// When the pricer came from [`HestonSlvPricer::new`] and `(r, q)` differs
  /// from the pair the leverage surface was calibrated at — see
  /// [`HestonSlvPricer`] for why substituting is not a valid reprojection.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_estimate(s, k, r, q, tau).mean
  }
}

/// European vanilla call: the particle simulation averages $(S_T-K)^+$
/// under a log-spot drifted at $r-q$, so the default forward applies.
impl VanillaEuropeanCall for HestonSlvPricer {}

#[cfg(test)]
mod tests;
