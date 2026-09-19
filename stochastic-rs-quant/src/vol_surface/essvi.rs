//! # eSSVI — extended SSVI slices
//!
//! $$
//! w_t(k) = \frac{\theta_t}{2}\left(1 + \rho_t\varphi_t k + \sqrt{(\varphi_t k + \rho_t)^2 + 1 - \rho_t^2}\right),\qquad \psi_t := \theta_t\varphi_t
//! $$
//!
//! The extended SSVI surface lets the correlation vary with maturity: every
//! slice carries its own `(θ_t, ρ_t, ψ_t)`. Calibration follows the anchored
//! scheme of Corbetta, Cohort, Laachir & Martini: a slice is pinned to its
//! data point closest to the money, `(k*, θ*)`, through `θ = θ* − ρ ψ k*`,
//! which leaves `(ρ, ψ)` free; the Gatheral–Jacquier butterfly bounds become
//! the cap `ψ ≤ min(ψ₊(ρ, k*, θ*), 4 / (1 + |ρ|))`, and the Hendriks–Martini
//! calendar-spread conditions add a floor on `ψ` and an upper bound that
//! keeps `φ = ψ / θ` non-increasing. Equal-variance slices preserve `ρψ`. Slices
//! are calibrated going forward in maturity, each by a bounded
//! one-dimensional search in `ρ` (coarse grid, then golden-section
//! refinement) with the best admissible `ψ` found by a golden-section search
//! inside its bounds — the note only states that one Brent search suffices;
//! the nesting is this crate's choice. Between slices the parameters
//! `(θ, ψ, ρψ)` are interpolated linearly in maturity, which the note shows
//! keeps the surface free of calendar-spread arbitrage.
//!
//! References: Corbetta, J., Cohort, P., Laachir, I. & Martini, C. (2019),
//! *Robust calibration and arbitrage-free interpolation of SSVI slices*,
//! arXiv:1804.04924; Hendriks, S. & Martini, C. (2019), *The extended SSVI
//! volatility surface*, Journal of Computational Finance 22(5); Gatheral, J.
//! & Jacquier, A. (2014), *Arbitrage-free SVI volatility surfaces*,
//! Quantitative Finance 14(1), 59–71. Calendar conditions include the
//! correction in Pasquazzi (2023), *eSSVI Surface Calibration*, Proposition
//! 4.14 and Section 2.1, arXiv:2304.02106.

mod calibrate;

pub use calibrate::calibrate_essvi;
pub use calibrate::try_calibrate_essvi;

use super::ssvi::SsviSlice;
use crate::traits::RealExt;

/// One eSSVI maturity slice `(θ, ρ, ψ)` with `ψ = θ φ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EssviSlice<T: RealExt> {
  /// Maturity in years.
  pub maturity: T,
  /// ATM total variance θ.
  pub theta: T,
  /// Correlation ρ ∈ (−1, 1).
  pub rho: T,
  /// Curvature ψ = θ φ > 0.
  pub psi: T,
}

impl<T: RealExt> EssviSlice<T> {
  pub fn new(maturity: T, theta: T, rho: T, psi: T) -> Self {
    Self {
      maturity,
      theta,
      rho,
      psi,
    }
  }

  /// Total variance `w(k)` of the slice.
  pub fn total_variance(&self, k: T) -> T {
    if self.theta <= T::zero() {
      return T::zero();
    }
    let one = T::one();
    let half = T::from_f64_fast(0.5);
    let phi = self.psi / self.theta;
    let u = phi * k + self.rho;
    half * self.theta * (one + self.rho * phi * k + (u * u + one - self.rho * self.rho).sqrt())
  }

  /// Implied volatility `√(w(k) / t)`.
  pub fn implied_vol(&self, k: T) -> T {
    let w = self.total_variance(k);
    if w > T::zero() && self.maturity > T::zero() {
      (w / self.maturity).sqrt()
    } else {
      T::nan()
    }
  }

  /// Gatheral–Jacquier sufficient no-butterfly conditions
  /// `θφ ≤ 4/(1+|ρ|)` and `θφ² ≤ 4/(1+|ρ|)`, i.e. `ψ ≤ 4/(1+|ρ|)` and
  /// `ψ² ≤ 4θ/(1+|ρ|)`.
  pub fn is_butterfly_free(&self) -> bool {
    let cap = T::from_f64_fast(4.0) / (T::one() + self.rho.abs());
    let tol = T::from_f64_fast(1e-12);
    self.psi <= cap + tol && self.psi * self.psi <= cap * self.theta + tol
  }
}

/// Hendriks–Martini conditions for two consecutive slices to be free of
/// calendar-spread arbitrage.
fn calendar_free_pair<T: RealExt>(earlier: &EssviSlice<T>, later: &EssviSlice<T>) -> bool {
  let tol = T::from_f64_fast(1e-12);
  if later.theta + tol < earlier.theta || later.psi + tol < earlier.psi {
    return false;
  }
  let d_psi = later.psi - earlier.psi;
  let d_skew = later.rho * later.psi - earlier.rho * earlier.psi;
  if d_skew.abs() > d_psi + tol {
    return false;
  }
  if later.psi * earlier.theta <= earlier.psi * later.theta {
    return true;
  }
  let bound = (later.theta - earlier.theta)
    * (later.psi * later.psi / later.theta - earlier.psi * earlier.psi / earlier.theta);
  d_skew * d_skew <= bound + tol * tol
}

/// Extended SSVI surface: calibrated slices plus the arbitrage-free linear
/// interpolation of `(θ, ψ, ρψ)` between them.
#[derive(Clone, Debug)]
pub struct EssviSurface<T: RealExt> {
  /// Slices in ascending maturity.
  pub slices: Vec<EssviSlice<T>>,
}

impl<T: RealExt> EssviSurface<T> {
  pub fn new(slices: Vec<EssviSlice<T>>) -> Self {
    assert!(
      !slices.is_empty(),
      "an eSSVI surface needs at least one slice"
    );
    assert!(
      slices.windows(2).all(|w| w[0].maturity < w[1].maturity),
      "slices must have increasing maturities"
    );
    Self { slices }
  }

  /// Slice parameters at `t`: linear in `(θ, ψ, ρψ)` between neighbouring
  /// slices, from the origin `(0, 0, 0)` before the first one, flat after
  /// the last.
  pub fn slice_at(&self, t: T) -> EssviSlice<T> {
    let last = self.slices.last().expect("non-empty");
    if t >= last.maturity {
      return EssviSlice::new(t, last.theta, last.rho, last.psi);
    }
    let zero = EssviSlice::new(T::zero(), T::zero(), T::zero(), T::zero());
    let (lo, hi) = match self.slices.iter().position(|s| s.maturity >= t) {
      Some(0) => (zero, self.slices[0]),
      Some(i) => (self.slices[i - 1], self.slices[i]),
      None => unreachable!("t is below the last maturity"),
    };
    let span = hi.maturity - lo.maturity;
    let a = if span > T::zero() {
      (t - lo.maturity) / span
    } else {
      T::one()
    };
    let theta = lo.theta + a * (hi.theta - lo.theta);
    let psi = lo.psi + a * (hi.psi - lo.psi);
    let rho_psi = lo.rho * lo.psi + a * (hi.rho * hi.psi - lo.rho * lo.psi);
    let rho = if psi > T::zero() {
      rho_psi / psi
    } else {
      hi.rho
    };
    EssviSlice::new(t, theta, rho, psi)
  }

  /// Total variance at `(k, t)`.
  pub fn total_variance(&self, k: T, t: T) -> T {
    self.slice_at(t).total_variance(k)
  }

  /// Implied volatility at `(k, t)`.
  pub fn implied_vol(&self, k: T, t: T) -> T {
    self.slice_at(t).implied_vol(k)
  }

  /// Every slice satisfies the Gatheral–Jacquier butterfly bounds.
  pub fn is_butterfly_free(&self) -> bool {
    self.slices.iter().all(EssviSlice::is_butterfly_free)
  }

  /// Every consecutive pair satisfies the Hendriks–Martini calendar-spread
  /// conditions.
  pub fn is_calendar_spread_free(&self) -> bool {
    self
      .slices
      .windows(2)
      .all(|w| calendar_free_pair(&w[0], &w[1]))
  }
}

#[cfg(test)]
mod tests;
