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
//! calendar-spread conditions between consecutive slices (`θ` and `ψ`
//! non-decreasing, `|Δ(ρψ) / Δψ| ≤ 1`) become the floor `ψ ≥ ψ₋(ρ)` plus the
//! `θ > θ_prev` side condition through `ψ̂ = (θ* − θ_prev) / (ρ k*)`. Slices
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
//! Quantitative Finance 14(1), 59–71.

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
  if d_psi <= tol {
    return (later.rho * later.psi - earlier.rho * earlier.psi).abs() <= tol;
  }
  ((later.rho * later.psi - earlier.rho * earlier.psi) / d_psi).abs() <= T::one() + tol
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

/// Golden-section minimisation of a unimodal function on `[lo, hi]`.
fn golden_section(f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64, iters: usize) -> f64 {
  let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
  let mut c = hi - inv_phi * (hi - lo);
  let mut d = lo + inv_phi * (hi - lo);
  let (mut fc, mut fd) = (f(c), f(d));
  for _ in 0..iters {
    if fc < fd {
      hi = d;
      d = c;
      fd = fc;
      c = hi - inv_phi * (hi - lo);
      fc = f(c);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + inv_phi * (hi - lo);
      fd = f(d);
    }
  }
  0.5 * (lo + hi)
}

/// Anchored slice data in `f64`.
struct AnchoredSlice {
  ks: Vec<f64>,
  ws: Vec<f64>,
  k_star: f64,
  theta_star: f64,
}

/// Admissible `ψ` interval for `ρ` on an anchored slice, given the previous
/// slice; `None` when the bounds cross.
fn psi_bounds(
  rho: f64,
  slice: &AnchoredSlice,
  previous: Option<&EssviSlice<f64>>,
) -> Option<(f64, f64)> {
  let abs = 1.0 + rho.abs();
  let (k, theta) = (slice.k_star, slice.theta_star);
  let psi_plus =
    -2.0 * rho * k / abs + (4.0 * rho * rho * k * k / (abs * abs) + 4.0 * theta / abs).sqrt();
  let mut hi = psi_plus.min(4.0 / abs);
  let mut lo = 1e-10_f64;
  let rk = rho * k;
  if rk > 0.0 {
    hi = hi.min(theta / rk * (1.0 - 1e-9));
  }
  if let Some(prev) = previous {
    let psi_minus = ((prev.psi - prev.rho * prev.psi) / (1.0 - rho))
      .max((prev.psi + prev.rho * prev.psi) / (1.0 + rho));
    lo = lo.max(psi_minus);
    if rk.abs() > 0.0 {
      let psi_hat = (theta - prev.theta) / rk;
      if rk > 0.0 {
        hi = hi.min(psi_hat);
      } else {
        lo = lo.max(psi_hat);
      }
    } else if theta <= prev.theta {
      return None;
    }
  }
  (lo < hi).then_some((lo, hi))
}

fn slice_sse(rho: f64, psi: f64, slice: &AnchoredSlice) -> f64 {
  let theta = slice.theta_star - rho * psi * slice.k_star;
  let model = EssviSlice::new(1.0, theta, rho, psi);
  slice
    .ks
    .iter()
    .zip(&slice.ws)
    .map(|(&k, &w)| (model.total_variance(k) - w).powi(2))
    .sum()
}

/// Best admissible `(ψ, sse)` for a given `ρ`.
fn best_psi(
  rho: f64,
  slice: &AnchoredSlice,
  previous: Option<&EssviSlice<f64>>,
) -> Option<(f64, f64)> {
  let (lo, hi) = psi_bounds(rho, slice, previous)?;
  let f = |psi: f64| slice_sse(rho, psi, slice);
  let psi = golden_section(&f, lo, hi, 80);
  Some((psi, f(psi)))
}

/// Calibrates one anchored slice going forward from `previous`.
fn calibrate_slice(
  slice: &AnchoredSlice,
  previous: Option<&EssviSlice<f64>>,
  maturity: f64,
) -> EssviSlice<f64> {
  let objective = |rho: f64| best_psi(rho, slice, previous).map_or(f64::INFINITY, |(_, sse)| sse);
  let grid: Vec<f64> = (0..81).map(|i| -0.99 + 1.98 * i as f64 / 80.0).collect();
  let (mut best_rho, mut best_sse) = (0.0_f64, f64::INFINITY);
  for &rho in &grid {
    let sse = objective(rho);
    if sse < best_sse {
      best_sse = sse;
      best_rho = rho;
    }
  }
  let step = 1.98 / 80.0;
  let refined = golden_section(
    &objective,
    (best_rho - step).max(-0.999),
    (best_rho + step).min(0.999),
    60,
  );
  let rho = if objective(refined) <= best_sse {
    refined
  } else {
    best_rho
  };
  let (psi, _) = best_psi(rho, slice, previous).expect("the grid optimum is admissible");
  EssviSlice::new(
    maturity,
    slice.theta_star - rho * psi * slice.k_star,
    rho,
    psi,
  )
}

/// Calibrates eSSVI slices to `slices` (ascending `maturities`, one per
/// slice) going forward in maturity, so that every slice is butterfly-free
/// and every consecutive pair calendar-spread-free by construction. Each
/// slice is anchored at its data point closest to the money.
pub fn calibrate_essvi<T: RealExt>(slices: &[SsviSlice<T>], maturities: &[T]) -> EssviSurface<T> {
  assert_eq!(slices.len(), maturities.len(), "one maturity per slice");
  assert!(
    !slices.is_empty(),
    "eSSVI calibration needs at least one slice"
  );
  let mut out: Vec<EssviSlice<f64>> = Vec::with_capacity(slices.len());
  for (slice, &maturity) in slices.iter().zip(maturities) {
    let ks: Vec<f64> = slice
      .log_moneyness
      .iter()
      .map(|k| k.to_f64().unwrap_or(0.0))
      .collect();
    let ws: Vec<f64> = slice
      .total_variance
      .iter()
      .map(|w| w.to_f64().unwrap_or(0.0))
      .collect();
    assert!(
      ks.len() >= 2 && ks.len() == ws.len(),
      "a slice needs at least two quotes"
    );
    let anchor = (0..ks.len())
      .min_by(|&i, &j| {
        ks[i]
          .abs()
          .partial_cmp(&ks[j].abs())
          .expect("finite log-moneyness")
      })
      .expect("non-empty");
    let anchored = AnchoredSlice {
      k_star: ks[anchor],
      theta_star: ws[anchor],
      ks,
      ws,
    };
    let previous = out.last();
    let calibrated = calibrate_slice(&anchored, previous, maturity.to_f64().unwrap_or(0.0));
    out.push(calibrated);
  }
  EssviSurface::new(
    out
      .into_iter()
      .map(|s| {
        EssviSlice::new(
          T::from_f64_fast(s.maturity),
          T::from_f64_fast(s.theta),
          T::from_f64_fast(s.rho),
          T::from_f64_fast(s.psi),
        )
      })
      .collect(),
  )
}

#[cfg(test)]
mod tests;
