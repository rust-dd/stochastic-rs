//! Guyon–Henry-Labordère particle calibration of the leverage `L(S, t)` by Gyöngy's theorem:
//!
//! $$
//! L^2(t,K) = \frac{\sigma_{\text{LV}}^2(t,K)}{\mathbb{E}[V_t \mid S_t = K]}
//! $$
//!
//! $$
//! \hat{\mathbb{E}}[V_t \mid S_t = K]
//!   = \frac{\sum_i V^i_t\,\delta_N(S^i_t - K)}{\sum_i \delta_N(S^i_t - K)},
//! \qquad
//! \delta_N(x) = \frac{e^{-\frac12 (x / h_N(t))^2}}{h_N(t)\sqrt{2\pi}},
//! $$
//!
//! $$
//! h_N(t) = \kappa\, S_0\, \sigma_{\text{LV}}(t, S_0)\,
//!   \sqrt{\max(t, t_{\min})}\; N^{-1/5},
//! \qquad \kappa = 1.5,\ t_{\min} = 0.25,
//! $$
//!
//! The cloud starts at `(S₀, v₀)` and steps under `L(t_k, ·)`, the variance by Euler under full
//! truncation, the recursion [`super::HestonSlvPricer`] and the `HestonSlv` process run.
//! Reference: Guyon & Henry-Labordère (2012), Risk 25(1), 88–93
//! Reference: Guyon & Henry-Labordère (2013), Nonlinear Option Pricing, Chapman & Hall, ch. 11
//! Reference: Cozma et al. (2017), arXiv:1701.06001, §3.4, Algorithm 1
//! Reference: Reisinger & Tsianni (2025), arXiv:2504.14343
//! Reference: Lord et al. (2010), Quant. Finance 10(2), 177–194, DOI 10.1080/14697680802392496

use std::cmp::Ordering;

use ndarray::Array1;
use ndarray::Array2;
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::traits::Grid2D;

use super::HestonSlvParams;
use super::LeverageSurface;
use crate::QuantError;
use crate::total_order::ascending;

/// Floor of the conditional-variance estimate before the square root.
pub(super) const CONDITIONAL_VARIANCE_FLOOR: f64 = 1e-8;

/// The leverage is clamped to this band so a degenerate local volatility or
/// an almost-empty kernel window cannot stall or blow up the cloud.
const LEVERAGE_BAND: (f64, f64) = (0.01, 10.0);

/// Kernel mass below which a particle is left out of a node's window,
/// Cozma et al.'s $\varepsilon$.
const WINDOW_MASS: f64 = 1e-5;

/// The particle method's tuning; the defaults are Guyon–Henry-Labordère's bandwidth constants and a
/// cloud large enough for a production surface.
#[derive(Clone, Debug)]
pub struct ParticleMethod<S: SeedExt = Unseeded> {
  /// Particles in the cloud, $N$.
  pub n_particles: usize,
  /// Euler steps per year; each snapshot maturity is a grid point, so an interval `Δt` between two
  /// takes `ceil(Δt · steps_per_year)` steps.
  pub steps_per_year: usize,
  /// The cloud's Gaussian stream draws from a clone, so a `Deterministic` seed replays.
  pub seed: S,
  /// $\kappa$ in the bandwidth, `1.5` in the reference.
  pub bandwidth_factor: f64,
  /// `t_min` of the bandwidth, `0.25` in the reference: below it the bandwidth stops shrinking with
  /// the cloud, which is still concentrated at the spot.
  pub bandwidth_t_min: f64,
}

impl Default for ParticleMethod {
  fn default() -> Self {
    Self {
      n_particles: 100_000,
      steps_per_year: 200,
      seed: Unseeded,
      bandwidth_factor: 1.5,
      bandwidth_t_min: 0.25,
    }
  }
}

impl<S: SeedExt> ParticleMethod<S> {
  pub fn with_particles(mut self, n_particles: usize) -> Self {
    self.n_particles = n_particles;
    self
  }

  pub fn with_steps_per_year(mut self, steps_per_year: usize) -> Self {
    self.steps_per_year = steps_per_year;
    self
  }

  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> ParticleMethod<S2> {
    ParticleMethod {
      n_particles: self.n_particles,
      steps_per_year: self.steps_per_year,
      seed,
      bandwidth_factor: self.bandwidth_factor,
      bandwidth_t_min: self.bandwidth_t_min,
    }
  }

  pub fn with_bandwidth_factor(mut self, factor: f64) -> Self {
    self.bandwidth_factor = factor;
    self
  }

  pub fn with_bandwidth_t_min(mut self, t_min: f64) -> Self {
    self.bandwidth_t_min = t_min;
    self
  }

  fn validate(&self) -> Result<(), QuantError> {
    if self.n_particles < 2 {
      return Err(QuantError::invalid(
        "n_particles",
        format!("must be at least 2, got {}", self.n_particles),
      ));
    }
    if self.steps_per_year == 0 {
      return Err(QuantError::invalid(
        "steps_per_year",
        "must be positive, got 0",
      ));
    }
    if !(self.bandwidth_factor.is_finite() && self.bandwidth_factor > 0.0) {
      return Err(QuantError::invalid(
        "bandwidth_factor",
        format!("must be finite and positive, got {}", self.bandwidth_factor),
      ));
    }
    if !(self.bandwidth_t_min.is_finite() && self.bandwidth_t_min >= 0.0) {
      return Err(QuantError::invalid(
        "bandwidth_t_min",
        format!(
          "must be finite and non-negative, got {}",
          self.bandwidth_t_min
        ),
      ));
    }
    Ok(())
  }
}

/// The leverage surface and the cloud's spots at each requested maturity, in the given order: the
/// particles the surface was read off, so the target grid prices without a second simulation.
#[derive(Clone, Debug)]
pub struct ParticleCalibration {
  pub leverage: LeverageSurface,
  pub snapshots: Vec<Array1<f64>>,
}

/// `L` on `local_vol`'s nodes up to the last maturity, at `(r, q)`; `InsufficientData` with no
/// maturity, `InvalidInput` for any input off its domain (`s0 ≤ 0`, `|ρ| > 1`, `< 2` particles, …).
pub fn calibrate_leverage(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  local_vol: &Grid2D<f64>,
  snapshot_maturities: &[f64],
  method: &ParticleMethod<impl SeedExt>,
) -> Result<ParticleCalibration, QuantError> {
  validate(params, s0, r, q, local_vol, snapshot_maturities)?;
  method.validate()?;

  let times = time_grid(snapshot_maturities, method.steps_per_year);
  let spots = local_vol.xs().to_vec();
  let n_eval = spots.len();
  let n = method.n_particles;
  let (kappa, theta, rho, v0) = (
    params.heston.kappa,
    params.heston.theta,
    params.heston.rho,
    params.heston.v0.max(0.0),
  );
  let sigma_mixed = params.sigma_mixed();
  let rho_bar = (1.0 - rho * rho).sqrt();
  let carry = r - q;
  let bandwidth_scale = method.bandwidth_factor * s0 * (n as f64).powf(-0.2);

  let mut leverage = Array2::<f64>::zeros((times.len(), n_eval));
  let sqrt_v0 = v0.max(CONDITIONAL_VARIANCE_FLOOR).sqrt();
  for (i, &node) in spots.iter().enumerate() {
    leverage[[0, i]] = clamp_leverage(local_vol.eval(0.0, node) / sqrt_v0);
  }

  let mut normals = SimdNormal::<f64>::new(0.0, 1.0).seeded(&method.seed.clone());
  let mut x = vec![s0.ln(); n];
  let mut v = vec![v0; n];
  let mut z_v = vec![0.0; n];
  let mut z_perp = vec![0.0; n];
  let mut sorted = vec![(0.0_f64, 0.0_f64); n];
  let mut conditional = vec![f64::NAN; n_eval];
  let mut snapshots = Vec::with_capacity(snapshot_maturities.len());
  let mut next_snapshot = 0;

  for k in 0..times.len() - 1 {
    let t_next = times[k + 1];
    let dt = t_next - times[k];
    let sqrt_dt = dt.sqrt();
    normals.fill_slice(&mut z_v);
    normals.fill_slice(&mut z_perp);

    let row = leverage.row(k);
    let row = row.as_slice().expect("a leverage row is contiguous");
    for p in 0..n {
      let l = interpolate_row(&spots, row, x[p].exp());
      // Full truncation, as `HestonSlv` steps: `v[p]` keeps its sign and every read is `v[p]⁺`.
      let vp = v[p].max(0.0);
      let sqrt_v = vp.sqrt();
      let dw_v = z_v[p] * sqrt_dt;
      let dw_s = rho * dw_v + rho_bar * z_perp[p] * sqrt_dt;
      x[p] += (carry - 0.5 * l * l * vp) * dt + l * sqrt_v * dw_s;
      v[p] = v[p] + kappa * (theta - vp) * dt + sigma_mixed * sqrt_v * dw_v;
    }

    if next_snapshot < snapshot_maturities.len() && t_next == snapshot_maturities[next_snapshot] {
      snapshots.push(Array1::from_iter(x.iter().map(|xp| xp.exp())));
      next_snapshot += 1;
    }

    for (slot, (xp, vp)) in sorted.iter_mut().zip(x.iter().zip(v.iter())) {
      *slot = (xp.exp(), vp.max(0.0));
    }
    sorted.sort_unstable_by(|a, b| ascending(&a.0, &b.0));
    let h =
      bandwidth_scale * local_vol.eval(t_next, s0) * t_next.max(method.bandwidth_t_min).sqrt();
    let half_width = window_half_width(h);
    conditional
      .par_iter_mut()
      .zip(spots.par_iter())
      .for_each(|(slot, &node)| *slot = conditional_variance(&sorted, node, h, half_width));
    let cloud_mean = v.iter().map(|vp| vp.max(0.0)).sum::<f64>() / n as f64;
    fill_gaps(&mut conditional, cloud_mean);

    for (i, &node) in spots.iter().enumerate() {
      let cond = conditional[i].max(CONDITIONAL_VARIANCE_FLOOR);
      leverage[[k + 1, i]] = clamp_leverage(local_vol.eval(t_next, node) / cond.sqrt());
    }
  }

  Ok(ParticleCalibration {
    leverage: LeverageSurface::new(Array1::from_vec(spots), Array1::from_vec(times), leverage),
    snapshots,
  })
}

pub(super) fn validate(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  local_vol: &Grid2D<f64>,
  snapshot_maturities: &[f64],
) -> Result<(), QuantError> {
  if !(s0.is_finite() && s0 > 0.0) {
    return Err(QuantError::invalid(
      "spot",
      format!("must be finite and positive, got {s0}"),
    ));
  }
  if !(r.is_finite() && q.is_finite()) {
    return Err(QuantError::invalid(
      "rates",
      format!("must be finite, got r = {r}, q = {q}"),
    ));
  }
  if !(params.heston.v0.is_finite() && params.heston.v0 >= 0.0) {
    return Err(QuantError::invalid(
      "v0",
      format!(
        "must be a finite non-negative variance, got {}",
        params.heston.v0
      ),
    ));
  }
  if !(params.eta.is_finite() && params.eta >= 0.0) {
    return Err(QuantError::invalid(
      "mixing fraction eta",
      format!("must be finite and non-negative, got {}", params.eta),
    ));
  }
  if !(params.heston.rho.is_finite() && params.heston.rho.abs() <= 1.0) {
    return Err(QuantError::invalid(
      "rho",
      format!(
        "must be a correlation in [-1, 1], got {}",
        params.heston.rho
      ),
    ));
  }
  let (kappa, theta, sigma) = (
    params.heston.kappa,
    params.heston.theta,
    params.heston.sigma,
  );
  if !(kappa.is_finite() && theta.is_finite() && sigma.is_finite()) {
    return Err(QuantError::invalid(
      "Heston parameters",
      format!("must be finite, got kappa = {kappa}, theta = {theta}, sigma = {sigma}"),
    ));
  }
  if let Some(((i, j), v)) = local_vol
    .values()
    .indexed_iter()
    .find(|(_, v)| !(v.is_finite() && **v >= 0.0))
  {
    return Err(QuantError::invalid(
      "local volatility",
      format!(
        "must be finite and non-negative everywhere, got {v} at [{i}, {j}]; clean the Dupire \
         surface first"
      ),
    ));
  }
  if snapshot_maturities.is_empty() {
    return Err(QuantError::InsufficientData {
      what: "maturities in the snapshot list",
      needed: 1,
      got: 0,
    });
  }
  if let Some((j, t)) = snapshot_maturities
    .iter()
    .enumerate()
    .find(|(_, t)| !(t.is_finite() && **t > 0.0))
  {
    return Err(QuantError::invalid(
      "snapshot maturities",
      format!("maturity {j} must be finite and positive, got {t}"),
    ));
  }
  if let Some(j) = snapshot_maturities.windows(2).position(|w| w[0] >= w[1]) {
    return Err(QuantError::invalid(
      "snapshot maturities",
      format!(
        "must be strictly ascending, got maturity {j} = {} before {}",
        snapshot_maturities[j],
        snapshot_maturities[j + 1]
      ),
    ));
  }
  Ok(())
}

/// `0 = t_0 < … < t_m`, every maturity a node, each interval split into
/// `ceil(Δ · steps_per_year)` equal steps.
pub(super) fn time_grid(maturities: &[f64], steps_per_year: usize) -> Vec<f64> {
  let mut times = vec![0.0];
  let mut prev = 0.0;
  for &maturity in maturities {
    let steps = ((maturity - prev) * steps_per_year as f64).ceil().max(1.0) as usize;
    for j in 1..steps {
      times.push(prev + (maturity - prev) * j as f64 / steps as f64);
    }
    times.push(maturity);
    prev = maturity;
  }
  times
}

pub(super) fn clamp_leverage(l: f64) -> f64 {
  l.clamp(LEVERAGE_BAND.0, LEVERAGE_BAND.1)
}

/// The row's value at `s`, linear between the nodes and held flat outside.
fn interpolate_row(spots: &[f64], row: &[f64], s: f64) -> f64 {
  let last = spots.len() - 1;
  if !matches!(s.partial_cmp(&spots[0]), Some(Ordering::Greater)) {
    return row[0];
  }
  if s >= spots[last] {
    return row[last];
  }
  let i1 = spots.partition_point(|&node| node <= s);
  let i0 = i1 - 1;
  let w = (s - spots[i0]) / (spots[i1] - spots[i0]);
  row[i0] + w * (row[i1] - row[i0])
}

/// `ΔK = √(−2h² ln(ε√(2π) h))`; unbounded once the kernel is so wide that no particle's mass falls
/// below `ε`.
fn window_half_width(h: f64) -> f64 {
  let scaled = WINDOW_MASS * (2.0 * std::f64::consts::PI).sqrt() * h;
  if scaled >= 1.0 {
    f64::INFINITY
  } else {
    (-2.0 * h * h * scaled.ln()).sqrt()
  }
}

/// The Nadaraya–Watson estimate of $\mathbb{E}[V \mid S = K]$ over the
/// particles inside the window, `NaN` when none is.
fn conditional_variance(sorted: &[(f64, f64)], node: f64, h: f64, half_width: f64) -> f64 {
  let lo = sorted.partition_point(|&(s, _)| s < node - half_width);
  let hi = sorted.partition_point(|&(s, _)| s <= node + half_width);
  let inv_h = 1.0 / h;
  let (mut sum_w, mut sum_wv) = (0.0, 0.0);
  for &(s, v) in &sorted[lo..hi] {
    let u = (s - node) * inv_h;
    let w = (-0.5 * u * u).exp();
    sum_w += w;
    sum_wv += w * v;
  }
  if sum_w > 0.0 {
    sum_wv / sum_w
  } else {
    f64::NAN
  }
}

/// A node no particle reaches takes the nearest node's estimate; a row no
/// particle reaches at all takes the cloud's mean variance.
fn fill_gaps(conditional: &mut [f64], fallback: f64) {
  let finite = conditional
    .iter()
    .enumerate()
    .filter(|(_, c)| c.is_finite())
    .map(|(i, &c)| (i, c))
    .collect::<Vec<_>>();
  if finite.is_empty() {
    conditional.fill(fallback);
    return;
  }
  for (i, slot) in conditional.iter_mut().enumerate() {
    if !slot.is_finite() {
      *slot = finite
        .iter()
        .min_by_key(|(j, _)| j.abs_diff(i))
        .map(|&(_, c)| c)
        .unwrap_or(fallback);
    }
  }
}

#[cfg(test)]
mod tests;
