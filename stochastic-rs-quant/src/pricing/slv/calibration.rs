//! Guyon–Henry-Labordère particle calibration of the leverage function
//! $L(S,t)$.
//!
//! The calibration condition is Gyöngy's theorem applied to the SLV model:
//!
//! $$
//! L^2(t,K) = \frac{\sigma_{\text{LV}}^2(t,K)}{\mathbb{E}[V_t \mid S_t = K]}
//! $$
//!
//! The conditional expectation is estimated from an interacting particle
//! system by Nadaraya–Watson regression with a Gaussian kernel in spot,
//!
//! $$
//! \hat{\mathbb{E}}[V_t \mid S_t = K]
//!   = \frac{\sum_i V^i_t\,\delta_N(S^i_t - K)}{\sum_i \delta_N(S^i_t - K)},
//! \qquad
//! \delta_N(x) = \frac{e^{-\frac12 (x / h_N(t))^2}}{h_N(t)\sqrt{2\pi}},
//! $$
//!
//! under the Silverman-type bandwidth
//!
//! $$
//! h_N(t) = \kappa\, S_0\, \sigma_{\text{LV}}(t, S_0)\,
//!   \sqrt{\max(t, t_{\min})}\; N^{-1/5},
//! \qquad \kappa = 1.5,\ t_{\min} = 0.25,
//! $$
//!
//! and, for speed, only the particles inside $[K - \Delta K, K + \Delta K]$
//! with $\Delta K = \sqrt{-2 h_N^2 \ln(\varepsilon \sqrt{2\pi}\, h_N)}$,
//! $\varepsilon = 10^{-5}$, contribute to a node's sums — the mass a
//! particle outside carries is below $\varepsilon$. The cloud starts at
//! $(S_0, v_0)$, so the $t = 0$ slice is exact: $L(0, s) =
//! \sigma_{\text{LV}}(0, s) / \sqrt{v_0}$. Each step evolves the cloud from
//! $t_k$ to $t_{k+1}$ under $L(t_k, \cdot)$ and regresses at $t_{k+1}$, the
//! log-spot by Euler–Maruyama and the variance by the crate's Euler scheme
//! with absorption at zero, the same recursion
//! [`HestonSlvPricer`](super::HestonSlvPricer) and the `HestonSlv` process
//! run.
//!
//! References: Guyon, J. & Henry-Labordère, P. (2012), *Being particular
//! about calibration*, Risk 25(1), 88–93; Guyon, J. & Henry-Labordère, P.
//! (2013), *Nonlinear Option Pricing*, Chapman & Hall, ch. 11; Cozma, A.,
//! Mariapragassam, M. & Reisinger, C. (2017), *Calibration of a hybrid
//! local-stochastic volatility stochastic rates model with a control variate
//! particle method*, arXiv:1701.06001, §3.4 and Algorithm 1 (the kernel, the
//! bandwidth constants and the window above); Reisinger, C. & Tsianni, M. O.
//! (2025), *Numerical analysis of a particle system for the calibrated
//! Heston-type local stochastic volatility model*, arXiv:2504.14343
//! (well-posedness of the kernel-regularised system and strong convergence of
//! its Euler scheme).

use std::cmp::Ordering;

use anyhow::Result;
use anyhow::bail;
use ndarray::Array1;
use ndarray::Array2;
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::traits::Grid2D;

use super::HestonSlvParams;
use super::LeverageSurface;

/// Floor of the conditional-variance estimate before the square root.
const CONDITIONAL_VARIANCE_FLOOR: f64 = 1e-8;

/// The leverage is clamped to this band so a degenerate local volatility or
/// an almost-empty kernel window cannot stall or blow up the cloud.
const LEVERAGE_BAND: (f64, f64) = (0.01, 10.0);

/// Kernel mass below which a particle is left out of a node's window,
/// Cozma et al.'s $\varepsilon$.
const WINDOW_MASS: f64 = 1e-5;

/// The particle method's tuning: everything besides the model and the
/// target surface. The defaults are Guyon–Henry-Labordère's bandwidth
/// constants and a cloud large enough for a production surface.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleMethod {
  /// Particles in the cloud, $N$.
  pub n_particles: usize,
  /// Euler steps per year; every snapshot maturity is a grid point, so an
  /// interval between two maturities takes `ceil(Δt · steps_per_year)` of
  /// them.
  pub steps_per_year: usize,
  /// Seed of the cloud's Gaussian stream.
  pub seed: u64,
  /// $\kappa$ in the bandwidth, `1.5` in the reference.
  pub bandwidth_factor: f64,
  /// $t_{\min}$ in the bandwidth, `0.25` in the reference: below it the
  /// bandwidth stops shrinking with the cloud, which is still concentrated
  /// at the spot.
  pub bandwidth_t_min: f64,
}

impl Default for ParticleMethod {
  fn default() -> Self {
    Self {
      n_particles: 100_000,
      steps_per_year: 200,
      seed: 42,
      bandwidth_factor: 1.5,
      bandwidth_t_min: 0.25,
    }
  }
}

impl ParticleMethod {
  pub fn with_particles(mut self, n_particles: usize) -> Self {
    self.n_particles = n_particles;
    self
  }

  pub fn with_steps_per_year(mut self, steps_per_year: usize) -> Self {
    self.steps_per_year = steps_per_year;
    self
  }

  pub fn with_seed(mut self, seed: u64) -> Self {
    self.seed = seed;
    self
  }

  pub fn with_bandwidth_factor(mut self, factor: f64) -> Self {
    self.bandwidth_factor = factor;
    self
  }

  pub fn with_bandwidth_t_min(mut self, t_min: f64) -> Self {
    self.bandwidth_t_min = t_min;
    self
  }

  fn validate(&self) -> Result<()> {
    if self.n_particles < 2 {
      bail!(
        "the particle method needs at least two particles, got {}",
        self.n_particles
      );
    }
    if self.steps_per_year == 0 {
      bail!("steps_per_year must be positive");
    }
    if !(self.bandwidth_factor.is_finite() && self.bandwidth_factor > 0.0) {
      bail!(
        "bandwidth_factor must be finite and positive, got {}",
        self.bandwidth_factor
      );
    }
    if !(self.bandwidth_t_min.is_finite() && self.bandwidth_t_min >= 0.0) {
      bail!(
        "bandwidth_t_min must be finite and non-negative, got {}",
        self.bandwidth_t_min
      );
    }
    Ok(())
  }
}

/// What one particle run produces: the leverage surface, and the cloud's
/// spots at every requested maturity, in the order the maturities were
/// given — the same particles the surface was read off, so a caller can
/// price the vanilla grid the calibration targeted without a second
/// simulation.
#[derive(Clone, Debug)]
pub struct ParticleCalibration {
  pub leverage: LeverageSurface,
  pub snapshots: Vec<Array1<f64>>,
}

/// Calibrate the leverage surface $L(S,t)$ by the Guyon–Henry-Labordère
/// particle method — the module doc has the formulas.
///
/// `local_vol` is the Dupire local volatility $\sigma_{\text{LV}}(t, S)$ on
/// a `(t, S)` grid; its spot nodes are the nodes the leverage is calibrated
/// on, and it is read by bilinear interpolation, held flat outside its own
/// extent, at every step time. `snapshot_maturities`, strictly ascending
/// and positive, are the times the cloud is recorded at; the last one is
/// the surface's horizon.
///
/// **The returned surface is anchored to `(r, q)`.** The cloud that supplies
/// $\mathbb{E}[V_t \mid S_t = K]$ is evolved under the risk-neutral drift
/// $r - q$, so a different rate produces a different conditional expectation
/// and hence a different $L$. Feed the same `(r, q)` to
/// [`HestonSlvPricer::new`](super::HestonSlvPricer::new) so the pricer can
/// reject queries the surface cannot honour.
///
/// Errors on an invalid input — a non-positive spot, a negative variance or
/// mixing fraction, a correlation outside `[-1, 1]`, a local volatility
/// with a non-finite or negative value, an empty or non-ascending maturity
/// list, or a [`ParticleMethod`] with fewer than two particles.
pub fn calibrate_leverage(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  local_vol: &Grid2D<f64>,
  snapshot_maturities: &[f64],
  method: &ParticleMethod,
) -> Result<ParticleCalibration> {
  validate(params, s0, r, q, local_vol, snapshot_maturities)?;
  method.validate()?;

  let times = time_grid(snapshot_maturities, method.steps_per_year);
  let spots = local_vol.xs().to_vec();
  let n_eval = spots.len();
  let n = method.n_particles;
  let (kappa, theta, rho, v0) = (params.kappa, params.theta, params.rho, params.v0.max(0.0));
  let sigma_mixed = params.sigma_mixed();
  let rho_bar = (1.0 - rho * rho).sqrt();
  let carry = r - q;
  let bandwidth_scale = method.bandwidth_factor * s0 * (n as f64).powf(-0.2);

  let mut leverage = Array2::<f64>::zeros((times.len(), n_eval));
  let sqrt_v0 = v0.max(CONDITIONAL_VARIANCE_FLOOR).sqrt();
  for (i, &node) in spots.iter().enumerate() {
    leverage[[0, i]] = clamp_leverage(local_vol.eval(0.0, node) / sqrt_v0);
  }

  let normals = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(method.seed));
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
      let vp = v[p];
      let sqrt_v = vp.sqrt();
      let dw_v = z_v[p] * sqrt_dt;
      let dw_s = rho * dw_v + rho_bar * z_perp[p] * sqrt_dt;
      x[p] += (carry - 0.5 * l * l * vp) * dt + l * sqrt_v * dw_s;
      v[p] = (vp + kappa * (theta - vp) * dt + sigma_mixed * sqrt_v * dw_v).max(0.0);
    }

    if next_snapshot < snapshot_maturities.len() && t_next == snapshot_maturities[next_snapshot] {
      snapshots.push(Array1::from_iter(x.iter().map(|xp| xp.exp())));
      next_snapshot += 1;
    }

    for (slot, (xp, vp)) in sorted.iter_mut().zip(x.iter().zip(v.iter())) {
      *slot = (xp.exp(), *vp);
    }
    sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let h = bandwidth_scale * local_vol.eval(t_next, s0) * t_next.max(method.bandwidth_t_min).sqrt();
    let half_width = window_half_width(h);
    conditional
      .par_iter_mut()
      .zip(spots.par_iter())
      .for_each(|(slot, &node)| *slot = conditional_variance(&sorted, node, h, half_width));
    let cloud_mean = v.iter().sum::<f64>() / n as f64;
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

fn validate(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  local_vol: &Grid2D<f64>,
  snapshot_maturities: &[f64],
) -> Result<()> {
  if !(s0.is_finite() && s0 > 0.0) {
    bail!("the spot must be finite and positive, got {s0}");
  }
  if !(r.is_finite() && q.is_finite()) {
    bail!("the rates must be finite, got r = {r}, q = {q}");
  }
  if !(params.v0.is_finite() && params.v0 >= 0.0) {
    bail!("v0 must be a finite non-negative variance, got {}", params.v0);
  }
  if !(params.eta.is_finite() && params.eta >= 0.0) {
    bail!("the mixing fraction eta must be finite and non-negative, got {}", params.eta);
  }
  if !(params.rho.is_finite() && params.rho.abs() <= 1.0) {
    bail!("rho must be a correlation in [-1, 1], got {}", params.rho);
  }
  for value in [params.kappa, params.theta, params.sigma] {
    if !value.is_finite() {
      bail!("the Heston parameters must be finite");
    }
  }
  if local_vol.values().iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
    bail!("the local volatility must be finite and non-negative everywhere; clean the Dupire surface first");
  }
  if snapshot_maturities.is_empty() {
    bail!("at least one snapshot maturity is needed: it is the surface's horizon");
  }
  if snapshot_maturities.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
    bail!("snapshot maturities must be finite and positive");
  }
  if snapshot_maturities.windows(2).any(|w| w[0] >= w[1]) {
    bail!("snapshot maturities must be strictly ascending");
  }
  Ok(())
}

/// `0 = t_0 < … < t_m`, every maturity a node, each interval split into
/// `ceil(Δ · steps_per_year)` equal steps.
fn time_grid(maturities: &[f64], steps_per_year: usize) -> Vec<f64> {
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

fn clamp_leverage(l: f64) -> f64 {
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

/// $\Delta K = \sqrt{-2 h^2 \ln(\varepsilon\sqrt{2\pi}\,h)}$; unbounded once
/// the kernel is so wide that no particle's mass falls below
/// $\varepsilon$.
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
  if sum_w > 0.0 { sum_wv / sum_w } else { f64::NAN }
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
