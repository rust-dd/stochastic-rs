//! Heston Stochastic Local Volatility (SLV) model.
//!
//! $$
//! \frac{dS_t}{S_t} = (r-q)\,dt + L(S_t,t)\,\sqrt{V_t}\,dW_t^S,\qquad
//! dV_t = \kappa(\theta-V_t)\,dt + \eta\sigma\sqrt{V_t}\,dW_t^V
//! $$
//!
//! The leverage function $L(S,t)$ is calibrated so that the model
//! reproduces the Dupire local-volatility surface:
//!
//! $$
//! L^2(K,t) = \frac{\sigma_{\text{LV}}^2(K,t)}{\mathbb{E}[V_t \mid S_t = K]}
//! $$
//!
//! Calibration uses the Guyon–Labordère particle method with
//! Nadaraya–Watson kernel regression for the conditional expectation.
//!
//! Reference: Guyon & Henry-Labordère, "Being particular about
//! calibration", *Risk*, 2012.
//! See also: arXiv 2208.09986 (Djete, McKean–Vlasov existence),
//! arXiv 2406.14074 (Mustapha, strong well-posedness),
//! arXiv 1701.06001 (Cozma et al., control-variate particle method).

use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::traits::ModelPricer;

/// Heston model parameters augmented with the SLV mixing factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonSlvParams {
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Vol-of-vol of the base Heston model.
  pub sigma: f64,
  /// Spot–variance correlation.
  pub rho: f64,
  /// Initial variance.
  pub v0: f64,
  /// Mixing factor in $[0,1]$. $\eta=0$: pure local vol,
  /// $\eta=1$: full stochastic vol.
  pub eta: f64,
}

impl HestonSlvParams {
  /// Effective vol-of-vol after mixing: $\sigma_{\text{mix}} = \eta\,\sigma$.
  pub fn sigma_mixed(&self) -> f64 {
    self.eta * self.sigma
  }
}

/// A 2-D grid storing the calibrated leverage function $L(S,t)$ with
/// bilinear interpolation.
#[derive(Debug, Clone)]
pub struct LeverageSurface {
  spots: Array1<f64>,
  times: Array1<f64>,
  values: Array2<f64>,
}

impl LeverageSurface {
  /// Build from pre-computed grid values. `values` has shape
  /// `(times.len(), spots.len())`.
  pub fn new(spots: Array1<f64>, times: Array1<f64>, values: Array2<f64>) -> Self {
    Self {
      spots,
      times,
      values,
    }
  }

  /// Bilinear interpolation of $L(S,t)$, clamped at the grid boundary.
  pub fn interpolate(&self, s: f64, t: f64) -> f64 {
    let si = fractional_index(&self.spots, s);
    let ti = fractional_index(&self.times, t);

    let i0 = (si.floor() as usize).min(self.spots.len() - 2);
    let j0 = (ti.floor() as usize).min(self.times.len() - 2);
    let i1 = i0 + 1;
    let j1 = j0 + 1;

    let ws = si - i0 as f64;
    let wt = ti - j0 as f64;
    let ws = ws.clamp(0.0, 1.0);
    let wt = wt.clamp(0.0, 1.0);

    let v00 = self.values[[j0, i0]];
    let v10 = self.values[[j0, i1]];
    let v01 = self.values[[j1, i0]];
    let v11 = self.values[[j1, i1]];

    (1.0 - wt) * ((1.0 - ws) * v00 + ws * v10) + wt * ((1.0 - ws) * v01 + ws * v11)
  }

  /// Spot grid.
  pub fn spots(&self) -> &Array1<f64> {
    &self.spots
  }

  /// Time grid.
  pub fn times(&self) -> &Array1<f64> {
    &self.times
  }

  /// Raw grid values (shape: times × spots).
  pub fn values(&self) -> &Array2<f64> {
    &self.values
  }
}

fn fractional_index(grid: &Array1<f64>, x: f64) -> f64 {
  if x <= grid[0] {
    return 0.0;
  }
  let n = grid.len();
  if x >= grid[n - 1] {
    return (n - 1) as f64;
  }
  for i in 0..n - 1 {
    if x >= grid[i] && x < grid[i + 1] {
      return i as f64 + (x - grid[i]) / (grid[i + 1] - grid[i]);
    }
  }
  (n - 1) as f64
}

/// Calibrate the leverage surface $L(S,t)$ using the Guyon–Labordère
/// particle method.
///
/// The local-volatility surface is provided as a grid:
/// `local_vol_values[j, i]` = $\sigma_\text{LV}(S_i, t_j)$.
pub fn calibrate_leverage(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  local_vol_spots: &Array1<f64>,
  local_vol_times: &Array1<f64>,
  local_vol_values: &Array2<f64>,
  eval_spots: &Array1<f64>,
  eval_times: &Array1<f64>,
  n_particles: usize,
  seed: u64,
) -> LeverageSurface {
  let sigma_mixed = params.sigma_mixed();
  let rho_bar = (1.0 - params.rho * params.rho).sqrt();
  let n_steps = eval_times.len();
  let n_eval = eval_spots.len();

  let lv_surf = LeverageSurface::new(
    local_vol_spots.clone(),
    local_vol_times.clone(),
    local_vol_values.clone(),
  );

  let mut leverage_grid = Array2::ones((n_steps, n_eval));

  let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
  let mut x_particles = Array1::from_elem(n_particles, s0.ln());
  let mut v_particles = Array1::from_elem(n_particles, params.v0);

  let mut t_prev = 0.0;

  for step in 0..n_steps {
    let t_curr = eval_times[step];
    let dt = t_curr - t_prev;
    if dt <= 0.0 {
      continue;
    }
    let sqrt_dt = dt.sqrt();

    // Compute leverage at eval_spots from current particle cloud
    let bandwidth = silverman_bandwidth(&x_particles);
    for i in 0..n_eval {
      let x_eval = eval_spots[i].ln();
      let (sum_v, sum_k) = kernel_conditional_mean(&x_particles, &v_particles, x_eval, bandwidth);
      let cond_v = if sum_k > 1e-12 { sum_v / sum_k } else { params.v0 };
      let cond_v = cond_v.max(1e-8);

      let local_vol = lv_surf.interpolate(eval_spots[i], t_curr);
      let lev = local_vol / cond_v.sqrt();
      leverage_grid[[step, i]] = lev.clamp(0.01, 10.0);
    }

    let lev_step = LeverageSurface::new(
      eval_spots.clone(),
      eval_times.clone(),
      leverage_grid.clone(),
    );

    // Evolve particles forward
    for p in 0..n_particles {
      let dw_v: f64 = Distribution::<f64>::sample(&StandardNormal, &mut rng) * sqrt_dt;
      let dw_ind: f64 = Distribution::<f64>::sample(&StandardNormal, &mut rng) * sqrt_dt;
      let dw_x = params.rho * dw_v + rho_bar * dw_ind;

      let v_curr = v_particles[p].max(0.0);
      let s_curr = x_particles[p].exp();
      let sqrt_v = v_curr.sqrt();

      let l = lev_step.interpolate(s_curr, t_curr);

      // Variance: truncated Euler–Maruyama
      v_particles[p] = (v_curr + params.kappa * (params.theta - v_curr) * dt
        + sigma_mixed * sqrt_v * dw_v)
        .max(0.0);

      // Log-spot
      let drift = (r - q) - 0.5 * l * l * v_curr;
      x_particles[p] += drift * dt + l * sqrt_v * dw_x;
    }

    t_prev = t_curr;
  }

  LeverageSurface::new(eval_spots.clone(), eval_times.clone(), leverage_grid)
}

/// Calibrate the leverage surface directly from a [`super::dupire::Dupire`]
/// instance. Computes the Dupire local-vol surface, then delegates to
/// [`calibrate_leverage`].
pub fn calibrate_from_dupire(
  params: &HestonSlvParams,
  dupire: &super::dupire::Dupire,
  n_particles: usize,
  seed: u64,
) -> LeverageSurface {
  let lv_surface = dupire.local_vol_surface();
  let nt = dupire.ts.len();
  let nk = dupire.ks.len();

  // Replace NaN boundary values with nearest valid interior value per row
  let mut lv_clean = lv_surface.clone();
  for j in 0..nt {
    let first_valid = (0..nk)
      .find(|&i| lv_clean[[j, i]].is_finite())
      .unwrap_or(1);
    let last_valid = (0..nk)
      .rfind(|&i| lv_clean[[j, i]].is_finite())
      .unwrap_or(nk - 2);
    for i in 0..first_valid {
      lv_clean[[j, i]] = lv_clean[[j, first_valid]];
    }
    for i in (last_valid + 1)..nk {
      lv_clean[[j, i]] = lv_clean[[j, last_valid]];
    }
    // Replace any remaining NaN with row mean
    let row = lv_clean.index_axis(Axis(0), j);
    let finite_vals: Vec<f64> = row.iter().filter(|x| x.is_finite()).copied().collect();
    let row_mean = if finite_vals.is_empty() {
      0.2
    } else {
      finite_vals.iter().sum::<f64>() / finite_vals.len() as f64
    };
    for i in 0..nk {
      if !lv_clean[[j, i]].is_finite() {
        lv_clean[[j, i]] = row_mean;
      }
    }
  }

  let spots = Array1::from_vec(dupire.ks.clone());
  let times = Array1::from_vec(dupire.ts.clone());
  let s0 = dupire.ks[nk / 2]; // mid-strike as proxy for spot

  calibrate_leverage(
    params,
    s0,
    dupire.r,
    dupire.q,
    &spots,
    &times,
    &lv_clean,
    &spots,
    &times,
    n_particles,
    seed,
  )
}

/// Heston-SLV Monte Carlo pricer with a pre-calibrated leverage surface.
#[derive(Debug, Clone)]
pub struct HestonSlvPricer {
  /// Model parameters.
  pub params: HestonSlvParams,
  /// Calibrated leverage surface.
  pub leverage: LeverageSurface,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Number of MC paths.
  pub n_paths: usize,
  /// Time-discretization steps per year.
  pub steps_per_year: usize,
  /// RNG seed.
  pub seed: u64,
}

impl HestonSlvPricer {
  pub fn new(
    params: HestonSlvParams,
    leverage: LeverageSurface,
    r: f64,
    q: f64,
  ) -> Self {
    Self {
      params,
      leverage,
      r,
      q,
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

  fn mc_call_price(&self, s: f64, k: f64, tau: f64) -> f64 {
    let n_steps = ((tau * self.steps_per_year as f64).round() as usize).max(1);
    let dt = tau / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let sigma_mixed = self.params.sigma_mixed();
    let rho_bar = (1.0 - self.params.rho * self.params.rho).sqrt();

    let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
    let mut payoff_sum = 0.0;

    for _ in 0..self.n_paths {
      let mut x = s.ln();
      let mut v = self.params.v0;

      for step in 0..n_steps {
        let t = (step as f64 + 1.0) * dt;
        let dw_v: f64 = Distribution::<f64>::sample(&StandardNormal, &mut rng) * sqrt_dt;
        let dw_ind: f64 = Distribution::<f64>::sample(&StandardNormal, &mut rng) * sqrt_dt;
        let dw_x = self.params.rho * dw_v + rho_bar * dw_ind;

        let v_pos = v.max(0.0);
        let sqrt_v = v_pos.sqrt();
        let s_curr = x.exp();

        let l = self.leverage.interpolate(s_curr, t);

        v = (v + self.params.kappa * (self.params.theta - v_pos) * dt
          + sigma_mixed * sqrt_v * dw_v)
          .max(0.0);

        let drift = (self.r - self.q) - 0.5 * l * l * v_pos;
        x += drift * dt + l * sqrt_v * dw_x;
      }

      let s_t = x.exp();
      payoff_sum += (s_t - k).max(0.0);
    }

    (-self.r * tau).exp() * payoff_sum / self.n_paths as f64
  }
}

impl ModelPricer for HestonSlvPricer {
  fn price_call(&self, s: f64, k: f64, _r: f64, _q: f64, tau: f64) -> f64 {
    self.mc_call_price(s, k, tau)
  }
}

// Silverman's rule-of-thumb bandwidth for Gaussian kernel.
fn silverman_bandwidth(x: &Array1<f64>) -> f64 {
  let n = x.len() as f64;
  let mean = x.sum() / n;
  let var = x.iter().map(|&xi| (xi - mean) * (xi - mean)).sum::<f64>() / n;
  let std = var.sqrt().max(1e-10);
  1.06 * std * n.powf(-0.2)
}

// Nadaraya–Watson kernel regression: returns (Σ V_i K(x_i - x), Σ K(x_i - x)).
fn kernel_conditional_mean(
  x_particles: &Array1<f64>,
  v_particles: &Array1<f64>,
  x_eval: f64,
  bandwidth: f64,
) -> (f64, f64) {
  let inv_h = 1.0 / bandwidth;
  let mut sum_vk = 0.0;
  let mut sum_k = 0.0;

  for i in 0..x_particles.len() {
    let u = (x_particles[i] - x_eval) * inv_h;
    let k = (-0.5 * u * u).exp();
    sum_k += k;
    sum_vk += v_particles[i] * k;
  }

  (sum_vk, sum_k)
}
