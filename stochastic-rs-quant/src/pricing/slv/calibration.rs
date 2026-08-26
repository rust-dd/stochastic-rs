//! Guyon–Labordère particle calibration of the leverage function $L(S,t)$.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

use super::HestonSlvParams;
use super::LeverageSurface;

/// Calibrate the leverage surface $L(S,t)$ using the Guyon–Labordère
/// particle method.
///
/// The local-volatility surface is provided as a grid:
/// `local_vol_values[j, i]` = $\sigma_\text{LV}(S_i, t_j)$.
///
/// **The returned surface is anchored to `(r, q)`.** The particle cloud that
/// supplies $\mathbb{E}[V_t \mid S_t = K]$ is evolved under the risk-neutral
/// drift $r - q$, so a different rate produces a different conditional
/// expectation and hence a different $L$. Feed the same `(r, q)` to
/// [`HestonSlvPricer::new`](super::HestonSlvPricer::new) so the pricer can
/// reject queries the surface cannot honour.
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

  let normals = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
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
      let cond_v = if sum_k > 1e-12 {
        sum_v / sum_k
      } else {
        params.v0
      };
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
      let dw_v = normals.sample_fast() * sqrt_dt;
      let dw_ind = normals.sample_fast() * sqrt_dt;
      let dw_x = params.rho * dw_v + rho_bar * dw_ind;

      let v_curr = v_particles[p].max(0.0);
      let s_curr = x_particles[p].exp();
      let sqrt_v = v_curr.sqrt();

      let l = lev_step.interpolate(s_curr, t_curr);

      // Variance: truncated Euler–Maruyama
      v_particles[p] =
        (v_curr + params.kappa * (params.theta - v_curr) * dt + sigma_mixed * sqrt_v * dw_v)
          .max(0.0);

      // Log-spot
      let drift = (r - q) - 0.5 * l * l * v_curr;
      x_particles[p] += drift * dt + l * sqrt_v * dw_x;
    }

    t_prev = t_curr;
  }

  LeverageSurface::new(eval_spots.clone(), eval_times.clone(), leverage_grid)
}

/// Calibrate the leverage surface directly from a
/// [`Dupire`](crate::pricing::dupire::Dupire) instance. Computes the Dupire
/// local-vol surface, then delegates to [`calibrate_leverage`].
///
/// The result is anchored to `dupire.r` / `dupire.q` twice over: the Dupire
/// numerator carries an explicit $(r-q)K\,\partial_K C + qC$ term, and the
/// particle drift in [`calibrate_leverage`] is $r-q$.
pub fn calibrate_from_dupire(
  params: &HestonSlvParams,
  dupire: &crate::pricing::dupire::Dupire,
  n_particles: usize,
  seed: u64,
) -> LeverageSurface {
  let lv_surface = dupire.local_vol_surface();
  let nt = dupire.ts.len();
  let nk = dupire.ks.len();

  // Replace NaN boundary values with nearest valid interior value per row
  let mut lv_clean = lv_surface.clone();
  for j in 0..nt {
    let first_valid = (0..nk).find(|&i| lv_clean[[j, i]].is_finite()).unwrap_or(1);
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
