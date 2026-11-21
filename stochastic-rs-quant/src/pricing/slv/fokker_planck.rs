//! Leverage calibration by the forward Kolmogorov (Fokker–Planck) equation of the Heston SLV model:
//!
//! $$
//! \partial_t p = \partial_{xx}\bigl(\tfrac12 L^2 v\, p\bigr) +
//!   \partial_{xv}\bigl(\rho\,\xi L v\, p\bigr) +
//!   \partial_{vv}\bigl(\tfrac12 \xi^2 v\, p\bigr) -
//!   \partial_x\bigl((r - q - \tfrac12 L^2 v)\, p\bigr) -
//!   \partial_v\bigl(\kappa(\theta - v)\, p\bigr),
//! \qquad p(0, \cdot) = \delta_{(x_0, v_0)},\quad \xi = \eta\sigma,\quad L = L(t, e^x),
//! $$
//!
//! $$
//! L^2(t, x) = \frac{\sigma_{\text{LV}}^2(t, e^x)}{\mathbb{E}[V_t \mid X_t = x]},
//! \qquad
//! \mathbb{E}[V_t \mid X_t = x] = \frac{\int_0^\infty v\, p(t, x, v)\, dv}{\int_0^\infty p(t, x, v)\, dv},
//! $$
//!
//! Finite volumes (§2), Hundsdorfer–Verwer ADI after a Rannacher start (§3), `Q` passes of §4.
//! Reference: Wyns & Du Toit (2016), Int. J. Comput. Math. 94(11), 2239–2267, arXiv:1611.02961
//! Reference: Ren et al. (2007), Risk 20(9), 138–143
//! Reference: Tian et al. (2015), J. Deriv. 22(3), 21–39
//! Reference: in 't Hout & Foulon (2010), Int. J. Numer. Anal. Model. 7(2), 303–320

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::traits::Fn2D;
use stochastic_rs_distributions::traits::Grid2D;

use self::mesh::Mesh;
use self::solver::Direction;
use self::solver::Level;
use self::solver::StepBuffers;
use self::solver::step;
use super::HestonSlvParams;
use super::LeverageSurface;
use super::calibration::CONDITIONAL_VARIANCE_FLOOR;
use super::calibration::clamp_leverage;
use super::calibration::time_grid;
use super::calibration::validate;
use crate::QuantError;

mod mesh;
mod solver;

/// A column whose mass is below this fraction of the heaviest column's is
/// taken as empty: its conditional expectation keeps the previous row's.
const MASS_THRESHOLD: f64 = 1e-12;

/// Truncation of the log-spot domain when none is given: `x_0 ± ln 30`, the
/// reference's.
const DEFAULT_LOG_SPOT_HALF_WIDTH: f64 = 3.401_197_381_662_155_4;

/// The finite-volume solver's tuning. The defaults are the reference's
/// scheme settings on a mesh half its size.
#[derive(Clone, Debug, PartialEq)]
pub struct FokkerPlanckMethod {
  /// Log-spot nodes; made odd so the spot is a node.
  pub log_spot_nodes: usize,
  /// Nodes of the variance mesh, uniform on `[0, v0]` and stretched above it.
  pub variance_nodes: usize,
  /// Time steps per year; every snapshot maturity is a step boundary.
  pub steps_per_year: usize,
  /// Inner iterations per step on the non-linearity, the reference's `Q`.
  pub inner_iterations: usize,
  /// The Hundsdorfer–Verwer parameter, `½ + √3/6` in the reference.
  pub theta: f64,
  /// Rannacher start-up: the first two steps as four fully implicit Euler
  /// half-steps, including the mixed derivative, solved by sparse LU.
  pub damping: bool,
  /// Half-width of the log-spot domain around `ln s0`; `None` takes `ln 30`.
  pub x_half_width: Option<f64>,
  /// Upper variance boundary; `None` takes `15 · max(v0, θ)`, at least `0.5`.
  pub v_max: Option<f64>,
  /// Clustering scale `c` of the log-spot mesh `x = x0 + c sinh ξ`.
  pub x_stretch: f64,
  /// Stretching scale above `v0` as a fraction of `max(v0, θ)`; `[0, v0]`
  /// is uniformly resolved with comparable spacing at the join.
  pub v_stretch: f64,
}

impl Default for FokkerPlanckMethod {
  fn default() -> Self {
    Self {
      log_spot_nodes: 201,
      variance_nodes: 100,
      steps_per_year: 200,
      inner_iterations: 2,
      theta: 0.5 + 3.0_f64.sqrt() / 6.0,
      damping: true,
      x_half_width: None,
      v_max: None,
      x_stretch: 0.2,
      v_stretch: 0.2,
    }
  }
}

impl FokkerPlanckMethod {
  pub fn with_nodes(mut self, log_spot_nodes: usize, variance_nodes: usize) -> Self {
    self.log_spot_nodes = log_spot_nodes;
    self.variance_nodes = variance_nodes;
    self
  }

  pub fn with_steps_per_year(mut self, steps_per_year: usize) -> Self {
    self.steps_per_year = steps_per_year;
    self
  }

  pub fn with_inner_iterations(mut self, inner_iterations: usize) -> Self {
    self.inner_iterations = inner_iterations;
    self
  }

  pub fn with_theta(mut self, theta: f64) -> Self {
    self.theta = theta;
    self
  }

  pub fn with_damping(mut self, damping: bool) -> Self {
    self.damping = damping;
    self
  }

  pub fn with_x_half_width(mut self, half_width: f64) -> Self {
    self.x_half_width = Some(half_width);
    self
  }

  pub fn with_v_max(mut self, v_max: f64) -> Self {
    self.v_max = Some(v_max);
    self
  }

  pub fn with_x_stretch(mut self, stretch: f64) -> Self {
    self.x_stretch = stretch;
    self
  }

  pub fn with_v_stretch(mut self, stretch: f64) -> Self {
    self.v_stretch = stretch;
    self
  }

  fn validate(&self, params: &HestonSlvParams) -> Result<(), QuantError> {
    let (nx, nv) = (self.log_spot_nodes, self.variance_nodes);
    if nx < 3 || nv < 3 {
      return Err(QuantError::invalid(
        "Fokker–Planck mesh",
        format!("needs at least three nodes per axis, got {nx} log-spot and {nv} variance nodes"),
      ));
    }
    let (steps, inner) = (self.steps_per_year, self.inner_iterations);
    if steps == 0 || inner == 0 {
      return Err(QuantError::invalid(
        "steps_per_year and inner_iterations",
        format!("must be positive, got {steps} and {inner}"),
      ));
    }
    if !(self.theta.is_finite() && self.theta > 0.0) {
      return Err(QuantError::invalid(
        "theta",
        format!("must be finite and positive, got {}", self.theta),
      ));
    }
    let (x_stretch, v_stretch) = (self.x_stretch, self.v_stretch);
    if !(x_stretch.is_finite() && x_stretch > 0.0) || !(v_stretch.is_finite() && v_stretch > 0.0) {
      return Err(QuantError::invalid(
        "mesh stretches",
        format!(
          "must be finite and positive, got x_stretch = {x_stretch}, v_stretch = {v_stretch}"
        ),
      ));
    }
    if let Some(w) = self.x_half_width.filter(|w| !(w.is_finite() && *w > 0.0)) {
      return Err(QuantError::invalid(
        "x_half_width",
        format!("must be finite and positive, got {w}"),
      ));
    }
    let v_scale = params.heston.v0.max(params.heston.theta);
    if let Some(v) = self.v_max.filter(|v| !(v.is_finite() && *v > v_scale)) {
      return Err(QuantError::invalid(
        "v_max",
        format!("must be finite and above both v0 and theta ({v_scale}), got {v}"),
      ));
    }
    Ok(())
  }

  fn mesh(&self, params: &HestonSlvParams, s0: f64) -> Mesh {
    let v_scale = params.heston.v0.max(params.heston.theta).max(1e-6);
    let v_max = self.v_max.unwrap_or((15.0 * v_scale).max(0.5));
    Mesh::new(
      s0.ln(),
      self.x_half_width.unwrap_or(DEFAULT_LOG_SPOT_HALF_WIDTH),
      self.x_stretch,
      self.log_spot_nodes,
      params.heston.v0.max(0.0),
      v_max,
      self.v_stretch * v_scale,
      self.variance_nodes,
    )
  }
}

/// The log-spot density at the snapshot maturities, with the mass and variance mean carried there:
/// the diagnostics the reference judges the scheme by.
#[derive(Clone, Debug)]
pub struct FokkerPlanckDensity {
  /// The log-spot nodes.
  pub log_spots: Array1<f64>,
  /// The cell width of each node, the trapezoid weight over the mesh.
  pub weights: Array1<f64>,
  /// The marginal density of the log-spot at each snapshot maturity.
  pub marginals: Vec<Array1<f64>>,
  /// The total numerical mass at each snapshot.
  pub mass: Vec<f64>,
  /// The mean of the variance at each snapshot.
  pub variance_means: Vec<f64>,
}

impl FokkerPlanckDensity {
  /// The fair value of a call at the snapshot `index` by the trapezoid rule
  /// over the marginal, `e^{-r τ} Σ_i ω_i (e^{x_i} - K)^+ P_i`.
  pub fn call_price(&self, index: usize, k: f64, r: f64, tau: f64) -> f64 {
    let payoff = self
      .log_spots
      .iter()
      .zip(&self.weights)
      .zip(&self.marginals[index])
      .map(|((x, w), p)| w * (x.exp() - k).max(0.0) * p)
      .sum::<f64>();
    (-r * tau).exp() * payoff
  }

  /// The mean of the spot under the marginal at the snapshot `index`.
  pub fn spot_mean(&self, index: usize) -> f64 {
    self
      .log_spots
      .iter()
      .zip(&self.weights)
      .zip(&self.marginals[index])
      .map(|((x, w), p)| w * x.exp() * p)
      .sum()
  }
}

/// What the finite-volume calibration produces: the leverage surface on the
/// log-spot mesh, and the density diagnostics at the snapshot maturities.
#[derive(Clone, Debug)]
pub struct FokkerPlanckCalibration {
  pub leverage: LeverageSurface,
  pub density: FokkerPlanckDensity,
}

/// Where a time level's leverage row comes from.
enum Source<'a> {
  /// A given leverage, read at the new level: the plain density solver.
  Fixed(&'a Fn2D<f64>),
  /// The calibration condition, with the previous level's conditional
  /// expectation as the fallback for an empty column.
  Calibrated {
    local_vol: &'a Grid2D<f64>,
    previous: Vec<f64>,
  },
  /// At `η = 0` the variance is `V_t = θ + (v₀ − θ)e^{−κt}`, so `L = σ_LV / √V_t` needs no density,
  /// whose central advection flux would leak that variance delta into its neighbour cells.
  Deterministic {
    local_vol: &'a Grid2D<f64>,
    kappa: f64,
    theta: f64,
    v0: f64,
  },
}

/// The leverage by the forward Kolmogorov equation from [`super::calibrate_leverage`]'s inputs and
/// anchored to `(r, q)` alike; the `t = 0` row is the first full step's, as in the reference.
pub fn calibrate_leverage_fokker_planck(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  local_vol: &Grid2D<f64>,
  snapshot_maturities: &[f64],
  method: &FokkerPlanckMethod,
) -> Result<FokkerPlanckCalibration, QuantError> {
  validate(params, s0, r, q, local_vol, snapshot_maturities)?;
  method.validate(params)?;
  let mesh = method.mesh(params, s0);
  let sqrt_v0 = params.heston.v0.max(CONDITIONAL_VARIANCE_FLOOR).sqrt();
  let first_row = mesh
    .x
    .iter()
    .map(|x| clamp_leverage(local_vol.eval(0.0, x.exp()) / sqrt_v0))
    .collect::<Vec<_>>();
  let deterministic = params.sigma_mixed() == 0.0;
  let source = if deterministic {
    Source::Deterministic {
      local_vol,
      kappa: params.heston.kappa,
      theta: params.heston.theta,
      v0: params.heston.v0.max(0.0),
    }
  } else {
    Source::Calibrated {
      local_vol,
      previous: vec![params.heston.v0.max(0.0); mesh.m1()],
    }
  };
  let (rows, density) = march(
    &mesh,
    params,
    r - q,
    snapshot_maturities,
    method,
    first_row,
    source,
  )?;
  let mut times = rows.iter().map(|(t, _)| *t).collect::<Vec<_>>();
  let mut values = Array2::<f64>::zeros((rows.len(), mesh.m1()));
  for (k, (_, row)) in rows.iter().enumerate() {
    values.row_mut(k).assign(&Array1::from_vec(row.clone()));
  }
  if rows.len() > 1 && !deterministic {
    let first_full_step = if method.damping { 2 } else { 1 };
    let first = values.row(first_full_step).to_owned();
    values.row_mut(0).assign(&first);
  }
  times[0] = 0.0;
  let spots = Array1::from_iter(mesh.x.iter().map(|x| x.exp()));
  Ok(FokkerPlanckCalibration {
    leverage: LeverageSurface::new(spots, Array1::from_vec(times), values),
    density,
  })
}

/// The Heston SLV density under a given leverage (`1` for Heston), by the same scheme without the
/// calibration loop: vanilla prices without Monte Carlo noise, and a check against closed forms.
pub fn heston_slv_density(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  leverage: &Fn2D<f64>,
  snapshot_maturities: &[f64],
  method: &FokkerPlanckMethod,
) -> Result<FokkerPlanckDensity, QuantError> {
  let unit = Grid2D::new(
    Array1::from_vec(vec![0.0]),
    Array1::from_vec(vec![1.0]),
    Array2::from_elem((1, 1), 1.0),
  );
  validate(params, s0, r, q, &unit, snapshot_maturities)?;
  method.validate(params)?;
  let mesh = method.mesh(params, s0);
  let first_row = mesh
    .x
    .iter()
    .map(|x| leverage.call(0.0, x.exp()))
    .collect::<Vec<_>>();
  let (_, density) = march(
    &mesh,
    params,
    r - q,
    snapshot_maturities,
    method,
    first_row,
    Source::Fixed(leverage),
  )?;
  Ok(density)
}

/// Marches the density from the Dirac start over the time grid, one leverage
/// row per (sub)step from `source`, and records the snapshots.
fn march(
  mesh: &Mesh,
  params: &HestonSlvParams,
  carry: f64,
  snapshot_maturities: &[f64],
  method: &FokkerPlanckMethod,
  first_row: Vec<f64>,
  mut source: Source<'_>,
) -> Result<(Vec<(f64, Vec<f64>)>, FokkerPlanckDensity), QuantError> {
  let (rho, xi, kappa, theta) = (
    params.heston.rho,
    params.sigma_mixed(),
    params.heston.kappa,
    params.heston.theta,
  );
  let attainable = 2.0 * kappa * theta < xi * xi;
  let along_v = Direction::along_v(mesh, kappa, theta, xi);
  let mut buffers = StepBuffers::new(mesh.len());
  let grid = time_grid(snapshot_maturities, method.steps_per_year);
  let mut substeps = Vec::with_capacity(grid.len() + 2);
  for (k, pair) in grid.windows(2).enumerate() {
    let (from, to) = (pair[0], pair[1]);
    if method.damping && k < 2 {
      let mid = 0.5 * (from + to);
      substeps.push((from, mid, true));
      substeps.push((mid, to, true));
    } else {
      substeps.push((from, to, false));
    }
  }

  let mut p = mesh.dirac();
  let mut p_next = vec![0.0; mesh.len()];
  let mut rows = vec![(0.0, first_row)];
  let mut density = FokkerPlanckDensity {
    log_spots: Array1::from_vec(mesh.x.clone()),
    weights: Array1::from_vec(mesh.wx.clone()),
    marginals: Vec::with_capacity(snapshot_maturities.len()),
    mass: Vec::with_capacity(snapshot_maturities.len()),
    variance_means: Vec::with_capacity(snapshot_maturities.len()),
  };
  let mut next_snapshot = 0;

  for &(from, to, damping) in &substeps {
    let dt = to - from;
    let lev_prev = rows.last().expect("a row per level").1.clone();
    let mut prev = Level::new(mesh, carry, rho, xi, &lev_prev, attainable);
    let lev_next = match &mut source {
      Source::Fixed(f) => {
        let row = mesh
          .x
          .iter()
          .map(|x| f.call(to, x.exp()))
          .collect::<Vec<_>>();
        let mut next = Level::new(mesh, carry, rho, xi, &row, attainable);
        step(
          mesh,
          &along_v,
          &mut prev,
          &mut next,
          dt,
          method.theta,
          damping,
          &p,
          &mut p_next,
          &mut buffers,
        )?;
        row
      }
      Source::Deterministic {
        local_vol,
        kappa,
        theta,
        v0,
      } => {
        let variance = *theta + (*v0 - *theta) * (-*kappa * to).exp();
        let sqrt_v = variance.max(CONDITIONAL_VARIANCE_FLOOR).sqrt();
        let row = mesh
          .x
          .iter()
          .map(|x| clamp_leverage(local_vol.eval(to, x.exp()) / sqrt_v))
          .collect::<Vec<_>>();
        let mut next = Level::new(mesh, carry, rho, xi, &row, attainable);
        step(
          mesh,
          &along_v,
          &mut prev,
          &mut next,
          dt,
          method.theta,
          damping,
          &p,
          &mut p_next,
          &mut buffers,
        )?;
        row
      }
      Source::Calibrated {
        local_vol,
        previous,
      } => {
        let mut guess = p.clone();
        let mut row = Vec::new();
        for _ in 0..method.inner_iterations {
          let conditional = mesh
            .conditional_variance(&guess, MASS_THRESHOLD)
            .into_iter()
            .zip(previous.iter())
            .map(|(e, fallback)| e.unwrap_or(*fallback))
            .collect::<Vec<_>>();
          row = mesh
            .x
            .iter()
            .zip(&conditional)
            .map(|(x, e)| {
              clamp_leverage(local_vol.eval(to, x.exp()) / e.max(CONDITIONAL_VARIANCE_FLOOR).sqrt())
            })
            .collect();
          let mut next = Level::new(mesh, carry, rho, xi, &row, attainable);
          step(
            mesh,
            &along_v,
            &mut prev,
            &mut next,
            dt,
            method.theta,
            damping,
            &p,
            &mut p_next,
            &mut buffers,
          )?;
          guess.copy_from_slice(&p_next);
          *previous = conditional;
        }
        row
      }
    };
    std::mem::swap(&mut p, &mut p_next);
    rows.push((to, lev_next));
    if next_snapshot < snapshot_maturities.len() && to == snapshot_maturities[next_snapshot] {
      density.marginals.push(Array1::from_vec(mesh.marginal(&p)));
      density.mass.push(mesh.mass(&p));
      density.variance_means.push(mesh.variance_mean(&p));
      next_snapshot += 1;
    }
  }
  Ok((rows, density))
}

#[cfg(test)]
mod tests;
