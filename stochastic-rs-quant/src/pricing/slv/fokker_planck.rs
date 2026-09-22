//! Leverage calibration by the forward Kolmogorov (Fokker–Planck) equation
//! of the Heston SLV model, after Wyns & Du Toit (2016).
//!
//! The joint density $p(t, x, v)$ of the log-spot and the variance satisfies
//!
//! $$
//! \partial_t p = \partial_{xx}\bigl(\tfrac12 L^2 v\, p\bigr)
//!   + \partial_{xv}\bigl(\rho\,\xi L v\, p\bigr)
//!   + \partial_{vv}\bigl(\tfrac12 \xi^2 v\, p\bigr)
//!   - \partial_x\bigl((r - q - \tfrac12 L^2 v)\, p\bigr)
//!   - \partial_v\bigl(\kappa(\theta - v)\, p\bigr),
//! \qquad p(0, \cdot) = \delta_{(x_0, v_0)},
//! $$
//!
//! with $\xi = \eta\sigma$ the mixed vol-of-vol and $L = L(t, e^x)$ the
//! leverage, and the calibration condition
//!
//! $$
//! L^2(t, x) = \frac{\sigma_{\text{LV}}^2(t, e^x)}{\mathbb{E}[V_t \mid X_t = x]},
//! \qquad
//! \mathbb{E}[V_t \mid X_t = x] = \frac{\int_0^\infty v\, p(t, x, v)\, dv}{\int_0^\infty p(t, x, v)\, dv},
//! $$
//!
//! makes the PDE non-linear. The equation is discretised on a vertex-centred
//! `(x, v)` mesh by the finite-volume scheme of the reference's §2 — central
//! advection fluxes, the conservative diffusion flux, the four-corner mixed
//! flux with ghost values and a first-order forward flux on the corner row
//! above the attainable boundary `v = 0`, every boundary flux zero so the
//! numerical mass is conserved — and marched by the Hundsdorfer–Verwer ADI
//! scheme of its §3 at $\theta = \tfrac12 + \tfrac{\sqrt3}{6}$, the first two
//! steps replaced by four implicit-Euler half-steps (Rannacher). The
//! non-linearity is handled by the inner iteration of its §4: at each step
//! the conditional expectation (4.5) is read off the current approximation
//! of the new density by the trapezoid rule on $|P|$, the leverage row (4.4)
//! is rebuilt from it, and the step is redone, `Q` times. A column carrying
//! no mass keeps the previous row's expectation, and the $t = 0$ row is
//! replaced by the first computed one, as the reference does. The density
//! starts as the cell average of the Dirac mass on the node `(x_0, v_0)`, so
//! both are exact nodes of the meshes, which cluster there.
//!
//! References: Wyns, M. & Du Toit, J. (2016), *A finite volume – alternating
//! direction implicit approach for the calibration of stochastic local
//! volatility models*, Int. J. Comput. Math. 94(11), 2239–2267,
//! arXiv:1611.02961; Ren, Y., Madan, D. & Qian, M. Q. (2007), *Calibrating
//! and pricing with embedded local volatility models*, Risk 20(9), 138–143;
//! Tian, Y., Zhu, Z., Lee, G., Klebaner, F. & Hamza, K. (2015), *Calibrating
//! and pricing with a stochastic-local volatility model*, J. Derivatives
//! 22(3), 21–39; in 't Hout, K. J. & Foulon, S. (2010), *ADI finite
//! difference schemes for option pricing in the Heston model with
//! correlation*, Int. J. Numer. Anal. Model. 7(2), 303–320 (the meshes and
//! the ADI schemes).

use anyhow::Result;
use anyhow::bail;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::traits::Fn2D;
use stochastic_rs_distributions::traits::Grid2D;

use self::solver::Direction;
use self::solver::Level;
use self::solver::Mesh;
use self::solver::StepBuffers;
use self::solver::step;
use super::HestonSlvParams;
use super::LeverageSurface;
use super::calibration::CONDITIONAL_VARIANCE_FLOOR;
use super::calibration::clamp_leverage;
use super::calibration::time_grid;
use super::calibration::validate;

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
  /// Variance nodes.
  pub variance_nodes: usize,
  /// Time steps per year; every snapshot maturity is a step boundary.
  pub steps_per_year: usize,
  /// Inner iterations per step on the non-linearity, the reference's `Q`.
  pub inner_iterations: usize,
  /// The Hundsdorfer–Verwer parameter, `½ + √3/6` in the reference.
  pub theta: f64,
  /// Rannacher start-up: the first two steps as four implicit-Euler
  /// half-steps.
  pub damping: bool,
  /// Half-width of the log-spot domain around `ln s0`; `None` takes `ln 30`.
  pub x_half_width: Option<f64>,
  /// Upper variance boundary; `None` takes `15 · max(v0, θ)`, at least `0.5`.
  pub v_max: Option<f64>,
  /// Clustering scale `c` of the log-spot mesh `x = x0 + c sinh ξ`.
  pub x_stretch: f64,
  /// Clustering scale of the variance mesh as a fraction of `max(v0, θ)`.
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

  fn validate(&self, params: &HestonSlvParams) -> Result<()> {
    if self.log_spot_nodes < 3 || self.variance_nodes < 3 {
      bail!("the Fokker–Planck mesh needs at least three nodes per axis");
    }
    if self.steps_per_year == 0 || self.inner_iterations == 0 {
      bail!("steps_per_year and inner_iterations must be positive");
    }
    if !(self.theta.is_finite() && self.theta > 0.0) {
      bail!("theta must be finite and positive, got {}", self.theta);
    }
    if !(self.x_stretch.is_finite() && self.x_stretch > 0.0)
      || !(self.v_stretch.is_finite() && self.v_stretch > 0.0)
    {
      bail!("the mesh stretches must be finite and positive");
    }
    if self.x_half_width.is_some_and(|w| !(w.is_finite() && w > 0.0)) {
      bail!("x_half_width must be finite and positive");
    }
    let v_scale = params.v0.max(params.theta);
    if self.v_max.is_some_and(|v| !(v.is_finite() && v > v_scale)) {
      bail!("v_max must be finite and above both v0 and theta");
    }
    Ok(())
  }

  fn mesh(&self, params: &HestonSlvParams, s0: f64) -> Mesh {
    let v_scale = params.v0.max(params.theta).max(1e-6);
    let v_max = self.v_max.unwrap_or((15.0 * v_scale).max(0.5));
    Mesh::new(
      s0.ln(),
      self.x_half_width.unwrap_or(DEFAULT_LOG_SPOT_HALF_WIDTH),
      self.x_stretch,
      self.log_spot_nodes,
      params.v0.max(0.0),
      v_max,
      self.v_stretch * v_scale,
      self.variance_nodes,
    )
  }
}

/// The density of the log-spot at the snapshot maturities, and the mass and
/// the variance mean the solver carried there — the diagnostics the
/// reference judges the scheme by.
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
}

/// Calibrate the leverage surface by the forward Kolmogorov equation — the
/// module doc has the scheme. Takes what
/// [`calibrate_leverage`](super::calibrate_leverage) takes, the method aside,
/// and is anchored to `(r, q)` the same way.
pub fn calibrate_leverage_fokker_planck(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  local_vol: &Grid2D<f64>,
  snapshot_maturities: &[f64],
  method: &FokkerPlanckMethod,
) -> Result<FokkerPlanckCalibration> {
  validate(params, s0, r, q, local_vol, snapshot_maturities)?;
  method.validate(params)?;
  let mesh = method.mesh(params, s0);
  let sqrt_v0 = params.v0.max(CONDITIONAL_VARIANCE_FLOOR).sqrt();
  let first_row = mesh
    .x
    .iter()
    .map(|x| clamp_leverage(local_vol.eval(0.0, x.exp()) / sqrt_v0))
    .collect::<Vec<_>>();
  let source = Source::Calibrated {
    local_vol,
    previous: vec![params.v0.max(0.0); mesh.m1()],
  };
  let (rows, density) = march(&mesh, params, r - q, snapshot_maturities, method, first_row, source);
  let mut times = rows.iter().map(|(t, _)| *t).collect::<Vec<_>>();
  let mut values = Array2::<f64>::zeros((rows.len(), mesh.m1()));
  for (k, (_, row)) in rows.iter().enumerate() {
    values.row_mut(k).assign(&Array1::from_vec(row.clone()));
  }
  if rows.len() > 1 {
    let first = values.row(1).to_owned();
    values.row_mut(0).assign(&first);
  }
  times[0] = 0.0;
  let spots = Array1::from_iter(mesh.x.iter().map(|x| x.exp()));
  Ok(FokkerPlanckCalibration {
    leverage: LeverageSurface::new(spots, Array1::from_vec(times), values),
    density,
  })
}

/// The density of the Heston SLV model under a **given** leverage — a
/// calibrated surface, or `1` for the Heston model itself — by the same
/// finite-volume scheme, without the calibration loop. Useful to price the
/// vanilla grid of a calibrated model without Monte Carlo noise, and to
/// check the scheme against closed forms.
pub fn heston_slv_density(
  params: &HestonSlvParams,
  s0: f64,
  r: f64,
  q: f64,
  leverage: &Fn2D<f64>,
  snapshot_maturities: &[f64],
  method: &FokkerPlanckMethod,
) -> Result<FokkerPlanckDensity> {
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
  );
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
) -> (Vec<(f64, Vec<f64>)>, FokkerPlanckDensity) {
  let (rho, xi, kappa, theta) = (params.rho, params.sigma_mixed(), params.kappa, params.theta);
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

  for &(from, to, douglas) in &substeps {
    let dt = to - from;
    let lev_prev = rows.last().expect("a row per level").1.clone();
    let mut prev = Level::new(mesh, carry, rho, xi, &lev_prev);
    let lev_next = match &mut source {
      Source::Fixed(f) => {
        let row = mesh.x.iter().map(|x| f.call(to, x.exp())).collect::<Vec<_>>();
        let mut next = Level::new(mesh, carry, rho, xi, &row);
        step(mesh, &along_v, &mut prev, &mut next, dt, method.theta, douglas, &p, &mut p_next, &mut buffers);
        row
      }
      Source::Calibrated { local_vol, previous } => {
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
            .map(|(x, e)| clamp_leverage(local_vol.eval(to, x.exp()) / e.max(CONDITIONAL_VARIANCE_FLOOR).sqrt()))
            .collect();
          let mut next = Level::new(mesh, carry, rho, xi, &row);
          step(mesh, &along_v, &mut prev, &mut next, dt, method.theta, douglas, &p, &mut p_next, &mut buffers);
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
  (rows, density)
}

#[cfg(test)]
mod tests;
