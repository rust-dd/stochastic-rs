//! One declaration has to produce a host step and a kernel body that agree,
//! so these tests pin both halves against the closed forms they came from.

use super::*;
use super::vocabulary::C_PRELUDE;

/// The generated step and report take the whole state and noise vectors, so
/// a one-component family is exercised through these two shims rather than by
/// spelling the four slots out at every call.
fn step1(family: Family, x: f64, params: &[f64], dt: f64, dz: f64) -> f64 {
  let mut out = [0.0; 4];
  host_step(
    family,
    &[x, 0.0, 0.0, 0.0],
    params,
    dt,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    &[dz, 0.0, 0.0, 0.0],
    &mut out,
  );
  out[0]
}

fn report1(family: Family, x: f64, params: &[f64]) -> f64 {
  let mut out = [0.0; 4];
  host_report(
    family,
    &[x, 0.0, 0.0, 0.0],
    params,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    &mut out,
  );
  out[0]
}

fn gbm_closed(x: f64, mu: f64, sigma: f64, dt: f64, dz: f64) -> f64 {
  x + mu * x * dt + sigma * x * dz
}

fn ou_closed(x: f64, theta: f64, mu: f64, sigma: f64, dt: f64, dz: f64) -> f64 {
  x + theta * (mu - x) * dt + sigma * dz
}

fn cir_closed(x: f64, kappa: f64, theta: f64, sigma: f64, dt: f64, dz: f64) -> f64 {
  let positive = if x > 0.0 { x } else { 0.0 };
  x + kappa * (theta - positive) * dt + sigma * positive.sqrt() * dz
}

/// The generated host step is the same operations in the same order as the
/// closed form, so it must agree bit for bit rather than approximately. The
/// closed forms take the noise **increment**, as the declarations do: the
/// diffusion multiplies by `dz`, not by `sqrt_dt` and then by a normal.
#[test]
fn the_host_step_matches_the_closed_forms() {
  let (dt, sqrt_dt) = (1.0 / 253.0, (1.0f64 / 253.0).sqrt());
  for (x, z) in [(100.0, 0.5), (1e-8, -2.25), (-0.01, 3.0)] {
    assert_eq!(
      step1(Family::GeometricBrownian, x, &[0.05, 0.2], dt, sqrt_dt * z),
      gbm_closed(x, 0.05, 0.2, dt, sqrt_dt * z)
    );
    assert_eq!(
      step1(
        Family::OrnsteinUhlenbeck,
        x,
        &[0.5, 0.02, 0.1],
        dt,
        sqrt_dt * z
      ),
      ou_closed(x, 0.5, 0.02, 0.1, dt, sqrt_dt * z)
    );
    assert_eq!(
      step1(Family::SquareRoot, x, &[0.5, 0.04, 0.1], dt, sqrt_dt * z),
      cir_closed(x, 0.5, 0.04, 0.1, dt, sqrt_dt * z)
    );
  }
}

/// Full truncation reports the positive part; every other family reports its
/// state unchanged, at `t = 0` as much as later.
#[test]
fn the_host_report_truncates_only_the_square_root_family() {
  // A report binds the family's whole parameter list before it runs, as a
  // step does, so it is handed a full buffer even where it names nothing.
  let p = &[0.0f64; 8];
  assert_eq!(report1(Family::GeometricBrownian, -2.0, p), -2.0);
  assert_eq!(report1(Family::OrnsteinUhlenbeck, -2.0, p), -2.0);
  assert_eq!(report1(Family::SquareRoot, -2.0, p), 0.0);
  assert_eq!(report1(Family::SquareRoot, 0.25, p), 0.25);
}

/// The family codes are the kernels' ABI: the generated C compares against
/// these numbers, so they are pinned rather than derived from declaration
/// order by accident.
#[test]
fn the_family_codes_are_pinned() {
  assert_eq!(Family::GeometricBrownian.code(), 0);
  assert_eq!(Family::OrnsteinUhlenbeck.code(), 1);
  assert_eq!(Family::SquareRoot.code(), 2);
}

/// The emitted C binds every parameter as a local in declaration order and
/// then assigns the step, which is what makes the same tokens compile in both
/// languages.
#[test]
fn the_emitted_c_binds_parameters_and_steps() {
  assert!(C_STEP.contains("if (family == 0u) {"));
  assert!(C_STEP.contains("const REAL mu = params[0];"));
  assert!(C_STEP.contains("const REAL sigma = params[1];"));
  // The state and the noise are bound by position from the launch's own
  // vectors, so a one-component family reads slot zero of each.
  assert!(C_STEP.contains("const REAL x = state[0];"));
  assert!(C_STEP.contains("const REAL dz = noise[0];"));
  // Every component is computed into a temporary before any is stored, so a
  // system's second component still sees the state as it stood.
  assert!(C_STEP.contains("const REAL __n0 = x + mu * x * dt + sigma * x * dz;"));
  assert!(C_STEP.contains("state[0] = __n0;"));
  assert!(C_STEP.contains("const REAL kappa = params[0];"));
  assert!(C_STEP.contains("const REAL theta = params[1];"));
  assert!(C_STEP.contains("const REAL sigma = params[2];"));
  assert!(C_STEP.contains("sqrt(positive(x))"));
}

/// The report blocks carry the same expression the host runs.
#[test]
fn the_emitted_c_reports_per_family() {
  assert!(C_REPORT.contains("if (family == 2u) {"));
  assert!(C_REPORT.contains("const REAL __n0 = positive(x);"));
  assert!(C_REPORT.contains("const REAL __n0 = x;"));
  assert!(C_REPORT.contains("reported[0] = __n0;"));
}

/// A `bind` becomes a local in every language: a `let` on the host and in a
/// CubeCL kernel, a `const REAL` in the emitted C. Without that the clamped
/// state a family names once would have to be spelled out at each use.
#[test]
fn a_bind_becomes_a_c_local() {
  assert!(C_STEP.contains("const REAL xi = positive(x);"));
  assert!(C_STEP.contains(
    "const REAL __n0 = positive(xi + kappa * (theta - xi) * xi * dt + sigma * sqrt(xi) * dz);"
  ));
  assert_eq!(
    step1(Family::FellerLogistic, -3.0, &[0.5, 0.04, 0.1], 0.01, 0.02),
    0.0,
    "a negative state truncates to zero before the coefficients see it"
  );
}

/// A report may name the family's parameters, which is what lets the
/// displaced diffusion step the shifted variable and report the shift back
/// out.
#[test]
fn a_report_sees_the_family_parameters() {
  assert!(C_REPORT.contains("const REAL beta = params[2];"));
  assert!(C_REPORT.contains("const REAL __n0 = x - beta;"));
  assert_eq!(report1(Family::Displaced, 105.0, &[0.05, 0.2, 5.0]), 100.0);
}

/// Every function the families use has a C definition, or the kernel would
/// not link.
#[test]
fn every_function_used_has_a_c_definition() {
  for name in [
    "sqrt", "exp", "ln", "pow", "abs", "negate", "tanh", "positive", "max", "min", "lit", "less",
    "leq", "geq", "pick",
  ] {
    assert!(
      C_PRELUDE.contains(&format!("#define {name}(")),
      "{name} has no C definition"
    );
  }
  for used in ["positive(", "sqrt("] {
    assert!(C_STEP.contains(used));
    assert!(C_PRELUDE.contains(&format!("#define {}", used.trim_end_matches('('))));
  }
}

/// Almost every family is the closed form's own operations in the same
/// order, so bit equality is the right assertion. Where a declaration had to
/// reassociate — a literal may not sit on the left of an operator, so
/// `r·X(1 − X/K)` is written `r·X·((K − X)/K)` — the two agree to floating
/// point noise instead, which this asserts explicitly rather than papering
/// over with a loose tolerance everywhere.
fn close(a: f64, b: f64) {
  assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{a} vs {b}");
}

/// The families with enough structure to be worth mis-transcribing: a folded
/// constant, a guarded division, a state stepped in a transformed space.
/// Each closed form here is written from the model, not from the
/// declaration, so a declaration that drifts from the paper fails here
/// rather than in a device comparison where noise could hide it.
#[test]
fn the_structured_families_match_their_models() {
  let (dt, dz) = (1.0 / 253.0, 0.031);
  let x = 0.05_f64;

  // Pearson: dX = κ(μ−X)dt + √|2κ(aX² + bX + c)| dW, with 2κ folded.
  let (kappa, mu, a, b, c) = (3.0, 0.05, 0.1, 0.2, 0.01);
  close(
    step1(
      Family::Pearson,
      x,
      &[kappa, mu, a, b, c, 2.0 * kappa],
      dt,
      dz,
    ),
    x + kappa * (mu - x) * dt + (2.0 * kappa * (a * x * x + b * x + c)).abs().sqrt() * dz,
  );

  // Feller root: dX = X(θ₁ − X(θ₃³ − θ₁θ₂))dt + θ₃|X|^{3/2} dW, with the
  // drift's constant folded.
  let (t1, t2, t3) = (0.5_f64, 0.3_f64, 0.2_f64);
  let decay = t3.powi(3) - t1 * t2;
  close(
    step1(Family::FellerRoot, x, &[t1, decay, t3], dt, dz),
    x + x * (t1 - x * decay) * dt + t3 * x.abs().powf(1.5) * dz,
  );

  // Aït-Sahalia: the 1/X drift, guarded away from the origin, over a
  // square-rooted diffusion.
  let p = [0.0001, 0.15, -3.0, 0.0, 0.0004, 0.0, 0.05, 1.5];
  let guarded = if x.abs() < 1e-12 { 1e-12 } else { x };
  close(
    step1(Family::AitSahalia, x, &p, dt, dz),
    x + (p[0] / guarded + p[1] + p[2] * x + p[3] * x * x) * dt
      + (p[4] + p[5] * x + p[6] * x.abs().powf(p[7])).abs().sqrt() * dz,
  );

  // The same drift with the diffusion left unsquared.
  close(
    step1(Family::NonLinear, x, &p, dt, dz),
    x + (p[0] / guarded + p[1] + p[2] * x + p[3] * x * x) * dt
      + (p[4] + p[5] * x + p[6] * x.abs().powf(p[7])) * dz,
  );

  // Hyperbolic diffusion: ½σ² folded out of the drift.
  let (beta, gamma, delta, m, sigma) = (0.5, 1.0, 1.0, 0.0, 0.3);
  close(
    step1(
      Family::HyperbolicDiffusion,
      x,
      &[beta, gamma, delta, m, sigma, 0.5 * sigma * sigma],
      dt,
      dz,
    ),
    x + 0.5 * sigma * sigma * (beta - gamma * x / (delta * delta + (x - m) * (x - m)).sqrt()) * dt
      + sigma * dz,
  );

  // Verhulst: the one reassociated declaration.
  let (r, k) = (1.0, 2.0);
  close(
    step1(Family::Verhulst, x, &[r, k, sigma], dt, dz),
    x + r * x * (1.0 - x / k) * dt + sigma * x * dz,
  );
}

/// The families that clamp, truncate or reflect: what matters is that the
/// boundary is applied where the model applies it — to the coefficients, to
/// the result, or to both.
#[test]
fn the_bounded_families_apply_their_boundaries() {
  let (dt, dz) = (1.0 / 253.0, 0.4);

  // Kimura clamps the coefficients into [0, 1] and clamps the result again.
  let (a, sigma) = (0.5, 0.2);
  let clamped = |x: f64| {
    let xi = x.clamp(0.0, 1.0);
    (xi + a * xi * (1.0 - xi) * dt + sigma * (xi * (1.0 - xi)).sqrt() * dz).clamp(0.0, 1.0)
  };
  for x in [-0.3, 0.0, 0.5, 1.0, 1.7] {
    assert_eq!(step1(Family::Kimura, x, &[a, sigma], dt, dz), clamped(x));
  }

  // The squared-Bessel recursion truncates; its reflected twin mirrors.
  for x in [-0.2, 0.0, 1.0] {
    let raw = x + 3.0 * dt + 2.0 * x.abs().sqrt() * dz;
    assert_eq!(
      step1(Family::SquaredBesselState, x, &[3.0, 2.0], dt, dz),
      raw.max(0.0)
    );
    assert_eq!(
      step1(Family::SquaredBesselStateReflected, x, &[3.0, 2.0], dt, dz),
      raw.abs()
    );
  }

  // A bounded correlation never leaves [−0.9999, 0.9999], whatever the noise.
  let stepped = step1(Family::BoundedCorrelation, 0.99, &[1.0, 0.3, 5.0], dt, 3.0);
  assert!((-0.9999..=0.9999).contains(&stepped), "{stepped}");

  // Teng's process is stepped unbounded and reported through a tanh, so the
  // reported value is in (−1, 1) however far the state has wandered.
  let p = [1.0, 0.3, 0.2];
  assert_eq!(
    report1(Family::TanhOrnsteinUhlenbeck, 40.0, &p),
    40.0_f64.tanh()
  );
  assert_eq!(
    step1(Family::TanhOrnsteinUhlenbeck, 0.4, &p, dt, dz),
    0.4 + p[0] * (p[1] - 0.4_f64.tanh()) * dt + p[2] * dz
  );
}

/// The generated step over the full state and noise, for the families whose
/// state has more than one slot.
#[allow(clippy::too_many_arguments)]
fn step_full(
  family: Family,
  state: [f64; 4],
  params: &[f64],
  dt: f64,
  ct: f64,
  u: f64,
  u2: f64,
  sj: f64,
  iv: f64,
  noise: [f64; 4],
) -> [f64; 4] {
  let mut out = [0.0; 4];
  host_step(
    family, &state, params, dt, ct, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, u, u2,
    0.0, 0.0, sj, 0.0, 0.0, 0.0, 0.0, iv, 0.0, &noise, &mut out,
  );
  out
}

/// No family declares two of the history, series and table clauses: the
/// frame keeps one per-path array for the three.
#[test]
fn a_family_declares_at_most_one_per_path_block() {
  for family in Family::ALL {
    let blocks = usize::from(family.history_slot().is_some())
      + usize::from(family.has_series())
      + usize::from(family.has_table());
    assert!(blocks <= 1, "{family:?} declares {blocks} per-path blocks");
  }
}

/// Rosiński's term size: the arrival bound `(Γ rate)^{-1/α}` against the
/// tempered draw `e_scale E^{e_pow} U^{1/α} / λ_side`, on the side the
/// uniform picks.
#[test]
fn series_size_is_the_smaller_of_bound_and_draw_on_the_drawn_side() {
  let (rate, alpha, e_scale, e_pow, w_plus, lp, lm) = (0.25, 0.5, 0.5, 0.5, 0.4, 2.0, 6.0);
  let params = [0.01, rate, 1.0 / alpha, e_scale, e_pow, w_plus, lp, lm];
  let (gj, ej, uj): (f64, f64, f64) = (2.0, 1.5, 0.3);
  let cap = (gj * rate).powf(-1.0 / alpha);
  for (uv, side, sign) in [(0.2, lp, 1.0), (0.9, lm, -1.0)] {
    let draw = e_scale * ej.powf(e_pow) * uj.powf(1.0 / alpha) / side;
    let size = host_series(Family::TemperedStableSeries, &[0.0; 4], &params, 0.01, gj, ej, uj, uv)
      .expect("a series family");
    assert!((size - sign * cap.min(draw)).abs() < 1e-12, "uv = {uv}: {size}");
  }
  assert!(host_series(Family::GeometricBrownian, &[0.0; 4], &[0.05, 0.2], 0.01, gj, ej, uj, 0.2).is_none());
}

/// The live series of the stochastic-volatility CGMY reads the variance: the
/// arrival bound's rate is `rate0 / v`, so a larger variance admits larger
/// terms, and the family says so of itself.
#[test]
fn live_series_size_reads_the_variance_slot() {
  assert!(Family::StochasticVolatilityCgmy.series_live());
  assert!(!Family::TemperedStableSeries.series_live());
  let (rate0, alpha, lp, lm) = (0.25_f64, 0.5_f64, 2.0_f64, 6.0_f64);
  let params = [rate0, 1.0 / alpha, lp, lm, -0.1, 400.0, 0.98, 0.3];
  let (gj, ej, uj, uv): (f64, f64, f64, f64) = (2.0, 1.5, 0.3, 0.2);
  for v in [0.04_f64, 0.16_f64] {
    let cap = (gj * rate0 / v).powf(-1.0 / alpha);
    let draw = ej * uj.powf(1.0 / alpha) / lp;
    let size = host_series(Family::StochasticVolatilityCgmy, &[0.0, v, 0.0, 0.0], &params, 0.01, gj, ej, uj, uv)
      .expect("a series family");
    assert!((size - cap.min(draw)).abs() < 1e-12, "v = {v}: {size}");
  }
}

/// Kanter's positive-stable draw at the table's own scale.
#[test]
fn table_increment_is_the_positive_stable_draw_at_the_spacing() {
  let (alpha, c, tv) = (0.7_f64, 1.5_f64, 0.05_f64);
  let params = [alpha, c, 1.0 / alpha, 1.0 - alpha, (1.0 - alpha) / alpha, std::f64::consts::PI];
  let (uj, uv) = (0.37_f64, 0.61_f64);
  let u = uj * std::f64::consts::PI;
  let w = -uv.ln();
  let s1 = (alpha * u).sin() / u.sin().powf(1.0 / alpha);
  let s2 = (((1.0 - alpha) * u).sin() / w).powf((1.0 - alpha) / alpha);
  let inc = host_table(Family::InverseStableSubordinator, &params, 0.01, uj, uv, tv)
    .expect("a table family");
  assert!(
    (inc - (c * tv).powf(1.0 / alpha) * s1 * s2).abs() < 1e-12,
    "increment {inc}"
  );
  assert!(host_table(Family::GeometricBrownian, &[0.05, 0.2], 0.01, uj, uv, tv).is_none());
}

/// Dassios and Zhao's exact recursion: the wait is the smaller of the
/// baseline's exponential and the excitation's own arrival, which never
/// comes once `1 + β ln u / s` is not positive.
#[test]
fn hawkes_step_is_the_dassios_zhao_recursion() {
  let (mu, alpha, beta) = (1.0_f64, 0.5_f64, 1.5_f64);
  let params = [mu, alpha, beta];
  let (t, s) = (1.0_f64, 0.8_f64);
  let (u2, s2) = (0.3_f64, -(0.3_f64).ln() / mu);
  // A uniform near one: the excitation's own arrival comes first.
  let u = 0.95_f64;
  let d = 1.0 + beta * u.ln() / s;
  let s1 = -d.ln() / beta;
  assert!(s1 < s2, "the case is meant to have the excitation fire first");
  let next = step_full(Family::HawkesEvents, [t, s, 0.0, 0.0], &params, 1.0, 0.0, u, u2, 0.0, 0.0, [0.0; 4]);
  assert!((next[0] - (t + s1)).abs() < 1e-12);
  assert!((next[1] - (s * (-beta * s1).exp() + alpha)).abs() < 1e-12);
  // A middling uniform: the excitation would fire, but the baseline is sooner.
  let u = 0.6_f64;
  let d = 1.0 + beta * u.ln() / s;
  assert!(d > 0.0 && -d.ln() / beta > s2, "the case is meant to have the baseline fire first");
  let next = step_full(Family::HawkesEvents, [t, s, 0.0, 0.0], &params, 1.0, 0.0, u, u2, 0.0, 0.0, [0.0; 4]);
  assert!((next[0] - (t + s2)).abs() < 1e-12);
  // Too small a uniform: the excitation never fires and the baseline does.
  let next = step_full(Family::HawkesEvents, [t, s, 0.0, 0.0], &params, 1.0, 0.0, 0.01, u2, 0.0, 0.0, [0.0; 4]);
  assert!((next[0] - (t + s2)).abs() < 1e-12);
  assert!((next[1] - (s * (-beta * s2).exp() + alpha)).abs() < 1e-12);
}

/// Three independent forwards with the first past its reset: it stays, the
/// live ones take the frozen-drift log-Euler step with their own drift, and
/// the absent fourth slot never moves.
#[test]
fn libor_market_step_freezes_reset_rates_and_steps_the_live_ones() {
  let (sigma, delta) = ([0.2_f64, 0.25, 0.3, 0.0], [0.5_f64, 0.75, 0.25, 0.0]);
  let identity = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
  let mut params = Vec::new();
  params.extend_from_slice(&sigma);
  params.extend_from_slice(&delta);
  params.extend_from_slice(&identity);
  let f = [0.03_f64, 0.035, 0.04, 0.0];
  let dz = [0.1_f64, -0.2, 0.3, 0.0];
  let dt = 0.01;
  // One reset date passed: rate 0 is frozen, the drift sums start at it.
  let next = step_full(Family::LiborMarket4, f, &params, dt, 1.0, 0.0, 0.0, 0.0, 0.0, dz);
  assert_eq!(next[0], f[0]);
  for n in 1..3 {
    let drift = sigma[n] * (delta[n] * sigma[n] * f[n] / (1.0 + delta[n] * f[n]));
    let expected = f[n] * ((drift - sigma[n] * sigma[n] / 2.0) * dt + sigma[n] * dz[n]).exp();
    assert!((next[n] - expected).abs() < 1e-12, "rate {n}: {} vs {expected}", next[n]);
  }
  assert_eq!(next[3], 0.0);
  // No reset date passed: every rate is live, including the first.
  let next = step_full(Family::LiborMarket4, f, &params, dt, 0.0, 0.0, 0.0, 0.0, 0.0, dz);
  assert!(next[0] > f[0]);
}
