use super::*;

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

/// The generated host step is the closed form's operations in the same order, so it agrees bit
/// for bit. The closed forms take the noise increment `dz`, as the declarations do.
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

/// For a declaration that had to reassociate (no literal on an operator's left, so
/// `r·X(1 − X/K)` is written `r·X·((K − X)/K)`): agreement to rounding, not bit equality.
fn close(a: f64, b: f64) {
  assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{a} vs {b}");
}

/// The families with a folded constant, a guarded division or a transformed state. Each closed
/// form is written from the model, not the declaration, so a drift from the paper fails here.
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

/// The families that clamp, truncate or reflect: the boundary is applied where the model
/// applies it, to the coefficients, to the result, or to both.
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

/// Three independent forwards with the first past its reset: it stays, the live ones take the
/// frozen-drift log-Euler step with their own drift, and the absent fourth slot never moves.
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
  let next = step_full(
    Family::LiborMarket4,
    f,
    &params,
    dt,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    dz,
  );
  assert_eq!(next[0], f[0]);
  for n in 1..3 {
    let drift = sigma[n] * (delta[n] * sigma[n] * f[n] / (1.0 + delta[n] * f[n]));
    let expected = f[n] * ((drift - sigma[n] * sigma[n] / 2.0) * dt + sigma[n] * dz[n]).exp();
    assert!(
      (next[n] - expected).abs() < 1e-12,
      "rate {n}: {} vs {expected}",
      next[n]
    );
  }
  assert_eq!(next[3], 0.0);
  // No reset date passed: every rate is live, including the first.
  let next = step_full(
    Family::LiborMarket4,
    f,
    &params,
    dt,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    dz,
  );
  assert!(next[0] > f[0]);
}
