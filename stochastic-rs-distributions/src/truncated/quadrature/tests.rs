use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Normal;

impl Kernel for Normal {
  fn ln_k(&self, z: f64) -> f64 {
    -0.5 * z * z
  }

  fn ln_rel(&self, _z: f64, y: f64, c: f64, _branch: Option<(f64, f64)>) -> f64 {
    -0.5 * y * (y + 2.0 * c)
  }

  fn d1(&self, z: f64) -> f64 {
    -z
  }

  fn d2(&self, _z: f64) -> f64 {
    -1.0
  }

  fn mode(&self) -> f64 {
    0.0
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }
}

/// `∫_{-1}^{1} x^126 dx = 2/127`: the highest degree a 64-node rule integrates exactly.
#[test]
fn gl64_is_exact_to_degree_127() {
  let sum: f64 = GL64_X
    .iter()
    .zip(&GL64_W)
    .map(|(x, w)| 2.0 * w * x.powi(126))
    .sum();
  assert!((sum - 2.0 / 127.0).abs() < 1e-16, "{sum}");
  let total: f64 = GL64_W.iter().sum::<f64>() * 2.0;
  assert!((total - 2.0).abs() < 4.0 * f64::EPSILON);
}

/// `∫ e^{−z²/2} = √(2π)` over the whole line, the clip's 1e-18 below it.
#[test]
fn whole_line_mass_is_sqrt_two_pi() {
  let t = Truncation::new(Normal, f64::NEG_INFINITY, f64::INFINITY);
  assert!(
    (t.kernel.ln_k(t.centre) + t.m0.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()).abs() < 1e-15
  );
  assert!(t.a < -9.1 && t.b > 9.1 && t.quantile(0.5).abs() < 1e-15);
  let [mean, var, skew, kurt] = t.moments();
  assert!(
    mean.abs() < 1e-15 && (var - 1.0).abs() < 1e-14 && skew.abs() < 1e-14 && kurt.abs() < 1e-13
  );
  assert!(
    (t.entropy() - 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln()).abs() < 1e-15
  );
}

/// A narrow interval far from the mode takes its midpoint as reference; mpmath's cdf at a quarter of the width.
#[test]
fn narrow_interval_centres_on_its_midpoint() {
  let t = Truncation::new(Normal, 0.3, 0.3 + 1e-6);
  assert_eq!(t.centre, 0.3 + 0.5e-6);
  assert!((t.moments()[3] + 1.1999999999999508).abs() < 1e-11);
  assert!((t.cdf(0.3 + 0.25e-6) - 0.25000002811116198).abs() < 1e-13);
}

/// Below the clip the cdf integrates from where the kernel is negligible against its value at `x`, and the quantile
/// inverts it there; `Φ(−12)/(Φ(2) − Φ(−∞))` and its level by mpmath.
#[test]
fn far_below_the_clip_keeps_its_digits() {
  let t = Truncation::new(Normal, f64::NEG_INFINITY, 2.0);
  let p = t.cdf(-12.0);
  assert!(
    ((p - 1.8178381703127302e-33) / 1.8178381703127302e-33).abs() < 1e-12,
    "{p}"
  );
  assert!((t.quantile(p) + 12.0).abs() < 1e-13);
}

/// `Φ⁻¹(1e-300 Φ(2))` by mpmath: a level far below the clip, found by walking the panels from where the kernel is
/// negligible.
#[test]
fn far_below_the_clip_the_quantile_walks_the_tail() {
  let t = Truncation::new(Normal, f64::NEG_INFINITY, 2.0);
  assert!((t.quantile(1e-300) + 37.04771702239952).abs() < 1e-13);
}

/// `x^0.1 e^{−x}` on `[0, ∞)`, the Gamma(1.1, 1) kernel, whose lower clip sits near 1e-181.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Power;

impl Kernel for Power {
  fn ln_k(&self, x: f64) -> f64 {
    if x < 0.0 {
      f64::NEG_INFINITY
    } else {
      0.1 * x.ln() - x
    }
  }

  fn ln_rel(&self, x: f64, y: f64, c: f64, _branch: Option<(f64, f64)>) -> f64 {
    0.1 * (x.ln() - c.ln()) - y
  }

  fn d1(&self, x: f64) -> f64 {
    0.1 / x - 1.0
  }

  fn d2(&self, x: f64) -> f64 {
    -0.1 / (x * x)
  }

  fn mode(&self) -> f64 {
    0.1
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [Some((0.0, 0.1)), None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }
}

/// Far below the clip the ladder reaches the range's own end instead of stopping at its 1e-18 mass rung, so
/// `P(1.1, 1e-200)` (mpmath) takes a few dozen panels.
#[test]
fn ladder_reaches_an_end_far_below_the_clip() {
  let t = Truncation::new(Power, 0.0, f64::INFINITY);
  let p = t.cdf(1e-200);
  assert!(
    ((p - 9.555790964652526e-221) / 9.555790964652526e-221).abs() < 1e-12,
    "{p}"
  );
  assert!(((t.quantile(p) - 1e-200) / 1e-200).abs() < 1e-12);
}
