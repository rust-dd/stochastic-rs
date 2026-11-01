use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Normal {
  c: f64,
}

impl Kernel for Normal {
  fn ln_k(&self, z: f64) -> f64 {
    -0.5 * z * z
  }

  fn anchor(&mut self, c: Cut, _at_mode: bool) {
    self.c = c.x;
  }

  fn ln_rel(&self, _z: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    -0.5 * y * (y + 2.0 * self.c)
  }

  fn d1(&self, z: f64, _y: f64) -> f64 {
    -z
  }

  fn d2(&self, _z: f64, _y: f64) -> f64 {
    -1.0
  }

  fn mode(&self) -> Cut {
    Cut::at(0.0)
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }

  fn support(&self) -> (f64, f64) {
    (f64::NEG_INFINITY, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    [0.0, 1.0, 0.0, 0.0]
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
  let t = Truncation::new(Normal::default(), f64::NEG_INFINITY, f64::INFINITY);
  assert!(
    (t.kernel.ln_k(t.centre.x) + t.mass().ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()).abs()
      < 1e-15
  );
  assert!(t.a.x < -9.1 && t.b.x > 9.1 && t.quantile(0.5).value().abs() < 1e-15);
  assert_eq!(t.moments(1.0), [0.0, 1.0, 0.0, 0.0]);
  let (frame, [unit, offset], [var, skew, kurt]) = t.summed_moments(1.0);
  let mean = frame.plus(unit * offset);
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
  let t = Truncation::new(Normal::default(), 0.3, 0.3 + 1e-6);
  assert_eq!(t.centre.x, 0.3 + 0.5e-6);
  assert!((t.moments(1.0)[3] + 1.1999999999999508).abs() < 1e-11);
  assert!((t.cdf(Cut::at(0.3 + 0.25e-6)) - 0.25000002811116198).abs() < 1e-13);
}

/// Below the clip the cdf integrates from where the kernel is negligible against its value at `x`, and the quantile
/// inverts it there; `Φ(−12)/(Φ(2) − Φ(−∞))` and its level by mpmath.
#[test]
fn far_below_the_clip_keeps_its_digits() {
  let t = Truncation::new(Normal::default(), f64::NEG_INFINITY, 2.0);
  let p = t.cdf(Cut::at(-12.0));
  assert!(
    ((p - 1.8178381703127302e-33) / 1.8178381703127302e-33).abs() < 1e-12,
    "{p}"
  );
  assert!((t.quantile(p).value() + 12.0).abs() < 1e-13);
}

/// `Φ⁻¹(1e-300 Φ(2))` by mpmath: a level far below the clip, found by walking the panels from where the kernel is
/// negligible.
#[test]
fn far_below_the_clip_the_quantile_walks_the_tail() {
  let t = Truncation::new(Normal::default(), f64::NEG_INFINITY, 2.0);
  assert!((t.quantile(1e-300).value() + 37.04771702239952).abs() < 1e-13);
}

/// `x^0.1 e^{−x}` on `[0, ∞)`, the Gamma(1.1, 1) kernel, whose lower clip sits near 1e-181.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Power {
  c: f64,
}

impl Kernel for Power {
  fn ln_k(&self, x: f64) -> f64 {
    if x < 0.0 {
      f64::NEG_INFINITY
    } else {
      0.1 * x.ln() - x
    }
  }

  fn anchor(&mut self, c: Cut, _at_mode: bool) {
    self.c = c.x;
  }

  fn ln_rel(&self, x: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    0.1 * (x.ln() - self.c.ln()) - y
  }

  fn d1(&self, x: f64, _y: f64) -> f64 {
    0.1 / x - 1.0
  }

  fn d2(&self, x: f64, _y: f64) -> f64 {
    -0.1 / (x * x)
  }

  fn mode(&self) -> Cut {
    Cut::at(0.1)
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [Some((0.0, 1.1)), None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }

  fn support(&self) -> (f64, f64) {
    (0.0, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    let k: f64 = 1.1;
    [k, k.sqrt(), 2.0 / k.sqrt(), 6.0 / k]
  }
}

/// Far below the clip the ladder reaches the range's own end instead of stopping at its 1e-18 mass rung, so
/// `P(1 + f64(0.1), 1e-200)` (mpmath, 60 and 120 digits) takes a few dozen panels.
#[test]
fn ladder_reaches_an_end_far_below_the_clip() {
  let t = Truncation::new(Power::default(), 0.0, f64::INFINITY);
  let p = t.cdf(Cut::at(1e-200));
  assert!(
    ((p - 9.5557909646525e-221) / 9.5557909646525e-221).abs() < 1e-12,
    "{p}"
  );
  assert!(((t.quantile(p).value() - 1e-200) / 1e-200).abs() < 1e-12);
}

/// `ln(1 + z) − z` to a few ulps across the series and the halved range below `−1/2` (mpmath, 60 and 120 digits).
#[test]
fn ln_1p_mx_keeps_its_digits_to_three_quarters() {
  for (z, want) in [
    (-0.75, -0.6362943611198906),
    (-0.7, -0.5039728043259359),
    (-0.65, -0.3998221244986777),
    (-0.6, -0.316290731874155),
    (-0.55, -0.24850769621777166),
    (-0.5000000000000001, -0.19314718055994543),
    (-0.5, -0.19314718055994531),
    (-0.25, -0.03768207245178093),
    (0.3, -0.03763573553250894),
  ] {
    let got = ln_1p_mx(z);
    assert!(
      ((got - want) / want).abs() <= 4.0 * f64::EPSILON,
      "ln_1p_mx({z}) = {got:e}, want {want:e}"
    );
  }
}
