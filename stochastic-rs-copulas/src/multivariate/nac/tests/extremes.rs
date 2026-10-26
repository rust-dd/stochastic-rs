use super::*;
use crate::multivariate::nac::sample::kanter;
use crate::multivariate::nac::sample::stable_power;

/// Kanter's draw at a subnormal `α` and the least angle, where `αΘ` underflows to 0: `Z` against mpmath at 120 digits.
#[test]
fn kanter_holds_where_alpha_theta_underflows() {
  let (alpha, theta) = (5.56e-310, std::f64::consts::PI * 1e-15);
  assert_eq!(alpha * theta, 0.0);
  let z = kanter(alpha, theta, (-f64::ln(0.5)).ln());
  assert_eq!(z, 0.36651292058166435);
}

/// `stable_power` is `α ln S` of the positive stable: at `t = e^{c/α}`, `c ∈ {−1, 0, 1}`, the empirical Laplace transform
/// is `exp(−e^c)` within 4 SE, `Var e^{−tS} = L(2t) − L(t)²`, down to `α = 1e-300`, and every draw is finite.
#[test]
fn stable_power_has_the_positive_stable_laplace_transform() {
  let n = 20_000;
  for (alpha, seed) in [(0.5, 1), (0.01, 2), (0.005, 3), (0.002, 4), (1e-300, 5)] {
    let mut rng = Deterministic::new(seed).rng();
    let z = (0..n)
      .map(|_| stable_power(&mut rng, alpha))
      .collect::<Vec<_>>();
    assert!(
      z.iter().all(|z| z.is_finite()),
      "α {alpha}: a non-finite draw"
    );
    for c in [-1.0_f64, 0.0, 1.0] {
      let lst = |scale: f64| (-scale.powf(alpha) * c.exp()).exp();
      let got = z
        .iter()
        .map(|&z| (-((c + z) / alpha).exp()).exp())
        .sum::<f64>()
        / n as f64;
      let se = ((lst(2.0) - lst(1.0).powi(2)) / n as f64).sqrt();
      assert!(
        (got - lst(1.0)).abs() < 4.0 * se,
        "α {alpha} c {c}: {got} vs {} ± {}",
        lst(1.0),
        4.0 * se
      );
    }
  }
}

/// The Clayton root frailty's `w = ln V / θ`, `V ~ Γ(1/θ, 1)`: mean `aψ(a)` and variance `a²ψ'(a)`, `a = 1/θ`, within 4 SE,
/// on both sides of the boost at `a = 1` and up to `θ = f64::MAX`, where `ln V` overflows.
#[test]
fn clayton_frailty_has_the_gamma_log_moments() {
  let n = 20_000;
  // `(θ, aψ(a), a²ψ'(a), a⁴ψ'''(a))` by mpmath at 80 digits, `a` the `f64` that `1/θ` rounds to.
  let cases = [
    (
      0.5,
      0.8455686701969343,
      2.5797362673929056,
      7.903030436269266,
    ),
    (
      200.0,
      -1.002845104556722,
      1.000040824853919,
      6.000000003981887,
    ),
    (
      500.0,
      -1.001147861192707,
      1.000006560555176,
      6.00000000010311,
    ),
    (1e10, -1.0000000000577216, 1.0, 6.0),
    (f64::MAX, -1.0, 1.0, 6.0),
  ];
  for (seed, (theta, mean, var, kappa4)) in (1..).zip(cases) {
    let mut rng = Deterministic::new(seed).rng();
    let w = (0..n)
      .map(|_| Generator::Clayton.root_frailty(&mut rng, theta).w)
      .collect::<Vec<_>>();
    let m = w.iter().sum::<f64>() / n as f64;
    let v = w.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let se_mean = (var / n as f64).sqrt();
    let se_var = ((kappa4 + 2.0 * var * var) / n as f64).sqrt();
    assert!(
      (m - mean).abs() < 4.0 * se_mean,
      "θ {theta}: mean {m} vs {mean} ± {}",
      4.0 * se_mean
    );
    assert!(
      (v - var).abs() < 4.0 * se_var,
      "θ {theta}: variance {v} vs {var} ± {}",
      4.0 * se_var
    );
  }
}
