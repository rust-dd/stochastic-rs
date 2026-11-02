//! The truncated exponential's closed-form fractions of `λL`: the mass per rate's `φ`, and the mean's and variance's
//! offsets with their Bernoulli series where the plain forms cancel.

/// `(1 − e^{−z})/z`, 1 where `z` is below the digits `e^{−z}` keeps.
pub(super) fn one_minus_exp_over(z: f64) -> f64 {
  if z < 1e-8 {
    1.0 - 0.5 * z
  } else {
    -(-z).exp_m1() / z
  }
}

/// The $x^{2k-1}$ coefficients $-B_{2k}/(2k)!$, $k = 1..=7$, of `mean_fraction` past its constant ½, rounded to `f64`.
const MEAN_SERIES: [f64; 7] = [
  -0.08333333333333333,
  0.001388888888888889,
  -3.306878306878307e-05,
  8.267195767195768e-07,
  -2.08767569878681e-08,
  5.284190138687493e-10,
  -1.3382536530684679e-11,
];

/// The $x^{2k-2}$ coefficients $(2k-1)B_{2k}/(2k)!$, $k = 1..=12$, of `variance_fraction`, rounded to `f64`.
const VARIANCE_SERIES: [f64; 12] = [
  0.08333333333333333,
  -0.004166666666666667,
  0.00016534391534391533,
  -5.787037037037037e-06,
  1.8789081289081288e-07,
  -5.812609152556243e-09,
  1.7397297489890083e-10,
  -5.084520444483875e-12,
  1.4596305495672335e-13,
  -4.1322505272603176e-15,
  1.1568905939556482e-16,
  -3.2095268777368804e-18,
];

/// $1/x - 1/(e^x - 1)$ at $x = \lambda L$: the mean's offset from `lower` over `L`; below `x = 0.5` the two terms
/// cancel, so it takes the Bernoulli series there.
pub(super) fn mean_fraction(x: f64) -> f64 {
  if x < 0.5 {
    let x2 = x * x;
    0.5 + x * MEAN_SERIES.iter().rev().fold(0.0, |acc, &c| acc * x2 + c)
  } else {
    1.0 / x - 1.0 / x.exp_m1()
  }
}

/// $1/x^2 - 1/(4\sinh^2(x/2))$ at $x = \lambda L$: the variance over $L^2$; the series below `x = 1`, where the
/// difference amplifies `sinh`'s rounding past 12-fold.
pub(super) fn variance_fraction(x: f64) -> f64 {
  if x < 1.0 {
    let x2 = x * x;
    VARIANCE_SERIES
      .iter()
      .rev()
      .fold(0.0, |acc, &c| acc * x2 + c)
  } else {
    let s = (0.5 * x).sinh();
    1.0 / (x * x) - 1.0 / (4.0 * s * s)
  }
}
