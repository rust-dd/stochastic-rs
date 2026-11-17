/// Cumulants of $X_T = \ln(S_T/F_T)$ under the model's own `chf`, so `c1` holds no carry
/// `(r − q)T`.
#[derive(Debug, Clone)]
pub struct Cumulants {
  /// First cumulant, the mean of $\ln(S_T/F_T)$.
  pub c1: f64,
  /// Second cumulant (variance of log-return).
  pub c2: f64,
  /// Fourth cumulant; `None` for a model that does not compute it (regime switching, CGMYSV).
  pub c4: Option<f64>,
}

impl Cumulants {
  /// Fang & Oosterlee (2008) half-width `l·√(c2 + √c4)`, `c4` counted as 0 when `None`, not finite
  /// or negative; NaN unless `c2` is finite and positive.
  pub fn truncation_half_width(&self, l: f64) -> f64 {
    if !(self.c2.is_finite() && self.c2 > 0.0) {
      return f64::NAN;
    }
    let c4 = self
      .c4
      .filter(|c4| c4.is_finite() && *c4 >= 0.0)
      .unwrap_or(0.0);
    l * (self.c2 + c4.sqrt()).sqrt()
  }
}

/// `c1` from Fang & Oosterlee's Table 11; `c2` and `c4` are `(−i)ⁿ∂ⁿᵤ ln chf` at `u = 0` of the
/// Little-Trap chf, grouped by powers of `e^{−κt}`.
/// Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31(2), 826–848.
pub(crate) fn heston_cumulants(
  v0: f64,
  kappa: f64,
  theta: f64,
  sigma: f64,
  rho: f64,
  t: f64,
) -> Cumulants {
  let ekt = (-kappa * t).exp();
  let c1 = (1.0 - ekt) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * t;

  let kappa2 = kappa * kappa;
  let kappa3 = kappa2 * kappa;
  let sigma2 = sigma * sigma;
  let rs = rho * sigma;
  let n0 = theta
    * (8.0 * kappa3 * t - 8.0 * kappa2 * rs * t - 8.0 * kappa2
      + 16.0 * kappa * rs
      + 2.0 * kappa * sigma2 * t
      - 5.0 * sigma2)
    + v0 * 2.0 * (4.0 * kappa2 - 4.0 * kappa * rs + sigma2);
  let n1 = theta
    * -4.0
    * (2.0 * kappa2 * rs * t - 2.0 * kappa2 + 4.0 * kappa * rs - kappa * sigma2 * t - sigma2)
    + v0 * 4.0 * kappa * (2.0 * kappa * rs * t - 2.0 * kappa + 2.0 * rs - sigma2 * t);
  let n2 = sigma2 * (theta - 2.0 * v0);
  let c2 = (n0 + n1 * ekt + n2 * ekt * ekt) / (8.0 * kappa3);

  Cumulants {
    c1,
    c2,
    c4: Some(heston_c4(v0, kappa, theta, sigma, rho, t, ekt)),
  }
}

/// Heston `κ₄ = Σₙ aₙeⁿ`, `e = e^{−κt}`, each `aₙ` collected in `θ` and `v0`: the exact Riccati
/// series of `∂⁴ᵤ ln chf` at `u = 0`.
fn heston_c4(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, t: f64, e: f64) -> f64 {
  let (rho2, rho3) = (rho * rho, rho * rho * rho);
  let (sigma2, sigma3) = (sigma * sigma, sigma * sigma * sigma);
  let (sigma4, sigma6) = (sigma2 * sigma2, sigma2 * sigma2 * sigma2);
  let (kappa2, kappa3) = (kappa * kappa, kappa * kappa * kappa);
  let (kappa4, kappa5) = (kappa2 * kappa2, kappa2 * kappa3);
  let (kappa6, kappa7) = (kappa3 * kappa3, kappa3 * kappa4);
  let (t2, t3) = (t * t, t * t * t);
  let a0 = 3.0
    * sigma2
    * (theta
      * (4.0 * rho2 * t + t
        - 4.0 * rho3 * sigma * t / kappa
        - 12.0 * rho2 / kappa
        - 6.0 * rho * sigma * t / kappa
        - 5.0 / (2.0 * kappa)
        + 16.0 * rho3 * sigma / kappa2
        + 6.0 * rho2 * sigma2 * t / kappa2
        + 20.0 * rho * sigma / kappa2
        + 3.0 * sigma2 * t / (2.0 * kappa2)
        - 25.0 * rho2 * sigma2 / kappa3
        - 5.0 * rho * sigma3 * t / (2.0 * kappa3)
        - 11.0 * sigma2 / (2.0 * kappa3)
        + 11.0 * rho * sigma3 / kappa4
        + 5.0 * sigma4 * t / (16.0 * kappa4)
        - 93.0 * sigma4 / (64.0 * kappa5))
      + v0
        * (4.0 * rho2 / kappa + 1.0 / kappa
          - 4.0 * rho3 * sigma / kappa2
          - 6.0 * rho * sigma / kappa2
          + 6.0 * rho2 * sigma2 / kappa3
          + 3.0 * sigma2 / (2.0 * kappa3)
          - 5.0 * rho * sigma3 / (2.0 * kappa4)
          + 5.0 * sigma4 / (16.0 * kappa5)))
    / kappa2;
  let a1 = sigma2
    * (theta
      * (-2.0 * rho3 * sigma * t3 + 6.0 * rho2 * t2 - 12.0 * rho3 * sigma * t2 / kappa
        + 3.0 * rho2 * sigma2 * t3 / kappa
        + 24.0 * rho2 * t / kappa
        - 12.0 * rho * sigma * t2 / kappa
        + 6.0 * t / kappa
        - 36.0 * rho3 * sigma * t / kappa2
        + 21.0 * rho2 * sigma2 * t2 / kappa2
        + 36.0 * rho2 / kappa2
        - 3.0 * rho * sigma3 * t3 / (2.0 * kappa2)
        - 48.0 * rho * sigma * t / kappa2
        + 9.0 * sigma2 * t2 / (2.0 * kappa2)
        + 6.0 / kappa2
        - 48.0 * rho3 * sigma / kappa3
        + 60.0 * rho2 * sigma2 * t / kappa3
        - 21.0 * rho * sigma3 * t2 / (2.0 * kappa3)
        - 48.0 * rho * sigma / kappa3
        + sigma4 * t3 / (4.0 * kappa3)
        + 27.0 * sigma2 * t / (2.0 * kappa3)
        + 60.0 * rho2 * sigma2 / kappa4
        - 105.0 * rho * sigma3 * t / (4.0 * kappa4)
        + 3.0 * sigma4 * t2 / (2.0 * kappa4)
        + 45.0 * sigma2 / (4.0 * kappa4)
        - 45.0 * rho * sigma3 / (2.0 * kappa5)
        + 27.0 * sigma4 * t / (8.0 * kappa5)
        + 21.0 * sigma4 / (8.0 * kappa6))
      + v0
        * (2.0 * rho3 * sigma * t3 - 6.0 * rho2 * t2 + 6.0 * rho3 * sigma * t2 / kappa
          - 3.0 * rho2 * sigma2 * t3 / kappa
          - 12.0 * rho2 * t / kappa
          + 12.0 * rho * sigma * t2 / kappa
          - 6.0 * t / kappa
          + 12.0 * rho3 * sigma * t / kappa2
          - 12.0 * rho2 * sigma2 * t2 / kappa2
          - 12.0 * rho2 / kappa2
          + 3.0 * rho * sigma3 * t3 / (2.0 * kappa2)
          + 24.0 * rho * sigma * t / kappa2
          - 9.0 * sigma2 * t2 / (2.0 * kappa2)
          + 12.0 * rho3 * sigma / kappa3
          - 18.0 * rho2 * sigma2 * t / kappa3
          + 6.0 * rho * sigma3 * t2 / kappa3
          - sigma4 * t3 / (4.0 * kappa3)
          - 9.0 * sigma2 * t / (2.0 * kappa3)
          + 21.0 * rho * sigma3 * t / (4.0 * kappa4)
          - 3.0 * sigma4 * t2 / (4.0 * kappa4)
          + 9.0 * sigma2 / (4.0 * kappa4)
          - 15.0 * rho * sigma3 / (4.0 * kappa5)
          - 3.0 * sigma4 * t / (8.0 * kappa5)
          + 3.0 * sigma4 / (4.0 * kappa6)))
    / kappa;
  let a2 = 3.0
    * sigma2
    * (theta
      * (rho2 * sigma2 * t2 - 2.0 * rho * sigma * t + 0.5 + 4.0 * rho2 * sigma2 * t / kappa
        - rho * sigma3 * t2 / kappa
        - 4.0 * rho * sigma / kappa
        + 3.0 * sigma2 * t / (2.0 * kappa)
        + 5.0 * rho2 * sigma2 / kappa2
        - 7.0 * rho * sigma3 * t / (2.0 * kappa2)
        + sigma4 * t2 / (4.0 * kappa2)
        + 3.0 * sigma2 / (2.0 * kappa2)
        - 3.0 * rho * sigma3 / kappa3
        + 5.0 * sigma4 * t / (8.0 * kappa3)
        + 7.0 * sigma4 / (16.0 * kappa4))
      + v0
        * (-2.0 * rho2 * sigma2 * t2 + 4.0 * rho * sigma * t
          - 1.0
          - 6.0 * rho2 * sigma2 * t / kappa
          + 2.0 * rho * sigma3 * t2 / kappa
          + 6.0 * rho * sigma / kappa
          - 3.0 * sigma2 * t / kappa
          - 6.0 * rho2 * sigma2 / kappa2
          + 5.0 * rho * sigma3 * t / kappa2
          - sigma4 * t2 / (2.0 * kappa2)
          - 3.0 * sigma2 / (2.0 * kappa2)
          + 5.0 * rho * sigma3 / (2.0 * kappa3)
          - 3.0 * sigma4 * t / (4.0 * kappa3)
          - sigma4 / (4.0 * kappa4)))
    / kappa3;
  let a3 = 3.0
    * sigma4
    * (theta
      * (-2.0 * rho * sigma * t + 2.0 - 4.0 * rho * sigma / kappa
        + sigma2 * t / kappa
        + sigma2 / kappa2)
      + v0
        * (6.0 * rho * sigma * t - 6.0 + 10.0 * rho * sigma / kappa
          - 3.0 * sigma2 * t / kappa
          - 2.0 * sigma2 / kappa2))
    / (8.0 * kappa5);
  let a4 = 3.0 * sigma6 * (theta - 4.0 * v0) / (64.0 * kappa7);
  a0 + e * (a1 + e * (a2 + e * (a3 + e * a4)))
}
