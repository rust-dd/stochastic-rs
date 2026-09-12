//! # SVI — Stochastic Volatility Inspired parameterization
//!
//! Raw SVI total-variance slice (Gatheral, *The Volatility Surface*, 2006):
//!
//! $$
//! w(k)=a+b\bigl(\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\bigr)
//! $$
//!
//! where $w=\sigma_{\mathrm{imp}}^2 T$ is total implied variance and
//! $k=\ln(K/F)$ is log-forward moneyness.
//!
//! Jump-Wings (JW) parameterization (Gatheral, 2004):
//!
//! $$
//! v_t = a+b\bigl(-\rho m+\sqrt{m^2+\sigma^2}\bigr),\quad
//! \psi_t = \frac{b}{2}\Bigl(\frac{-m}{\sqrt{m^2+\sigma^2}}+\rho\Bigr)
//! $$
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

use nalgebra::DMatrix;
use nalgebra::DVector;

use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::LmOptions;
use crate::calibration::least_squares::minimize;
use crate::traits::RealExt;

/// Raw SVI parameters: $\{a, b, \rho, m, \sigma\}$.
#[derive(Clone, Copy, Debug)]
pub struct SviRawParams<T: RealExt> {
  /// Vertical shift
  pub a: T,
  /// Slope magnitude ($b \geq 0$)
  pub b: T,
  /// Correlation ($|\rho| < 1$)
  pub rho: T,
  /// Horizontal shift
  pub m: T,
  /// Curvature ($\sigma > 0$)
  pub sigma: T,
}

impl<T: RealExt> SviRawParams<T> {
  pub fn new(a: T, b: T, rho: T, m: T, sigma: T) -> Self {
    Self {
      a,
      b,
      rho,
      m,
      sigma,
    }
  }

  /// Evaluate total variance $w(k)$ at log-moneyness $k$.
  #[inline]
  pub fn total_variance(&self, k: T) -> T {
    let dk = k - self.m;
    self.a + self.b * (self.rho * dk + (dk * dk + self.sigma * self.sigma).sqrt())
  }

  /// Evaluate implied volatility $\sigma(k, T)$ at log-moneyness $k$ for expiry $T$.
  #[inline]
  pub fn implied_vol(&self, k: T, tau: T) -> T {
    let w = self.total_variance(k);
    let zero = T::zero();
    if w > zero && tau > zero {
      (w / tau).sqrt()
    } else {
      T::nan()
    }
  }

  /// First derivative $w'(k)$.
  #[inline]
  pub fn w_prime(&self, k: T) -> T {
    let dk = k - self.m;
    let r = (dk * dk + self.sigma * self.sigma).sqrt();
    self.b * (self.rho + dk / r)
  }

  /// Second derivative $w''(k)$.
  #[inline]
  pub fn w_double_prime(&self, k: T) -> T {
    let dk = k - self.m;
    let s2 = self.sigma * self.sigma;
    let r = (dk * dk + s2).sqrt();
    self.b * s2 / (r * r * r)
  }

  /// Minimum total variance $\tilde{v} = a + b\sigma\sqrt{1-\rho^2}$.
  #[inline]
  pub fn min_variance(&self) -> T {
    let one = T::one();
    self.a + self.b * self.sigma * (one - self.rho * self.rho).sqrt()
  }

  /// Check whether raw parameters satisfy basic admissibility:
  /// $b \geq 0$, $|\rho| < 1$, $\sigma > 0$, $\tilde{v} \geq 0$.
  pub fn is_admissible(&self) -> bool {
    let zero = T::zero();
    let one = T::one();
    self.b >= zero && self.rho.abs() < one && self.sigma > zero && self.min_variance() >= zero
  }

  /// Project parameters to satisfy admissibility constraints (parameter
  /// bounds + min-variance non-negativity). Does **not** enforce
  /// butterfly arbitrage-freeness — use [`Self::is_butterfly_arb_free`]
  /// (or [`Self::project_with_butterfly_check`]) when butterfly
  /// arbitrage matters for downstream calibration.
  pub fn project(&mut self) {
    let zero = T::zero();
    let bound = T::from_f64_fast(0.9999);
    let eps = T::from_f64_fast(1e-8);
    self.b = if self.b > zero { self.b } else { zero };
    self.rho = self.rho.max(-bound).min(bound);
    self.sigma = self.sigma.max(eps);
    let v_min = self.min_variance();
    if v_min < zero {
      self.a -= v_min;
    }
  }

  /// Check butterfly arbitrage-freeness via Durrleman (2009) condition:
  /// `g(k) := (1 - k·w'/(2w))^2 − (w')^2·(1/w + 1/4)/4 + w''/2 ≥ 0`
  /// for every log-moneyness `k` on the supplied grid. `w(k)` is the
  /// total variance returned by [`Self::total_variance`]; `w'`, `w''`
  /// are its analytic derivatives.
  ///
  /// Returns `true` if `g(k) ≥ -tol` for all grid points; `tol`
  /// defaults to `1e-9` to absorb floating-point noise around the
  /// boundary.
  ///
  /// Reference: Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility
  /// surfaces", §4 / Durrleman (2009), eq. (1.2).
  pub fn is_butterfly_arb_free(&self, ks: &[T]) -> bool {
    self.is_butterfly_arb_free_with_tol(ks, T::from_f64_fast(1e-9))
  }

  /// Like [`Self::is_butterfly_arb_free`] with an explicit tolerance.
  pub fn is_butterfly_arb_free_with_tol(&self, ks: &[T], tol: T) -> bool {
    for &k in ks {
      let g = self.durrleman_g(k);
      if g < -tol {
        return false;
      }
    }
    true
  }

  /// Durrleman's `g(k)` density-positivity functional (Gatheral-Jacquier
  /// 2014, eq. 4.1). The risk-neutral density of the underlying is
  /// proportional to `g(k)`, so a negative `g(k)` at any strike implies
  /// a butterfly-arbitrage opportunity. Exposed so callers can integrate
  /// it into custom calibration penalty terms.
  pub fn durrleman_g(&self, k: T) -> T {
    let half = T::from_f64_fast(0.5);
    let quarter = T::from_f64_fast(0.25);
    let two = T::from_f64_fast(2.0);
    let one = T::one();
    let w = self.total_variance(k);
    if w <= T::zero() {
      // Below-zero total variance: outside the valid SVI domain. Return
      // -∞ so the boolean check fails loudly.
      return T::neg_infinity();
    }
    let km = k - self.m;
    let r2 = (km * km + self.sigma * self.sigma).sqrt();
    let w_prime = self.b * (self.rho + km / r2);
    let w_double = self.b * self.sigma * self.sigma / (r2 * r2 * r2);
    let term1 = {
      let inner = one - k * w_prime / (two * w);
      inner * inner
    };
    let term2 = w_prime * w_prime * quarter * (one / w + quarter);
    let term3 = half * w_double;
    term1 - term2 + term3
  }

  /// Project, then run [`Self::is_butterfly_arb_free`] over `ks`.
  /// Returns `true` if butterfly arbitrage-free after projection,
  /// `false` if the projected parameters still violate butterfly
  /// arb-freeness on the grid (caller is then free to apply an
  /// additional penalty / re-fit).
  pub fn project_with_butterfly_check(&mut self, ks: &[T]) -> bool {
    self.project();
    self.is_butterfly_arb_free(ks)
  }

  /// Convert to Jump-Wings parameterization.
  pub fn jump_wings(self, tau: T) -> SviJumpWings<T> {
    let half = T::from_f64_fast(0.5);
    let one = T::one();
    let sqrt_m2s2 = (self.m * self.m + self.sigma * self.sigma).sqrt();

    let v_t = self.a + self.b * (-self.rho * self.m + sqrt_m2s2);
    let psi_t = self.b * half * (-self.m / sqrt_m2s2 + self.rho);
    let p_t = self.b * (one - self.rho) * half;
    let c_t = self.b * (one + self.rho) * half;
    let v_tilde = self.a + self.b * self.sigma * (one - self.rho * self.rho).sqrt();

    SviJumpWings {
      v_t,
      psi_t,
      p_t,
      c_t,
      v_tilde,
      tau,
    }
  }

  /// Convert to f64 for calibration internals.
  fn as_f64(self) -> SviRawParams<f64> {
    SviRawParams {
      a: self.a.to_f64().unwrap_or(0.0),
      b: self.b.to_f64().unwrap_or(0.0),
      rho: self.rho.to_f64().unwrap_or(0.0),
      m: self.m.to_f64().unwrap_or(0.0),
      sigma: self.sigma.to_f64().unwrap_or(0.0),
    }
  }
}

impl SviRawParams<f64> {
  fn into_dvector(self) -> DVector<f64> {
    DVector::from_vec(vec![self.a, self.b, self.rho, self.m, self.sigma])
  }

  fn from_dvector(v: &DVector<f64>) -> Self {
    SviRawParams {
      a: v[0],
      b: v[1],
      rho: v[2],
      m: v[3],
      sigma: v[4],
    }
  }
}

/// Jump-Wings (JW) SVI parameterization.
///
/// Parametrizes the smile in terms of observable market quantities:
/// ATM variance, ATM skew, put/call wing slopes, and minimum variance.
#[derive(Clone, Copy, Debug)]
pub struct SviJumpWings<T: RealExt> {
  /// ATM total variance
  pub v_t: T,
  /// ATM skew
  pub psi_t: T,
  /// Left (put) slope: $p = b(1-\rho)/2$
  pub p_t: T,
  /// Right (call) slope: $c = b(1+\rho)/2$
  pub c_t: T,
  /// Minimum total variance
  pub v_tilde: T,
  /// Time to maturity in years.
  pub tau: T,
}

/// Calibrate raw SVI parameters to observed total-variance data via
/// Levenberg-Marquardt least squares.
///
/// # Arguments
/// * `log_moneyness` - Log-forward moneyness values $k_i$
/// * `total_variance` - Observed total variance $w_i = \sigma_i^2 T$
/// * `initial` - Optional initial guess; defaults to a heuristic
pub fn calibrate_svi<T: RealExt>(
  log_moneyness: &[T],
  total_variance: &[T],
  initial: Option<SviRawParams<T>>,
) -> SviRawParams<T> {
  assert_eq!(
    log_moneyness.len(),
    total_variance.len(),
    "k and w must have the same length"
  );

  let ks_f64: Vec<f64> = log_moneyness
    .iter()
    .map(|x| x.to_f64().unwrap_or(0.0))
    .collect();
  let ws_f64: Vec<f64> = total_variance
    .iter()
    .map(|x| x.to_f64().unwrap_or(0.0))
    .collect();

  let init_f64 = initial
    .map(|p| p.as_f64())
    .unwrap_or_else(|| svi_initial_guess(&ks_f64, &ws_f64));

  let problem = SviLmProblem {
    ks: ks_f64,
    ws: ws_f64,
    params: init_f64.into_dvector(),
  };

  let (result, _report) = minimize(
    problem,
    LmOptions {
      tolerance: Some(1e-12),
      patience: 200,
      pivoted_qr: false,
    },
  );

  let mut p64 = SviRawParams::<f64>::from_dvector(&result.params);
  // The objective depends on sigma squared, so either sign represents the fit.
  // Reflect before projecting to preserve its curvature at a negative solution.
  p64.sigma = p64.sigma.abs();
  p64.project();

  SviRawParams {
    a: T::from_f64_fast(p64.a),
    b: T::from_f64_fast(p64.b),
    rho: T::from_f64_fast(p64.rho),
    m: T::from_f64_fast(p64.m),
    sigma: T::from_f64_fast(p64.sigma),
  }
}

/// Heuristic initial guess for SVI from data moments.
fn svi_initial_guess(ks: &[f64], ws: &[f64]) -> SviRawParams<f64> {
  let n = ks.len() as f64;
  let w_mean: f64 = ws.iter().sum::<f64>() / n;
  let k_mean: f64 = ks.iter().sum::<f64>() / n;

  let slope = if n > 1.0 {
    let num: f64 = ks
      .iter()
      .zip(ws.iter())
      .map(|(&k, &w)| (k - k_mean) * (w - w_mean))
      .sum();
    let denom: f64 = ks.iter().map(|&k| (k - k_mean).powi(2)).sum();
    if denom.abs() > 1e-14 {
      num / denom
    } else {
      0.0
    }
  } else {
    0.0
  };

  let k_range = ks.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    - ks.iter().cloned().fold(f64::INFINITY, f64::min);

  SviRawParams {
    a: w_mean * 0.5,
    b: slope.abs().max(0.01),
    rho: slope.signum() * 0.3,
    m: k_mean,
    sigma: (k_range * 0.3).max(0.01),
  }
}

struct SviLmProblem {
  ks: Vec<f64>,
  ws: Vec<f64>,
  params: DVector<f64>,
}

impl LeastSquaresProblem for SviLmProblem {
  fn set_params(&mut self, params: &DVector<f64>) {
    self.params.copy_from(params);
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let p = SviRawParams::<f64>::from_dvector(&self.params);
    let n = self.ks.len();
    let mut r = DVector::zeros(n);
    for i in 0..n {
      r[i] = p.total_variance(self.ks[i]) - self.ws[i];
    }
    Some(r)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let p = SviRawParams::<f64>::from_dvector(&self.params);
    let n = self.ks.len();
    let mut jac = DMatrix::zeros(n, 5);

    for i in 0..n {
      let dk = self.ks[i] - p.m;
      let r = (dk * dk + p.sigma * p.sigma).sqrt();

      jac[(i, 0)] = 1.0;
      jac[(i, 1)] = p.rho * dk + r;
      jac[(i, 2)] = p.b * dk;
      jac[(i, 3)] = p.b * (-p.rho - dk / r);
      jac[(i, 4)] = p.b * p.sigma / r;
    }

    Some(jac)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn svi_evaluation() {
    let p = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.1, 0.2);
    assert!(p.is_admissible());

    let w0 = p.total_variance(0.0);
    assert!(w0 > 0.0);

    let w_right = p.total_variance(0.5);
    let w_left = p.total_variance(-0.5);
    assert!(w_left > w0, "put-side should be higher with negative rho");
    assert!(w_right > 0.0);
  }

  #[test]
  fn svi_derivatives() {
    let p = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.1, 0.2);
    let k = 0.1;
    let h = 1e-7;

    let num_first = (p.total_variance(k + h) - p.total_variance(k - h)) / (2.0 * h);
    let num_second =
      (p.total_variance(k + h) - 2.0 * p.total_variance(k) + p.total_variance(k - h)) / (h * h);

    assert!(
      (p.w_prime(k) - num_first).abs() < 1e-5,
      "w'(k) mismatch: analytic={} numeric={}",
      p.w_prime(k),
      num_first
    );
    let rel_err = (p.w_double_prime(k) - num_second).abs() / p.w_double_prime(k).abs().max(1e-14);
    assert!(
      rel_err < 1e-3,
      "w''(k) mismatch: analytic={} numeric={} rel_err={}",
      p.w_double_prime(k),
      num_second,
      rel_err
    );
  }

  #[test]
  fn svi_calibration_exact() {
    let true_params = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
    let ks: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
    let ws: Vec<f64> = ks.iter().map(|&k| true_params.total_variance(k)).collect();

    let fitted = calibrate_svi(&ks, &ws, None);

    assert!((fitted.a - true_params.a).abs() < 1e-4, "a mismatch");
    assert!((fitted.b - true_params.b).abs() < 1e-4, "b mismatch");
    assert!((fitted.rho - true_params.rho).abs() < 1e-4, "rho mismatch");
    assert!((fitted.m - true_params.m).abs() < 1e-4, "m mismatch");
    assert!(
      (fitted.sigma - true_params.sigma).abs() < 1e-4,
      "sigma mismatch"
    );
  }

  #[test]
  fn jump_wings_round_trip() {
    let raw = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
    let jw = raw.jump_wings(1.0);

    assert!(jw.v_t > 0.0);
    assert!(jw.p_t > 0.0);
    assert!(jw.c_t > 0.0);
    assert!(jw.v_tilde > 0.0);
  }

  #[test]
  fn calibration_preserves_the_smile_with_negative_sigma() {
    let truth = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
    let ks = (-10..=10).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
    let ws = ks
      .iter()
      .map(|&k| truth.total_variance(k))
      .collect::<Vec<_>>();
    let initial = SviRawParams {
      sigma: -0.3,
      ..truth
    };
    let fitted = calibrate_svi(&ks, &ws, Some(initial));
    assert!((fitted.sigma - truth.sigma).abs() < 1e-6);
    for (&k, &w) in ks.iter().zip(&ws) {
      assert!((fitted.total_variance(k) - w).abs() < 1e-10);
    }
  }

  #[test]
  fn calibration_recovers_poorly_conditioned_smiles() {
    for (name, width, truth) in [
      (
        "narrow",
        0.025,
        SviRawParams::new(0.04, 0.2, -0.4, 0.05, 0.15),
      ),
      ("flat", 0.5, SviRawParams::new(0.04, 1e-5, -0.4, 0.05, 0.5)),
      (
        "correlated",
        0.5,
        SviRawParams::new(0.04, 0.2, -0.999, 0.05, 0.05),
      ),
    ] {
      let ks = (-10..=10)
        .map(|i| width * i as f64 / 10.0)
        .collect::<Vec<_>>();
      let ws = ks
        .iter()
        .map(|&k| truth.total_variance(k))
        .collect::<Vec<_>>();
      let fitted = calibrate_svi(&ks, &ws, None);
      // Weakly identified parameters can differ while describing the same smile.
      for (&k, &w) in ks.iter().zip(&ws) {
        assert!(
          (fitted.total_variance(k) - w).abs() < 1e-10,
          "{name}: {fitted:?}"
        );
      }
    }
  }
}
