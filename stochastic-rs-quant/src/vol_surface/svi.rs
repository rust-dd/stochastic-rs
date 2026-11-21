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

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use crate::LossMetric;
use crate::calibration::Calibrated;
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

  /// Project onto the admissible box (`b ≥ 0`, `|ρ| ≤ 0.9999`, `σ ≥ 1e-8`, non-negative minimum variance), not butterfly-free (see
  /// [`Self::project_with_butterfly_check`]); a NaN parameter stays NaN instead of landing on a bound.
  pub fn project(&mut self) {
    let zero = T::zero();
    let bound = T::from_f64_fast(0.9999);
    let eps = T::from_f64_fast(1e-8);
    self.b = if self.b > zero || self.b.is_nan() {
      self.b
    } else {
      zero
    };
    self.rho = self.rho.max_or_nan(-bound).min_or_nan(bound);
    self.sigma = self.sigma.max_or_nan(eps);
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
  fn into_dvector(self) -> Array1<f64> {
    Array1::from_vec(vec![self.a, self.b, self.rho, self.m, self.sigma])
  }

  fn from_dvector(v: &Array1<f64>) -> Self {
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

/// Raw SVI fitted to total variances by Levenberg–Marquardt from `initial` or a heuristic start;
/// the loss is in total variance, and `converged` and `message` are the stop's.
pub fn calibrate_svi<T: RealExt>(
  log_moneyness: &[T],
  total_variance: &[T],
  initial: Option<SviRawParams<T>>,
) -> Calibrated<SviRawParams<T>> {
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

  let (result, report) = minimize(
    problem,
    LmOptions {
      tolerance: Some(1e-12),
      patience: 200,
      pivoted_qr: false,
      ..LmOptions::default()
    },
  );

  let mut p64 = SviRawParams::<f64>::from_dvector(&result.params);
  // The objective depends on sigma squared, so either sign represents the fit.
  // Reflect before projecting to preserve its curvature at a negative solution.
  p64.sigma = p64.sigma.abs();
  p64.project();

  let model = result
    .ks
    .iter()
    .map(|&k| p64.total_variance(k))
    .collect::<Vec<_>>();
  let params = SviRawParams {
    a: T::from_f64_fast(p64.a),
    b: T::from_f64_fast(p64.b),
    rho: T::from_f64_fast(p64.rho),
    m: T::from_f64_fast(p64.m),
    sigma: T::from_f64_fast(p64.sigma),
  };
  report.fit(params, &result.ws, &model, &LossMetric::ALL, Vec::new())
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

  let k_range = f64::max_or_nan_fold(ks.iter().copied()) - f64::min_or_nan_fold(ks.iter().copied());

  SviRawParams {
    a: w_mean * 0.5,
    b: slope.abs().max_or_nan(0.01),
    rho: slope.signum() * 0.3,
    m: k_mean,
    sigma: (k_range * 0.3).max_or_nan(0.01),
  }
}

struct SviLmProblem {
  ks: Vec<f64>,
  ws: Vec<f64>,
  params: Array1<f64>,
}

impl LeastSquaresProblem for SviLmProblem {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    self.params.assign(&params);
  }

  fn params(&self) -> Array1<f64> {
    self.params.clone()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let p = SviRawParams::<f64>::from_dvector(&self.params);
    let n = self.ks.len();
    let mut r = Array1::zeros(n);
    for i in 0..n {
      r[i] = p.total_variance(self.ks[i]) - self.ws[i];
    }
    Some(r)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let p = SviRawParams::<f64>::from_dvector(&self.params);
    let n = self.ks.len();
    let mut jac = Array2::zeros((n, 5));

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
mod tests;
