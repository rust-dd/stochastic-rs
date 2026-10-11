//! # Frank
//!
//! $$
//! C_\theta(u,v)=-\frac1\theta\log\!\left(1+\frac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\right)
//! $$
//!
//! Reference: Domino, K. (2018), "Selected Methods for non-Gaussian Data Analysis", arXiv:1811.10486, Example 4.3.2, eq. (4.54) (the conditional inverse; its printed fraction bar wrongly spans the leading 1, which belongs outside the fraction).
//! Reference: Hofert, M., Mächler, M. & McNeil, A. J. (2011), "Likelihood inference for Archimedean copulas", arXiv:1108.6032v1, Table 2 (Kendall's τ); Journal of Multivariate Analysis 110 (2012), 133–150.

use std::sync::OnceLock;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::conditional::conditional_cdf;
use crate::bivariate::conditional::conditional_quantiles;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;
use crate::traits::bivariate::admitted;
use crate::traits::bivariate::invert_tau;
use crate::traits::bivariate::tau_out_of_range;

/// `4B₂ₖ/((2k + 1)(2k)!)`, the coefficients of Frank's τ in odd powers of θ (convergent for `|θ| < 2π`); 18 of them
/// reach `1e-20` at `|θ| = 2`.
const TAU_SERIES: [f64; 18] = [
  0.1111111111111111,
  -0.0011111111111111111,
  1.889644746787604e-05,
  -3.6743092298647856e-07,
  7.5915479955884e-09,
  -1.6259046580576902e-10,
  3.568676408182581e-12,
  -7.97571834428843e-14,
  1.8075920118479673e-15,
  -4.142607044872499e-17,
  9.580874484104748e-19,
  -2.2327143497300036e-20,
  5.236603021673285e-22,
  -1.2349679209706961e-23,
  2.926390261080881e-25,
  -6.963382628936004e-27,
  1.66305425784556e-28,
  -3.984859395313849e-30,
];

#[derive(Debug, Clone)]
pub struct Frank {
  pub r#type: BivariateFamily,
  pub theta: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Frank {
  pub fn new(theta: Option<f64>) -> Self {
    Self {
      r#type: BivariateFamily::Frank,
      theta,
      theta_bounds: (f64::NEG_INFINITY, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }

  /// `h⁻¹(y | v)` at `θ > 0` given `v̄ = 1 − v`, before the domain rule: `−ln(1 + x)/θ`,
  /// `x = y·expm1(−θ)/(y + (1 − y)e^{−θv})`, `1 + x` as a ratio once `x ≤ −½`, its logarithm term by term past `θ ≈ 708`.
  pub(crate) fn inverse(theta: f64) -> impl Fn(f64, f64, f64) -> f64 {
    let (expm1_theta, exp_theta) = ((-theta).exp_m1(), (-theta).exp());
    let ratio_is_normal = exp_theta >= f64::MIN_POSITIVE;
    move |y, v, v_bar| {
      let exp_theta_v = (-theta * v).exp();
      let denominator = y + (1.0 - y) * exp_theta_v;
      let x = y * expm1_theta / denominator;
      if x > -0.5 {
        -x.ln_1p() / theta
      } else if ratio_is_normal {
        -(((1.0 - y) * exp_theta_v + y * exp_theta) / denominator).ln() / theta
      } else if y == 0.0 || y == 1.0 {
        y
      } else {
        let numerator = (1.0 - y) + y * (-theta * v_bar).exp();
        v - (numerator.ln() - denominator.ln()) / theta
      }
    }
  }

  /// `∂_v C = g(u)/(e^{θ(v − u)}g(v) + g(1 − v))`, `g(z) = expm1(−θz)`: same-signed terms, free of the `0/0` an
  /// underflowing `e^{−θv}` leaves in the unfactored form; `θ < 0` through the same reflection.
  fn h(theta: f64) -> impl Fn(f64, f64) -> f64 {
    let magnitude = theta.abs();
    move |u, v| {
      if theta == 0.0 {
        return u;
      }
      let (v, v_bar) = if theta > 0.0 {
        (v, 1.0 - v)
      } else {
        (1.0 - v, v)
      };
      let denominator =
        (magnitude * (v - u)).exp() * (-magnitude * v).exp_m1() + (-magnitude * v_bar).exp_m1();
      (-magnitude * u).exp_m1() / denominator
    }
  }
}

impl Default for Frank {
  fn default() -> Self {
    Self::new(None)
  }
}

impl BivariateExt for Frank {
  fn family(&self) -> BivariateFamily {
    self.r#type
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn theta_bounds(&self) -> (f64, f64) {
    self.theta_bounds
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    self.invalid_thetas.clone()
  }

  fn set_theta(&mut self, theta: f64) {
    self.theta = Some(theta);
  }

  fn generator(&self, t: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    let theta = self.theta.unwrap();
    let a = ((-theta * &t).exp() - 1.0) / ((-theta).exp() - 1.0);
    let out = -(a.ln());
    Ok(out)
  }

  /// Density $c(u,v) = -\theta\,g(1)\,e^{-\theta(u+v)} \big/ \big(g(1)+g(u)g(v)\big)^2$,
  /// $g(z) = e^{-\theta z}-1$ — the mixed second partial of this family's
  /// own `cdf`, obtained by direct differentiation and cross-checked
  /// against a finite-difference probe of `cdf`. The previous denominator,
  /// `g(u)+g(v)+g(1)` (a sum), was wrong for every `θ ≠ 0` — up to 268×
  /// off at `θ=5` — because nothing checked `pdf` against `cdf` directly;
  /// `θ = 0` was unaffected (a disjoint, separately special-cased path).
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd
  /// ed., Springer, Example 4.23 / Table 4.1 (Frank generator
  /// $\varphi_\theta(t) = -\ln\frac{e^{-\theta t}-1}{e^{-\theta}-1}$).
  fn pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;

    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(Array1::ones(U.len()));
    }

    let num =
      (-theta * self._g(Array1::ones(U.len()).view())?) * (1.0 + self._g((&U + &V).view())?);
    let aux = self._g(Array1::ones(U.len()).view())? + self._g(U.view())? * self._g(V.view())?;
    let den = aux.pow2();
    Ok(num / den)
  }

  fn cdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(&U * &V);
    }

    let num = ((-theta * &U).exp() - 1.0) * ((-theta * &V).exp() - 1.0);
    let den = (-theta).exp() - 1.0;
    let out = -1.0 / theta * (1.0 + num / den).ln();
    Ok(out)
  }

  /// `Frank::inverse`, `θ < 0` through `C_{−θ}(u, v) = u − C_θ(u, 1 − v)`, which swaps `v` and `1 − v`; the identity at
  /// `θ = 0`, NaN for `y` or `v` outside `[0, 1]`.
  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    V: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let theta = self.theta.unwrap();
    if theta == 0.0 {
      return conditional_quantiles(y, V, |y, _| y);
    }
    // One closure per sign keeps the reflection out of the per-pair work.
    let inverse = Self::inverse(theta.abs());
    if theta > 0.0 {
      conditional_quantiles(y, V, |y, v| inverse(y, v, 1.0 - v))
    } else {
      conditional_quantiles(y, V, |y, v| inverse(y, 1.0 - v, v))
    }
  }

  /// `Frank::h`'s factored form, `u` at `θ = 0`; NaN for `v` outside `[0, 1]`.
  fn partial_derivative(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    Ok(conditional_cdf(X, Self::h(self.theta.unwrap())))
  }

  /// `τ(θ) = 1 − 4/θ + 4D₁(θ)/θ`, `D₁(θ) = θ⁻¹∫₀^θ t/(eᵗ − 1) dt` (Hofert, Mächler & McNeil 2011, Table 2), odd in θ:
  /// its Taylor series below `|θ| = 2`, above it `∫₀^θ = π²/6 − Σ_k e^{−kθ}(θ/k + 1/k²)`; constant cost at every θ.
  fn tau_from_theta(&self, theta: f64) -> f64 {
    let x = theta.abs();
    let tau = if x < 2.0 {
      let x2 = x * x;
      x * TAU_SERIES.iter().rev().fold(0.0, |acc, c| acc * x2 + c)
    } else {
      let q = (-x).exp();
      let (mut q_k, mut tail) = (1.0, 0.0);
      for k in 1..=20 {
        q_k *= q;
        if q_k == 0.0 {
          break;
        }
        let k = f64::from(k);
        tail += q_k * (x / k + 1.0 / (k * k));
      }
      let pi = std::f64::consts::PI;
      1.0 - 4.0 / x + 4.0 * (pi * pi / 6.0 - tail) / (x * x)
    };
    tau.copysign(theta)
  }

  fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
    static CAP: OnceLock<f64> = OnceLock::new();
    let cap = *CAP.get_or_init(|| self.tau_from_theta(1e4));
    if !(tau > -cap && tau < cap) {
      return Err(tau_out_of_range(tau, format!("{:?} < tau < {cap:?}", -cap)));
    }
    let theta = if tau.abs() < 1e-12 {
      0.0
    } else {
      // `τ(0) = 0` exactly, so the bracket changes sign for every `τ` that passed the guards.
      let (lo, hi) = if tau > 0.0 { (0.0, 1e4) } else { (-1e4, 0.0) };
      invert_tau(|theta| self.tau_from_theta(theta) - tau, lo, hi, 1e-14)?
    };
    admitted(self, tau, theta)
  }

  /// Frank has no tail dependence in either tail, for any $\theta$.
  /// Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed.,
  /// Springer, Table 5.1.
  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    self.check_theta()?;
    Ok(TailDependence {
      lower: 0.0,
      upper: 0.0,
    })
  }
}

impl Frank {
  fn _g(&self, z: ArrayView1<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok((-self.theta.unwrap() * &z).exp() - 1.0)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn frank_tail_dependence_is_zero() {
    let f = Frank::new(Some(6.0));
    let td = f.tail_dependence().unwrap();
    assert_eq!(td.lower, 0.0);
    assert_eq!(td.upper, 0.0);
  }

  /// θ=0 is Frank's independence limit (`invalid_thetas` no longer rejects
  /// it) and must actually behave like one: `pdf(u,v) = 1` everywhere.
  #[test]
  fn frank_pdf_at_independence_is_one() {
    let f = Frank::new(Some(0.0));
    let pdf = f.pdf(array![[0.2_f64, 0.9], [0.5, 0.5]].view()).unwrap();
    assert!(approx(pdf[0], 1.0, 1e-12), "got {}", pdf[0]);
    assert!(approx(pdf[1], 1.0, 1e-12), "got {}", pdf[1]);
  }

  /// `cdf` had no `theta == 0.0` branch at all: unblocking the boundary
  /// without one would leave `C(u,v) = -1/theta * ln(1 + 0/0)` (both the
  /// generator ratio's numerator and denominator vanish identically at
  /// theta=0) evaluating to `NaN` for every input. The independence copula
  /// is `C(u,v) = uv`.
  #[test]
  fn frank_cdf_at_independence_is_uv() {
    let f = Frank::new(Some(0.0));
    let cdf = f.cdf(array![[0.3_f64, 0.7], [0.2, 0.9]].view()).unwrap();
    assert!(approx(cdf[0], 0.21, 1e-12), "got {}", cdf[0]);
    assert!(approx(cdf[1], 0.18, 1e-12), "got {}", cdf[1]);
  }

  /// `partial_derivative` computes $\partial_v C(u,v)$ (the same "second
  /// argument, fixed conditioning value" convention as
  /// [`crate::bivariate::gaussian::GaussianCopula::partial_derivative`]).
  /// At independence, $C(u,v)=uv \Rightarrow \partial_v C = u$ — not `v`.
  #[test]
  fn frank_partial_derivative_at_independence_returns_u() {
    let f = Frank::new(Some(0.0));
    let pd = f
      .partial_derivative(array![[0.3_f64, 0.6], [0.2, 0.9]].view())
      .unwrap();
    assert!(approx(pd[0], 0.3, 1e-12), "got {}", pd[0]);
    assert!(approx(pd[1], 0.2, 1e-12), "got {}", pd[1]);
  }

  /// `τ(θ)` against mpmath at 50 digits on both sides of the switch at `|θ| = 2`, near 0 and far out; constant cost at
  /// `θ = ±1e300` and `±∞`, where a quadrature over `[0, θ]` would not return.
  #[test]
  fn frank_tau_matches_mpmath() {
    let f = Frank::new(None);
    for (theta, want) in [
      (1e-8, 1.111111111111111e-9),
      (0.05, 0.00555541667257152),
      (0.1, 0.01111000018892774),
      (1.999, 0.21379542313291425),
      (2.0, 0.21389456921962013),
      (3.0, 0.3072469594307238),
      (-4.0, -0.38814802129793785),
      (20.0, 0.81644934023564),
      (1e7, 0.9999996000000658),
    ] {
      let tau = f.tau_from_theta(theta);
      assert!((tau - want).abs() <= 1e-15, "θ={theta}: {tau} vs {want}");
    }
    assert_eq!(f.tau_from_theta(1e300), 1.0);
    assert_eq!(f.tau_from_theta(-1e300), -1.0);
    assert_eq!(f.tau_from_theta(f64::INFINITY), 1.0);
    assert_eq!(f.tau_from_theta(f64::NEG_INFINITY), -1.0);
    assert!(f.tau_from_theta(f64::NAN).is_nan());
    assert_eq!(Frank::new(Some(1e300)).tau(), Some(1.0));
  }

  /// A `τ` near independence inverts: the bracket ends at `θ = 0`, where `τ` is exactly 0.
  #[test]
  fn frank_inverts_a_tau_near_independence() {
    let f = Frank::new(None);
    for tau in [1e-11, -1e-9, 1e-6, -0.003] {
      let theta = f.theta_from_tau(tau).unwrap();
      assert!(
        (f.tau_from_theta(theta) - tau).abs() < 1e-10,
        "τ = {tau}: θ = {theta}"
      );
    }
  }

  /// The test that decides the θ=0 boundary question: at Frank's
  /// independence limit, `percent_point`'s conditional inverse must not
  /// depend on the conditioning variable `V` at all, so sampled pairs
  /// must be empirically independent — Kendall's τ statistically
  /// indistinguishable from 0.
  ///
  /// Tolerance is `6` standard errors of τ under the null of independence
  /// (Kendall & Gibbons (1990), *Rank Correlation Methods*: `Var(τ) =
  /// 2(2n+5) / (9n(n-1))` for large `n`), derived from the sample size
  /// rather than tuned; three pinned seeds, best-of, per this crate's
  /// statistical-assertion convention.
  ///
  /// This must fail (and, before the `percent_point` fix, did fail) against
  /// the pre-fix `if theta == 0.0 { return Ok(V.clone()) }`: returning the
  /// conditioning variable instead of the fresh uniform `y` makes `U = V`
  /// exactly for every sampled pair — perfectly comonotonic, τ ≈ 1 — the
  /// exact opposite of the independence the caller asked for.
  #[test]
  fn frank_independence_sample_kendall_tau_near_zero() {
    let n = 4000usize;
    let se = (2.0 * (2.0 * n as f64 + 5.0) / (9.0 * n as f64 * (n as f64 - 1.0))).sqrt();
    let tol = 6.0 * se;
    let f = Frank::new(Some(0.0));
    let best_abs_tau = [2718_u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let samples = f.sample(n, &Deterministic::new(seed)).unwrap();
        let u = samples.column(0).to_vec();
        let v = samples.column(1).to_vec();
        let (tau, ..) =
          kendalls::tau_b_with_comparator(&u, &v, |a: &f64, b: &f64| a.partial_cmp(b).unwrap())
            .unwrap();
        tau.abs()
      })
      .fold(f64::INFINITY, f64::min);
    assert!(
      best_abs_tau < tol,
      "best |tau| across 3 seeds = {best_abs_tau}, expected < {tol} (6 SE, SE={se}, n={n})"
    );
  }
}
