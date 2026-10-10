//! # Marshall-Olkin (1967) bivariate copula
//!
//! $$
//! C_{\alpha,\beta}(u,v) =
//! \begin{cases}
//!   u^{1-\alpha}\, v & \text{if } u^\alpha \ge v^\beta,\\
//!   u\, v^{1-\beta} & \text{if } u^\alpha < v^\beta,
//! \end{cases}
//! \qquad \alpha, \beta \in (0, 1].
//! $$
//!
//! Equivalently $C(u,v) = \min\!\big(u^{1-\alpha} v,\, u v^{1-\beta}\big)$.
//! Non-exchangeable in general; **has a singular component** on the curve
//! $u^\alpha = v^\beta$ carrying mass $\alpha\beta / (\alpha + \beta -
//! \alpha\beta)$. The remaining $1 - \alpha\beta/(\alpha+\beta-\alpha\beta)$
//! of probability mass is absolutely continuous on the two open sectors.
//!
//! Tail dependence: $\lambda_U = \min(\alpha, \beta)$ (upper),
//! $\lambda_L = 0$ (lower).
//!
//! Kendall's tau:
//! $\tau = \alpha\beta / (\alpha + \beta - \alpha\beta)$.
//!
//! [`BivariateExt::set_theta`] / [`BivariateExt::compute_theta`] operate on
//! the *symmetric* slice $\alpha = \beta = \theta$, in which case
//! $\tau = \theta / (2 - \theta) \iff \theta = 2\tau/(1 + \tau)$. The
//! asymmetric two-parameter case is supplied through
//! [`MarshallOlkin::with_alpha_beta`].
//!
//! Reference: Marshall, A.W., Olkin, I. (1967), "A multivariate exponential distribution", *JASA* 62(317), 30-44.
//! Reference: Nelsen, R.B. (2006), "An Introduction to Copulas", 2nd ed., Springer, §3.1.1, eq. 3.1.3; §2.9 (the conditional-distribution method with a quasi-inverse).
//! Reference: Mai, J.-F., Scherer, M. (2012), "Simulating Copulas", Imperial College Press, §1.2.3 (the exponential-shock representation).

use ndarray::Array1;
use ndarray::Array2;

use crate::bivariate::CopulaType;
use crate::error::CopulaError;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

#[derive(Debug, Clone)]
pub struct MarshallOlkin {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
  /// Asymmetric Marshall-Olkin parameter $\alpha \in (0, 1]$. When
  /// `Some`, takes precedence over the symmetric `theta` field.
  pub alpha: Option<f64>,
  /// Asymmetric Marshall-Olkin parameter $\beta \in (0, 1]$.
  pub beta: Option<f64>,
}

impl Default for MarshallOlkin {
  fn default() -> Self {
    Self {
      r#type: CopulaType::MarshallOlkin,
      theta: None,
      tau: None,
      theta_bounds: (0.0, 1.0),
      invalid_thetas: vec![],
      alpha: None,
      beta: None,
    }
  }
}

impl MarshallOlkin {
  pub fn new() -> Self {
    Self::default()
  }

  /// Construct the asymmetric two-parameter variant $C_{\alpha,\beta}$.
  /// Both parameters must lie in $(0, 1]$.
  pub fn with_alpha_beta(alpha: f64, beta: f64) -> Self {
    assert!(
      alpha > 0.0 && alpha <= 1.0,
      "alpha must lie in (0, 1], got {alpha}"
    );
    assert!(
      beta > 0.0 && beta <= 1.0,
      "beta must lie in (0, 1], got {beta}"
    );
    Self {
      alpha: Some(alpha),
      beta: Some(beta),
      ..Self::default()
    }
  }

  /// Resolve the effective `(alpha, beta)` pair from either the asymmetric
  /// fields or the symmetric `theta` fallback.
  fn resolve_params(&self) -> (f64, f64) {
    match (self.alpha, self.beta) {
      (Some(a), Some(b)) => (a, b),
      _ => {
        let theta = self
          .theta
          .expect("Marshall-Olkin: neither (alpha, beta) nor theta is set");
        (theta, theta)
      }
    }
  }
}

impl BivariateExt for MarshallOlkin {
  fn r#type(&self) -> CopulaType {
    self.r#type
  }

  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn set_tau(&mut self, tau: f64) {
    self.tau = Some(tau);
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
    // Reset the asymmetric slot so the symmetric value wins on resolve.
    self.alpha = None;
    self.beta = None;
  }

  /// Validates the parameterisation `resolve_params` uses: `alpha` and `beta` in (0, 1] when both
  /// are set, otherwise `theta` through `check_theta`.
  fn check_fit(&self) -> Result<(), CopulaError> {
    let (Some(alpha), Some(beta)) = (self.alpha, self.beta) else {
      return self.check_theta();
    };
    for (name, value) in [("alpha", alpha), ("beta", beta)] {
      if !(value > 0.0 && value <= 1.0) {
        return Err(CopulaError::InvalidParameter {
          name,
          value,
          constraint: format!("0.0 < {name} <= 1.0"),
        });
      }
    }
    Ok(())
  }

  /// Absolutely continuous density. Returns `0` exactly on the singular
  /// curve $u^\alpha = v^\beta$ and `(1 - \alpha) u^{-\alpha}` /
  /// `(1 - \beta) v^{-\beta}` in the two open sectors.
  fn pdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
        out[i] = 0.0;
        continue;
      }
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      if (lhs - rhs).abs() < 1e-15 {
        // Singular curve: absolutely continuous density is 0; the singular
        // component carries the curve mass and is not returned here.
        out[i] = 0.0;
      } else if lhs > rhs {
        out[i] = (1.0 - alpha) * u.powf(-alpha);
      } else {
        out[i] = (1.0 - beta) * v.powf(-beta);
      }
    }
    Ok(out)
  }

  /// CDF $C_{\alpha,\beta}(u,v) = \min(u^{1-\alpha} v, u v^{1-\beta})$.
  fn cdf(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 || v <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if u >= 1.0 {
        out[i] = v;
        continue;
      }
      if v >= 1.0 {
        out[i] = u;
        continue;
      }
      let lhs = u.powf(alpha);
      let rhs = v.powf(beta);
      out[i] = if lhs >= rhs {
        u.powf(1.0 - alpha) * v
      } else {
        u * v.powf(1.0 - beta)
      };
    }
    Ok(out)
  }

  /// `∂_v C`: `u^{1-α}` where `u^α ≥ v^β`, `(1 - β) u v^{-β}` below; the jump `β v^{β(1-α)/α}` at `u = v^{β/α}`
  /// is the singular component's conditional mass.
  fn partial_derivative(&self, x: &Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    let u_col = x.column(0);
    let v_col = x.column(1);
    let mut out = Array1::<f64>::zeros(u_col.len());
    for i in 0..u_col.len() {
      let u = u_col[i];
      let v = v_col[i];
      if u <= 0.0 {
        out[i] = 0.0;
        continue;
      }
      if u >= 1.0 {
        out[i] = 1.0;
        continue;
      }
      out[i] = if u >= v.powf(beta / alpha) {
        u.powf(1.0 - alpha)
      } else {
        (1.0 - beta) * u * v.powf(-beta)
      };
    }
    Ok(out)
  }

  /// Generalised inverse of `∂_v C(· | v)`: closed form on both continuous pieces and the atom `v^{β/α}` for `y`
  /// inside the jump `[(1-β) w, w]`, `w = v^{β(1-α)/α}`; the `α = 1` and `β = 1` branches never divide by zero.
  fn percent_point(&self, y: &Array1<f64>, v: &Array1<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit()?;
    let (alpha, beta) = self.resolve_params();
    Ok(
      y.iter()
        .zip(v.iter())
        .map(|(&y, &v)| {
          let w = v.powf(beta * (1.0 - alpha) / alpha);
          if y < (1.0 - beta) * w {
            y * v.powf(beta) / (1.0 - beta)
          } else if y <= w {
            v.powf(beta / alpha)
          } else {
            y.powf(1.0 / (1.0 - alpha))
          }
        })
        .collect(),
    )
  }

  /// Symmetric-slice Kendall's tau inversion: $\theta = 2\tau / (1 + \tau)$.
  /// For the asymmetric case, $\tau$ alone underdetermines $(\alpha, \beta)$ —
  /// supply both directly via [`MarshallOlkin::with_alpha_beta`].
  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();
    if tau <= 0.0 {
      return 0.0_f64.max(f64::EPSILON);
    }
    if tau >= 1.0 - 1e-12 {
      return 1.0;
    }
    (2.0 * tau / (1.0 + tau)).clamp(0.0, 1.0)
  }

  /// Upper-tail dependence $\lambda_U = \min(\alpha,\beta)$, carried
  /// entirely by the singular component on $u^\alpha = v^\beta$; no
  /// lower-tail dependence.
  /// Reference: Marshall, A.W., Olkin, I. (1967); Nelsen (2006) Example
  /// 5.22 (module header).
  ///
  /// Cannot use [`BivariateExt::assert_theta_valid_for_tail_dependence`]
  /// directly: the legitimate [`MarshallOlkin::with_alpha_beta`]
  /// construction path leaves `theta` as `None`, which the generic
  /// `theta`-based guard would reject as unfit even though it's a valid
  /// configuration. Validates the *resolved* `(alpha, beta)` against
  /// their own domain instead — e.g. a raw `set_theta(5.0)` bypasses
  /// `with_alpha_beta`'s constructor asserts and would otherwise report a
  /// nonsensical `\lambda_U > 1`.
  fn tail_dependence(&self) -> TailDependence<f64> {
    let (alpha, beta) = self.resolve_params();
    if !(alpha > 0.0 && alpha <= 1.0) || !(beta > 0.0 && beta <= 1.0) {
      panic!(
        "tail_dependence requires a valid theta: alpha={alpha} and beta={beta} must each lie in (0, 1]"
      );
    }
    TailDependence {
      lower: 0.0,
      upper: alpha.min(beta),
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn mo_cdf_marginal_recovers_input() {
    let c = MarshallOlkin::with_alpha_beta(0.5, 0.3);
    let x = array![[0.4_f64, 1.0], [1.0, 0.7]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.4, 1e-12));
    assert!(approx(cdf[1], 0.7, 1e-12));
  }

  #[test]
  fn mo_alpha_eq_one_beta_eq_one_is_comonotone() {
    let c = MarshallOlkin::with_alpha_beta(1.0, 1.0);
    let x = array![[0.3_f64, 0.7], [0.6, 0.2], [0.5, 0.5]];
    let cdf = c.cdf(&x).unwrap();
    for i in 0..x.nrows() {
      let expected = x[[i, 0]].min(x[[i, 1]]);
      assert!(approx(cdf[i], expected, 1e-12), "row {i}");
    }
  }

  #[test]
  fn mo_alpha_zero_or_beta_zero_is_independence() {
    // α → 0 with β fixed: C(u,v) = u^{1-0} v = u v. Use α just above 0.
    let c = MarshallOlkin::with_alpha_beta(1e-12, 0.5);
    let x = array![[0.4_f64, 0.6]];
    let cdf = c.cdf(&x).unwrap();
    assert!(approx(cdf[0], 0.24, 1e-6), "α→0: got {}", cdf[0]);
  }

  #[test]
  fn mo_compute_theta_via_symmetric_inversion() {
    // Symmetric MO: τ = θ/(2-θ). Pick θ = 0.5 ⇒ τ = 1/3; invert to recover.
    let mut c = MarshallOlkin::new();
    c.set_tau(1.0 / 3.0);
    let theta = c.compute_theta();
    assert!(approx(theta, 0.5, 1e-12), "expected θ=0.5, got {theta}");
  }

  #[test]
  fn mo_singular_curve_total_mass_matches_paper() {
    // Singular component carries mass αβ/(α+β-αβ). Verify against
    // Monte-Carlo on a fine grid: count fraction of unit-square sectors
    // dominated by the absolutely continuous density vs total.
    let alpha = 0.6_f64;
    let beta = 0.4_f64;
    let mass_singular_paper = alpha * beta / (alpha + beta - alpha * beta);

    // Integrate the absolutely continuous density on a 200×200 grid.
    let c = MarshallOlkin::with_alpha_beta(alpha, beta);
    let n = 200usize;
    let h = 1.0 / n as f64;
    let mut points = Array2::<f64>::zeros((n * n, 2));
    for i in 0..n {
      for j in 0..n {
        let row = i * n + j;
        points[[row, 0]] = (i as f64 + 0.5) * h;
        points[[row, 1]] = (j as f64 + 0.5) * h;
      }
    }
    let pdf_vals = c.pdf(&points).unwrap();
    let mass_abs: f64 = pdf_vals.iter().sum::<f64>() * h * h;
    let mass_singular_grid = 1.0 - mass_abs;

    assert!(
      (mass_singular_grid - mass_singular_paper).abs() < 0.02,
      "singular mass grid={mass_singular_grid:.4}, paper={mass_singular_paper:.4}"
    );
  }

  /// `∂_v C` from mpmath's derivative of `C` in `v`, 17 digits, in both sectors.
  #[test]
  fn mo_partial_derivative_is_the_v_derivative_in_each_sector() {
    let cases = [
      (0.5, 0.5, 0.9, 0.5, 0.9486832980505138),
      (0.5, 0.5, 0.3, 0.8, 0.16770509831248423),
      (0.5, 0.5, 0.6, 0.4, 0.7745966692414834),
      (0.3, 0.6, 0.9, 0.5, 0.928901697685371),
      (0.3, 0.6, 0.3, 0.8, 0.1371915155781979),
      (0.3, 0.6, 0.6, 0.4, 0.6993681904144294),
      (0.8, 0.2, 0.9, 0.5, 0.9791483623609768),
      (0.8, 0.2, 0.3, 0.8, 0.2509534926219056),
      (0.8, 0.2, 0.6, 0.4, 0.576539728311087),
    ];
    for (alpha, beta, u, v, want) in cases {
      let c = MarshallOlkin::with_alpha_beta(alpha, beta);
      let h = c.partial_derivative(&array![[u, v]]).unwrap()[0];
      assert!(
        (h - want).abs() < 1e-15,
        "α={alpha} β={beta} (u,v)=({u},{v}): {h} vs {want}"
      );
    }
  }

  /// The generalised inverse at `v = 0.37`: both continuous pieces and the atom, with the `α = 1`, `β = 1` and
  /// `α = β = 1` branches.
  #[test]
  fn mo_percent_point_matches_the_reference_table() {
    let v = 0.37_f64;
    let cases = [
      (0.5, 0.5, 0.25 * v.sqrt(), 0.185),
      (0.5, 0.5, 0.3041381265, 0.37),
      (0.5, 0.5, 0.45620719, 0.37),
      (0.5, 0.5, 0.608276253, 0.37),
      (0.5, 0.5, 0.5 + 0.5 * v.sqrt(), 0.6466381265149109),
      (0.3, 0.6, 0.2 * v.powf(1.4), 0.06845),
      (0.3, 0.6, 0.1740125, 0.1369),
      (0.3, 0.6, 0.5 + 0.5 * v.powf(1.4), 0.5101518223511724),
      (1.0, 0.4, 0.3, 0.33593147277382094),
      (1.0, 0.4, 0.8, 0.6718629455476419),
      (0.7, 1.0, 0.32652287, 0.2416269211589283),
      (0.7, 1.0, 0.5 + 0.5 * v.powf(3.0 / 7.0), 0.5298865677782774),
      (1.0, 1.0, 0.5, 0.37),
    ];
    for (alpha, beta, y, want) in cases {
      let c = MarshallOlkin::with_alpha_beta(alpha, beta);
      let u = c.percent_point(&array![y], &array![v]).unwrap()[0];
      assert!(
        (u - want).abs() < 1e-9,
        "α={alpha} β={beta} y={y}: {u} vs {want}"
      );
    }
  }

  /// Off the atom the inverse round-trips through `h`; on the atom `h(u−) ≤ y ≤ h(u)`.
  #[test]
  fn mo_percent_point_inverts_the_h_function() {
    for (alpha, beta) in [(0.5, 0.5), (0.3, 0.6), (0.8, 0.2), (1.0, 0.4), (0.7, 1.0)] {
      let c = MarshallOlkin::with_alpha_beta(alpha, beta);
      for v in [0.05_f64, 0.37, 0.9] {
        let w = v.powf(beta * (1.0 - alpha) / alpha);
        for y in [0.02, 0.2, 0.5, 0.8, 0.98] {
          let u = c.percent_point(&array![y], &array![v]).unwrap()[0];
          let h_at = c.partial_derivative(&array![[u, v]]).unwrap()[0];
          if y < (1.0 - beta) * w || y > w {
            assert!(
              (h_at - y).abs() < 1e-12,
              "α={alpha} β={beta} v={v} y={y}: h({u})={h_at}"
            );
          } else {
            let h_left = c.partial_derivative(&array![[u - 1e-12, v]]).unwrap()[0];
            assert!(
              h_left <= y + 1e-9 && y <= h_at + 1e-9,
              "atom: {h_left} ≤ {y} ≤ {h_at}"
            );
          }
        }
      }
    }
  }

  /// τ, the singular mass and the cdf on a grid, against Nelsen's closed forms; no `Numerical` failure at any `n`.
  #[test]
  fn mo_sampler_reproduces_tau_the_singular_mass_and_the_cdf() {
    for n in [1_usize, 5, 20, 200] {
      assert!(
        MarshallOlkin::with_alpha_beta(0.5, 0.5)
          .sample_with_seed(n, 7)
          .is_ok(),
        "n = {n}"
      );
    }
    let grid = [(0.2, 0.3), (0.5, 0.5), (0.7, 0.4), (0.3, 0.9), (0.85, 0.65)];
    for (alpha, beta) in [(0.5, 0.5), (0.3, 0.6), (0.6, 0.4), (1.0, 0.4), (0.7, 1.0)] {
      let c = MarshallOlkin::with_alpha_beta(alpha, beta);
      let n = 40_000_usize;
      let uv = c.sample_with_seed(n, 7).unwrap();
      let (u, v) = (uv.column(0).to_vec(), uv.column(1).to_vec());
      let (tau, ..) =
        kendalls::tau_b_with_comparator(&u, &v, |a: &f64, b: &f64| a.partial_cmp(b).unwrap())
          .unwrap();
      let want = alpha * beta / (alpha + beta - alpha * beta);
      assert!(
        (tau - want).abs() < 0.02,
        "α={alpha} β={beta}: τ {tau} vs {want}"
      );
      let atoms = u
        .iter()
        .zip(&v)
        .filter(|(a, b)| (a.powf(alpha) - b.powf(beta)).abs() < 1e-9)
        .count() as f64
        / n as f64;
      assert!(
        (atoms - want).abs() < 0.02,
        "α={alpha} β={beta}: atom share {atoms} vs {want}"
      );
      for (gu, gv) in grid {
        let empirical = u
          .iter()
          .zip(&v)
          .filter(|(a, b)| **a <= gu && **b <= gv)
          .count() as f64
          / n as f64;
        let exact = (gu.powf(1.0 - alpha) * gv).min(gu * gv.powf(1.0 - beta));
        assert!(
          (empirical - exact).abs() < 0.02,
          "α={alpha} β={beta} ({gu},{gv}): {empirical} vs {exact}"
        );
      }
    }
    let comonotone = MarshallOlkin::with_alpha_beta(1.0, 1.0)
      .sample_with_seed(500, 7)
      .unwrap();
    assert!(comonotone.rows().into_iter().all(|r| r[0] == r[1]));
  }

  #[test]
  fn mo_tail_dependence_matches_min_alpha_beta() {
    let c = MarshallOlkin::with_alpha_beta(0.6, 0.4);
    let td = c.tail_dependence();
    assert!(approx(td.upper, 0.4, 1e-12), "got {}", td.upper);
    assert_eq!(td.lower, 0.0);
  }

  /// A raw `set_theta(5.0)` bypasses `with_alpha_beta`'s constructor
  /// asserts, resolving to `(alpha, beta) = (5.0, 5.0)` — outside `(0, 1]`.
  /// Must panic, not silently report `λ_U = 5.0`.
  #[test]
  #[should_panic(expected = "tail_dependence requires a valid theta")]
  fn mo_tail_dependence_panics_on_invalid_theta() {
    let mut c = MarshallOlkin::new();
    c.set_theta(5.0);
    let _ = c.tail_dependence();
  }

  /// pdf/cdf/partial_derivative on an unfit `MarshallOlkin` (neither
  /// `theta` nor `(alpha, beta)` set) must return `Err`, matching every
  /// sibling copula's `check_fit`-gated contract — not panic via
  /// `resolve_params().expect(..)`.
  #[test]
  fn marshall_olkin_unfit_errs_like_siblings() {
    let c = MarshallOlkin::new();
    let x = array![[0.4_f64, 0.6]];
    assert!(c.pdf(&x).is_err(), "pdf must Err, not panic, when unfit");
    assert!(c.cdf(&x).is_err(), "cdf must Err, not panic, when unfit");
    assert!(
      c.partial_derivative(&x).is_err(),
      "partial_derivative must Err, not panic, when unfit"
    );
  }

  /// `generator` has no override in this family, so this exercises
  /// `BivariateExt::generator`'s trait-default body directly.
  #[test]
  fn marshall_olkin_generator_returns_err_not_archimedean() {
    let c = MarshallOlkin::with_alpha_beta(0.5, 0.3);
    let t = array![0.5_f64, 0.8];
    assert!(c.generator(&t).is_err());
  }

  #[test]
  fn an_out_of_domain_parameter_neither_samples_nor_evaluates() {
    let x = array![[0.3_f64, 0.4]];
    for theta in [1.5, -0.5] {
      let c = MarshallOlkin {
        theta: Some(theta),
        ..MarshallOlkin::new()
      };
      assert!(
        matches!(
          c.sample_with_seed(16, 7),
          Err(CopulaError::InvalidParameter { name: "theta", .. })
        ),
        "theta = {theta}"
      );
      assert!(
        matches!(
          c.pdf(&x),
          Err(CopulaError::InvalidParameter { name: "theta", .. })
        ),
        "theta = {theta}"
      );
    }
    let c = MarshallOlkin {
      alpha: Some(1.5),
      beta: Some(0.5),
      ..MarshallOlkin::new()
    };
    assert!(matches!(
      c.cdf(&x),
      Err(CopulaError::InvalidParameter { name: "alpha", .. })
    ));
  }
}
