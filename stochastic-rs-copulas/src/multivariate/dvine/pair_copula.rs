//! Pair-copula family enum with h-function, inverse h-function and log
//! density implementations used by the D-vine / C-vine / R-vine pair-copula
//! constructions (Aas, Czado, Frigessi, Bakken 2009).
//!
//! The h-function of a bivariate copula $C(u, v)$ is the conditional CDF
//!
//! $$ h(u \mid v) := \frac{\partial C(u, v)}{\partial v}, $$
//!
//! and `h_inverse(p, v)` solves $h(u \mid v) = p$ for $u$. Both are the
//! workhorses of the Joe (1997) / Aas-Czado (2009) recursive sampling and
//! density evaluation algorithms — sampling traverses h-inverses,
//! conditional pseudo-observations traverse h-functions.
//!
//! Closed-form $h$ and $h^{-1}$ are provided for the five families with
//! tractable analytic inverses:
//!
//! - **Independence:** trivial identities.
//! - **Gaussian** $\rho \in (-1, 1)$.
//! - **Clayton** $\theta > 0$.
//! - **Frank** $\theta \neq 0$.
//! - **Student-$t$** $\rho \in (-1, 1),\ \nu > 0$.
//!
//! The Gumbel / Joe pair families have no closed-form $h^{-1}$ (only an
//! iterative bisection) and are not yet included.
//!
//! Reference: Aas, Czado, Frigessi, Bakken (2009), "Pair-copula
//! constructions of multiple dependence", *Insurance: Mathematics and
//! Economics* 44(2), 182-198, Table 1 and Appendix.

use std::f64;

use stochastic_rs_distributions::special::beta_i;
use stochastic_rs_distributions::special::ln_gamma;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

/// Bivariate pair-copula family used as the building block of every
/// PCC-based multivariate copula in this crate (D-vine, C-vine, R-vine).
///
/// Every variant carries the family's parameters as fields so PCC trees
/// can be stored as `Vec<Vec<PairCopula>>` without trait objects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairCopula {
  /// Product copula: $C(u, v) = u \cdot v$.
  Independence,
  /// Gaussian: $C(u, v) = \Phi_\rho(\Phi^{-1}(u), \Phi^{-1}(v))$.
  Gaussian {
    /// Correlation $\rho \in (-1, 1)$.
    rho: f64,
  },
  /// Clayton: $C(u, v) = (u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}$,
  /// $\theta > 0$. Lower-tail dependent.
  Clayton {
    /// Generator parameter $\theta > 0$.
    theta: f64,
  },
  /// Frank: $C(u, v) = -\tfrac{1}{\theta}\ln\!\bigl(1 + \frac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\bigr)$,
  /// $\theta \neq 0$. Symmetric tail-independent.
  Frank {
    /// Generator parameter $\theta \neq 0$.
    theta: f64,
  },
  /// Student-$t$: $C(u, v) = t_{\rho,\nu}(t_\nu^{-1}(u), t_\nu^{-1}(v))$.
  /// Symmetric tail-dependent, $\nu \to \infty$ collapses to Gaussian.
  StudentT {
    /// Correlation $\rho \in (-1, 1)$.
    rho: f64,
    /// Degrees of freedom $\nu > 0$.
    nu: f64,
  },
}

impl PairCopula {
  /// Family-specific h-function $h(u \mid v) = \partial C / \partial v$.
  /// Returns the conditional CDF $F_{U \mid V = v}(u)$.
  pub fn h(&self, u: f64, v: f64) -> f64 {
    let u = u.clamp(EPS, 1.0 - EPS);
    let v = v.clamp(EPS, 1.0 - EPS);
    let h = match *self {
      PairCopula::Independence => u,
      PairCopula::Gaussian { rho } => {
        let x = ndtri(u);
        let y = ndtri(v);
        let arg = (x - rho * y) / (1.0 - rho * rho).sqrt();
        norm_cdf(arg)
      }
      PairCopula::Clayton { theta } => {
        // Aas-Czado (2009) Table 1, Clayton row.
        // h(u|v) = v^{-θ-1} (u^{-θ} + v^{-θ} - 1)^{-(θ+1)/θ}
        let s = u.powf(-theta) + v.powf(-theta) - 1.0;
        v.powf(-theta - 1.0) * s.powf(-(theta + 1.0) / theta)
      }
      PairCopula::Frank { theta } => {
        // Aas-Czado (2009) Table 1, Frank row.
        // h(u|v) = e^{-θv} (e^{-θu} - 1) / ((e^{-θ} - 1) + (e^{-θu} - 1)(e^{-θv} - 1))
        let eu = (-theta * u).exp();
        let ev = (-theta * v).exp();
        let e1 = (-theta).exp();
        let num = ev * (eu - 1.0);
        let den = (e1 - 1.0) + (eu - 1.0) * (ev - 1.0);
        num / den
      }
      PairCopula::StudentT { rho, nu } => {
        // Aas-Czado (2009) Appendix A.2.
        // x = t_ν^{-1}(u), y = t_ν^{-1}(v)
        // h(u|v) = t_{ν+1}((x - ρ y) · √((ν+1) / ((ν + y²)(1 - ρ²))))
        let x = student_t_quantile(u, nu);
        let y = student_t_quantile(v, nu);
        let scale = ((nu + 1.0) / ((nu + y * y) * (1.0 - rho * rho))).sqrt();
        student_t_cdf((x - rho * y) * scale, nu + 1.0)
      }
    };
    h.clamp(EPS, 1.0 - EPS)
  }

  /// Inverse h-function: returns the $u$ that satisfies $h(u \mid v) = p$.
  /// Used in PCC sampling (Aas-Czado 2009, Algorithm 4 backward step).
  pub fn h_inverse(&self, p: f64, v: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    let v = v.clamp(EPS, 1.0 - EPS);
    let u = match *self {
      PairCopula::Independence => p,
      PairCopula::Gaussian { rho } => {
        let z = ndtri(p);
        let y = ndtri(v);
        let arg = z * (1.0 - rho * rho).sqrt() + rho * y;
        norm_cdf(arg)
      }
      PairCopula::Clayton { theta } => {
        // Inverse derived from h(u|v) = v^{-θ-1}(u^{-θ} + v^{-θ} - 1)^{-(θ+1)/θ}:
        //   p · v^{θ+1} = (u^{-θ} + v^{-θ} - 1)^{-(θ+1)/θ}
        //   u^{-θ} + v^{-θ} - 1 = (p · v^{θ+1})^{-θ/(θ+1)}
        //   u = ((p · v^{θ+1})^{-θ/(θ+1)} - v^{-θ} + 1)^{-1/θ}
        let base = p * v.powf(theta + 1.0);
        let inner = base.powf(-theta / (theta + 1.0)) - v.powf(-theta) + 1.0;
        inner.max(1.0 + EPS).powf(-1.0 / theta)
      }
      PairCopula::Frank { theta } => {
        // Solve p = h(u|v) algebraically. Let A = e^{-θ} - 1, E = e^{-θv}.
        //   p = E(e^{-θu} - 1) / (A + (e^{-θu} - 1) E - E e^{-θv?}... ): direct from the h formula gives
        //   e^{-θu} = 1 + A·p / (1 - p·(E - 1))
        //   u = -ln(1 + A·p / (1 - p·(E - 1))) / θ
        //
        // Where E = (e^{-θv} - 1). Re-derive:
        //   Let X = e^{-θu} - 1, Y = e^{-θv} - 1, A = e^{-θ} - 1
        //   h(u|v) = (Y + 1)·X / (A + X·Y)
        //   p · (A + X·Y) = (Y + 1) · X
        //   p·A + p·X·Y = X·Y + X
        //   p·A = X·(Y + 1 - p·Y) = X·(1 + Y·(1 - p))
        //   X = p·A / (1 + Y·(1 - p))
        //   ⟹ u = -ln(X + 1) / θ
        let y = (-theta * v).exp() - 1.0;
        let a = (-theta).exp() - 1.0;
        let x_val = p * a / (1.0 + y * (1.0 - p));
        let arg = x_val + 1.0;
        if arg <= 0.0 {
          return EPS;
        }
        -arg.ln() / theta
      }
      PairCopula::StudentT { rho, nu } => {
        // Inverse of h(u|v):
        //   z = t_{ν+1}^{-1}(p),  y = t_ν^{-1}(v)
        //   x = z / √((ν+1) / ((ν + y²)(1 - ρ²)))  +  ρ y
        //     = z · √((ν + y²)(1 - ρ²)/(ν+1))  +  ρ y
        //   u = t_ν(x)
        let z = student_t_quantile(p, nu + 1.0);
        let y = student_t_quantile(v, nu);
        let scale = ((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0)).sqrt();
        let x = z * scale + rho * y;
        student_t_cdf(x, nu)
      }
    };
    u.clamp(EPS, 1.0 - EPS)
  }

  /// Log copula density $\log c(u, v) = \log \frac{\partial^2 C(u, v)}{\partial u\, \partial v}$.
  /// Used for sequential MLE fitting and direct density evaluation on the
  /// PCC tree.
  pub fn log_density(&self, u: f64, v: f64) -> f64 {
    let u = u.clamp(EPS, 1.0 - EPS);
    let v = v.clamp(EPS, 1.0 - EPS);
    match *self {
      PairCopula::Independence => 0.0,
      PairCopula::Gaussian { rho } => {
        // c(u, v) = (1 - ρ²)^{-1/2} · exp(-(ρ² (x² + y²) - 2ρ x y) / (2(1 - ρ²)))
        let x = ndtri(u);
        let y = ndtri(v);
        let one_minus_rho2 = 1.0 - rho * rho;
        let q = rho * rho * (x * x + y * y) - 2.0 * rho * x * y;
        -0.5 * one_minus_rho2.ln() - 0.5 * q / one_minus_rho2
      }
      PairCopula::Clayton { theta } => {
        // c(u, v) = (1 + θ) (u v)^{-θ - 1} (u^{-θ} + v^{-θ} - 1)^{-1/θ - 2}
        let s = u.powf(-theta) + v.powf(-theta) - 1.0;
        (1.0 + theta).ln()
          + (-theta - 1.0) * u.ln()
          + (-theta - 1.0) * v.ln()
          + (-1.0 / theta - 2.0) * s.ln()
      }
      PairCopula::Frank { theta } => {
        // c(u, v) = θ(1 - e^{-θ}) e^{-θ(u+v)} / ((1 - e^{-θ}) - (1 - e^{-θu})(1 - e^{-θv}))²
        let a = 1.0 - (-theta).exp();
        let eu = (-theta * u).exp();
        let ev = (-theta * v).exp();
        let denom = a - (1.0 - eu) * (1.0 - ev);
        (theta * a).ln() + (-theta * (u + v)) - 2.0 * denom.abs().ln()
      }
      PairCopula::StudentT { rho, nu } => {
        // c(u, v) = (Γ((ν+2)/2) Γ(ν/2)) / (Γ((ν+1)/2)² √(1-ρ²))
        //          · (1 + (x² - 2ρxy + y²)/(ν(1-ρ²)))^{-(ν+2)/2}
        //          / ((1 + x²/ν)(1 + y²/ν))^{-(ν+1)/2}
        let x = student_t_quantile(u, nu);
        let y = student_t_quantile(v, nu);
        let one_minus_rho2 = 1.0 - rho * rho;
        let log_norm = ln_gamma(0.5 * (nu + 2.0)) + ln_gamma(0.5 * nu)
          - 2.0 * ln_gamma(0.5 * (nu + 1.0))
          - 0.5 * one_minus_rho2.ln();
        let log_joint =
          -0.5 * (nu + 2.0) * (1.0 + (x * x - 2.0 * rho * x * y + y * y) / (nu * one_minus_rho2)).ln();
        let log_marg = 0.5 * (nu + 1.0) * ((1.0 + x * x / nu).ln() + (1.0 + y * y / nu).ln());
        log_norm + log_joint + log_marg
      }
    }
  }

  /// Copula density $c(u, v)$ (exponentiated log density).
  pub fn density(&self, u: f64, v: f64) -> f64 {
    self.log_density(u, v).exp()
  }
}

const EPS: f64 = 1e-12;

/// Standard Student-$t$ CDF $F_\nu(x)$ via the regularised incomplete-beta
/// identity. Mirrors `crate::bivariate::t_copula::TCopula::t_cdf`.
fn student_t_cdf(x: f64, nu: f64) -> f64 {
  if !x.is_finite() {
    return if x > 0.0 { 1.0 } else { 0.0 };
  }
  let t = nu / (nu + x * x);
  let half = 0.5 * beta_i(0.5 * nu, 0.5, t);
  if x >= 0.0 { 1.0 - half } else { half }
}

/// Standard Student-$t$ density log-PDF at `x`. Useful for the
/// Newton-refined quantile and `log_density` evaluation.
fn student_t_log_pdf(x: f64, nu: f64) -> f64 {
  let log_norm =
    ln_gamma(0.5 * (nu + 1.0)) - 0.5 * (nu * f64::consts::PI).ln() - ln_gamma(0.5 * nu);
  let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
  log_norm + log_kernel
}

/// Standard Student-$t$ quantile $t_\nu^{-1}(p)$: Cornish-Fisher normal
/// seed refined by 40 Newton steps on `[0, 1]`. Identical routine to the
/// bivariate t-copula and multivariate t-copula modules.
fn student_t_quantile(p: f64, nu: f64) -> f64 {
  if p <= 0.0 {
    return f64::NEG_INFINITY;
  }
  if p >= 1.0 {
    return f64::INFINITY;
  }
  let z = ndtri(p);
  let mut x = z * (1.0 + (z * z + 1.0) / (4.0 * nu));
  for _ in 0..40 {
    let cdf = student_t_cdf(x, nu);
    let f = cdf - p;
    let pdf = student_t_log_pdf(x, nu).exp();
    if pdf <= 0.0 {
      break;
    }
    let dx = f / pdf;
    let new_x = x - dx;
    if (new_x - x).abs() < 1e-14 * (1.0 + x.abs()) {
      return new_x;
    }
    x = new_x;
  }
  x
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Independence h-function must be the identity; inverse same.
  #[test]
  fn independence_h_identity() {
    let p = PairCopula::Independence;
    for u in [0.1, 0.3, 0.5, 0.7, 0.9] {
      for v in [0.1, 0.5, 0.9] {
        assert!((p.h(u, v) - u).abs() < 1e-12);
        assert!((p.h_inverse(u, v) - u).abs() < 1e-12);
      }
    }
  }

  /// Round-trip h ∘ h_inverse = identity (to numerical tolerance) across
  /// all closed-form families. Tolerances reflect the underlying numerical
  /// chain: Gaussian / StudentT compose ndtri + norm_cdf or `t_cdf` +
  /// quantile (≈ 1e-7), Clayton / Frank are pure algebraic (≈ 1e-10).
  #[test]
  fn h_inverse_round_trips_all_families() {
    let cases: Vec<(PairCopula, f64)> = vec![
      (PairCopula::Gaussian { rho: 0.5 }, 1e-6),
      (PairCopula::Gaussian { rho: -0.3 }, 1e-6),
      (PairCopula::Clayton { theta: 2.0 }, 1e-10),
      (PairCopula::Clayton { theta: 0.5 }, 1e-10),
      (PairCopula::Frank { theta: 3.0 }, 1e-10),
      (PairCopula::Frank { theta: -2.0 }, 1e-10),
      (PairCopula::StudentT { rho: 0.4, nu: 4.0 }, 1e-6),
      (PairCopula::StudentT { rho: -0.5, nu: 8.0 }, 1e-6),
    ];
    for (cop, tol) in cases {
      for u in [0.1, 0.25, 0.5, 0.75, 0.9] {
        for v in [0.1, 0.5, 0.9] {
          let p = cop.h(u, v);
          let u_back = cop.h_inverse(p, v);
          assert!(
            (u_back - u).abs() < tol,
            "{cop:?}: u={u}, v={v}, h(u|v)={p}, h⁻¹(p|v)={u_back}, err={}",
            (u_back - u).abs()
          );
        }
      }
    }
  }

  /// Gaussian h(u, 0.5) should equal Φ(Φ⁻¹(u) - 0) when ρ=0 (independence
  /// boundary), i.e. the identity (within ndtri/norm_cdf round-trip error,
  /// which is ≈ 1e-9 in the bulk and degrades near the tails).
  #[test]
  fn gaussian_h_at_zero_rho_is_identity() {
    let p = PairCopula::Gaussian { rho: 0.0 };
    for u in [0.1, 0.5, 0.9] {
      assert!(
        (p.h(u, 0.4) - u).abs() < 1e-6,
        "h(u={u}, v=0.4, ρ=0) = {} should ≈ {u}",
        p.h(u, 0.4)
      );
    }
  }

  /// Clayton density at u = v = 0.5 with θ = 2 matches the analytic
  /// formula c(0.5, 0.5) = 3 · 0.5^{-3} · (2·0.5^{-2} - 1)^{-2.5}.
  #[test]
  fn clayton_density_closed_form() {
    let p = PairCopula::Clayton { theta: 2.0 };
    let theta: f64 = 2.0;
    let u = 0.5_f64;
    let v = 0.5_f64;
    let s = u.powf(-theta) + v.powf(-theta) - 1.0;
    let expected = (1.0 + theta) * (u * v).powf(-theta - 1.0) * s.powf(-1.0 / theta - 2.0);
    let got = p.density(u, v);
    assert!(
      (got - expected).abs() / expected < 1e-10,
      "Clayton(θ=2) density at (0.5, 0.5): got {got}, expected {expected}"
    );
  }

  /// In the ν → ∞ limit the Student-t h-function collapses to the
  /// Gaussian one. ν = 200 gives a ~1% agreement.
  #[test]
  fn student_t_h_large_nu_approaches_gaussian() {
    let g = PairCopula::Gaussian { rho: 0.4 };
    let t = PairCopula::StudentT { rho: 0.4, nu: 200.0 };
    for u in [0.2, 0.5, 0.8] {
      for v in [0.2, 0.5, 0.8] {
        let hg = g.h(u, v);
        let ht = t.h(u, v);
        assert!(
          (ht - hg).abs() < 0.01,
          "ν=200 t-h({u},{v})={ht} vs Gaussian-h={hg}"
        );
      }
    }
  }
}
