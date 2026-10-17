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

use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use crate::error::CopulaError;
use crate::student_t;

/// Bivariate pair-copula family of the C-, D- and R-vines, held by value in their `Vec<Vec<PairCopula>>` trees.
/// Pair families must be exchangeable: the vine recursions call `h` from either end of an edge.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
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
    /// Degrees of freedom, finite with $\nu / 2 > 0$, which the vine constructors check; `h`, `h_inverse` and
    /// `log_density` panic otherwise.
    nu: f64,
  },
  /// BB1 (Clayton–Gumbel) copula — see [`crate::bivariate::bb1::Bb1`].
  Bb1 { theta: f64, delta: f64 },
  /// BB7 (Joe–Clayton) copula — see [`crate::bivariate::bb7::Bb7`].
  Bb7 { theta: f64, delta: f64 },
}

impl PairCopula {
  /// Family-specific h-function $h(u \mid v) = \partial C / \partial v$.
  /// Returns the conditional CDF $F_{U \mid V = v}(u)$.
  pub fn h(&self, u: f64, v: f64) -> f64 {
    let u = clamp(u);
    let v = clamp(v);
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
      PairCopula::Bb1 { theta, delta } => crate::bivariate::bb1::Bb1::h_scalar(u, v, theta, delta),
      PairCopula::Bb7 { theta, delta } => crate::bivariate::bb7::Bb7::h_scalar(u, v, theta, delta),
      PairCopula::StudentT { rho, nu } => {
        // Aas-Czado (2009) Appendix A.2.
        // x = t_ν^{-1}(u), y = t_ν^{-1}(v)
        // h(u|v) = t_{ν+1}((x - ρ y) · √((ν+1) / ((ν + y²)(1 - ρ²))))
        let x = student_t::quantile(u, nu);
        let y = student_t::quantile(v, nu);
        let scale = ((nu + 1.0) / ((nu + y * y) * (1.0 - rho * rho))).sqrt();
        student_t::cdf((x - rho * y) * scale, nu + 1.0)
      }
    };
    clamp(h)
  }

  /// Inverse h-function: returns the $u$ that satisfies $h(u \mid v) = p$.
  /// Used by the vine samplers (Aas et al. 2009, Algorithms 1 and 2).
  pub fn h_inverse(&self, p: f64, v: f64) -> f64 {
    let p = clamp(p);
    let v = clamp(v);
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
      PairCopula::Bb1 { .. } | PairCopula::Bb7 { .. } => {
        crate::bivariate::two_parameter::invert_h(|u| self.h(u, v), p)
      }
      PairCopula::StudentT { rho, nu } => {
        // Inverse of h(u|v):
        //   z = t_{ν+1}^{-1}(p),  y = t_ν^{-1}(v)
        //   x = z / √((ν+1) / ((ν + y²)(1 - ρ²)))  +  ρ y
        //     = z · √((ν + y²)(1 - ρ²)/(ν+1))  +  ρ y
        //   u = t_ν(x)
        let z = student_t::quantile(p, nu + 1.0);
        let y = student_t::quantile(v, nu);
        let scale = ((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0)).sqrt();
        let x = z * scale + rho * y;
        student_t::cdf(x, nu)
      }
    };
    clamp(u)
  }

  /// Log copula density $\log c(u, v) = \log \frac{\partial^2 C(u, v)}{\partial u\, \partial v}$.
  /// Used for sequential MLE fitting and direct density evaluation on the
  /// PCC tree.
  pub fn log_density(&self, u: f64, v: f64) -> f64 {
    let u = clamp(u);
    let v = clamp(v);
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
        // c = (1 + θ)(uv)^{−θ−1} s^{−1/θ−2} with ln s = M + ln(1 + e^{m−M}(1 − e^{−m})), {M ≥ m} = {−θ ln u, −θ ln v}: s in
        // logarithms, so `u^{−θ}` never overflows.
        let (a, b) = (-theta * u.ln(), -theta * v.ln());
        let (big, small) = (a.max(b), a.min(b));
        let ln_s = big + ((small - big).exp() * -(-small).exp_m1()).ln_1p();
        theta.ln_1p() + (-theta - 1.0) * (u.ln() + v.ln()) + (-1.0 / theta - 2.0) * ln_s
      }
      PairCopula::Frank { theta } => {
        // c_θ(u, v) = θ(1 − e^{−θ}) e^{−θ(u+v)} / D² with D = e^{−θu}(1 − e^{−θv}) + e^{−θv}(1 − e^{−θ(1−v)}), two positive terms
        // for θ > 0; c_θ(u, v) = c_{−θ}(u, 1 − v) for θ < 0.
        let (t, p, q) = if theta > 0.0 {
          (theta, v, 1.0 - v)
        } else {
          (-theta, 1.0 - v, v)
        };
        let first = -t * u + (-(-t * p).exp_m1()).ln();
        let second = -t * p + (-(-t * q).exp_m1()).ln();
        let ln_d = first.max(second) + (-(first - second).abs()).exp().ln_1p();
        t.ln() + (-(-t).exp_m1()).ln() - t * (u + p) - 2.0 * ln_d
      }
      PairCopula::Bb1 { theta, delta } => {
        crate::bivariate::bb1::Bb1::log_density_scalar(u, v, theta, delta)
      }
      PairCopula::Bb7 { theta, delta } => {
        crate::bivariate::bb7::Bb7::log_density_scalar(u, v, theta, delta)
      }
      PairCopula::StudentT { rho, nu } => student_t::ln_copula_density(
        student_t::quantile(u, nu),
        student_t::quantile(v, nu),
        rho,
        nu,
      ),
    }
  }

  /// Copula density $c(u, v)$ (exponentiated log density).
  pub fn density(&self, u: f64, v: f64) -> f64 {
    self.log_density(u, v).exp()
  }
}

const EPS: f64 = 1e-12;

/// The bounds every pair evaluation holds its inputs and outputs in.
pub(crate) fn clamp(u: f64) -> f64 {
  u.clamp(EPS, 1.0 - EPS)
}

/// `InvalidParameter` for a Student-t pair whose `nu` the Student-t law refuses, which its evaluation would panic on.
pub(crate) fn check_pairs(trees: &[Vec<PairCopula>]) -> Result<(), CopulaError> {
  for pair in trees.iter().flatten() {
    if let PairCopula::StudentT { nu, .. } = *pair {
      student_t::check_nu(nu)?;
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bivariate::bb1::Bb1;
  use crate::traits::BivariateExt;

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
    let t = PairCopula::StudentT {
      rho: 0.4,
      nu: 200.0,
    };
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

  #[test]
  fn bb_pairs_round_trip_h_and_match_the_bivariate_structs() {
    let bb1 = PairCopula::Bb1 {
      theta: 0.8,
      delta: 1.6,
    };
    let bb7 = PairCopula::Bb7 {
      theta: 1.7,
      delta: 0.9,
    };
    for pair in [bb1, bb7] {
      for (u, v) in [(0.2, 0.7), (0.55, 0.35), (0.9, 0.1)] {
        let p = pair.h(u, v);
        assert!(p > 0.0 && p < 1.0);
        let back = pair.h_inverse(p, v);
        assert!((back - u).abs() < 1e-9, "{pair:?} u {u} back {back}");
        assert!(pair.density(u, v).is_finite() && pair.density(u, v) > 0.0);
      }
    }
    let bb1_struct = Bb1::new(0.8, 1.6);
    let x = ndarray::array![[0.2, 0.7]];
    let reference = bb1_struct.pdf(x.view()).unwrap()[0];
    assert!((bb1.density(0.2, 0.7) - reference).abs() < 1e-12);
  }

  /// 50-digit mpmath values of the expanded formulas where f64 loses them: Frank's denominator cancels near (1, 1) and
  /// Clayton's `u^{−θ}` overflows near (0, 0); the θ < 0 Frank rows cross the rotation.
  #[test]
  fn log_density_holds_at_strong_dependence() {
    let frank = [
      (30.0, 0.95, 0.95, 2.2514907829411306),
      (30.0, 0.999, 0.995, 3.229447778423594),
      (35.0, 0.99, 0.99, 3.037845014469233),
      (35.0, 0.995, 0.999, 3.3564222824584893),
      (40.0, 0.95, 0.95, 2.442716933314608),
      (40.0, 0.999, 0.9995, 3.6304328995062574),
      (50.0, 0.99, 0.99, 3.2484298739257733),
      (50.0, 0.9999, 0.9995, 3.8822693052484136),
      (-40.0, 0.99, 0.01, 3.1190028997956873),
      (-50.0, 0.0005, 0.9999, 3.882269305248411),
    ];
    let clayton = [
      (50.0, 1e-7, 1e-7, 18.649763978951555),
      (50.0, 1e-4, 1.2e-4, 3.843544653068734),
      (100.0, 1e-4, 1e-4, 12.432235055891953),
      (100.0, 1e-3, 1.01e-3, 9.885546533017381),
      (200.0, 1e-3, 1e-3, 10.821300090018523),
      (200.0, 0.01, 0.0101, 7.651583313741083),
    ];
    let pins = frank
      .map(|(theta, u, v, want)| (PairCopula::Frank { theta }, u, v, want))
      .into_iter()
      .chain(clayton.map(|(theta, u, v, want)| (PairCopula::Clayton { theta }, u, v, want)));
    for (pair, u, v, want) in pins {
      let got = pair.log_density(u, v);
      assert!(
        (got - want).abs() <= 1e-13 * want.abs(),
        "{pair:?} at ({u}, {v}): {got} vs {want}"
      );
    }
  }

  #[test]
  #[should_panic(expected = "nu must satisfy `nu > T::zero()`, got nu = -1.0")]
  fn a_student_t_pair_outside_the_nu_rule_panics_with_the_laws_message() {
    PairCopula::StudentT { rho: 0.5, nu: -1.0 }.h(0.3, 0.6);
  }

  /// The match has no wildcard arm, so a new family does not compile until it joins this exchangeability check.
  #[test]
  fn every_pair_family_is_exchangeable() {
    let families = [
      PairCopula::Independence,
      PairCopula::Gaussian { rho: 0.6 },
      PairCopula::Gaussian { rho: -0.4 },
      PairCopula::Clayton { theta: 2.5 },
      PairCopula::Frank { theta: 4.0 },
      PairCopula::Frank { theta: -3.0 },
      PairCopula::StudentT { rho: 0.5, nu: 4.0 },
      PairCopula::Bb1 {
        theta: 0.8,
        delta: 1.6,
      },
      PairCopula::Bb7 {
        theta: 1.7,
        delta: 0.9,
      },
    ];
    let grid = [0.05, 0.2, 0.45, 0.7, 0.95];
    for pair in families {
      match pair {
        PairCopula::Independence
        | PairCopula::Gaussian { .. }
        | PairCopula::Clayton { .. }
        | PairCopula::Frank { .. }
        | PairCopula::StudentT { .. }
        | PairCopula::Bb1 { .. }
        | PairCopula::Bb7 { .. } => {}
      }
      for u in grid {
        for v in grid {
          let gap = (pair.log_density(u, v) - pair.log_density(v, u)).abs();
          assert!(gap <= 1e-12, "{pair:?} at ({u}, {v}): gap {gap:e}");
        }
      }
    }
  }
}
