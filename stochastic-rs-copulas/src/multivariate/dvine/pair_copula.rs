//! # Pair-copula families
//!
//! $$
//! h(u \mid v) = \frac{\partial C(u, v)}{\partial v}, \qquad
//! h^{-1}(p, v) \text{ solves } h(u \mid v) = p
//! $$
//!
//! the conditional cdf the vines propagate pseudo-observations through (`h`) and sample through
//! (`h_inverse`): each family's bivariate kernels, closed-form but for BB1's and BB7's `h_inverse`, a Brent root.
//!
//! Reference: Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009), "Pair-copula constructions of multiple dependence", Insurance: Mathematics and Economics 44(2), 182–198, doi:10.1016/j.insmatheco.2007.02.001 (preprint: SFB 386 Discussion Paper 487), eq. (7) for h and App. B for the Gaussian, Student-t and Clayton h-functions and inverses.

use crate::bivariate::bb1::Bb1;
use crate::bivariate::bb7::Bb7;
use crate::bivariate::clayton::Clayton;
use crate::bivariate::conditional::confine;
use crate::bivariate::conditional::inside;
use crate::bivariate::conditional::numerical_quantile;
use crate::bivariate::frank::Frank;
use crate::bivariate::gaussian::GaussianCopula;
use crate::bivariate::t_copula::TCopula;
use crate::error::CopulaError;

/// Bivariate pair-copula family of the C-, D- and R-vines, held by value in their `Vec<Vec<PairCopula>>` trees; a vine
/// constructor admits the parameters the family's bivariate `try_new` does. Pairs must be exchangeable.
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
  /// any real $\theta$. Symmetric tail-independent.
  Frank {
    /// Generator parameter $\theta$; $\theta = 0$ is the independence copula.
    theta: f64,
  },
  /// Student-$t$: $C(u, v) = t_{\rho,\nu}(t_\nu^{-1}(u), t_\nu^{-1}(v))$.
  /// Symmetric tail-dependent, $\nu \to \infty$ collapses to Gaussian.
  StudentT {
    /// Correlation $\rho \in (-1, 1)$.
    rho: f64,
    /// Degrees of freedom, finite with $\nu / 2 > 0$; `h`, `h_inverse` and `log_density` panic otherwise, which the
    /// vine constructors rule out.
    nu: f64,
  },
  /// BB1 (Clayton–Gumbel) copula — see [`crate::bivariate::bb1::Bb1`].
  Bb1 { theta: f64, delta: f64 },
  /// BB7 (Joe–Clayton) copula — see [`crate::bivariate::bb7::Bb7`].
  Bb7 { theta: f64, delta: f64 },
}

impl PairCopula {
  /// `h(u | v) = ∂C/∂v`, the conditional cdf $F_{U \mid V = v}(u)$: the bivariate family's own, an exact 0 or 1 moved
  /// one ulp inside so the vine recursions never meet an edge they did not start from.
  pub fn h(&self, u: f64, v: f64) -> f64 {
    let h = match *self {
      PairCopula::Independence => u,
      PairCopula::Gaussian { rho } => GaussianCopula::h(rho)(u, v),
      PairCopula::Clayton { theta } => Clayton::h_at(theta, u, v),
      PairCopula::Frank { theta } => Frank::h(theta)(u, v),
      PairCopula::Bb1 { theta, delta } => Bb1::h_scalar(u, v, theta, delta),
      PairCopula::Bb7 { theta, delta } => Bb7::h_scalar(u, v, theta, delta),
      PairCopula::StudentT { rho, nu } => TCopula::h(rho, nu)(u, v),
    };
    inside(h)
  }

  /// The `u` with `h(u | v) = p`, the vine samplers' step (Aas et al. 2009, Algorithms 1 and 2): the bivariate family's
  /// own inverse under its quantile rule, BB1 and BB7 by Brent's `zero` on their `ln h`.
  pub fn h_inverse(&self, p: f64, v: f64) -> f64 {
    let u = match *self {
      PairCopula::Independence => p,
      PairCopula::Gaussian { rho: 0.0 } => p,
      PairCopula::Gaussian { rho } => GaussianCopula::inverse(rho)(p, v),
      PairCopula::Clayton { theta } => Clayton::inverse_at(theta, p, v),
      PairCopula::Frank { theta } => Frank::inverse_at(theta, p, v),
      PairCopula::Bb1 { theta, delta } => {
        numerical_quantile(p, v, &|u, v| Bb1::ln_h(u, v, theta, delta)).unwrap_or(f64::NAN)
      }
      PairCopula::Bb7 { theta, delta } => {
        numerical_quantile(p, v, &|u, v| Bb7::ln_h(u, v, theta, delta)).unwrap_or(f64::NAN)
      }
      PairCopula::StudentT { rho, nu } => TCopula::inverse(rho, nu)(p, v),
    };
    confine(u)
  }

  /// Log copula density $\log c(u, v) = \log \frac{\partial^2 C(u,v)}{\partial u\,\partial v}$, an edge's term of the
  /// vine likelihood: the bivariate family's own.
  pub fn log_density(&self, u: f64, v: f64) -> f64 {
    match *self {
      PairCopula::Independence => 0.0,
      PairCopula::Gaussian { rho } => GaussianCopula::ln_density(rho, u, v),
      PairCopula::Clayton { theta } => Clayton::ln_density(theta, u, v),
      PairCopula::Frank { theta } => Frank::ln_density(theta, u, v),
      PairCopula::Bb1 { theta, delta } => Bb1::log_density_scalar(u, v, theta, delta),
      PairCopula::Bb7 { theta, delta } => Bb7::log_density_scalar(u, v, theta, delta),
      PairCopula::StudentT { rho, nu } => TCopula::ln_density_at(rho, nu)(u, v),
    }
  }

  /// Copula density $c(u, v)$ (exponentiated log density).
  pub fn density(&self, u: f64, v: f64) -> f64 {
    self.log_density(u, v).exp()
  }
}

/// The `InvalidParameter` of the first pair whose parameters its bivariate family's `try_new` refuses, so a vine holds
/// only pairs that family would build.
pub(crate) fn check_parameters(trees: &[Vec<PairCopula>]) -> Result<(), CopulaError> {
  for pair in trees.iter().flatten() {
    match *pair {
      PairCopula::Independence => {}
      PairCopula::Gaussian { rho } => GaussianCopula::try_new(rho).map(drop)?,
      PairCopula::Clayton { theta } => Clayton::try_new(theta).map(drop)?,
      PairCopula::Frank { theta } => Frank::try_new(theta).map(drop)?,
      PairCopula::StudentT { rho, nu } => TCopula::try_new(rho, nu).map(drop)?,
      PairCopula::Bb1 { theta, delta } => Bb1::try_new(theta, delta).map(drop)?,
      PairCopula::Bb7 { theta, delta } => Bb7::try_new(theta, delta).map(drop)?,
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

  /// `h ∘ h⁻¹` is the identity in every closed-form family: to 1e-10 for Clayton and Frank, to 1e-6 where the chain
  /// passes `ndtri` and `norm_cdf` (Gaussian) or the Student-t quantile and cdf.
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

  /// At ρ = 0 the Gaussian `h(u, 0.5)` is the identity up to the `ndtri`/`norm_cdf` round trip,
  /// ≈ 1e-9 in the bulk and worse in the tails.
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

  /// The Frank and Student-t pairs are the bivariate families' closed forms: Frank at `θ = 30` and through the `θ < 0`
  /// reflection against 60-digit values, where an unfactored form lost up to four digits; the t pair bit for bit.
  #[test]
  fn frank_and_t_pairs_are_the_bivariate_closed_forms() {
    let frank = PairCopula::Frank { theta: 30.0 };
    for (u, v, want) in [
      (0.999, 0.999, 0.9712939262423305),
      (0.529, 0.99, 9.856146466724287e-7),
    ] {
      let got = frank.h(u, v);
      assert!(
        (got - want).abs() <= 1e-14 * want,
        "h({u} | {v}) = {got}, want {want}"
      );
    }
    for (y, v, want) in [
      (0.99, 0.999, 0.9996548388390495),
      (1e-6, 0.99, 0.52948299425107),
    ] {
      let got = frank.h_inverse(y, v);
      assert!(
        (got - want).abs() <= 1e-14 * want,
        "h⁻¹({y} | {v}) = {got}, want {want}"
      );
    }
    let reflected = PairCopula::Frank { theta: -3.0 }.h_inverse(1e-6, 0.99);
    assert!(
      (reflected - 3.2638389105694917e-7).abs() <= 1e-14 * 3.2638389105694917e-7,
      "θ = −3: {reflected}"
    );
    let (rho, nu) = (-0.5, 8.0);
    let t = PairCopula::StudentT { rho, nu };
    for (u, v) in [(0.1, 0.5), (0.25, 0.9), (0.75, 0.1)] {
      assert_eq!(t.h(u, v), TCopula::h(rho, nu)(u, v));
      assert_eq!(t.h_inverse(u, v), TCopula::inverse(rho, nu)(u, v));
    }
  }

  /// Each vine constructor refuses, with its error, a pair its bivariate family's `try_new` refuses, in any tree; the
  /// match has no wildcard arm, so a new family does not compile until it has rows here.
  #[test]
  fn the_vine_constructors_refuse_the_pairs_their_bivariate_families_refuse() {
    use crate::multivariate::cvine::CVine;
    use crate::multivariate::dvine::DVine;

    let rows = [
      (PairCopula::Independence, None),
      (PairCopula::Gaussian { rho: -0.5 }, None),
      (PairCopula::Gaussian { rho: 1.0 }, Some("theta")),
      (PairCopula::Gaussian { rho: 1.5 }, Some("theta")),
      (PairCopula::Gaussian { rho: f64::NAN }, Some("theta")),
      (PairCopula::Clayton { theta: 0.0 }, None),
      (PairCopula::Clayton { theta: -1.0 }, Some("theta")),
      (
        PairCopula::Clayton {
          theta: f64::INFINITY,
        },
        Some("theta"),
      ),
      (PairCopula::Frank { theta: -3.0 }, None),
      (PairCopula::Frank { theta: f64::NAN }, Some("theta")),
      (PairCopula::StudentT { rho: 0.5, nu: 4.0 }, None),
      (PairCopula::StudentT { rho: 2.0, nu: 4.0 }, Some("theta")),
      (PairCopula::StudentT { rho: 0.5, nu: -1.0 }, Some("nu")),
      (
        PairCopula::Bb1 {
          theta: 0.8,
          delta: 1.6,
        },
        None,
      ),
      (
        PairCopula::Bb1 {
          theta: -1.0,
          delta: 0.5,
        },
        Some("delta"),
      ),
      (
        PairCopula::Bb1 {
          theta: 0.0,
          delta: 1.6,
        },
        Some("theta"),
      ),
      (
        PairCopula::Bb7 {
          theta: 1.7,
          delta: 0.9,
        },
        None,
      ),
      (
        PairCopula::Bb7 {
          theta: 1.7,
          delta: -1.0,
        },
        Some("delta"),
      ),
      (
        PairCopula::Bb7 {
          theta: 0.5,
          delta: 0.9,
        },
        Some("theta"),
      ),
    ];
    for (pair, refused) in rows {
      let bivariate = match pair {
        PairCopula::Independence => Ok(()),
        PairCopula::Gaussian { rho } => GaussianCopula::try_new(rho).map(drop),
        PairCopula::Clayton { theta } => Clayton::try_new(theta).map(drop),
        PairCopula::Frank { theta } => Frank::try_new(theta).map(drop),
        PairCopula::StudentT { rho, nu } => TCopula::try_new(rho, nu).map(drop),
        PairCopula::Bb1 { theta, delta } => Bb1::try_new(theta, delta).map(drop),
        PairCopula::Bb7 { theta, delta } => Bb7::try_new(theta, delta).map(drop),
      };
      let in_first = || {
        vec![
          vec![pair, PairCopula::Independence],
          vec![PairCopula::Independence],
        ]
      };
      let in_last = || vec![vec![PairCopula::Independence; 2], vec![pair]];
      let verdicts = [
        DVine::new(3, in_first()).map(drop),
        DVine::new(3, in_last()).map(drop),
        CVine::new(3, in_first()).map(drop),
        CVine::new(3, in_last()).map(drop),
      ];
      for verdict in verdicts {
        match (refused, &verdict, &bivariate) {
          (None, Ok(()), Ok(())) => {}
          (Some(want), Err(CopulaError::InvalidParameter { name, .. }), Err(expected)) => {
            assert_eq!(*name, want, "{pair:?}");
            assert_eq!(
              verdict.unwrap_err().to_string(),
              expected.to_string(),
              "{pair:?}"
            );
          }
          _ => panic!("{pair:?}: vine {verdict:?}, bivariate {bivariate:?}"),
        }
      }
    }
  }

  #[test]
  #[should_panic(expected = "nu must satisfy")]
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
