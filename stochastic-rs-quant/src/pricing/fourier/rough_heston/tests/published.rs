//! Numbers printed by the papers' authors.

use num_complex::Complex64;

use super::super::Riccati;
use super::super::pade::Pade;
use super::implied_vol;
use crate::pricing::fourier::Cumulants;
use crate::pricing::fourier::FourierModelExt;

/// Gatheral & Radoičić's forward-variance CGF `∫ξ(t − s)(D^αh + λh)(s)ds` by their 200-point
/// trapezoid, with the curve `ξ(t) = 0.04 − 0.0336e^{−3t}` of their notebook.
struct NotebookModel {
  pade: Pade,
  riccati: Riccati,
}

fn curve(t: f64) -> f64 {
  0.04 - 0.0336 * (-3.0 * t).exp()
}

impl FourierModelExt for NotebookModel {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    const POINTS: usize = 200;
    let z = Complex64::i() * xi;
    let moment = Riccati::moment(z);
    let coefficients = self.pade.coefficients(z, &self.riccati);
    let alpha = self.pade.alpha();
    let g = |i: usize| {
      let s = t * i as f64 / POINTS as f64;
      let h = Pade::h(&coefficients, moment, s.powf(alpha));
      (self.riccati.rhs(moment, z, h) + h * self.riccati.kappa) * curve(t - s)
    };
    let ends = 0.5 * (g(0) + g(POINTS));
    let sum = (0..=POINTS).map(g).sum::<Complex64>() - ends;
    (sum * (t / POINTS as f64)).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let variance = 0.04 * t - 0.0336 * (1.0 - (-3.0 * t).exp()) / 3.0;
    Cumulants {
      c1: -0.5 * variance,
      c2: variance,
      c4: None,
    }
  }
}

/// "Rough Heston smiles using the Padé approximations", github.com/jgatheral/RationalRoughHeston:
/// `H = 0.05, ν = 0.4, ρ = −0.65, λ = 1`, `k = −0.1`, `t = 0.019`, orders 3 to 6. An independent
/// quadrature sits 2.7e-6 from their seven printed digits; the bound is four times that.
/// Reference: Gatheral & Radoičić (2024), arXiv:2310.09181, (1.2), (4.1)
#[test]
fn the_authors_notebook_vols_are_reproduced() {
  let published = [
    (3, 0.2757039),
    (4, 0.2761866),
    (5, 0.2762137),
    (6, 0.2757119),
  ];
  for (order, want) in published {
    let model = NotebookModel {
      pade: Pade::new(0.55, order),
      riccati: Riccati {
        kappa: 1.0,
        nu: 0.4,
        rho: -0.65,
      },
    };
    let got = implied_vol(&model, (-0.1_f64).exp(), 0.019);
    assert!((got - want).abs() < 1e-5, "order {order}: {got} vs {want}");
  }
}
