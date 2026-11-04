use super::*;

/// mpmath at 60 and 120 digits: `A` at a double node `(|t|, g)` within `4e-18` relative, at the anchors of the two worst
/// cells of the review's class, a node off the crossing and one at `|t| = 0.8`; the plain `ln|A|` was off by up to `1.2e-15`.
#[test]
fn the_argument_holds_to_twice_the_precision() {
  for ((rho, nu, sign_y, sign_t, abs_t, gap), (hi, lo)) in [
    (
      (
        -0.999_985_343_937_099_9,
        483_983_037_239.454_1,
        1.0,
        -1.0,
        (9.532_544_390_900_752e-7, 0.0),
        (-2.680_167_112_137_406e-7, 0.0),
      ),
      (-34.437_549_326_194_97, -6.097_830_646_012_912e-17),
    ),
    (
      (
        -0.99996,
        1e6,
        1.0,
        -1.0,
        (9.850_583_652_133_188e-4, 0.0),
        (-3.220_298_556_437_716e-4, 0.0),
      ),
      (-36.000_000_000_000_064, 1.097_599_852_978_172_1e-15),
    ),
    (
      (0.9999, 1e8, -1.0, 1.0, (1e-3, 0.0), (-5e-4, 0.0)),
      (-1_060.615_584_401_933_6, -5.160_090_872_022_009_5e-14),
    ),
    (
      (-0.999, 100.0, 1.0, -1.0, (0.8, 1e-17), (-0.05, 2e-18)),
      (-10.907_719_893_640_676, 8.287_804_793_608_489e-16),
    ),
  ] {
    let cond = Conditional {
      c0: f64::NAN,
      rho,
      ln_abs_rho: f64::NAN,
      sign_y,
      ln_y: f64::NAN,
      e2y: f64::NAN,
      abs_ty: abs_t.0 + gap.0,
    };
    let (a, a_lo) = cond.argument_parts(nu, sign_t, abs_t, gap);
    let miss = ((a - hi) + (a_lo - lo)) / hi;
    assert!(
      miss.abs() <= 4e-18,
      "ρ = {rho}, ν = {nu}: {a} + {a_lo:e}, want {hi} + {lo:e}"
    );
  }
}

#[test]
fn tmp_parts() {
  for x in [0.025_f64, 0.775, 0.8] {
    let (s, s_lo) = sinh_parts((x, 0.0));
    let (c, c_lo) = cosh_parts((x, 0.0));
    eprintln!(
      "x {x}: sinh {s:e} + {s_lo:e} (std {:e}), cosh {c:e} + {c_lo:e} (std {:e})",
      x.sinh(),
      x.cosh()
    );
  }
}
