macro_rules! euler_truncations {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $u3:ident, $u4:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    $emit! {
      $($previous)*
  /// [`Heston`](Family::Heston) under full truncation (Lord, Koekkoek & van Dijk 2010): the
  /// variance slot keeps its sign, and drift, diffusion and the report read its positive part.
  121 => HestonFullTruncation { mu, kappa, theta, sigma, rho, pow_v }
    state (s, v)
    noise (dw, dz)
    step {
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      s + mu * s * $dt + s * sqrt(vp) * dw,
      v + kappa * (theta - vp) * $dt + sigma * pow(vp, pow_v) * db
    }
    report { s, positive(v) },

  /// [`LogHeston`](Family::LogHeston) under full truncation.
  122 => LogHestonFullTruncation { drift, kappa, theta, xi, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dwv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift - vp / lit(2.0)) * $dt + sv * dw),
      v + kappa * (theta - vp) * $dt + xi * sv * dwv
    }
    report { s, positive(v) },

  /// [`DoubleHeston`](Family::DoubleHeston) with both variances under full truncation.
  123 => DoubleHestonFullTruncation {
    mu, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2
  }
    state (s, v1, v2)
    noise (ds1, dq1, ds2, dq2)
    step {
      bind p1 = positive(v1);
      bind p2 = positive(v2);
      bind dv1 = rho1 * ds1 + sqrt(negate(rho1 * rho1 - lit(1.0))) * dq1;
      bind dv2 = rho2 * ds2 + sqrt(negate(rho2 * rho2 - lit(1.0))) * dq2;
      s + mu * s * $dt + s * sqrt(p1) * ds1 + s * sqrt(p2) * ds2,
      v1 + kappa1 * (theta1 - p1) * $dt + sigma1 * sqrt(p1) * dv1,
      v2 + kappa2 * (theta2 - p2) * $dt + sigma2 * sqrt(p2) * dv2
    }
    report { s, positive(v1), positive(v2) },

  /// [`TwoAssetHeston`](Family::TwoAssetHeston) with both variances under full truncation.
  124 => TwoAssetHestonFullTruncation {
    mu1, mu2, kappa1, theta1, sigma1, kappa2, theta2, sigma2,
    l11, l21, l22, l31, l32, l33, l41, l42, l43, l44
  }
    state (x1, v1, x2, v2)
    noise (e1, e2, e3, e4)
    step {
      bind dz1 = l11 * e1;
      bind dz2 = l21 * e1 + l22 * e2;
      bind dw1 = l31 * e1 + l32 * e2 + l33 * e3;
      bind dw2 = l41 * e1 + l42 * e2 + l43 * e3 + l44 * e4;
      bind p1 = positive(v1);
      bind p2 = positive(v2);
      x1 + (mu1 - p1 / lit(2.0)) * $dt + sqrt(p1) * dw1,
      v1 + kappa1 * (theta1 - p1) * $dt + sigma1 * sqrt(p1) * dz1,
      x2 + (mu2 - p2 / lit(2.0)) * $dt + sqrt(p2) * dw2,
      v2 + kappa2 * (theta2 - p2) * $dt + sigma2 * sqrt(p2) * dz2
    }
    report { x1, positive(v1), x2, positive(v2) },

  /// [`BatesJump`](Family::BatesJump) under full truncation.
  125 => BatesJumpFullTruncation { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * $dt + sv * dw + $js),
      v + (alpha - beta * vp) * $dt + sigma * sv * dv
    }
    report { s, positive(v) },

  /// [`Bates1996`](Family::Bates1996) under full truncation.
  126 => Bates1996FullTruncation { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s + drift_c * s * $dt + s * sv * dw + s * $js,
      v + (alpha - beta * vp) * $dt + sigma * sv * dv
    }
    report { s, positive(v) },

  /// [`KouJumpHeston`](Family::KouJumpHeston) under full truncation.
  127 => KouJumpHestonFullTruncation { drift_c, kappa, theta, sigma_v, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * $dt + sv * dw + $js),
      v + kappa * (theta - vp) * $dt + sigma_v * sv * dv
    }
    report { s, positive(v) },

  /// [`ShiftedSquareRoot`](Family::ShiftedSquareRoot) under full truncation: CIR++'s default.
  128 => ShiftedSquareRootFullTruncation { theta, mu, sigma }
    state (x)
    noise (dz)
    step { x + theta * (mu - positive(x)) * $dt + sigma * sqrt(positive(x)) * dz }
    report { positive(x) + $ct },

  /// [`TwoFactorSquareRoot`](Family::TwoFactorSquareRoot) with a scheme per factor: `0` is full
  /// truncation, `1` absorption, `2` reflection. The process launches it when a factor reads `0`.
  129 => TwoFactorSquareRootSchemes {
    theta1, mu1, sigma1, theta2, mu2, sigma2, scheme1, scheme2
  }
    state (a, b)
    noise (dw, dq)
    step {
      bind oa = pick(geq(scheme1, lit(2.0)), abs(a), positive(a));
      bind ob = pick(geq(scheme2, lit(2.0)), abs(b), positive(b));
      bind v1 = a + theta1 * (mu1 - oa) * $dt + sigma1 * sqrt(oa) * dw;
      bind v2 = b + theta2 * (mu2 - ob) * $dt + sigma2 * sqrt(ob) * dq;
      pick(leq(scheme1, lit(0.0)), v1, pick(geq(scheme1, lit(2.0)), abs(v1), positive(v1))),
      pick(leq(scheme2, lit(0.0)), v2, pick(geq(scheme2, lit(2.0)), abs(v2), positive(v2)))
    }
    report {
      bind ob = pick(geq(scheme2, lit(2.0)), abs(b), positive(b));
      pick(geq(scheme1, lit(2.0)), abs(a), positive(a)) + ob + $ct,
      ob
    },

  /// [`SquaredBesselState`](Family::SquaredBesselState) under full truncation.
  130 => SquaredBesselFullTruncation { delta, two }
    state (x)
    noise (dz)
    step { x + delta * $dt + two * sqrt(positive(x)) * dz }
    report { positive(x) },

  /// [`BesselFromSquared`](Family::BesselFromSquared) under full truncation.
  131 => BesselFromSquaredFullTruncation { delta, two }
    state (x)
    noise (dz)
    step { x + delta * $dt + two * sqrt(positive(x)) * dz }
    report { sqrt(positive(x)) },
    }
  };
}
