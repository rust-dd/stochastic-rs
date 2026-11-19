macro_rules! euler_multifactor {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $u3:ident, $u4:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    $emit! {
      $($previous)*
  /// The two-dimensional Wishart process by the exact Ahdida–Alfonsi step: carried to
  /// `y = θ⁻¹ m X mᵀ θ⁻ᵀ`, stepped one coordinate at a time, and carried back by `θ Y θᵀ`.
  116 => WishartTwo { m00, m01, m10, m11, t00, t01, t10, t11, i00, i01, i10, i11, two }
    state (a, b, c)
    noise (z1, w1, z2, w2)
    step {
      bind p00 = m00 * a + m01 * b;
      bind p01 = m00 * b + m01 * c;
      bind p10 = m10 * a + m11 * b;
      bind p11 = m10 * b + m11 * c;
      bind q00 = p00 * m00 + p01 * m01;
      bind q01 = p00 * m10 + p01 * m11;
      bind q11 = p10 * m10 + p11 * m11;
      bind r00 = i00 * q00 + i01 * q01;
      bind r01 = i00 * q01 + i01 * q11;
      bind r10 = i10 * q00 + i11 * q01;
      bind r11 = i10 * q01 + i11 * q11;
      bind y00 = r00 * i00 + r01 * i01;
      bind y01 = r00 * i10 + r01 * i11;
      bind y11 = r10 * i10 + r11 * i11;
      bind s1 = sqrt(max(y11, lit(1.0e-30)));
      bind u1 = y01 / s1;
      bind h1 = max(y00 - u1 * u1, lit(0.0));
      bind zz1 = z1 / sqrt($dt) + sqrt(h1 / $dt);
      bind hn1 = $dt * (zz1 * zz1 + $gm);
      bind un1 = u1 + w1;
      bind a1 = hn1 + un1 * un1;
      bind b1 = s1 * un1;
      bind s2 = sqrt(max(a1, lit(1.0e-30)));
      bind $u2 = b1 / s2;
      bind h2 = max(y11 - $u2 * $u2, lit(0.0));
      bind zz2 = z2 / sqrt($dt) + sqrt(h2 / $dt);
      bind hn2 = $dt * (zz2 * zz2 + $gm2);
      bind un2 = $u2 + w2;
      bind c2 = pick(two, hn2 + un2 * un2, y11);
      bind b2 = pick(two, s2 * un2, b1);
      bind e00 = t00 * a1 + t01 * b2;
      bind e01 = t00 * b2 + t01 * c2;
      bind e10 = t10 * a1 + t11 * b2;
      bind e11 = t10 * b2 + t11 * c2;
      e00 * t00 + e01 * t01,
      e00 * t10 + e01 * t11,
      e10 * t10 + e11 * t11
    }
    report { a, b, c },

  /// A bivariate Hawkes process, one event per step: the earliest of the targets' exact excess
  /// clocks (Dassios–Zhao, from `u` and `u3`) and the baselines' clock (`u2`, its mark from `u4`).
  117 => HawkesEvents2 { mu1, mu2, a11, a12, a21, a22, b1, b2 }
    state (t, s1, s2, k)
    noise ()
    step {
      bind se1 = max(s1, lit(1.0e-12));
      bind d1 = b1 * ln($u) / se1 + lit(1.0);
      bind w1 = pick(less(negate(d1), lit(0.0)), negate(ln(max(d1, lit(1.0e-30)))) / b1, lit(1.0e30));
      bind se2 = max(s2, lit(1.0e-12));
      bind d2 = b2 * ln($u3) / se2 + lit(1.0);
      bind w2 = pick(less(negate(d2), lit(0.0)), negate(ln(max(d2, lit(1.0e-30)))) / b2, lit(1.0e30));
      bind w0 = negate(ln($u2)) / (mu1 + mu2);
      bind w = min(w0, min(w1, w2));
      bind base1 = less($u4 * (mu1 + mu2), mu1);
      bind kn = pick(leq(w1, w), lit(0.0), pick(leq(w2, w), lit(1.0), pick(base1, lit(0.0), lit(1.0))));
      bind col1 = less(kn, lit(0.5));
      t + w,
      s1 * exp(negate(b1 * w)) + pick(col1, a11, a12),
      s2 * exp(negate(b2 * w)) + pick(col1, a21, a22),
      kn
    }
    report { t, k },

  /// The Cheyette quasi-Gaussian state by Euler–Maruyama, `dx = (y − κx) dt + σ(t, x) dW`,
  /// `dy = (σ² − 2κy) dt`, with `σ(t, x)` the launch's first program at the step's start, `pv`.
  118 => CheyetteLocalVol { kappa }
    state (x, y)
    noise (dz)
    step {
      bind s = $pv;
      x + (y - kappa * x) * $dt + s * dz,
      y + (s * s - kappa * y * lit(2.0)) * $dt
    }
    report { x, y },

  /// A stochastic Volterra equation under its kernel's Markov lift: the drift and diffusion are
  /// the launch's two programs at the step's start, `pv` and `pv2`; the step is the lifted value.
  119 => VolterraProgram { }
    state (x)
    noise (dz)
    step { $lv }
    report { x }
    lift { drift ($pv) diffusion ($pv2) shock (dz) },

  /// Heston stochastic-local volatility: the log-spot by Euler–Maruyama under the leverage `pv`,
  /// the variance truncated at zero under the vol-of-vol `η σ` the host folds into `sigma`.
  120 => HestonSlv { mu, kappa, theta, sigma, rho }
    state (s, v)
    noise (dw, dz)
    step {
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      bind l = $pv;
      s * exp((mu - l * l * vp * lit(0.5)) * $dt + l * sqrt(vp) * dw),
      positive(v + kappa * (theta - vp) * $dt + sigma * sqrt(vp) * db)
    }
    report { s, v },
    }
  };
}
