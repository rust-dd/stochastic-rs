macro_rules! euler_memory {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $u3:ident, $u4:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_multifactor! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $u3, $u4, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// Rough Heston in the two-term rational approximation of its kernel; the history block
  /// convolves the pushed `z` with the weights in `ct`. Reports `y + c1 ν z + c2 ν ∫ / Γ(H − 1/2)`.
  105 => RoughHestonMemory { mu, theta, ek, nu, c1, c2, inv_g, rho }
    state (s, v, y, z)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind st = positive(y + nu * z);
      bind dsp = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      bind y1 = theta + (y - theta) * ek;
      bind z1 = z * ek + sqrt(st) * dw;
      s * exp((mu - vp / lit(2.0)) * $dt + sqrt(vp) * dsp),
      y1 + c1 * nu * z1 + c2 * nu * $cv * inv_g,
      y1,
      z1
    }
    report { s, v }
    history { push (z) weights ($ct) },

  /// The fractional Bates model: [`RoughHestonMemory`](Family::RoughHestonMemory) with unit
  /// calibration coefficients, `xi` for the vol-of-vol, a compensated drift and normal jumps `js`.
  106 => FractionalBatesMemory { mu_c, theta, ek, xi, inv_g, rho }
    state (s, v, y, z)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind st = positive(y + xi * z);
      bind dsp = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      bind y1 = theta + (y - theta) * ek;
      bind z1 = z * ek + sqrt(st) * dw;
      s * exp((mu_c - vp / lit(2.0)) * $dt + sqrt(vp) * dsp + $js),
      y1 + xi * z1 + xi * $cv * inv_g,
      y1,
      z1
    }
    report { s, v }
    history { push (z) weights ($ct) },

  /// Rough Bergomi under the hybrid scheme: the history block's convolution plus a residual
  /// (`de / √dt`) drive `v0² exp(ν √(2H) X − ½ ν² t^{2H})`, the exponent's last term being `ct1`.
  107 => RoughBergomiMemory { r, a, v0sq, residual_sd, rho }
    state (s, v, x)
    noise (dw, dq, de)
    step {
      bind xv = $cv + residual_sd * (de / sqrt($dt));
      s + r * s * $dt + sqrt(v) * s * dw,
      v0sq * exp(a * xv - $ct1),
      xv
    }
    report { s, v }
    history { push (rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq) weights ($ct) },

  /// Linear fractional stable motion: α-stable Chambers–Mallows–Stuck innovations from the step's
  /// two uniforms, averaged under the weights `dt^d ((k+1)^d − k^d)` the host tabulates as `ct`.
  108 => LinearFractionalStable { alpha, inv_alpha, tail_exp, b, scale_s, pi, half_pi }
    state (x)
    noise (dz)
    step { x + $cv }
    report { x }
    history {
      push (
        bind va = pi * ($u - lit(0.5));
        bind w = negate(ln($u2));
        bind phi = alpha * (va + b);
        bind cs = max(sin(va + half_pi), lit(1e-20));
        bind ratio = max(sin(va - phi + half_pi) / w, lit(1e-30));
        scale_s * sin(phi) / pow(cs, inv_alpha) * pow(ratio, tail_exp)
      )
      weights ($ct)
    },

  /// A moving-average filter of white noise: the pushed innovations `σ z` convolved with the
  /// impulse response the host tabulates, so any ARMA recursion is exact. The first point is drawn.
  109 => MovingAverageFilter { sigma }
    state (x)
    noise (dz)
    step { $cv }
    report { x }
    history { push (sigma * dz / sqrt($dt)) weights ($ct) },

  /// A tempered-stable process by its shot-noise series (Rosiński 2007): each term is sized
  /// `min((Γ_j rate)^{-1/α}, e_scale E^{e_pow} U^{1/α} / λ_side)`; `rem_sd dz`: the dropped terms.
  110 => TemperedStableSeries { b_t, rate, inv_alpha, e_scale, e_pow, w_plus, lambda_plus, lambda_minus, rem_sd }
    state (x)
    noise (dz)
    step { x + b_t * $dt + rem_sd * dz + $sj }
    report { x }
    series {
      size (
        bind up = less($uv, w_plus);
        bind side = pick(up, lambda_plus, lambda_minus);
        bind cap = pow($gj * rate, negate(inv_alpha));
        bind draw = e_scale * pow($ej, e_pow) * pow($uj, inv_alpha) / side;
        bind mag = min(cap, draw);
        pick(up, mag, negate(mag))
      )
    },

  /// A Gaussian Volterra process by the reference quadrature, `X_i = Σ_{k<i} K((i − k) dt) ΔW_k`:
  /// the pushed value is the increment and the weights in `ct` are `K((m + 1) dt)` by lag.
  111 => VolterraReference { }
    state (x)
    noise (dz)
    step { $cv }
    report { x }
    history { push (dz) weights ($ct) },

  /// A Hawkes process, one event per step, by the exact recursion of Dassios and Zhao (2013): the
  /// wait is the smaller of `Exp(mu)` and `−ln(d) / beta`, `d = 1 + beta ln(u) / s`, when `d > 0`.
  112 => HawkesEvents { mu, alpha, beta }
    state (t, s)
    noise (dz)
    step {
      bind se = max(s, lit(1.0e-12));
      bind d = beta * ln($u) / se + lit(1.0);
      bind pos = less(negate(d), lit(0.0));
      bind s1 = pick(pos, negate(ln(max(d, lit(1.0e-30)))) / beta, lit(1.0e30));
      bind s2 = negate(ln($u2)) / mu;
      bind st = min(s1, s2);
      t + st,
      s * exp(negate(beta * st)) + alpha
    }
    report { t },

  /// Up to four forward LIBOR rates under the spot measure, by log-Euler with the drift frozen at
  /// the step's start. `ct` is `η`, the resets passed: `n < η` stays, a live `n` sums `j = η..=n`.
  113 => LiborMarket4 { s0, s1, s2, s3, d0, d1, d2, d3, l00, l10, l11, l20, l21, l22, l30, l31, l32, l33 }
    state (f0, f1, f2, f3)
    noise (z0, z1, z2, z3)
    step {
      bind a0 = leq($ct, lit(0.0));
      bind a1 = leq($ct, lit(1.0));
      bind a2 = leq($ct, lit(2.0));
      bind a3 = leq($ct, lit(3.0));
      bind g0 = a0 * d0 * s0 * f0 / (d0 * f0 + lit(1.0));
      bind g1 = a1 * d1 * s1 * f1 / (d1 * f1 + lit(1.0));
      bind g2 = a2 * d2 * s2 * f2 / (d2 * f2 + lit(1.0));
      bind g3 = a3 * d3 * s3 * f3 / (d3 * f3 + lit(1.0));
      bind r00 = l00 * l00;
      bind r10 = l10 * l00;
      bind r11 = l10 * l10 + l11 * l11;
      bind r20 = l20 * l00;
      bind r21 = l20 * l10 + l21 * l11;
      bind r22 = l20 * l20 + l21 * l21 + l22 * l22;
      bind r30 = l30 * l00;
      bind r31 = l30 * l10 + l31 * l11;
      bind r32 = l30 * l20 + l31 * l21 + l32 * l22;
      bind r33 = l30 * l30 + l31 * l31 + l32 * l32 + l33 * l33;
      bind m0 = s0 * (r00 * g0);
      bind m1 = s1 * (r10 * g0 + r11 * g1);
      bind m2 = s2 * (r20 * g0 + r21 * g1 + r22 * g2);
      bind m3 = s3 * (r30 * g0 + r31 * g1 + r32 * g2 + r33 * g3);
      bind w0 = l00 * z0;
      bind w1 = l10 * z0 + l11 * z1;
      bind w2 = l20 * z0 + l21 * z1 + l22 * z2;
      bind w3 = l30 * z0 + l31 * z1 + l32 * z2 + l33 * z3;
      pick(leq($ct, lit(0.0)), f0 * exp((m0 - s0 * s0 / lit(2.0)) * $dt + s0 * w0), f0),
      pick(leq($ct, lit(1.0)), f1 * exp((m1 - s1 * s1 / lit(2.0)) * $dt + s1 * w1), f1),
      pick(leq($ct, lit(2.0)), f2 * exp((m2 - s2 * s2 / lit(2.0)) * $dt + s2 * w2), f2),
      pick(leq($ct, lit(3.0)), f3 * exp((m3 - s3 * s3 / lit(2.0)) * $dt + s3 * w3), f3)
    }
    report { f0, f1, f2, f3 },

  /// The inverse α-stable subordinator `E(t) = inf{u : D(u) > t}`: a table of Kanter–CMS increments
  /// of `D` in `u`, summed as logarithms as the subordinator is, inverted at each step's time.
  114 => InverseStableSubordinator { alpha, c, inv_alpha, one_minus_alpha, tail_exp, pi }
    state (x)
    noise (dz)
    step { $iv }
    report { x }
    table {
      increment (
        bind uu = $uj * pi;
        bind w = negate(ln($uv));
        bind lsin = ln(c * $tv) - ln(max(sin(uu), lit(1e-20)));
        bind ltail = ln(max(sin(one_minus_alpha * uu), lit(1e-20))) - ln(w);
        bind lg = inv_alpha * lsin + tail_exp * ltail;
        sin(alpha * uu) * exp(lg)
      )
    },

  /// CGMY under a CIR variance (Kim 2021): the variance takes the exact square-root step, and
  /// each series term is sized in its own step against the variance. Records `y + ρ v` and `v`.
  115 => StochasticVolatilityCgmy { rate0, inv_alpha, lambda_plus, lambda_minus, bcoef, twoc, ek, rho }
    state (y, v)
    noise (dz)
    step {
      bind zn = dz / sqrt($dt);
      bind root = zn + sqrt(twoc * v * ek);
      y + bcoef * v * $dt + $sj,
      (root * root + $gm) / twoc
    }
    report { y + rho * v, v }
    series {
      live size (
        bind up = less($uv, lit(0.5));
        bind side = pick(up, lambda_plus, lambda_minus);
        bind cap = pow($gj * rate0 / max(v, lit(1.0e-12)), negate(inv_alpha));
        bind draw = $ej * pow($uj, inv_alpha) / side;
        bind mag = min(cap, draw);
        pick(up, mag, negate(mag))
      )
    },
    }
  };
}
