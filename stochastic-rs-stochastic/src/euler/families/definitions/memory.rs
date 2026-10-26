macro_rules! euler_memory {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_multifactor! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// The rough Heston model in the two-term rational approximation of its
  /// kernel — a mean-reverting factor `y`, a local factor `z` and the memory
  /// integral of `z` under `(t − s)^{H − 1/2}` — with the integral kept as
  /// the history block's exact convolution of the pushed `z` against the
  /// kernel weights the host tabulates as `ct`, so the device runs the host's
  /// own O(n²) scheme. The variance reported is
  /// `y + c1 ν z + c2 ν ∫ / Γ(H − 1/2)`; the spot steps in the log under the
  /// truncated previous variance, its shock correlated with the factor's
  /// through `rho`.
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

  /// The fractional Bates model:
  /// [`RoughHestonMemory`](Family::RoughHestonMemory) with unit calibration
  /// coefficients, `xi` for the vol-of-vol, a jump-compensated drift and
  /// normal jumps in the log-spot summed by the step's `js`.
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

  /// Rough Bergomi under the hybrid scheme: the Volterra driver of the
  /// log-variance is the history block's convolution of the correlated shock
  /// with the scheme's weights — the last interval's exact kernel integral
  /// first, then the kernel at the optimal points — plus an independent
  /// residual of standard-normal size (`de / √dt`, since `residual_sd`
  /// already carries the interval's `dt^{H}`), and the variance is
  /// `v0² exp(ν √(2H) X − ½ ν² t^{2H})` with the deterministic exponent
  /// tabulated as `ct1`. The spot steps by Euler under the previous variance.
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

  /// Linear fractional stable motion: a moving average of α-stable
  /// innovations under the weights `dt^d ((k+1)^d − k^d)`, `d = H − 1/α`,
  /// which the host tabulates as `ct`. The innovation is the
  /// Chambers–Mallows–Stuck draw from the step's two uniforms — the angle
  /// `π (u − ½)`, the exponential `−ln u2` — scaled by the host-folded
  /// `S · scale · dt^{1/α}`, `cos` written as a shifted `sin`, and the
  /// uniforms clamped as the stable subordinator clamps them.
  108 => LinearFractionalStable { alpha, inv_alpha, tail_exp, b, scale_s, pi, half_pi }
    state (x)
    noise (dz)
    step { x + $cv }
    report { x }
    history {
      push (
        bind uu = min(max($u, lit(1e-7)), lit(0.9999999));
        bind ee = min(max($u2, lit(1e-7)), lit(0.9999999));
        bind va = pi * (uu - lit(0.5));
        bind w = negate(ln(ee));
        bind phi = alpha * (va + b);
        bind cs = max(sin(va + half_pi), lit(1e-20));
        bind ratio = max(sin(va - phi + half_pi) / w, lit(1e-30));
        scale_s * sin(phi) / pow(cs, inv_alpha) * pow(ratio, tail_exp)
      )
      weights ($ct)
    },

  /// A moving-average filter of white noise: the state is the convolution of
  /// the pushed innovations `σ z` with the weight curve, which the host fills
  /// with the impulse response of whatever linear recursion it declares — an
  /// ARMA with its differencing, seasonal or not — so the device runs that
  /// recursion exactly, at any order. The first point is itself a draw.
  109 => MovingAverageFilter { sigma }
    state (x)
    noise (dz)
    step { $cv }
    report { x }
    history { push (sigma * dz / sqrt($dt)) weights ($ct) },

  /// A tempered-stable process by its shot-noise series (Rosiński 2007):
  /// the launch draws `J` terms per path before the steps — the arrival
  /// `Γ_j` of a unit-rate Poisson process, an `Exp(1)`, a uniform, the side
  /// by a uniform against `w_plus`, and a uniform arrival time — sizes each
  /// as `min((Γ_j rate)^{-1/α}, e_scale E^{e_pow} U^{1/α} / λ_side)` on the
  /// side's sign and sums it into the grid cell its time falls in; the step
  /// then adds the cell's jumps and the compensating drift `b_t dt`. CGMY,
  /// the classical and the rapidly decreasing tempered stable laws and KoBoL
  /// differ only in the folded `rate`, the exponential's scale and power and
  /// the side probability.
  110 => TemperedStableSeries { b_t, rate, inv_alpha, e_scale, e_pow, w_plus, lambda_plus, lambda_minus }
    state (x)
    noise (dz)
    step { x + b_t * $dt + $sj }
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

  /// A Gaussian Volterra process by the reference quadrature: the path is the
  /// convolution of the Brownian increments with the kernel on the grid,
  /// `X_i = Σ_{k<i} K((i − k) dt) ΔW_k`, so the pushed value is the increment
  /// itself and the weights the host tabulates as `ct` are `K((m + 1) dt)` by
  /// lag. Any kernel the host can evaluate rides it; the rough ones take the
  /// Markov lift instead.
  111 => VolterraReference { }
    state (x)
    noise (dz)
    step { $cv }
    report { x }
    history { push (dz) weights ($ct) },

  /// A Hawkes process with exponential excitation, one event per step, by the
  /// exact recursion of Dassios and Zhao (2013): with `s` the excess intensity
  /// just after the last event, the wait to the next is the smaller of an
  /// `Exp(mu)` from the baseline and the excitation's own arrival, which is
  /// `−ln(d) / beta` for `d = 1 + beta ln(u) / s` when that is positive and
  /// never otherwise; the excess then decays over the wait and jumps by
  /// `alpha`. The state keeps the time and the excess, the path records the
  /// times.
  112 => HawkesEvents { mu, alpha, beta }
    state (t, s)
    noise (dz)
    step {
      bind se = max(s, lit(1.0e-12));
      bind d = beta * ln(max($u, lit(1.0e-7))) / se + lit(1.0);
      bind pos = less(negate(d), lit(0.0));
      bind s1 = pick(pos, negate(ln(max(d, lit(1.0e-30)))) / beta, lit(1.0e30));
      bind s2 = negate(ln(max($u2, lit(1.0e-7)))) / mu;
      bind st = min(s1, s2);
      t + st,
      s * exp(negate(beta * st)) + alpha
    }
    report { t },

  /// Up to four forward LIBOR rates under the spot measure, by the log-Euler
  /// step with the drift frozen at the step's start. The curve `ct` is the
  /// number of reset dates passed at the step's start, `η`: a rate whose reset
  /// date has passed (`n < η`) stays where it is, and a live rate `n` takes
  /// `σ_n Σ_{j=max(η−1, 0)}^{n} ρ_nj δ_j σ_j L_j / (1 + δ_j L_j)` from the rates
  /// between the last reset and itself, `ρ_nj` from the lower Cholesky factor
  /// `L` whose rows also correlate the four shocks. An absent rate travels
  /// with a zero volatility and an identity row and never moves nor enters a
  /// drift.
  113 => LiborMarket4 { s0, s1, s2, s3, d0, d1, d2, d3, l00, l10, l11, l20, l21, l22, l30, l31, l32, l33 }
    state (f0, f1, f2, f3)
    noise (z0, z1, z2, z3)
    step {
      bind ei = max($ct - lit(1.0), lit(0.0));
      bind a0 = leq(ei, lit(0.0));
      bind a1 = leq(ei, lit(1.0));
      bind a2 = leq(ei, lit(2.0));
      bind a3 = leq(ei, lit(3.0));
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

  /// The inverse of an α-stable subordinator, `E(t) = inf{u : D(u) > t}`: the
  /// table block builds `D` on a grid in `u` from positive-stable increments —
  /// Kanter's form of the Chambers–Mallows–Stuck draw, its uniforms clamped
  /// away from the ends of the unit interval, at scale `(c Δu)^{1/α}` — and the
  /// step takes the interpolated first passage over the horizon's time. The
  /// state is that inverse; there is nothing to step.
  ///
  /// The increment sums its powers and its scale as logarithms and raises them
  /// once, for the reason
  /// [`StableSubordinator`](Family::StableSubordinator) does: the factors
  /// leave the representable range for small `α` well before their product
  /// does, and `(c Δu)^{1/α}` underflows to zero before any of them.
  114 => InverseStableSubordinator { alpha, c, inv_alpha, one_minus_alpha, tail_exp, pi }
    state (x)
    noise (dz)
    step { $iv }
    report { x }
    table {
      increment (
        bind uu = min(max($uj, lit(1e-7)), lit(0.9999999)) * pi;
        bind w = negate(ln(min(max($uv, lit(1e-7)), lit(0.9999999))));
        bind lsin = ln(c * $tv) - ln(max(sin(uu), lit(1e-20)));
        bind ltail = ln(max(sin(one_minus_alpha * uu), lit(1e-20))) - ln(w);
        bind lg = inv_alpha * lsin + tail_exp * ltail;
        sin(alpha * uu) * exp(lg)
      )
    },

  /// CGMY under a CIR stochastic volatility (Kim 2021): the variance takes
  /// the exact square-root step — a non-central χ² as the square of a shifted
  /// normal plus a central χ²(df − 1), which is the frame's gamma draw `gm` —
  /// and the log-price is a tempered-stable series whose terms are sized in
  /// their own step, each arrival's bound scaled by the variance at its time,
  /// plus the compensating drift `bcoef · v · dt`. The state keeps the jump
  /// part `y` and the variance; the path records `y + ρ v` and `v`.
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
