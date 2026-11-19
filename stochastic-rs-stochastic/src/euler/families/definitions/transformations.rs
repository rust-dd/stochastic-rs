macro_rules! euler_transformations {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $u3:ident, $u4:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_lifts! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $u3, $u4, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// `σ²_t = ω + αX²_{t−1} + βσ²_{t−1}`, `X_t = σ_t z_t`: GARCH(1,1), ARCH(1) at `β = 0`. The
  /// series starts at `σ₀ z₀`; `warm` holds the variance at its seed until the first step is taken.
  70 => Garch { omega, alpha, beta }
    state (x, s2, warm)
    noise (dz)
    step {
      bind v = pick(warm, omega + alpha * x * x + beta * s2, s2);
      sqrt(max(v, lit(1e-12))) * (dz / sqrt($dt)),
      v,
      lit(1.0)
    }
    report { x, s2, warm },

  /// [`Garch`](Family::Garch) with a threshold term: `γX²_{t−1}` enters only after a negative
  /// return, the GJR asymmetry, also named the asymmetric GARCH.
  71 => ThresholdGarch { omega, alpha, gamma, beta }
    state (x, s2, warm)
    noise (dz)
    step {
      bind lev = pick(less(x, lit(0.0)), gamma * x * x, lit(0.0));
      bind v = pick(warm, omega + alpha * x * x + lev + beta * s2, s2);
      sqrt(max(v, lit(1e-12))) * (dz / sqrt($dt)),
      v,
      lit(1.0)
    }
    report { x, s2, warm },

  /// `ln σ²_t = ω + α(|z_{t−1}| − E|z|) + γ z_{t−1} + β ln σ²_{t−1}`, `X_t = σ_t z_t`: EGARCH(1,1).
  /// The step recovers `z_{t−1}` from the state; `E|z| = √(2/π)` is folded on the host.
  72 => ExponentialGarch { omega, alpha, gamma, beta, e_abs_z }
    state (x, ls2, warm)
    noise (dz)
    step {
      bind sd = sqrt(exp(ls2));
      bind zl = x / sd;
      bind shock = alpha * (abs(zl) - e_abs_z) + gamma * zl;
      bind v = pick(warm, omega + shock + beta * ls2, ls2);
      sqrt(exp(v)) * (dz / sqrt($dt)),
      v,
      lit(1.0)
    }
    report { x, ls2, warm },

  /// One draw per grid point, with no recursion: white noise takes a mean and a standard
  /// deviation, and Gaussian noise is the same family at zero mean and `√dt`.
  73 => Innovation { mean, sd }
    state (x)
    noise (dz)
    step { mean + sd * (dz / sqrt($dt)) }
    report { x },

  /// A correlated pair of innovations: the second is drawn independently and correlated in the
  /// step, as every two-factor model here does with its shocks.
  74 => CorrelatedInnovation { rho }
    state (a, b)
    noise (dw, dq)
    step {
      dw,
      rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq
    }
    report { a, b },

  /// `X_t = φX_{t−1} + σz_t`: a first-order autoregression.
  75 => Autoregressive { phi, sigma }
    state (x)
    noise (dz)
    step { phi * x + sigma * (dz / sqrt($dt)) }
    report { x },

  /// `X_t = σz_t + θσz_{t−1}`: a first-order moving average. The lagged innovation is state,
  /// since the step cannot see the previous draw any other way.
  76 => MovingAverage { theta, sigma }
    state (x, e)
    noise (dz)
    step {
      bind now = sigma * (dz / sqrt($dt));
      now + theta * e,
      now
    }
    report { x, e },

  /// A gamma subordinator: each increment is one Gamma draw.
  77 => GammaSubordinator { }
    state (x)
    noise (dz)
    step { x + $gm }
    report { x },

  /// Brownian motion under a gamma clock: the variance gamma process.
  78 => VarianceGamma { mu, sigma }
    state (x)
    noise (dz)
    step { x + mu * $gm + sigma * sqrt($gm) * (dz / sqrt($dt)) }
    report { x },

  /// The difference of two gamma processes: the bilateral gamma process.
  79 => BilateralGamma { }
    state (x)
    noise (dz)
    step { x + $gm - $gm2 }
    report { x },

  /// [`BilateralGamma`](Family::BilateralGamma) with a Brownian part.
  80 => BilateralGammaMotion { sigma }
    state (x)
    noise (dz)
    step { x + sigma * dz + $gm - $gm2 }
    report { x },

  /// A tempered-stable subordinator: the deterministic drift the small jumps
  /// below the truncation contribute, plus the step's own thinned jumps.
  81 => TemperedStableSubordinator { drift }
    state (x)
    noise (dz)
    step { x + drift + $js }
    report { x },

  /// `dσ² = −λσ² dt + dZ` with gamma compound-Poisson jumps under a log-Euler asset: BNS. The
  /// decay is exact, and the jump sum is one gamma draw of the jump count times a jump's shape.
  82 => BarndorffNielsenShephard { decay, mu }
    state (s, v)
    noise (dw)
    step {
      s * exp((mu - v / lit(2.0)) * $dt + sqrt(v) * dw),
      decay * v + $gm
    }
    report { s, v },

  /// Two fractional rows from one embedding, which a shared Hurst exponent allows: the second
  /// reads the buffer's next `paths` rows, and the step correlates them as `CorrelatedInnovation`.
  83 => CorrelatedFractionalMotion { rho }
    state (a, b)
    noise (dz1, dz2)
    step {
      a + dz1,
      b + rho * dz1 + sqrt(negate(rho * rho - lit(1.0))) * dz2
    }
    report { a, b },

  /// The complex fractional Ornstein-Uhlenbeck process in real and imaginary parts: the reversion
  /// `lambda - i·omega` on `x1 + i·x2`, with the noise intensity `scale = sqrt(a / 2)` host-folded.
  84 => ComplexFractionalOu { lambda, omega, scale }
    state (x1, x2)
    noise (dz1, dz2)
    step {
      x1 - (lambda * x1 + omega * x2) * $dt + scale * dz1,
      x2 - (lambda * x2 - omega * x1) * $dt + scale * dz2
    }
    report { x1, x2 },

  /// An Ornstein-Uhlenbeck process reported through a bounded map onto `(-1, 1)`: `tanh x` at
  /// `arctan = 0`, `(2/pi) arctan(pi x / 2)` at one, with `pi / 2` passed as `half_pi`.
  85 => TransformedOrnsteinUhlenbeck { kappa, mu, sigma, arctan, half_pi }
    state (x)
    noise (dz)
    step { x + kappa * (mu - x) * $dt + sigma * dz }
    report {
      pick(
        geq(arctan, lit(0.5)),
        atan(x * half_pi) * recip(half_pi),
        tanh(x)
      )
    },

  /// The arrival times of a Poisson process sampled to a fixed count: a running sum of exponential
  /// waits, each by inverse CDF from the step's own uniform.
  86 => PoissonArrivals { lambda }
    state (x)
    noise (dz)
    step { x + negate(ln($u)) * recip(lambda) }
    report { x },

  /// The dynamic SABR with its term structure as curves: `ct` is `beta(t)`, `ct1` is `rho(t)`,
  /// `ct2` is `nu(t)`. The volatility takes the exact step for `d(alpha) = nu alpha dW`.
  87 => DynamicSabr { }
    state (f, v)
    noise (dw1, dq)
    step {
      bind dw2 = $ct1 * dw1 + sqrt(negate($ct1 * $ct1 - lit(1.0))) * dq;
      positive(f + positive(v) * pow(positive(f), $ct) * dw1),
      positive(v) * exp($ct2 * dw2 - $ct2 * $ct2 * $dt * lit(0.5))
    }
    report { f, v },

  /// Heath-Jarrow-Morton in three rows, each with a drift and a diffusion curve: `ct`/`ct1` the
  /// short rate, `ct2`/`ct3` the bond price (outer scale folded in), `ct4`/`ct5` the forward rate.
  88 => HeathJarrowMorton { }
    state (r, p, f)
    noise (dr, dp, df)
    step {
      r + $ct * $dt + $ct1 * dr,
      p + $ct2 * $dt + $ct3 * dp,
      f + $ct4 * $dt + $ct5 * df
    }
    report { r, p, f },

  /// One factor of the affine-diffusion Gaussian model: level `ct = k(t)`, speed `ct1 = theta(t)`,
  /// reported as `phi(t) + b(t) x + c(t) x^2` from `ct2`, `ct3`, `ct4`. One launch per factor.
  89 => AffineDiffusionGaussian { sigma }
    state (x)
    noise (dz)
    step { x + ($ct - $ct1 * x) * $dt + sigma * dz }
    report { $ct2 + $ct3 * x + $ct4 * x * x },

  /// One Wu-Zhang pair: a square-root variance reverting to `alpha` at speed `beta`, and a rate
  /// whose diffusion is `lambda` times the root of the variance just stepped, so it is bound first.
  90 => WuZhang { alpha, beta, nu, lambda }
    state (f, v)
    noise (df, dv)
    step {
      bind vn = positive(positive(v) + beta * (alpha - positive(v)) * $dt + nu * sqrt(positive(v)) * dv);
      positive(f) + positive(f) * lambda * sqrt(vn) * df,
      vn
    }
    report { f, v },

  /// Up to four correlated geometric Brownian motions under one lower Cholesky factor, each by
  /// the exact log-normal step. A missing asset has zero drift and volatility and a unit diagonal.
  91 => CorrelatedGeometric4 { m0, m1, m2, m3, s0, s1, s2, s3, l00, l10, l11, l20, l21, l22, l30, l31, l32, l33 }
    state (a, b, c, d)
    noise (dz0, dz1, dz2, dz3)
    step {
      bind e0 = l00 * dz0;
      bind e1 = l10 * dz0 + l11 * dz1;
      bind e2 = l20 * dz0 + l21 * dz1 + l22 * dz2;
      bind e3 = l30 * dz0 + l31 * dz1 + l32 * dz2 + l33 * dz3;
      a * exp((m0 - s0 * s0 * lit(0.5)) * $dt + s0 * e0),
      b * exp((m1 - s1 * s1 * lit(0.5)) * $dt + s1 * e1),
      c * exp((m2 - s2 * s2 * lit(0.5)) * $dt + s2 * e2),
      d * exp((m3 - s3 * s3 * lit(0.5)) * $dt + s3 * e3)
    }
    report { a, b, c, d },

  /// Up to four correlated Gaussian noises under one lower Cholesky factor; the state is the
  /// combined increment. A missing stream gets a unit diagonal, and its row is ignored.
  92 => CorrelatedNoises4 { l00, l10, l11, l20, l21, l22, l30, l31, l32, l33 }
    state (a, b, c, d)
    noise (dz0, dz1, dz2, dz3)
    step {
      bind e0 = l00 * dz0;
      bind e1 = l10 * dz0 + l11 * dz1;
      bind e2 = l20 * dz0 + l21 * dz1 + l22 * dz2;
      bind e3 = l30 * dz0 + l31 * dz1 + l32 * dz2 + l33 * dz3;
      e0,
      e1,
      e2,
      e3
    }
    report { a, b, c, d },

  /// A log-normal spot under a Markov chain of up to four regimes (slot value `0..3`): `g0..g3`
  /// are volatilities, `c_r1..c_r3` row `r`'s cumulative thresholds from `exp(Q dt)`, padded by 1.
  93 => RegimeSwitching {
    mu, g0, g1, g2, g3, c01, c02, c03, c11, c12, c13, c21, c22, c23, c31, c32, c33
  }
    state (s, z)
    noise (dw)
    step {
      bind r1 = geq(z, lit(0.5));
      bind r2 = geq(z, lit(1.5));
      bind r3 = geq(z, lit(2.5));
      bind sig = pick(r3, g3, pick(r2, g2, pick(r1, g1, g0)));
      bind k1 = pick(r3, c31, pick(r2, c21, pick(r1, c11, c01)));
      bind k2 = pick(r3, c32, pick(r2, c22, pick(r1, c12, c02)));
      bind k3 = pick(r3, c33, pick(r2, c23, pick(r1, c13, c03)));
      s * exp((mu - sig * sig * lit(0.5)) * $dt + sig * dw),
      pick(leq($u, k1), lit(0.0), pick(leq($u, k2), lit(1.0), pick(leq($u, k3), lit(2.0), lit(3.0))))
    }
    report { s, z },
    }
  };
}
