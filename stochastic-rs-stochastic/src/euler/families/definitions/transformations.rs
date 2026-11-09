macro_rules! euler_transformations {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_lifts! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// `σ²_t = ω + αX²_{t−1} + βσ²_{t−1}`, `X_t = σ_t z_t`: GARCH(1,1), and at
  /// `β = 0` ARCH(1). The series starts at `σ₀ z₀` with `σ₀²` the
  /// unconditional variance, so the launch steps before writing its first
  /// point and the third component marks whether that first step has been
  /// taken — until it has, the variance stays at the level the host seeds it
  /// with rather than running the recursion on a state that does not exist
  /// yet.
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

  /// [`Garch`](Family::Garch) with a threshold term: `γX²_{t−1}` enters only
  /// when the previous return was negative, which is the GJR asymmetry and,
  /// under the other name its author gave it, the asymmetric GARCH.
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

  /// `ln σ²_t = ω + α(|z_{t−1}| − E|z|) + γ z_{t−1} + β ln σ²_{t−1}`,
  /// `X_t = σ_t z_t`: EGARCH(1,1). The lagged standardised residual is the
  /// previous return over the previous standard deviation, both of which the
  /// state carries, so the step recovers it rather than keeping a third
  /// series. `E|z| = √(2/π)` is folded on the host.
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

  /// One draw per grid point: the innovations themselves, with no recursion
  /// over them. White noise takes a mean and a standard deviation; Gaussian
  /// noise is the same family at zero mean and `√dt`.
  73 => Innovation { mean, sd }
    state (x)
    noise (dz)
    step { mean + sd * (dz / sqrt($dt)) }
    report { x },

  /// A correlated pair of innovations: the second is drawn independently and
  /// correlated in the step, which is what every two-factor model here does
  /// with its own shocks.
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

  /// `X_t = σz_t + θσz_{t−1}`: a first-order moving average. The lagged
  /// innovation is state of its own, since the step cannot see the previous
  /// draw any other way.
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

  /// `dσ² = −λσ² dt + dZ` with `Z` a compound-Poisson subordinator of gamma
  /// jumps, and a log-Euler asset over it: the Barndorff-Nielsen-Shephard
  /// model. The variance step is exact in the decay, as the host takes it,
  /// and the jump sum is one gamma draw whose shape is the step's own jump
  /// count times a single jump's.
  82 => BarndorffNielsenShephard { decay, mu }
    state (s, v)
    noise (dw)
    step {
      s * exp((mu - v / lit(2.0)) * $dt + sqrt(v) * dw),
      decay * v + $gm
    }
    report { s, v },

  /// Two fractional rows out of one embedding: the first reads the increment
  /// buffer's leading `paths` rows, the second the next block, and the step
  /// correlates them exactly as `CorrelatedInnovation` correlates two
  /// Brownian shocks. Both rows share a Hurst exponent, which is what lets a
  /// single embedding feed the pair.
  83 => CorrelatedFractionalMotion { rho }
    state (a, b)
    noise (dz1, dz2)
    step {
      a + dz1,
      b + rho * dz1 + sqrt(negate(rho * rho - lit(1.0))) * dz2
    }
    report { a, b },

  /// The complex fractional Ornstein-Uhlenbeck process in its real and
  /// imaginary parts: one complex mean reversion `lambda - i·omega` acting on
  /// `x1 + i·x2`, driven by a complex fractional noise whose two parts are
  /// the pair of streams the one embedding produces. `scale` is the noise
  /// intensity `sqrt(a / 2)`, folded in on the host so the step carries no
  /// square root of its own.
  84 => ComplexFractionalOu { lambda, omega, scale }
    state (x1, x2)
    noise (dz1, dz2)
    step {
      x1 - (lambda * x1 + omega * x2) * $dt + scale * dz1,
      x2 - (lambda * x2 - omega * x1) * $dt + scale * dz2
    }
    report { x1, x2 },

  /// An Ornstein-Uhlenbeck process reported through a bounded map onto
  /// `(-1, 1)`, which is how a stochastic correlation is built from an
  /// unbounded state. `arctan` selects the map: at zero it is `tanh x`, and
  /// at one the shallower `(2/pi) arctan(pi x / 2)`, whose `pi / 2` arrives
  /// as `half_pi` rather than as a literal the kernel would carry at the
  /// wrong precision — `2 / pi` is its reciprocal. Both branches are
  /// evaluated and one is picked, since neither can fault.
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

  /// The arrival times of a Poisson process sampled to a fixed count: a
  /// running sum of exponential inter-arrival times, each drawn by inverse
  /// CDF from the step's own uniform. The uniform is floored before the
  /// logarithm because the hash stream can land on exactly zero, which is a
  /// clamp of probability `1e-7` rather than a change of law.
  86 => PoissonArrivals { lambda }
    state (x)
    noise (dz)
    step { x + negate(ln(max($u, lit(1.0e-7)))) * recip(lambda) }
    report { x },

  /// The dynamic SABR: a forward rate under a backbone exponent and a
  /// log-normal volatility, with all three coefficients time-varying. The
  /// term structure arrives as curves rather than parameters — `ct` is
  /// `beta(t)`, `ct1` is `rho(t)` and `ct2` is `nu(t)` — which is what turns
  /// the host's per-step bucket search into a buffer read. The volatility
  /// takes the exact step for `d(alpha) = nu alpha dW`, not an Euler one.
  87 => DynamicSabr { }
    state (f, v)
    noise (dw1, dq)
    step {
      bind dw2 = $ct1 * dw1 + sqrt(negate($ct1 * $ct1 - lit(1.0))) * dq;
      positive(f + positive(v) * pow(positive(f), $ct) * dw1),
      positive(v) * exp($ct2 * dw2 - $ct2 * $ct2 * $dt * lit(0.5))
    }
    report { f, v },

  /// Heath-Jarrow-Morton in its three-row discretisation: the short rate,
  /// the bond price and the forward rate each take a drift and a diffusion
  /// coefficient that are functions of time alone, so all six travel as
  /// curves — `ct`/`ct1` for the rate, `ct2`/`ct3` for the bond price with
  /// its outer scale already folded into both, `ct4`/`ct5` for the forward
  /// rate — and the family carries no parameter at all.
  88 => HeathJarrowMorton { }
    state (r, p, f)
    noise (dr, dp, df)
    step {
      r + $ct * $dt + $ct1 * dr,
      p + $ct2 * $dt + $ct3 * dp,
      f + $ct4 * $dt + $ct5 * df
    }
    report { r, p, f },

  /// One factor of the affine-diffusion Gaussian short-rate model: a
  /// mean-reverting state under a time-varying level `ct = k(t)` and speed
  /// `ct1 = theta(t)`, reported through the quadratic observation
  /// `phi(t) + b(t) x + c(t) x^2` whose three coefficients are `ct2`, `ct3`
  /// and `ct4`. Each factor carries its own diffusion scale, and a model of
  /// several is that many independent launches of this family.
  89 => AffineDiffusionGaussian { sigma }
    state (x)
    noise (dz)
    step { x + ($ct - $ct1 * x) * $dt + sigma * dz }
    report { $ct2 + $ct3 * x + $ct4 * x * x },

  /// One forward-rate / volatility pair of the Wu-Zhang model: a square-root
  /// variance mean-reverting to `alpha` at speed `beta` with vol-of-vol `nu`,
  /// and a forward rate whose diffusion is `lambda` times the square root of
  /// the variance *just stepped*, which is why the variance is bound first.
  /// Both are floored at zero as the host does, the rate before its step and
  /// the variance before and after. A model of several pairs is that many
  /// independent launches of this family.
  90 => WuZhang { alpha, beta, nu, lambda }
    state (f, v)
    noise (df, dv)
    step {
      bind vn = positive(positive(v) + beta * (alpha - positive(v)) * $dt + nu * sqrt(positive(v)) * dv);
      positive(f) + positive(f) * lambda * sqrt(vn) * df,
      vn
    }
    report { f, v },

  /// Up to four correlated geometric Brownian motions under one lower
  /// Cholesky factor: the four shocks are combined through `L` and each asset
  /// takes the exact log-normal step in its own combined increment. A model
  /// with fewer assets pads the rest with zero drift, zero volatility and a
  /// unit diagonal, which leaves them constant at their (zero) start.
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

  /// Up to four correlated Gaussian noises under one lower Cholesky factor:
  /// the four-stream generalisation of `CorrelatedInnovation`, whose state
  /// is the combined increment itself. A model with fewer streams pads the
  /// factor with a unit diagonal and ignores the rows it did not ask for.
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

  /// A log-normal spot whose volatility is set by a Markov chain of up to
  /// four regimes. The regime rides in a state slot as `0..3`; `g0..g3` are
  /// the regimes' volatilities and `c_r1..c_r3` the cumulative one-step
  /// transition thresholds of row `r`, tabulated on the host from `exp(Q dt)`.
  /// The spot steps under the regime it is in, then the regime draws its
  /// successor from that row by inverse CDF on the step's uniform — the host's
  /// own order. A chain with fewer regimes pads the rest with thresholds of
  /// one, which no draw reaches.
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
