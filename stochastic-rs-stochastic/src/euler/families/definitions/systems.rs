macro_rules! euler_systems {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_jump_schemes! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// `dS = μS dt + S√V dW`, `dV = κ(θ − V) dt + σV^p dB` with `corr(W, B) = ρ`
  /// and the variance truncated at zero: the Heston model under its Euler
  /// scheme. The two noise components are drawn independently and correlated
  /// here, which is what the host sampler does with its own pair.
  35 => Heston { mu, kappa, theta, sigma, rho, pow_v }
    state (s, v)
    noise (dw, dz)
    step {
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      s + mu * s * $dt + s * sqrt(vp) * dw,
      positive(v + kappa * (theta - vp) * $dt + sigma * pow(vp, pow_v) * db)
    }
    report { s, v },

  /// [`Heston`](Family::Heston) with the variance reflected at zero instead
  /// of truncated.
  36 => HestonReflected { mu, kappa, theta, sigma, rho, pow_v }
    state (s, v)
    noise (dw, dz)
    step {
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      s + mu * s * $dt + s * sqrt(vp) * dw,
      abs(v + kappa * (theta - vp) * $dt + sigma * pow(vp, pow_v) * db)
    }
    report { s, v },

  /// `dF = α F^β dW`, `dα = ν α dB` with `corr(W, B) = ρ`: SABR, with the
  /// volatility stepped by its exact log-normal solution so it stays
  /// positive. `½ν²` is folded on the host.
  37 => Sabr { beta, rho, nu, half_nu_sq }
    state (f, v)
    noise (dw, dz)
    step {
      bind fp = positive(f);
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      f + vp * pow(fp, beta) * dw,
      vp * exp(nu * db - half_nu_sq * $dt)
    }
    report { f, v },

  /// `dS = rS dt + S√V dW`, `V_t = V₀ exp(ν Z_t − ½ν² t)` with `Z` the
  /// running sum of the correlated increments: the Bergomi variance is a
  /// function of that sum, so the sum and the elapsed time are stepped as
  /// state of their own rather than recomputed from the whole history.
  38 => Bergomi { r, nu, half_nu_sq, v0_sq, rho }
    state (s, v, z, elapsed)
    noise (dw, dq)
    step {
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s + r * s * $dt + sqrt(v) * s * dw,
      v0_sq * exp(nu * (z + db) - half_nu_sq * (elapsed + $dt)),
      z + db,
      elapsed + $dt
    }
    report { s, v, z, elapsed },

  /// Two Ornstein–Uhlenbeck factors on one clock, a slow one and a fast one:
  /// the Fouque–Papanicolaou–Sircar two-scale volatility driver.
  39 => TwoScaleOrnsteinUhlenbeck { kappa, theta, eps, alpha, eps_inv, sqrt_eps_inv }
    state (x, y)
    noise (dx, dy)
    step {
      x + kappa * (theta - x) * $dt + eps * dx,
      y + eps_inv * (alpha - y) * $dt + sqrt_eps_inv * dy
    }
    report { x, y },

  /// The Heston model stepped in log-price: `S` advances by the exponential
  /// of its log increment, so it stays positive whatever the variance does.
  /// The variance is truncated at zero and, unlike the arithmetic form, the
  /// truncated value is what the next step starts from.
  40 => LogHeston { drift, kappa, theta, xi, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dwv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift - vp / lit(2.0)) * $dt + sv * dw),
      positive(vp + kappa * (theta - vp) * $dt + xi * sv * dwv)
    }
    report { s, v },

  /// [`LogHeston`](Family::LogHeston) with the variance reflected at zero.
  41 => LogHestonReflected { drift, kappa, theta, xi, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = abs(v);
      bind sv = sqrt(vp);
      bind dwv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift - vp / lit(2.0)) * $dt + sv * dw),
      abs(vp + kappa * (theta - vp) * $dt + xi * sv * dwv)
    }
    report { s, v },

  /// Two independent square-root variance factors driving one spot, each with
  /// its own correlation to the spot's own shock: the double Heston model,
  /// with both variances truncated at zero.
  42 => DoubleHeston {
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
      positive(v1 + kappa1 * (theta1 - p1) * $dt + sigma1 * sqrt(p1) * dv1),
      positive(v2 + kappa2 * (theta2 - p2) * $dt + sigma2 * sqrt(p2) * dv2)
    }
    report { s, v1, v2 },

  /// [`DoubleHeston`](Family::DoubleHeston) with both variances reflected.
  43 => DoubleHestonReflected {
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
      abs(v1 + kappa1 * (theta1 - p1) * $dt + sigma1 * sqrt(p1) * dv1),
      abs(v2 + kappa2 * (theta2 - p2) * $dt + sigma2 * sqrt(p2) * dv2)
    }
    report { s, v1, v2 },

  /// A Heston spot whose correlation to its own variance is itself a
  /// mean-reverting process, stepped on the unbounded variable and reported
  /// through a `tanh`. The log increment reads the correlation *after* its
  /// own step, so the third component is computed once and used by both.
  44 => StochasticCorrelationHeston {
    kappa_r, mu_r, sigma_r, kappa_v, mu_v, sigma_v, r, rho2
  }
    state (s, v, x)
    noise (dv_w, drho, dx_w)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind xc = x + kappa_r * (mu_r - tanh(x)) * $dt + sigma_r * drho;
      bind rt = tanh(xc);
      bind indep = sqrt(positive(negate(rt * rt + rho2 * rho2 - lit(1.0))));
      s * exp(
        (r - vp / lit(2.0)) * $dt
          + rt * sv * dv_w
          + rho2 * sv * drho
          + indep * sv * dx_w
      ),
      positive(vp + kappa_v * (mu_v - vp) * $dt + sigma_v * sv * dv_w),
      xc
    }
    report { s, v, tanh(x) },

  /// `dr = (θ(t) − αr) dt + σ dW`: Hull-White, whose mean-reversion level is
  /// the time-varying coefficient the launch carries.
  45 => HullWhite { alpha, sigma }
    state (x)
    noise (dz)
    step { x + ($ct - alpha * x) * $dt + sigma * dz }
    report { x },

  /// `dr = θ(t) dt + σ dW`: a drift that is entirely the curve, which is Ho-Lee
  /// under either of its two drift forms.
  46 => CurveDrift { sigma }
    state (x)
    noise (dz)
    step { x + $ct * $dt + sigma * dz }
    report { x },

  /// The exact one-step Ornstein–Uhlenbeck transition in log space, reported
  /// exponentiated: Black-Karasinski. `decay` is `exp(−a·dt)` and the noise
  /// scale folds the exact transition standard deviation, both of which
  /// depend on `dt` alone.
  47 => LogMeanReverting { decay, a, sigma_eff }
    state (y)
    noise (dz)
    step { y * decay + ($ct / a) * negate(decay - lit(1.0)) + sigma_eff * dz }
    report { exp(y) },

  /// A square-root diffusion shifted by a deterministic curve, truncated at
  /// zero before the shift: CIR++.
  48 => ShiftedSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { positive(x + theta * (mu - x) * $dt + sigma * sqrt(abs(x)) * dz) }
    report { x + $ct },

  /// [`ShiftedSquareRoot`](Family::ShiftedSquareRoot) reflected at zero.
  49 => ShiftedSquareRootMirrored { theta, mu, sigma }
    state (x)
    noise (dz)
    step { abs(x + theta * (mu - x) * $dt + sigma * sqrt(abs(x)) * dz) }
    report { x + $ct },

  /// `dX = μX dt + σ(t)X dW`: geometric Brownian motion over a term
  /// structure of volatilities.
  50 => TimeVaryingGeometricBrownian { mu }
    state (x)
    noise (dz)
    step { x + mu * x * $dt + $ct * x * dz }
    report { x },

  /// Two Brownian motions correlated by `ρ`: the pair every two-factor model
  /// here draws its shocks from, as a process in its own right.
  51 => CorrelatedBrownian { rho }
    state (a, b)
    noise (dw, dq)
    step {
      a + dw,
      b + rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq
    }
    report { a, b },

  /// A Brownian bridge from `X₀` to `xt`, stepped by the exact conditional
  /// law rather than by Euler's own variance. The curve carries `1/(T − s)`,
  /// from which both the drift and the per-step variance ratio follow; at the
  /// last step that ratio is zero and the drift is the whole remaining gap,
  /// so the path lands on `xt` exactly rather than by a diffusion kick.
  52 => BrownianBridge { xt, sigma }
    state (x)
    noise (dz)
    step {
      x + (xt - x) * $ct * $dt + sigma * sqrt(positive(negate($dt * $ct - lit(1.0)))) * dz
    }
    report { x },

  /// The two-factor Hull-White model: a short rate pulled toward the curve
  /// and a second, zero-reverting factor added to its drift.
  53 => TwoFactorHullWhite { a, b, sigma1, sigma2, rho }
    state (x, $u)
    noise (dw, dq)
    step {
      bind du = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      x + ($ct + $u - a * x) * $dt + sigma1 * dw,
      $u - b * $u * $dt + sigma2 * du
    }
    report { x, $u },

  /// Two independent square-root factors whose sum, shifted by the curve, is
  /// the reported short rate: the two-factor CIR model. Each factor chooses
  /// reflection or truncation through a flag rather than a family of its own,
  /// since the pair would otherwise need four.
  54 => TwoFactorSquareRoot {
    theta1, mu1, sigma1, theta2, mu2, sigma2, sym1, sym2
  }
    state (a, b)
    noise (dw, dq)
    step {
      bind v1 = a + theta1 * (mu1 - a) * $dt + sigma1 * sqrt(abs(a)) * dw;
      bind v2 = b + theta2 * (mu2 - b) * $dt + sigma2 * sqrt(abs(b)) * dq;
      pick(sym1, abs(v1), positive(v1)),
      pick(sym2, abs(v2), positive(v2))
    }
    report { a + b + $ct, b },

  /// The Duffie-Kan two-factor affine model: both factors drift affinely in
  /// the pair and share one affine volatility.
  55 => DuffieKan {
    a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho
  }
    state (r, x)
    noise (dw, dq)
    step {
      bind dx = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      bind vol = alpha * r + beta * x + gamma;
      r + (a1 * r + b1 * x + c1) * $dt + sigma1 * vol * dw,
      x + (a2 * r + b2 * x + c2) * $dt + sigma2 * vol * dx
    }
    report { r, x },

  /// Two Heston assets under one 4×4 Cholesky factor: both log-prices and
  /// both variances step together, so every cross-correlation the factor
  /// encodes is present in one launch. The variances are truncated at zero.
  56 => TwoAssetHeston {
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
      positive(v1 + kappa1 * (theta1 - p1) * $dt + sigma1 * sqrt(p1) * dz1),
      x2 + (mu2 - p2 / lit(2.0)) * $dt + sqrt(p2) * dw2,
      positive(v2 + kappa2 * (theta2 - p2) * $dt + sigma2 * sqrt(p2) * dz2)
    }
    report { x1, v1, x2, v2 },

  /// [`TwoAssetHeston`](Family::TwoAssetHeston) with both variances
  /// reflected at zero.
  57 => TwoAssetHestonReflected {
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
      abs(v1 + kappa1 * (theta1 - p1) * $dt + sigma1 * sqrt(p1) * dz1),
      x2 + (mu2 - p2 / lit(2.0)) * $dt + sqrt(p2) * dw2,
      abs(v2 + kappa2 * (theta2 - p2) * $dt + sigma2 * sqrt(p2) * dz2)
    }
    report { x1, v1, x2, v2 },
    }
  };
}
