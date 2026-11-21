macro_rules! euler_jump_schemes {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $u3:ident, $u4:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_transformations! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $u3, $u4, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// `d ln S = (μ − λκ − ½σ²) dt + σ dW + Σ Y_i` with lognormal jump sizes: Merton's jump
  /// diffusion in log-price. The jump sum is the kernel's, from the size law the process declares.
  58 => MertonJumpLog { drift_ln, sigma }
    state (x)
    noise (dw)
    step { x * exp(drift_ln + sigma * dw + $js) }
    report { x },

  /// Bates: a Heston variance under a log-price that jumps, the compensated drift folded on the
  /// host and the sizes one normal draw. The truncated variance is what the next step starts from.
  59 => BatesJump { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * $dt + sv * dw + $js),
      positive(vp + (alpha - beta * vp) * $dt + sigma * sv * dv)
    }
    report { s, v },

  /// [`BatesJump`](Family::BatesJump) with the variance reflected at zero.
  60 => BatesJumpReflected { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = abs(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * $dt + sv * dw + $js),
      abs(vp + (alpha - beta * vp) * $dt + sigma * sv * dv)
    }
    report { s, v },

  /// Andersen's quadratic-exponential Heston step: the variance is drawn from a moment-matched
  /// law, the correlation lives in the host-folded `k` constants. Both branches run, one is picked.
  61 => AndersenQe { theta, e_kd, c1, c2, k0, k1, k2, k34, mu }
    state (y, v)
    noise (dzv, dz)
    step {
      bind omekd = negate(e_kd - lit(1.0));
      bind m = theta + (v - theta) * e_kd;
      bind s2 = v * c1 * omekd + c2 * omekd * omekd;
      bind psi = s2 / (m * m);
      bind inv = recip(psi) * lit(2.0);
      bind b2 = inv - lit(1.0) + sqrt(positive(inv * (inv - lit(1.0))));
      bind zv = dzv / sqrt($dt);
      bind quad = m / (b2 + lit(1.0)) * (sqrt(b2) + zv) * (sqrt(b2) + zv);
      bind p = (psi - lit(1.0)) / (psi + lit(1.0));
      bind tail = ln(negate(p - lit(1.0)) / negate($u - lit(1.0))) * m
        * recip(negate(p - lit(1.0)));
      bind vn = pick(leq(psi, lit(1.5)), quad, pick(leq($u, p), lit(0.0), tail));
      y + mu * $dt + k0 + k1 * v + k2 * vn + sqrt(positive(k34 * (v + vn))) * (dz / sqrt($dt)),
      vn
    }
    report { exp(y), v },

  /// A Poisson counting process on the grid: the step adds the jumps it saw.
  62 => CountingProcess { }
    state (x)
    noise (dz)
    step { x + $nj }
    report { x },

  /// An inverse-Gaussian subordinator, one Michael-Schucany-Haas draw per increment. Its roots
  /// are a sum and a quotient: as a difference they cancel to zero or below in single precision.
  63 => InverseGaussianSubordinator { mu_ig, two_lam, four_mu_lam }
    state (x)
    noise (dz)
    step {
      bind w = (dz / sqrt($dt)) * (dz / sqrt($dt));
      bind rad = sqrt(four_mu_lam * w + mu_ig * mu_ig * w * w);
      bind big = mu_ig + mu_ig * mu_ig * w / two_lam + mu_ig / two_lam * rad;
      bind xr = mu_ig * mu_ig / big;
      x + pick(less($u, mu_ig / (mu_ig + xr)), xr, big)
    }
    report { x },

  /// Brownian motion under an inverse-Gaussian clock: the normal inverse Gaussian process. The
  /// clock's draw is `InverseGaussianSubordinator`'s; the second noise is the shock it scales.
  64 => NormalInverseGaussian { theta, sigma, mu_ig, two_lam, four_mu_lam }
    state (x)
    noise (dz, dq)
    step {
      bind w = (dz / sqrt($dt)) * (dz / sqrt($dt));
      bind rad = sqrt(four_mu_lam * w + mu_ig * mu_ig * w * w);
      bind big = mu_ig + mu_ig * mu_ig * w / two_lam + mu_ig / two_lam * rad;
      bind xr = mu_ig * mu_ig / big;
      bind ig = pick(less($u, mu_ig / (mu_ig + xr)), xr, big);
      x + theta * ig + sigma * sqrt(ig) * (dq / sqrt($dt))
    }
    report { x },

  /// A positive-stable subordinator by Chambers–Mallows–Stuck from the step's two uniforms, its two
  /// powers and scale summed as logarithms: each factor leaves `f32`'s range for small `α` first.
  65 => StableSubordinator { alpha, inv_alpha, one_minus_alpha, tail_exp, log_scale, pi }
    state (x)
    noise (dz)
    step {
      bind uu = $u * pi;
      bind w = negate(ln($u2));
      bind lsin = negate(ln(max(sin(uu), lit(1e-20))));
      bind ltail = ln(max(sin(one_minus_alpha * uu), lit(1e-20))) - ln(w);
      bind lg = log_scale + inv_alpha * lsin + tail_exp * ltail;
      x + sin(alpha * uu) * exp(lg)
    }
    report { x },

  /// A Heston variance under a log-price with Kou's double-exponential jumps: the sum has no
  /// closed form, so the kernel sums the sizes in a bounded loop. The variance is truncated.
  66 => KouJumpHeston { drift_c, kappa, theta, sigma_v, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * $dt + sv * dw + $js),
      positive(vp + kappa * (theta - vp) * $dt + sigma_v * sv * dv)
    }
    report { s, v },

  /// [`KouJumpHeston`](Family::KouJumpHeston) with the variance reflected.
  67 => KouJumpHestonReflected { drift_c, kappa, theta, sigma_v, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = abs(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * $dt + sv * dw + $js),
      abs(vp + kappa * (theta - vp) * $dt + sigma_v * sv * dv)
    }
    report { s, v },

  /// [`DuffieKan`](Family::DuffieKan) with a compound-Poisson jump on the second factor: a
  /// step sees a Poisson count with mean `λ·dt`, whose normal sizes aggregate in the kernel's sum.
  68 => DuffieKanJump {
    a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho
  }
    state (r, x)
    noise (dw, dq)
    step {
      bind dx = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      bind vol = alpha * r + beta * x + gamma;
      r + (a1 * r + b1 * x + c1) * $dt + sigma1 * vol * dw,
      x + (a2 * r + b2 * x + c2) * $dt + sigma2 * vol * dx + $js
    }
    report { r, x },

  /// The Hawkes jump diffusion: the intensity is a second state component the step excites and
  /// reverts, at most one jump per step as in the host's Bernoulli test. Only the path is reported.
  69 => HawkesJumpDiffusion {
    drift_c, sigma, alpha, beta, mu_lambda, jump_mu, jump_sigma
  }
    state (x, lam)
    noise (dw, dj)
    step {
      bind fired = less($u, lam * $dt);
      bind size = pick(fired, jump_mu + jump_sigma * (dj / sqrt($dt)), lit(0.0));
      bind excited = lam + pick(fired, alpha, lit(0.0));
      x + drift_c * $dt + sigma * dw + size,
      positive(excited + beta * (mu_lambda - excited) * $dt)
    }
    report { x, lam },
    }
  };
}
