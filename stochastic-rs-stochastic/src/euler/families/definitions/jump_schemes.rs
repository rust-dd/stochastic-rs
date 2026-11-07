macro_rules! euler_jump_schemes {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_transformations! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// `d ln S = (μ − λκ − ½σ²) dt + σ dW + Σ Y_i`, the jump sizes lognormal:
  /// Merton's jump diffusion in log-price. The jump sum is the kernel's, from
  /// the size law the process declares.
  58 => MertonJumpLog { drift_ln, sigma }
    state (x)
    noise (dw)
    step { x * exp(drift_ln + sigma * dw + $js) }
    report { x },

  /// A Heston variance under a log-price that also jumps: the Bates
  /// stochastic-volatility jump model. The compensated drift is folded on the
  /// host, and the jump sizes aggregate into one normal draw as they do for
  /// [`MertonJumpLog`](Family::MertonJumpLog). The variance is truncated at
  /// zero and the truncated value is what the next step starts from.
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

  /// Andersen's quadratic-exponential Heston step: the variance is drawn from
  /// a moment-matched law rather than stepped, so the spot needs no correlated
  /// Brownian pair — the correlation lives in the `k` constants, all of which
  /// depend on the parameters and `dt` alone and so are folded on the host.
  ///
  /// Both branches are evaluated and one is selected, which is what a kernel
  /// can do. The quadratic branch's square root is guarded because it is
  /// evaluated on the exponential branch's side too, where its argument is
  /// negative; the selected value is never the guarded one.
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

  /// An inverse-Gaussian subordinator: each increment is one Michael-Schucany-
  /// Haas draw, which needs a standard normal and a uniform and no rejection,
  /// so it is one expression. `2λ` and `4μλ` depend on the parameters and `dt`
  /// alone and are folded on the host.
  ///
  /// The draw's two roots are a sum and a quotient rather than a difference:
  /// once `w` is large the subtracted terms agree to five digits, and in
  /// single precision their difference comes out zero or negative — where the
  /// law is strictly positive, and where the caller of this clock takes its
  /// square root. Multiplying by the conjugate leaves `μ²` over a sum of
  /// positives, and the large root is that same sum.
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

  /// Brownian motion subordinated by an inverse-Gaussian clock: the normal
  /// inverse Gaussian process. The clock's draw is the same one
  /// [`InverseGaussianSubordinator`](Family::InverseGaussianSubordinator)
  /// takes, and the second noise component is the Brownian shock it scales.
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

  /// A positive-stable subordinator by the Chambers-Mallows-Stuck transform:
  /// one uniform on `(0, π)` and one exponential, both from the step's own
  /// uniforms, with no rejection. The two exponents depend on `α` alone and
  /// are folded on the host, as are `π` and the scale — the latter as
  /// `ln((c·dt)^{1/α})`, since the scale itself underflows to zero for `α`
  /// below about 0.011 on a fine grid.
  ///
  /// The uniforms are clamped into the open interval at a bound `f32` can
  /// hold below one: at exactly one the angle is `π`, whose sine is a small
  /// *negative* in single precision, and the logarithm of that is a NaN. The
  /// sines are floored for the same reason the clamp exists.
  ///
  /// The two powers and the scale are summed as logarithms and raised once,
  /// rather than multiplied as three factors. Each factor leaves the
  /// representable range long before their product does — at `α = 0.01` the
  /// exponent is 100, so `sin(u)^{1/α}` underflows for `sin u < 8e-4` and the
  /// tail factor overflows for `w < 7e-4` — and the product then came back
  /// `+inf`, or NaN against an underflowed scale, where the factors would have
  /// cancelled. `sin(αu)` stays outside: it is O(1) over the clamped range and
  /// is the one factor that never needed a logarithm.
  65 => StableSubordinator { alpha, inv_alpha, one_minus_alpha, tail_exp, log_scale, pi }
    state (x)
    noise (dz)
    step {
      bind uu = min(max($u, lit(1e-7)), lit(0.9999999)) * pi;
      bind w = negate(ln(min(max($u2, lit(1e-7)), lit(0.9999999))));
      bind lsin = negate(ln(max(sin(uu), lit(1e-20))));
      bind ltail = ln(max(sin(one_minus_alpha * uu), lit(1e-20))) - ln(w);
      bind lg = log_scale + inv_alpha * lsin + tail_exp * ltail;
      x + sin(alpha * uu) * exp(lg)
    }
    report { x },

  /// A Heston variance under a log-price whose jumps are Kou's
  /// double-exponential: the sum has no closed form, so the kernel sums the
  /// sizes in a bounded loop and the step reads that sum. The variance is
  /// truncated at zero.
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

  /// [`DuffieKan`](Family::DuffieKan) with a compound-Poisson jump on the
  /// second factor. The host walks its jump times sequentially; the waiting
  /// time is memoryless, so the number of jumps a step sees is Poisson with
  /// mean `λ·dt` and their normal sizes aggregate into the kernel's own sum.
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

  /// A jump diffusion whose jump intensity is itself excited by its jumps:
  /// the Hawkes jump diffusion. At most one jump per step, as the host's own
  /// Bernoulli test takes it, so the intensity is a state component the step
  /// excites and then mean-reverts. The process reports the path alone; the
  /// intensity is the second component it carries.
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
