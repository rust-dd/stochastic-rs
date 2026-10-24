macro_rules! euler_lifts {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_memory! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// Riemann-Liouville fractional Brownian motion under the Markov lift of
  /// its kernel: no drift, unit diffusion, driven by the step's own shock,
  /// and the state *is* the lifted value the frame hands back as `lv`. The
  /// same declaration serves any rough Volterra kernel whose nodes and
  /// weights the host supplies through `lift_spec()`.
  94 => RiemannLiouville { }
    state (x)
    noise (dz)
    step { $lv }
    report { x }
    lift { drift (lit(0.0)) diffusion (lit(1.0)) shock (dz) },

  /// A mean-reverting process driven by Riemann-Liouville fBm: the lifted
  /// fBm rides in the second slot and the state takes its increment,
  /// `x + kappa (mu - x) dt + nu (B_{i} - B_{i-1})`, exactly as the host
  /// feeds the lift's path into its own Euler loop.
  95 => RiemannLiouvilleOu { kappa, mu, nu }
    state (x, b)
    noise (dz)
    step {
      x + kappa * (mu - x) * $dt + nu * ($lv - b),
      $lv
    }
    report { x, b }
    lift { drift (lit(0.0)) diffusion (lit(1.0)) shock (dz) },

  /// Black-Scholes under Riemann-Liouville fBm in closed form: the spot is
  /// `s0 exp(r t - sigma^2 t^{2H} / 2 + sigma B_t)`, with the deterministic
  /// part tabulated as the curve `ct` at each grid point and the lifted fBm
  /// in the second slot.
  96 => RiemannLiouvilleBlackScholes { s0, sigma }
    state (s, b)
    noise (dz)
    step {
      s0 * exp($ct + sigma * $lv),
      $lv
    }
    report { s, b }
    lift { drift (lit(0.0)) diffusion (lit(1.0)) shock (dz) },

  /// The rough Heston model: a square-root variance whose mean reversion and
  /// diffusion enter through the Markov lift of a rough kernel, so the lifted
  /// value is the next variance, and a spot stepped by Euler under the
  /// truncated variance. The variance's shock is correlated with the spot's
  /// through `rho`; the state carries the raw lifted variance and reports it
  /// truncated, as the host does.
  97 => RiemannLiouvilleHeston { mu, kappa, theta, nu, rho }
    state (s, v)
    noise (ds, dq)
    step {
      s + mu * s * $dt + s * sqrt(positive(v)) * ds,
      $lv
    }
    report { s, positive(v) }
    lift {
      drift (kappa * (theta - positive(v)))
      diffusion (nu * sqrt(positive(v)))
      shock (rho * ds + sqrt(negate(rho * rho - lit(1.0))) * dq)
    },

  /// A diffusion with additive compound-Poisson jumps, `x + drift dt + sigma
  /// dW + J`, the jump sum `js` under whatever size law the host declares.
  /// Merton's and Kou's jump diffusions and the Lévy diffusion all step this,
  /// differing only in the drift they hand over per step — compensated for
  /// the jumps in the first two, the bare `gamma dt` in the third.
  98 => AdditiveJumpDiffusion { drift_dt, sigma }
    state (x)
    noise (dw)
    step { x + drift_dt + sigma * dw + $js }
    report { x },

  /// Bates (1996) by Euler in the spot rather than the log-spot: the drift
  /// arrives compensated, the jump multiplies the spot by `1 + js` with `js`
  /// under a product size law, `∏ (1 + y_j) − 1`, and the variance is
  /// truncated at zero.
  99 => Bates1996 { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s + drift_c * s * $dt + s * sv * dw + s * $js,
      positive(vp + (alpha - beta * vp) * $dt + sigma * sv * dv)
    }
    report { s, v },

  /// [`Bates1996`](Family::Bates1996) with the variance reflected at zero.
  100 => Bates1996Reflected { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = abs(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s + drift_c * s * $dt + s * sv * dw + s * $js,
      abs(vp + (alpha - beta * vp) * $dt + sigma * sv * dv)
    }
    report { s, v },

  /// An event-indexed compound Poisson path: every step is one arrival, so
  /// the first slot advances by an exponential waiting time, the third takes
  /// the arrival's single size — `js` under a one-per-step law — and the
  /// second accumulates it.
  101 => CompoundPoissonEvents { lambda }
    state (t, c, j)
    noise (dz)
    step {
      t + negate(ln(max($u, lit(1.0e-7)))) * recip(lambda),
      c + $js,
      $js
    }
    report { t, c, j },

  /// A fractional Ornstein-Uhlenbeck process with compound-Poisson jumps:
  /// the fractional increments come from the device's embedding and the jump
  /// sum from the step's own draw.
  102 => JumpFractionalOu { theta, mu, sigma }
    state (x)
    noise (dz)
    step { x + theta * (mu - x) * $dt + sigma * dz + $js }
    report { x },

  /// A square-root Volterra process under the Markov lift of its kernel:
  /// mean reversion and diffusion enter through the lift, the lifted value
  /// is the next state, and the state is reported truncated at zero as the
  /// host does — rough Heston's variance when the kernel is
  /// Riemann-Liouville. One difference of kind: the host's truncation lets a
  /// NaN through so a caller sees it, `positive` here maps it to zero.
  103 => VolterraSquareRoot { kappa, theta, nu }
    state (v)
    noise (dz)
    step { $lv }
    report { positive(v) }
    lift {
      drift (kappa * (theta - positive(v)))
      diffusion (nu * sqrt(positive(v)))
      shock (dz)
    },

  /// A polynomial of a Gaussian Volterra process: the state is the lifted
  /// Gaussian — no drift, unit diffusion — and the report evaluates the
  /// polynomial by Horner's rule over up to eight coefficients in rising
  /// order, the unused ones zero.
  104 => GaussianPolynomialVolatility { c0, c1, c2, c3, c4, c5, c6, c7 }
    state (x)
    noise (dz)
    step { $lv }
    report { c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * (c6 + x * c7)))))) }
    lift { drift (lit(0.0)) diffusion (lit(1.0)) shock (dz) },
    }
  };
}
