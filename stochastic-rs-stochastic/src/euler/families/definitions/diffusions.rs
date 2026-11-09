macro_rules! euler_diffusions {
  ($emit:ident; ($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident); $($previous:tt)*) => {
    euler_systems! {
      $emit;
      ($params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      $($previous)*
  /// `dX = μX dt + σX dW`.
  0 => GeometricBrownian { mu, sigma }
    state (x)
    noise (dz)
    step { x + mu * x * $dt + sigma * x * dz }
    report { x },

  /// `dX = θ(μ − X) dt + σ dW`.
  1 => OrnsteinUhlenbeck { theta, mu, sigma }
    state (x)
    noise (dz)
    step { x + theta * (mu - x) * $dt + sigma * dz }
    report { x },

  2 => SquareRoot { kappa, theta, sigma }
    state (x)
    noise (dz)
    step { x + kappa * (theta - positive(x)) * $dt + sigma * sqrt(positive(x)) * dz }
    report { positive(x) },

  /// `dX = κ(θ − X) dt + σ√X dW`, stepped with full truncation (Lord,
  /// Koekkoek & van Dijk 2010): the recursion runs on an auxiliary state
  /// whose positive part enters drift, diffusion and the reported path.
  /// `dX = dW`: the increment accumulates, which is fractional Brownian
  /// motion when the increments come from an fGN pipeline.
  3 => Additive { }
    state (x)
    noise (dz)
    step { x + dz }
    report { x },

  /// `dX = θ(μ − X) dt + σ√|X| dW`, clamped at zero after the step — the
  /// fractional CIR recursion, which truncates the *result* rather than
  /// stepping on a truncated state.
  4 => ReflectedSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { positive(x + theta * (mu - x) * $dt + sigma * sqrt(abs(x)) * dz) }
    report { x },

  /// The same with the symmetric reflection: the step's absolute value.
  5 => MirroredSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { abs(x + theta * (mu - x) * $dt + sigma * sqrt(abs(x)) * dz) }
    report { x },

  /// `dX = (α − βX) dt + σ√(X(1−X)) dW` on the unit interval, absorbing at
  /// both ends — the fractional Jacobi recursion. `X(1−X)` is written
  /// `x - x * x`, which needs no literal and so no type to infer, and both
  /// arms of a `pick` are evaluated, so the diffusion guards its own root.
  6 => Jacobi { alpha, beta, sigma }
    state (x)
    noise (dz)
    step {
      pick(
        leq(x, lit(0.0)),
        lit(0.0),
        pick(
          geq(x, lit(1.0)),
          lit(1.0),
          x + (alpha - beta * x) * $dt + sigma * sqrt(positive(x - x * x)) * dz
        )
      )
    }
    report { x },

  /// `dX = μX dt + σ|X|^γ dW`: constant elasticity of variance, the power
  /// taken off `|X|` so a negative excursion stays defined.
  7 => ConstantElasticity { mu, sigma, gamma }
    state (x)
    noise (dz)
    step { x + mu * x * $dt + sigma * pow(abs(x), gamma) * dz }
    report { x },

  /// `dX = (θ₁ + θ₂X) dt + θ₃|X|^θ₄ dW`: the Chan–Karolyi–Longstaff–Sanders
  /// family, whose four parameters nest most one-factor short-rate models.
  8 => Ckls { theta1, theta2, theta3, theta4 }
    state (x)
    noise (dz)
    step { x + (theta1 + theta2 * x) * $dt + theta3 * pow(abs(x), theta4) * dz }
    report { x },

  /// `dX = X(1 − aX) dt + bX dW`: logistic growth with multiplicative noise.
  /// `X(1 − aX)` is written `x - a * x * x`, which needs no literal.
  9 => Logistic { a, b }
    state (x)
    noise (dz)
    step { x + (x - a * x * x) * $dt + b * x * dz }
    report { x },

  /// `dX = κX(μ − X) dt + σ|X|^{3/2} dW`: the 3/2 model, whose variance
  /// mean-reverts faster than a square-root diffusion.
  10 => ThreeHalf { kappa, mu, sigma }
    state (x)
    noise (dz)
    step { x + kappa * x * (mu - x) * $dt + sigma * pow(abs(x), lit(1.5)) * dz }
    report { x },

  /// Geometric Brownian motion stepped in logs: `X ← X·exp(m + σ dW)`, with
  /// `m = (μ − σ²/2)Δt` computed once on the host, so the kernel needs no
  /// literal and the exponential is exact rather than a first-order step.
  11 => LogGeometric { drift_ln, sigma }
    state (x)
    noise (dz)
    step { x * exp(drift_ln + sigma * dz) }
    report { x },

  /// `dX = (κ/X − X) dt + σ dW`: the radial Ornstein–Uhlenbeck process, whose
  /// drift is guarded away from the origin exactly as the host sampler guards
  /// it.
  12 => RadialOrnsteinUhlenbeck { kappa, sigma }
    state (x)
    noise (dz)
    step {
      x + (kappa / pick(leq(abs(x), lit(1e-12)), lit(1e-12), x) - x) * $dt + sigma * dz
    }
    report { x },

  /// `dX = (a + bX) dt + cX dW`: the linear scalar SDE.
  13 => LinearSde { a, b, c }
    state (x)
    noise (dz)
    step { x + (a + b * x) * $dt + c * x * dz }
    report { x },

  /// `dX = −κX/√(1+X²) dt + σ dW`: a hyperbolic drift, bounded in `X`.
  14 => Hyperbolic { kappa, sigma }
    state (x)
    noise (dz)
    step { x - kappa * x / sqrt(x * x + lit(1.0)) * $dt + sigma * dz }
    report { x },

  /// `dX = −κX dt + σ√(1+X²) dW`: the modified CIR process, whose diffusion
  /// never vanishes.
  15 => ModifiedSquareRoot { kappa, sigma }
    state (x)
    noise (dz)
    step { x - kappa * x * $dt + sigma * sqrt(x * x + lit(1.0)) * dz }
    report { x },

  /// `dX = X(θ₁ − X(θ₃³ − θ₁θ₂)) dt + θ₃|X|^{3/2} dW`: the Feller root
  /// process, with the drift's constant folded on the host.
  16 => FellerRoot { theta1, decay, theta3 }
    state (x)
    noise (dz)
    step { x + x * (theta1 - x * decay) * $dt + theta3 * pow(abs(x), lit(1.5)) * dz }
    report { x },

  /// `dX = (a₋₁/X + a₀ + a₁X + a₂X²) dt + √|b₀ + b₁X + b₂|X|^{b₃}| dW`: the
  /// Aït-Sahalia short-rate model, whose drift is guarded away from the origin
  /// exactly as the host sampler guards it.
  17 => AitSahalia { am1, a0, a1, a2, b0, b1, b2, b3 }
    state (x)
    noise (dz)
    step {
      x + (am1 / pick(less(abs(x), lit(1e-12)), lit(1e-12), x)
        + a0 + a1 * x + a2 * x * x) * $dt
        + sqrt(abs(b0 + b1 * x + b2 * pow(abs(x), b3))) * dz
    }
    report { x },

  /// `dX = (a − b·ln X)X dt + σX dW`, floored at `1e-12`. The step's own
  /// floor makes the state positive from the first step on, and the process
  /// floors `X₀` the same way, so the guard the host applies to every
  /// coefficient is already true of `x` here.
  18 => Gompertz { a, b, sigma }
    state (x)
    noise (dz)
    step { max(x + (a - b * ln(x)) * x * $dt + sigma * x * dz, lit(1e-12)) }
    report { x },

  /// `dX = aX(1−X) dt + σ√(X(1−X)) dW` on `[0, 1]`: the Kimura diffusion of
  /// population genetics. As with [`Gompertz`](Family::Gompertz) the step's
  /// own clamp is what keeps the coefficients in range.
  19 => Kimura { a, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = min(max(x, lit(0.0)), lit(1.0));
      min(
        max(
          xi + a * xi * negate(xi - lit(1.0)) * $dt
            + sigma * sqrt(xi * negate(xi - lit(1.0))) * dz,
          lit(0.0)
        ),
        lit(1.0)
      )
    }
    report { x },

  /// `dX = (α + βX + γX²) dt + σX dW`: a quadratic drift with proportional
  /// noise.
  20 => Quadratic { alpha, beta, gamma, sigma }
    state (x)
    noise (dz)
    step { x + (alpha + beta * x + gamma * x * x) * $dt + sigma * x * dz }
    report { x },

  /// `dX = κ(μ − X) dt + √|2κ(aX² + bX + c)| dW`: the Pearson diffusion
  /// family. `2κ` is folded on the host so the step needs no literal.
  21 => Pearson { kappa, mu, a, b, c, two_kappa }
    state (x)
    noise (dz)
    step {
      x + kappa * (mu - x) * $dt + sqrt(abs(two_kappa * (a * x * x + b * x + c))) * dz
    }
    report { x },

  /// `dX = rX(1 − X/K) dt + σX dW`: logistic growth in its Verhulst
  /// parametrisation, run unclamped.
  22 => Verhulst { r, k, sigma }
    state (x)
    noise (dz)
    step { x + r * x * ((k - x) / k) * $dt + sigma * x * dz }
    report { x },

  /// [`Verhulst`](Family::Verhulst) with the state confined to `[0, K]`.
  23 => VerhulstClamped { r, k, sigma }
    state (x)
    noise (dz)
    step { min(max(x + r * x * ((k - x) / k) * $dt + sigma * x * dz, lit(0.0)), k) }
    report { x },

  /// `dX = κ(θ − X)X dt + σ√X dW`: Feller's logistic diffusion, truncated at
  /// zero.
  24 => FellerLogistic { kappa, theta, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = positive(x);
      positive(xi + kappa * (theta - xi) * xi * $dt + sigma * sqrt(xi) * dz)
    }
    report { x },

  /// [`FellerLogistic`](Family::FellerLogistic) reflected at zero instead of
  /// truncated.
  25 => FellerLogisticReflected { kappa, theta, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = positive(x);
      abs(xi + kappa * (theta - xi) * xi * $dt + sigma * sqrt(xi) * dz)
    }
    report { x },

  /// `dX = δ dt + 2√|X| dW`: the squared-Bessel recursion, truncated at zero.
  26 => SquaredBesselState { delta, two }
    state (x)
    noise (dz)
    step { positive(x + delta * $dt + two * sqrt(abs(x)) * dz) }
    report { x },

  /// [`SquaredBesselState`](Family::SquaredBesselState) reflected at zero.
  27 => SquaredBesselStateReflected { delta, two }
    state (x)
    noise (dz)
    step { abs(x + delta * $dt + two * sqrt(abs(x)) * dz) }
    report { x },

  /// [`SquaredBesselState`](Family::SquaredBesselState) reporting `√X`: the
  /// Bessel process itself, stepped in squared space so the `(δ−1)/2X`
  /// singularity never enters the recursion.
  28 => BesselFromSquared { delta, two }
    state (x)
    noise (dz)
    step { positive(x + delta * $dt + two * sqrt(abs(x)) * dz) }
    report { sqrt(x) },

  /// [`BesselFromSquared`](Family::BesselFromSquared) reflected at zero.
  29 => BesselFromSquaredReflected { delta, two }
    state (x)
    noise (dz)
    step { abs(x + delta * $dt + two * sqrt(abs(x)) * dz) }
    report { sqrt(x) },

  /// `dX = ½σ²(β − γ(X−μ)/√(δ² + (X−μ)²)) dt + σ dW`: the hyperbolic
  /// diffusion whose stationary law is the hyperbolic distribution. `½σ²` is
  /// folded on the host.
  30 => HyperbolicDiffusion { beta, gamma, delta, mu, sigma, half_var }
    state (x)
    noise (dz)
    step {
      x + half_var * (beta - gamma * x / sqrt(delta * delta + (x - mu) * (x - mu))) * $dt
        + sigma * dz
    }
    report { x },

  /// `dX = (a₋₁/X + a₀ + a₁X + a₂X²) dt + (b₀ + b₁X + b₂|X|^{b₃}) dW`: the
  /// Aït-Sahalia drift with the diffusion left unsquared, guarded away from
  /// the origin exactly as the host sampler guards it.
  31 => NonLinear { am1, a0, a1, a2, b0, b1, b2, b3 }
    state (x)
    noise (dz)
    step {
      x + (am1 / pick(less(abs(x), lit(1e-12)), lit(1e-12), x)
        + a0 + a1 * x + a2 * x * x) * $dt
        + (b0 + b1 * x + b2 * pow(abs(x), b3)) * dz
    }
    report { x },

  /// Geometric Brownian motion on the shifted variable `Y = S + β`, reported
  /// as `Y − β`: the displaced diffusion. The shift lives in the report, so
  /// the step is the geometric one term for term.
  32 => Displaced { mu, sigma, beta }
    state (x)
    noise (dz)
    step { x + mu * x * $dt + sigma * x * dz }
    report { x - beta },

  /// `dX = κ(μ − tanh X) dt + σ dW` reported as `tanh X`: Teng's stochastic
  /// correlation process, stepped on the unbounded variable so the reported
  /// correlation stays in `(−1, 1)` by construction.
  33 => TanhOrnsteinUhlenbeck { kappa, mu, sigma }
    state (x)
    noise (dz)
    step { x + kappa * (mu - tanh(x)) * $dt + sigma * dz }
    report { tanh(x) },

  /// `dρ = κ(μ − ρ) dt + σ√(1 − ρ²) dW` confined to `[−0.9999, 0.9999]`: the
  /// Van Emmerich stochastic correlation process.
  34 => BoundedCorrelation { kappa, mu, sigma }
    state (x)
    noise (dz)
    step {
      min(
        max(
          x + kappa * (mu - x) * $dt + sigma * sqrt(positive(negate(x * x - lit(1.0)))) * dz,
          lit(-0.9999)
        ),
        lit(0.9999)
      )
    }
    report { x },
    }
  };
}
