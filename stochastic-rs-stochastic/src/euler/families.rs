//! The drift / diffusion families of the Euler engine, declared once.
//!
//! A family is written in a subset that is simultaneously a valid Rust
//! expression and a valid C expression: infix arithmetic, parentheses and a
//! small vocabulary of functions. [`euler_families!`] takes that one
//! declaration and emits every form the engine needs — the host step, and the
//! C body the native CUDA and Metal kernels render — from the same tokens, so
//! the operation order cannot drift between them.
//!
//! The CubeCL kernel is written by hand instead: its `#[cube]` attribute
//! cannot see through a macro expansion, and calls into a generic helper need
//! a turbofish the shared tokens must not carry. What keeps it honest is
//! `euler::family_parity`, which launches every declared family on both
//! kernels and compares them point for point, so a family missing from the
//! hand-written dispatch fails a test rather than quietly returning a flat
//! path.
//!
//! A step or report may open with `bind name = expr;` lines before its final
//! expression. Each becomes a `let` on the host and in a CubeCL kernel and a
//! `const REAL` in the emitted C, which is how a family names a clamped or
//! guarded state once and then reads like the host sampler it came from.
//!
//! A step or report may read `u` and `u2`, two uniforms in `[0, 1)` for the
//! step, `nj`, the number of jumps it saw, `js`, the sum of their sizes, `gm`
//! and `gm2`, one or two Gamma draws, and `ct`, the step's value of a time-varying
//! coefficient the host supplies as one value per grid point. A family that
//! never names it costs nothing for it.
//!
//! The CubeCL functions take the four state and four noise slots as
//! parameters and bind the family's own names from them, so those parameters
//! are named `slot_a`..`slot_d` and `shock_a`..`shock_d`: a family whose
//! state were called `x1` would otherwise shadow the slot it was being read
//! from, and every later binding would read the shadowed value.
//!
//! The names a step may use are fixed: `x` for the state, `dt` for the step
//! size, `dz` for the step's noise **increment** — `sqrt_dt · z` for Gaussian
//! noise, the fractional increment itself for fGN, which is what lets one
//! declaration serve both — and the family's own
//! parameters, which the generated code binds as locals from the parameter
//! buffer in declaration order. The `report` expression maps the state to what
//! the path records and is evaluated at `t = 0` as well, where no noise exists.
//!
//! A family with a `history { push (..) weights (ctK) }` clause reads `cv`,
//! the convolution of what it has pushed so far with the weights the launch
//! tabulated in curve slot `K`, indexed by lag: at the step that produces
//! grid point `i` the frame appends the pushed value `p_i` (evaluated on the
//! pre-step state, this step's noise and uniforms) and hands the step
//! `cv = Σ_{k≤i} w_k p_{i-k}`. That is the exact O(n²) scheme of a moving
//! average over the whole past — a Volterra integral on the grid, an ARMA
//! filter, a fractional stable motion — for grids up to
//! [`HISTORY_SLOTS`](crate::euler::HISTORY_SLOTS) points.
//!
//! A family with a `series { size (..) }` clause reads `sj`, the sum of the
//! terms that fell into the step's grid cell: before the steps the frame
//! draws, per path, as many terms as the launch's `series_terms()` names —
//! for each, `gj` the arrival of a unit-rate Poisson process, `ej` an
//! `Exp(1)`, `uj` and `uv` two uniforms, and a uniform arrival time on the
//! horizon — sizes each with the clause's expression and adds it to the cell
//! its time falls in. That is the shot-noise series representation of a
//! Lévy process, the sort with a rejection-free size bound, run without the
//! host's sort: cells are addressed, not ordered. Grids up to
//! [`SERIES_SLOTS`](crate::euler::SERIES_SLOTS) points. A clause written
//! `series { live size (..) }` sizes each term in the step of its cell
//! instead, where the size may read the state — a scale that follows a
//! simulated variance — and the frame keeps the arrivals per term rather
//! than the sums per cell, so the term count is what the slots then bound.
//!
//! A family with a `table { increment (..) }` clause reads `iv`, the inverse
//! of a monotone table at the step's time: before the steps the frame builds,
//! per path, a table of as many points as the launch's `table_spec()` names
//! over `[0, u_max]` — each increment from the clause's expression of the two
//! uniforms `uj`, `uv` and the spacing `tv` — trying up to ten extents, each
//! double the last, until the table reaches the horizon, exactly as the host
//! does; each
//! step then finds the first table value at or past its time and
//! interpolates the abscissa linearly. That is the inverse subordinator, the
//! first-passage clock of a positive process. Tables up to
//! [`TABLE_SLOTS`](crate::euler::TABLE_SLOTS) points; the grid is free.
//!
//! The function vocabulary is `sqrt`, `exp`, `ln`, `pow`, `abs`, `negate`,
//! `tanh`, `atan`, `sin`, `recip`, `positive`, `max`, `min`, the literal
//! `lit`, the
//! comparisons
//! `less`, `leq` and `geq`, and the branch-free `pick`. Each has a host
//! implementation in [`ops`], a C definition in [`vocabulary::C_PRELUDE`] and a CubeCL
//! one in `cube_ops`, the three kept together in [`vocabulary`]; anything
//! outside it fails to compile on the host, which is the intended way to find
//! out that a kernel could not have run it either.
//!
//! This file is the table: the declarations, in code order, and nothing
//! else. The generator they feed — the `euler_families!` macro — lives in
//! [`codegen`], and the tests that pin the generated step and report against
//! the closed forms they came from in `tests`.

use crate::traits::FloatExt;

// Textual scope, not a path import: the macro's generated `cube*` modules
// recurse into it, and a child module created by an expansion sees only what
// is textually in scope at that point, not the parent's `use` items.
#[macro_use]
pub(crate) mod codegen;
pub(crate) mod vocabulary;

#[cfg(feature = "cubecl")]
pub(crate) use vocabulary::cube_ops;
pub(crate) use vocabulary::ops;

euler_families! {
  step_inputs(
    params, dt, ct, ct1, ct2, ct3, ct4, ct5, ct6, ct7, nj, js, gm, gm2, u, u2, lv, cv, sj, gj, ej, uj, uv, iv, tv,
    state(slot_a, slot_b, slot_c, slot_d),
    noise(shock_a, shock_b, shock_c, shock_d),
    select(component, produced)
  );

  /// `dX = μX dt + σX dW`.
  0 => GeometricBrownian { mu, sigma }
    state (x)
    noise (dz)
    step { x + mu * x * dt + sigma * x * dz }
    report { x },

  /// `dX = θ(μ − X) dt + σ dW`.
  1 => OrnsteinUhlenbeck { theta, mu, sigma }
    state (x)
    noise (dz)
    step { x + theta * (mu - x) * dt + sigma * dz }
    report { x },

  2 => SquareRoot { kappa, theta, sigma }
    state (x)
    noise (dz)
    step { x + kappa * (theta - positive(x)) * dt + sigma * sqrt(positive(x)) * dz }
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
    step { positive(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
    report { x },

  /// The same with the symmetric reflection: the step's absolute value.
  5 => MirroredSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { abs(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
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
          x + (alpha - beta * x) * dt + sigma * sqrt(positive(x - x * x)) * dz
        )
      )
    }
    report { x },

  /// `dX = μX dt + σ|X|^γ dW`: constant elasticity of variance, the power
  /// taken off `|X|` so a negative excursion stays defined.
  7 => ConstantElasticity { mu, sigma, gamma }
    state (x)
    noise (dz)
    step { x + mu * x * dt + sigma * pow(abs(x), gamma) * dz }
    report { x },

  /// `dX = (θ₁ + θ₂X) dt + θ₃|X|^θ₄ dW`: the Chan–Karolyi–Longstaff–Sanders
  /// family, whose four parameters nest most one-factor short-rate models.
  8 => Ckls { theta1, theta2, theta3, theta4 }
    state (x)
    noise (dz)
    step { x + (theta1 + theta2 * x) * dt + theta3 * pow(abs(x), theta4) * dz }
    report { x },

  /// `dX = X(1 − aX) dt + bX dW`: logistic growth with multiplicative noise.
  /// `X(1 − aX)` is written `x - a * x * x`, which needs no literal.
  9 => Logistic { a, b }
    state (x)
    noise (dz)
    step { x + (x - a * x * x) * dt + b * x * dz }
    report { x },

  /// `dX = κX(μ − X) dt + σ|X|^{3/2} dW`: the 3/2 model, whose variance
  /// mean-reverts faster than a square-root diffusion.
  10 => ThreeHalf { kappa, mu, sigma }
    state (x)
    noise (dz)
    step { x + kappa * x * (mu - x) * dt + sigma * pow(abs(x), lit(1.5)) * dz }
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
      x + (kappa / pick(leq(abs(x), lit(1e-12)), lit(1e-12), x) - x) * dt + sigma * dz
    }
    report { x },

  /// `dX = (a + bX) dt + cX dW`: the linear scalar SDE.
  13 => LinearSde { a, b, c }
    state (x)
    noise (dz)
    step { x + (a + b * x) * dt + c * x * dz }
    report { x },

  /// `dX = −κX/√(1+X²) dt + σ dW`: a hyperbolic drift, bounded in `X`.
  14 => Hyperbolic { kappa, sigma }
    state (x)
    noise (dz)
    step { x - kappa * x / sqrt(x * x + lit(1.0)) * dt + sigma * dz }
    report { x },

  /// `dX = −κX dt + σ√(1+X²) dW`: the modified CIR process, whose diffusion
  /// never vanishes.
  15 => ModifiedSquareRoot { kappa, sigma }
    state (x)
    noise (dz)
    step { x - kappa * x * dt + sigma * sqrt(x * x + lit(1.0)) * dz }
    report { x },

  /// `dX = X(θ₁ − X(θ₃³ − θ₁θ₂)) dt + θ₃|X|^{3/2} dW`: the Feller root
  /// process, with the drift's constant folded on the host.
  16 => FellerRoot { theta1, decay, theta3 }
    state (x)
    noise (dz)
    step { x + x * (theta1 - x * decay) * dt + theta3 * pow(abs(x), lit(1.5)) * dz }
    report { x },

  /// `dX = (a₋₁/X + a₀ + a₁X + a₂X²) dt + √|b₀ + b₁X + b₂|X|^{b₃}| dW`: the
  /// Aït-Sahalia short-rate model, whose drift is guarded away from the origin
  /// exactly as the host sampler guards it.
  17 => AitSahalia { am1, a0, a1, a2, b0, b1, b2, b3 }
    state (x)
    noise (dz)
    step {
      x + (am1 / pick(less(abs(x), lit(1e-12)), lit(1e-12), x)
        + a0 + a1 * x + a2 * x * x) * dt
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
    step { max(x + (a - b * ln(x)) * x * dt + sigma * x * dz, lit(1e-12)) }
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
          xi + a * xi * negate(xi - lit(1.0)) * dt
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
    step { x + (alpha + beta * x + gamma * x * x) * dt + sigma * x * dz }
    report { x },

  /// `dX = κ(μ − X) dt + √|2κ(aX² + bX + c)| dW`: the Pearson diffusion
  /// family. `2κ` is folded on the host so the step needs no literal.
  21 => Pearson { kappa, mu, a, b, c, two_kappa }
    state (x)
    noise (dz)
    step {
      x + kappa * (mu - x) * dt + sqrt(abs(two_kappa * (a * x * x + b * x + c))) * dz
    }
    report { x },

  /// `dX = rX(1 − X/K) dt + σX dW`: logistic growth in its Verhulst
  /// parametrisation, run unclamped.
  22 => Verhulst { r, k, sigma }
    state (x)
    noise (dz)
    step { x + r * x * ((k - x) / k) * dt + sigma * x * dz }
    report { x },

  /// [`Verhulst`](Family::Verhulst) with the state confined to `[0, K]`.
  23 => VerhulstClamped { r, k, sigma }
    state (x)
    noise (dz)
    step { min(max(x + r * x * ((k - x) / k) * dt + sigma * x * dz, lit(0.0)), k) }
    report { x },

  /// `dX = κ(θ − X)X dt + σ√X dW`: Feller's logistic diffusion, truncated at
  /// zero.
  24 => FellerLogistic { kappa, theta, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = positive(x);
      positive(xi + kappa * (theta - xi) * xi * dt + sigma * sqrt(xi) * dz)
    }
    report { x },

  /// [`FellerLogistic`](Family::FellerLogistic) reflected at zero instead of
  /// truncated.
  25 => FellerLogisticReflected { kappa, theta, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = positive(x);
      abs(xi + kappa * (theta - xi) * xi * dt + sigma * sqrt(xi) * dz)
    }
    report { x },

  /// `dX = δ dt + 2√|X| dW`: the squared-Bessel recursion, truncated at zero.
  26 => SquaredBesselState { delta, two }
    state (x)
    noise (dz)
    step { positive(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { x },

  /// [`SquaredBesselState`](Family::SquaredBesselState) reflected at zero.
  27 => SquaredBesselStateReflected { delta, two }
    state (x)
    noise (dz)
    step { abs(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { x },

  /// [`SquaredBesselState`](Family::SquaredBesselState) reporting `√X`: the
  /// Bessel process itself, stepped in squared space so the `(δ−1)/2X`
  /// singularity never enters the recursion.
  28 => BesselFromSquared { delta, two }
    state (x)
    noise (dz)
    step { positive(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { sqrt(x) },

  /// [`BesselFromSquared`](Family::BesselFromSquared) reflected at zero.
  29 => BesselFromSquaredReflected { delta, two }
    state (x)
    noise (dz)
    step { abs(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { sqrt(x) },

  /// `dX = ½σ²(β − γ(X−μ)/√(δ² + (X−μ)²)) dt + σ dW`: the hyperbolic
  /// diffusion whose stationary law is the hyperbolic distribution. `½σ²` is
  /// folded on the host.
  30 => HyperbolicDiffusion { beta, gamma, delta, mu, sigma, half_var }
    state (x)
    noise (dz)
    step {
      x + half_var * (beta - gamma * x / sqrt(delta * delta + (x - mu) * (x - mu))) * dt
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
        + a0 + a1 * x + a2 * x * x) * dt
        + (b0 + b1 * x + b2 * pow(abs(x), b3)) * dz
    }
    report { x },

  /// Geometric Brownian motion on the shifted variable `Y = S + β`, reported
  /// as `Y − β`: the displaced diffusion. The shift lives in the report, so
  /// the step is the geometric one term for term.
  32 => Displaced { mu, sigma, beta }
    state (x)
    noise (dz)
    step { x + mu * x * dt + sigma * x * dz }
    report { x - beta },

  /// `dX = κ(μ − tanh X) dt + σ dW` reported as `tanh X`: Teng's stochastic
  /// correlation process, stepped on the unbounded variable so the reported
  /// correlation stays in `(−1, 1)` by construction.
  33 => TanhOrnsteinUhlenbeck { kappa, mu, sigma }
    state (x)
    noise (dz)
    step { x + kappa * (mu - tanh(x)) * dt + sigma * dz }
    report { tanh(x) },

  /// `dρ = κ(μ − ρ) dt + σ√(1 − ρ²) dW` confined to `[−0.9999, 0.9999]`: the
  /// Van Emmerich stochastic correlation process.
  34 => BoundedCorrelation { kappa, mu, sigma }
    state (x)
    noise (dz)
    step {
      min(
        max(
          x + kappa * (mu - x) * dt + sigma * sqrt(positive(negate(x * x - lit(1.0)))) * dz,
          lit(-0.9999)
        ),
        lit(0.9999)
      )
    }
    report { x },

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
      s + mu * s * dt + s * sqrt(vp) * dw,
      positive(v + kappa * (theta - vp) * dt + sigma * pow(vp, pow_v) * db)
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
      s + mu * s * dt + s * sqrt(vp) * dw,
      abs(v + kappa * (theta - vp) * dt + sigma * pow(vp, pow_v) * db)
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
      vp * exp(nu * db - half_nu_sq * dt)
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
      s + r * s * dt + sqrt(v) * s * dw,
      v0_sq * exp(nu * (z + db) - half_nu_sq * (elapsed + dt)),
      z + db,
      elapsed + dt
    }
    report { s, v, z, elapsed },

  /// Two Ornstein–Uhlenbeck factors on one clock, a slow one and a fast one:
  /// the Fouque–Papanicolaou–Sircar two-scale volatility driver.
  39 => TwoScaleOrnsteinUhlenbeck { kappa, theta, eps, alpha, eps_inv, sqrt_eps_inv }
    state (x, y)
    noise (dx, dy)
    step {
      x + kappa * (theta - x) * dt + eps * dx,
      y + eps_inv * (alpha - y) * dt + sqrt_eps_inv * dy
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
      s * exp((drift - vp / lit(2.0)) * dt + sv * dw),
      positive(vp + kappa * (theta - vp) * dt + xi * sv * dwv)
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
      s * exp((drift - vp / lit(2.0)) * dt + sv * dw),
      abs(vp + kappa * (theta - vp) * dt + xi * sv * dwv)
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
      s + mu * s * dt + s * sqrt(p1) * ds1 + s * sqrt(p2) * ds2,
      positive(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dv1),
      positive(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dv2)
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
      s + mu * s * dt + s * sqrt(p1) * ds1 + s * sqrt(p2) * ds2,
      abs(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dv1),
      abs(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dv2)
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
      bind xc = x + kappa_r * (mu_r - tanh(x)) * dt + sigma_r * drho;
      bind rt = tanh(xc);
      bind indep = sqrt(positive(negate(rt * rt + rho2 * rho2 - lit(1.0))));
      s * exp(
        (r - vp / lit(2.0)) * dt
          + rt * sv * dv_w
          + rho2 * sv * drho
          + indep * sv * dx_w
      ),
      positive(vp + kappa_v * (mu_v - vp) * dt + sigma_v * sv * dv_w),
      xc
    }
    report { s, v, tanh(x) },

  /// `dr = (θ(t) − αr) dt + σ dW`: Hull-White, whose mean-reversion level is
  /// the time-varying coefficient the launch carries.
  45 => HullWhite { alpha, sigma }
    state (x)
    noise (dz)
    step { x + (ct - alpha * x) * dt + sigma * dz }
    report { x },

  /// `dr = θ(t) dt + σ dW`: a drift that is entirely the curve, which is Ho-Lee
  /// under either of its two drift forms.
  46 => CurveDrift { sigma }
    state (x)
    noise (dz)
    step { x + ct * dt + sigma * dz }
    report { x },

  /// The exact one-step Ornstein–Uhlenbeck transition in log space, reported
  /// exponentiated: Black-Karasinski. `decay` is `exp(−a·dt)` and the noise
  /// scale folds the exact transition standard deviation, both of which
  /// depend on `dt` alone.
  47 => LogMeanReverting { decay, a, sigma_eff }
    state (y)
    noise (dz)
    step { y * decay + (ct / a) * negate(decay - lit(1.0)) + sigma_eff * dz }
    report { exp(y) },

  /// A square-root diffusion shifted by a deterministic curve, truncated at
  /// zero before the shift: CIR++.
  48 => ShiftedSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { positive(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
    report { x + ct },

  /// [`ShiftedSquareRoot`](Family::ShiftedSquareRoot) reflected at zero.
  49 => ShiftedSquareRootMirrored { theta, mu, sigma }
    state (x)
    noise (dz)
    step { abs(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
    report { x + ct },

  /// `dX = μX dt + σ(t)X dW`: geometric Brownian motion over a term
  /// structure of volatilities.
  50 => TimeVaryingGeometricBrownian { mu }
    state (x)
    noise (dz)
    step { x + mu * x * dt + ct * x * dz }
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
      x + (xt - x) * ct * dt + sigma * sqrt(positive(negate(dt * ct - lit(1.0)))) * dz
    }
    report { x },

  /// The two-factor Hull-White model: a short rate pulled toward the curve
  /// and a second, zero-reverting factor added to its drift.
  53 => TwoFactorHullWhite { a, b, sigma1, sigma2, rho }
    state (x, u)
    noise (dw, dq)
    step {
      bind du = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      x + (ct + u - a * x) * dt + sigma1 * dw,
      u - b * u * dt + sigma2 * du
    }
    report { x, u },

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
      bind v1 = a + theta1 * (mu1 - a) * dt + sigma1 * sqrt(abs(a)) * dw;
      bind v2 = b + theta2 * (mu2 - b) * dt + sigma2 * sqrt(abs(b)) * dq;
      pick(sym1, abs(v1), positive(v1)),
      pick(sym2, abs(v2), positive(v2))
    }
    report { a + b + ct, b },

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
      r + (a1 * r + b1 * x + c1) * dt + sigma1 * vol * dw,
      x + (a2 * r + b2 * x + c2) * dt + sigma2 * vol * dx
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
      x1 + (mu1 - p1 / lit(2.0)) * dt + sqrt(p1) * dw1,
      positive(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dz1),
      x2 + (mu2 - p2 / lit(2.0)) * dt + sqrt(p2) * dw2,
      positive(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dz2)
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
      x1 + (mu1 - p1 / lit(2.0)) * dt + sqrt(p1) * dw1,
      abs(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dz1),
      x2 + (mu2 - p2 / lit(2.0)) * dt + sqrt(p2) * dw2,
      abs(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dz2)
    }
    report { x1, v1, x2, v2 },

  /// `d ln S = (μ − λκ − ½σ²) dt + σ dW + Σ Y_i`, the jump sizes lognormal:
  /// Merton's jump diffusion in log-price. The jump sum is the kernel's, from
  /// the size law the process declares.
  58 => MertonJumpLog { drift_ln, sigma }
    state (x)
    noise (dw)
    step { x * exp(drift_ln + sigma * dw + js) }
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
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      positive(vp + (alpha - beta * vp) * dt + sigma * sv * dv)
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
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      abs(vp + (alpha - beta * vp) * dt + sigma * sv * dv)
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
      bind zv = dzv / sqrt(dt);
      bind quad = m / (b2 + lit(1.0)) * (sqrt(b2) + zv) * (sqrt(b2) + zv);
      bind p = (psi - lit(1.0)) / (psi + lit(1.0));
      bind tail = ln(negate(p - lit(1.0)) / negate(u - lit(1.0))) * m
        * recip(negate(p - lit(1.0)));
      bind vn = pick(leq(psi, lit(1.5)), quad, pick(leq(u, p), lit(0.0), tail));
      y + mu * dt + k0 + k1 * v + k2 * vn + sqrt(positive(k34 * (v + vn))) * (dz / sqrt(dt)),
      vn
    }
    report { exp(y), v },

  /// A Poisson counting process on the grid: the step adds the jumps it saw.
  62 => CountingProcess { }
    state (x)
    noise (dz)
    step { x + nj }
    report { x },

  /// An inverse-Gaussian subordinator: each increment is one Michael-Schucany-
  /// Haas draw, which needs a standard normal and a uniform and no rejection,
  /// so it is one expression. `2λ` and `4μλ` depend on the parameters and `dt`
  /// alone and are folded on the host.
  63 => InverseGaussianSubordinator { mu_ig, two_lam, four_mu_lam }
    state (x)
    noise (dz)
    step {
      bind w = (dz / sqrt(dt)) * (dz / sqrt(dt));
      bind rad = sqrt(four_mu_lam * w + mu_ig * mu_ig * w * w);
      bind xr = mu_ig + mu_ig * mu_ig * w / two_lam - mu_ig / two_lam * rad;
      x + pick(less(u, mu_ig / (mu_ig + xr)), xr, mu_ig * mu_ig / xr)
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
      bind w = (dz / sqrt(dt)) * (dz / sqrt(dt));
      bind rad = sqrt(four_mu_lam * w + mu_ig * mu_ig * w * w);
      bind xr = mu_ig + mu_ig * mu_ig * w / two_lam - mu_ig / two_lam * rad;
      bind ig = pick(less(u, mu_ig / (mu_ig + xr)), xr, mu_ig * mu_ig / xr);
      x + theta * ig + sigma * sqrt(ig) * (dq / sqrt(dt))
    }
    report { x },

  /// A positive-stable subordinator by the Chambers-Mallows-Stuck transform:
  /// one uniform on `(0, π)` and one exponential, both from the step's own
  /// uniforms, with no rejection. The two exponents depend on `α` alone and
  /// are folded on the host, as are the scale `(c·dt)^{1/α}` and `π`.
  ///
  /// The uniforms are clamped into the open interval at a bound `f32` can
  /// hold below one: at exactly one the angle is `π`, whose sine is a small
  /// *negative* in single precision, and raising that to a fractional power
  /// is a NaN. The sines are floored for the same reason the clamp exists.
  65 => StableSubordinator { alpha, inv_alpha, one_minus_alpha, tail_exp, scale, pi }
    state (x)
    noise (dz)
    step {
      bind uu = min(max(u, lit(1e-7)), lit(0.9999999)) * pi;
      bind w = negate(ln(min(max(u2, lit(1e-7)), lit(0.9999999))));
      bind s1 = sin(alpha * uu) / pow(max(sin(uu), lit(1e-20)), inv_alpha);
      bind s2 = pow(max(sin(one_minus_alpha * uu), lit(1e-20)) / w, tail_exp);
      x + scale * s1 * s2
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
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      positive(vp + kappa * (theta - vp) * dt + sigma_v * sv * dv)
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
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      abs(vp + kappa * (theta - vp) * dt + sigma_v * sv * dv)
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
      r + (a1 * r + b1 * x + c1) * dt + sigma1 * vol * dw,
      x + (a2 * r + b2 * x + c2) * dt + sigma2 * vol * dx + js
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
      bind fired = less(u, lam * dt);
      bind size = pick(fired, jump_mu + jump_sigma * (dj / sqrt(dt)), lit(0.0));
      bind excited = lam + pick(fired, alpha, lit(0.0));
      x + drift_c * dt + sigma * dw + size,
      positive(excited + beta * (mu_lambda - excited) * dt)
    }
    report { x, lam },

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
      sqrt(max(v, lit(1e-12))) * (dz / sqrt(dt)),
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
      sqrt(max(v, lit(1e-12))) * (dz / sqrt(dt)),
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
      sqrt(exp(v)) * (dz / sqrt(dt)),
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
    step { mean + sd * (dz / sqrt(dt)) }
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
    step { phi * x + sigma * (dz / sqrt(dt)) }
    report { x },

  /// `X_t = σz_t + θσz_{t−1}`: a first-order moving average. The lagged
  /// innovation is state of its own, since the step cannot see the previous
  /// draw any other way.
  76 => MovingAverage { theta, sigma }
    state (x, e)
    noise (dz)
    step {
      bind now = sigma * (dz / sqrt(dt));
      now + theta * e,
      now
    }
    report { x, e },

  /// A gamma subordinator: each increment is one Gamma draw.
  77 => GammaSubordinator { }
    state (x)
    noise (dz)
    step { x + gm }
    report { x },

  /// Brownian motion under a gamma clock: the variance gamma process.
  78 => VarianceGamma { mu, sigma }
    state (x)
    noise (dz)
    step { x + mu * gm + sigma * sqrt(gm) * (dz / sqrt(dt)) }
    report { x },

  /// The difference of two gamma processes: the bilateral gamma process.
  79 => BilateralGamma { }
    state (x)
    noise (dz)
    step { x + gm - gm2 }
    report { x },

  /// [`BilateralGamma`](Family::BilateralGamma) with a Brownian part.
  80 => BilateralGammaMotion { sigma }
    state (x)
    noise (dz)
    step { x + sigma * dz + gm - gm2 }
    report { x },

  /// A tempered-stable subordinator: the deterministic drift the small jumps
  /// below the truncation contribute, plus the step's own thinned jumps.
  81 => TemperedStableSubordinator { drift }
    state (x)
    noise (dz)
    step { x + drift + js }
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
      s * exp((mu - v / lit(2.0)) * dt + sqrt(v) * dw),
      decay * v + gm
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
      x1 - (lambda * x1 + omega * x2) * dt + scale * dz1,
      x2 - (lambda * x2 - omega * x1) * dt + scale * dz2
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
    step { x + kappa * (mu - x) * dt + sigma * dz }
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
    step { x + negate(ln(max(u, lit(1.0e-7)))) * recip(lambda) }
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
      bind dw2 = ct1 * dw1 + sqrt(negate(ct1 * ct1 - lit(1.0))) * dq;
      positive(f + positive(v) * pow(positive(f), ct) * dw1),
      positive(v) * exp(ct2 * dw2 - ct2 * ct2 * dt * lit(0.5))
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
      r + ct * dt + ct1 * dr,
      p + ct2 * dt + ct3 * dp,
      f + ct4 * dt + ct5 * df
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
    step { x + (ct - ct1 * x) * dt + sigma * dz }
    report { ct2 + ct3 * x + ct4 * x * x },

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
      bind vn = positive(positive(v) + beta * (alpha - positive(v)) * dt + nu * sqrt(positive(v)) * dv);
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
      a * exp((m0 - s0 * s0 * lit(0.5)) * dt + s0 * e0),
      b * exp((m1 - s1 * s1 * lit(0.5)) * dt + s1 * e1),
      c * exp((m2 - s2 * s2 * lit(0.5)) * dt + s2 * e2),
      d * exp((m3 - s3 * s3 * lit(0.5)) * dt + s3 * e3)
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
      s * exp((mu - sig * sig * lit(0.5)) * dt + sig * dw),
      pick(leq(u, k1), lit(0.0), pick(leq(u, k2), lit(1.0), pick(leq(u, k3), lit(2.0), lit(3.0))))
    }
    report { s, z },

  /// Riemann-Liouville fractional Brownian motion under the Markov lift of
  /// its kernel: no drift, unit diffusion, driven by the step's own shock,
  /// and the state *is* the lifted value the frame hands back as `lv`. The
  /// same declaration serves any rough Volterra kernel whose nodes and
  /// weights the host supplies through `lift_spec()`.
  94 => RiemannLiouville { }
    state (x)
    noise (dz)
    step { lv }
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
      x + kappa * (mu - x) * dt + nu * (lv - b),
      lv
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
      s0 * exp(ct + sigma * lv),
      lv
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
      s + mu * s * dt + s * sqrt(positive(v)) * ds,
      lv
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
    step { x + drift_dt + sigma * dw + js }
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
      s + drift_c * s * dt + s * sv * dw + s * js,
      positive(vp + (alpha - beta * vp) * dt + sigma * sv * dv)
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
      s + drift_c * s * dt + s * sv * dw + s * js,
      abs(vp + (alpha - beta * vp) * dt + sigma * sv * dv)
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
      t + negate(ln(max(u, lit(1.0e-7)))) * recip(lambda),
      c + js,
      js
    }
    report { t, c, j },

  /// A fractional Ornstein-Uhlenbeck process with compound-Poisson jumps:
  /// the fractional increments come from the device's embedding and the jump
  /// sum from the step's own draw.
  102 => JumpFractionalOu { theta, mu, sigma }
    state (x)
    noise (dz)
    step { x + theta * (mu - x) * dt + sigma * dz + js }
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
    step { lv }
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
    step { lv }
    report { c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * (c6 + x * c7)))))) }
    lift { drift (lit(0.0)) diffusion (lit(1.0)) shock (dz) },

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
      s * exp((mu - vp / lit(2.0)) * dt + sqrt(vp) * dsp),
      y1 + c1 * nu * z1 + c2 * nu * cv * inv_g,
      y1,
      z1
    }
    report { s, v }
    history { push (z) weights (ct) },

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
      s * exp((mu_c - vp / lit(2.0)) * dt + sqrt(vp) * dsp + js),
      y1 + xi * z1 + xi * cv * inv_g,
      y1,
      z1
    }
    report { s, v }
    history { push (z) weights (ct) },

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
      bind xv = cv + residual_sd * (de / sqrt(dt));
      s + r * s * dt + sqrt(v) * s * dw,
      v0sq * exp(a * xv - ct1),
      xv
    }
    report { s, v }
    history { push (rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq) weights (ct) },

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
    step { x + cv }
    report { x }
    history {
      push (
        bind uu = min(max(u, lit(1e-7)), lit(0.9999999));
        bind ee = min(max(u2, lit(1e-7)), lit(0.9999999));
        bind va = pi * (uu - lit(0.5));
        bind w = negate(ln(ee));
        bind phi = alpha * (va + b);
        bind cs = max(sin(va + half_pi), lit(1e-20));
        bind ratio = max(sin(va - phi + half_pi) / w, lit(1e-30));
        scale_s * sin(phi) / pow(cs, inv_alpha) * pow(ratio, tail_exp)
      )
      weights (ct)
    },

  /// A moving-average filter of white noise: the state is the convolution of
  /// the pushed innovations `σ z` with the weight curve, which the host fills
  /// with the impulse response of whatever linear recursion it declares — an
  /// ARMA with its differencing, seasonal or not — so the device runs that
  /// recursion exactly, at any order. The first point is itself a draw.
  109 => MovingAverageFilter { sigma }
    state (x)
    noise (dz)
    step { cv }
    report { x }
    history { push (sigma * dz / sqrt(dt)) weights (ct) },

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
    step { x + b_t * dt + sj }
    report { x }
    series {
      size (
        bind up = less(uv, w_plus);
        bind side = pick(up, lambda_plus, lambda_minus);
        bind cap = pow(gj * rate, negate(inv_alpha));
        bind draw = e_scale * pow(ej, e_pow) * pow(uj, inv_alpha) / side;
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
    step { cv }
    report { x }
    history { push (dz) weights (ct) },

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
      bind d = beta * ln(max(u, lit(1.0e-7))) / se + lit(1.0);
      bind pos = less(negate(d), lit(0.0));
      bind s1 = pick(pos, negate(ln(max(d, lit(1.0e-30)))) / beta, lit(1.0e30));
      bind s2 = negate(ln(max(u2, lit(1.0e-7)))) / mu;
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
      bind ei = max(ct - lit(1.0), lit(0.0));
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
      pick(leq(ct, lit(0.0)), f0 * exp((m0 - s0 * s0 / lit(2.0)) * dt + s0 * w0), f0),
      pick(leq(ct, lit(1.0)), f1 * exp((m1 - s1 * s1 / lit(2.0)) * dt + s1 * w1), f1),
      pick(leq(ct, lit(2.0)), f2 * exp((m2 - s2 * s2 / lit(2.0)) * dt + s2 * w2), f2),
      pick(leq(ct, lit(3.0)), f3 * exp((m3 - s3 * s3 / lit(2.0)) * dt + s3 * w3), f3)
    }
    report { f0, f1, f2, f3 },

  /// The inverse of an α-stable subordinator, `E(t) = inf{u : D(u) > t}`: the
  /// table block builds `D` on a grid in `u` from positive-stable increments —
  /// Kanter's form of the Chambers–Mallows–Stuck draw, its uniforms clamped
  /// away from the ends of the unit interval, at scale `(c Δu)^{1/α}` — and the
  /// step takes the interpolated first passage over the horizon's time. The
  /// state is that inverse; there is nothing to step.
  114 => InverseStableSubordinator { alpha, c, inv_alpha, one_minus_alpha, tail_exp, pi }
    state (x)
    noise (dz)
    step { iv }
    report { x }
    table {
      increment (
        bind uu = min(max(uj, lit(1e-7)), lit(0.9999999)) * pi;
        bind w = negate(ln(min(max(uv, lit(1e-7)), lit(0.9999999))));
        bind s1 = sin(alpha * uu) / pow(max(sin(uu), lit(1e-20)), inv_alpha);
        bind s2 = pow(max(sin(one_minus_alpha * uu), lit(1e-20)) / w, tail_exp);
        pow(c * tv, inv_alpha) * s1 * s2
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
      bind zn = dz / sqrt(dt);
      bind root = zn + sqrt(twoc * v * ek);
      y + bcoef * v * dt + sj,
      (root * root + gm) / twoc
    }
    report { y + rho * v, v }
    series {
      live size (
        bind up = less(uv, lit(0.5));
        bind side = pick(up, lambda_plus, lambda_minus);
        bind cap = pow(gj * rate0 / max(v, lit(1.0e-12)), negate(inv_alpha));
        bind draw = ej * pow(uj, inv_alpha) / side;
        bind mag = min(cap, draw);
        pick(up, mag, negate(mag))
      )
    },
}

#[cfg(test)]
mod tests;
