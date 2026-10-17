use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;
use rand::RngExt;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::gamma::SimdGamma;

use super::Generator;
use super::NacNode;
use super::NestedArchimedean;
use crate::bivariate::conditional::softplus;

/// A node's frailty `V` as `ln V` and `w = ln V / θ`: the first is finite at small `θ`, where `V` is large, the second at
/// large `θ`, where `V` leaves the floating-point range.
#[derive(Debug, Clone, Copy)]
pub(super) struct Frailty {
  pub(super) ln_v: f64,
  pub(super) w: f64,
}

/// `α ln S` of the positive stable `S` with Laplace transform `e^{−t^α}`: Kanter's `S = (A(Θ)/W)^{(1−α)/α}` (Devroye 2009,
/// eq. (2)) in logarithms, finite for every `0 < α < 1`, where the product of powers underflows to `0/0` below `α ≈ 0.01`.
pub(super) fn stable_power<R: Rng + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
  let theta = std::f64::consts::PI * rng.random::<f64>().clamp(1e-15, 1.0 - 1e-15);
  let ln_w = (-rng.random::<f64>().clamp(1e-15, 1.0 - 1e-15).ln()).ln();
  let beta = 1.0 - alpha;
  alpha * (alpha * theta).sin().ln() + beta * (beta * theta).sin().ln()
    - theta.sin().ln()
    - beta * ln_w
}

/// The child frailty: Laplace transform `exp(−V₀((1 + t)^α − 1))` by Hofert's (2011) fast rejection, a sum of `m` positive
/// stables of transform `exp(−(V₀/m) t^α)` each kept with probability `e^{−S}`, `m` minimising `m e^{V₀/m}`.
pub(super) fn tilted_stable<R: Rng + ?Sized>(
  rng: &mut R,
  alpha: f64,
  parent: Frailty,
  theta_parent: f64,
  theta: f64,
) -> Frailty {
  let v0 = parent.ln_v.exp();
  let m = if v0 <= 1.0 {
    1.0
  } else {
    let (lo, hi) = (v0.floor(), v0.ceil());
    if lo * (v0 / lo).exp() <= hi * (v0 / hi).exp() {
      lo
    } else {
      hi
    }
  };
  let ln_scale = parent.ln_v - m.ln();
  let mut summand = || loop {
    let z = stable_power(rng, alpha);
    // Strict on the [0, 1) uniform, so an infinite proposal is never kept.
    if rng.random::<f64>() < (-((ln_scale + z) / alpha).exp()).exp() {
      break z;
    }
  };
  if m == 1.0 {
    let z = summand();
    return Frailty {
      ln_v: (ln_scale + z) / alpha,
      w: parent.w + z / theta_parent,
    };
  }
  // `ln Σ S_i = y_max/α + ln Σ e^{(y_i − y_max)/α}` for `ln S_i = y_i/α`, kept as a running maximum.
  let (mut y_max, mut sum) = (f64::NEG_INFINITY, 0.0);
  for _ in 0..m as u64 {
    let y = ln_scale + summand();
    if y > y_max {
      sum = sum * ((y_max - y) / alpha).exp() + 1.0;
      y_max = y;
    } else {
      sum += ((y - y_max) / alpha).exp();
    }
  }
  Frailty {
    ln_v: y_max / alpha + sum.ln(),
    w: y_max / theta_parent + sum.ln() / theta,
  }
}

impl Generator {
  /// The root frailty: Clayton `Γ(1/θ, 1)`, below shape 1 as `G U^{1/a}` with `G ~ Γ(1 + a)` (Marsaglia & Tsang 2000, the
  /// note on `α < 1`), whose logarithms stay exact where `V` underflows; Gumbel `S(1/θ)`, the constant 1 at `θ = 1`.
  pub(super) fn root_frailty<R: Rng + ?Sized>(self, rng: &mut R, theta: f64) -> Frailty {
    match self {
      Self::Clayton => {
        let shape = 1.0 / theta;
        // The sub-seed comes off `rng`: an `Unseeded` source would break the replay of every Clayton tree.
        let sub_seed = Deterministic::new(rng.random::<u64>());
        if shape >= 1.0 {
          let ln_v = SimdGamma::<f64>::new(shape, 1.0)
            .seeded(&sub_seed)
            .sample()
            .ln();
          return Frailty {
            ln_v,
            w: ln_v / theta,
          };
        }
        let ln_g = SimdGamma::<f64>::new(1.0 + shape, 1.0)
          .seeded(&sub_seed)
          .sample()
          .ln();
        let ln_u = (1.0 - rng.random::<f64>()).ln();
        Frailty {
          ln_v: ln_g + theta * ln_u,
          w: ln_g / theta + ln_u,
        }
      }
      Self::Gumbel if (theta - 1.0).abs() < 1e-12 => Frailty { ln_v: 0.0, w: 0.0 },
      Self::Gumbel => {
        let z = stable_power(rng, 1.0 / theta);
        Frailty {
          ln_v: theta * z,
          w: z,
        }
      }
    }
  }

  /// A child's frailty given its parent's, `α = θ_parent / θ` (Hofert & Mächler 2011, Table 3): Gumbel `V₀^{1/α} S(α)`,
  /// Clayton the tilted stable; `α = 1` is the no-nesting case, the parent's own `V`.
  pub(super) fn child_frailty<R: Rng + ?Sized>(
    self,
    rng: &mut R,
    parent: Frailty,
    theta_parent: f64,
    theta: f64,
  ) -> Frailty {
    let alpha = theta_parent / theta;
    if (alpha - 1.0).abs() < 1e-12 {
      return Frailty {
        ln_v: parent.ln_v,
        w: parent.w * alpha,
      };
    }
    match self {
      Self::Clayton => tilted_stable(rng, alpha, parent, theta_parent, theta),
      Self::Gumbel => {
        let z = stable_power(rng, alpha);
        Frailty {
          ln_v: (parent.ln_v + z) / alpha,
          w: parent.w + z / theta_parent,
        }
      }
    }
  }

  /// A leaf `ψ(E/V)`, `E ~ Exp(1)`, from the logarithms: Clayton `exp(−ln(1 + E/V)/θ)`, `exp(w − ln E/θ)` where `E/V`
  /// overflows; Gumbel `exp(−e^{ln E/θ − w})`.
  pub(super) fn leaf<R: Rng + ?Sized>(self, rng: &mut R, theta: f64, v: Frailty) -> f64 {
    let ln_e = (-rng.random::<f64>().clamp(1e-15, 1.0 - 1e-15).ln()).ln();
    let minus_ln_u = match self {
      Self::Clayton => {
        let x = ln_e - v.ln_v;
        if x == f64::INFINITY {
          ln_e / theta - v.w
        } else {
          softplus(x) / theta
        }
      }
      Self::Gumbel => (ln_e / theta - v.w).exp(),
    };
    (-minus_ln_u).exp()
  }
}

impl NestedArchimedean {
  pub(super) fn sample_with<R: Rng + ?Sized>(
    &self,
    root: &NacNode,
    n: usize,
    rng: &mut R,
  ) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, self.dim));
    let mut row = Array1::<f64>::zeros(self.dim);
    for r in 0..n {
      self.sample_node(rng, root, None, &mut row);
      out.row_mut(r).assign(&row);
    }
    out
  }

  /// Hofert & Mächler (2011), Algorithm 3.1: the node's frailty (from its parent's below the root), its leaves, then its
  /// subtrees.
  fn sample_node<R: Rng + ?Sized>(
    &self,
    rng: &mut R,
    node: &NacNode,
    parent: Option<(f64, Frailty)>,
    out: &mut Array1<f64>,
  ) {
    let v = match parent {
      None => self.generator.root_frailty(rng, node.theta),
      Some((theta_parent, frailty)) => {
        self
          .generator
          .child_frailty(rng, frailty, theta_parent, node.theta)
      }
    };
    for &j in &node.leaves {
      out[j] = self.generator.leaf(rng, node.theta, v);
    }
    for child in &node.children {
      self.sample_node(rng, child, Some((node.theta, v)), out);
    }
  }
}
