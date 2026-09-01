//! Bijection between the unconstrained optimisation coordinates and the
//! natural parameters, so neither the simplex nor L-BFGS can leave the
//! stationarity region:
//!
//! - GARCH / GJR: $\omega = e^{\theta_\omega}$; the persistence
//!   $s = \sigma(\theta_s) \in (0, 1)$ is split by a softmax whose first
//!   logit is pinned at zero — GARCH over the $p + q$ coefficients,
//!   $\alpha_i = s w_i$, $\beta_j = s w_{p+j}$; GJR over $2p + q$ slots,
//!   $\alpha_i = 2 s w_i$, $\alpha_i + \gamma_i = 2 s w_{p+i}$,
//!   $\beta_j = s w_{2p+j}$, which is exactly $\alpha_i \ge 0$,
//!   $\alpha_i + \gamma_i \ge 0$, $\beta_j \ge 0$ and
//!   $\sum\alpha_i + \tfrac12\sum\gamma_i + \sum\beta_j = s$.
//! - EGARCH: $\omega$, $\alpha_i$, $\gamma_i$ free; $\sum\beta_j =
//!   \tanh(\theta_s) \in (-1, 1)$ split by a softmax, so the $\beta_j$ share
//!   the sign of their sum.
//! - $\mu$ is free throughout.

use super::GarchKind;
use super::GarchSpec;
use super::MeanSpec;

fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}

fn logit(p: f64) -> f64 {
  (p / (1.0 - p)).ln()
}

/// Softmax over `logits.len() + 1` slots, the first logit pinned at zero.
fn softmax(logits: &[f64]) -> Vec<f64> {
  let max = logits.iter().copied().fold(0.0_f64, f64::max);
  let mut w: Vec<f64> = std::iter::once(0.0)
    .chain(logits.iter().copied())
    .map(|l| (l - max).exp())
    .collect();
  let sum: f64 = w.iter().sum();
  for x in &mut w {
    *x /= sum;
  }
  w
}

/// Logits of positive `weights` with the first pinned at zero.
fn logits(weights: &[f64]) -> Vec<f64> {
  weights[1..].iter().map(|w| (w / weights[0]).ln()).collect()
}

/// Unconstrained coordinates → natural parameters.
pub(super) fn to_natural(spec: &GarchSpec, theta: &[f64]) -> Vec<f64> {
  let mut out = Vec::with_capacity(theta.len());
  let mut i = 0;
  if spec.mean == MeanSpec::Constant {
    out.push(theta[0]);
    i = 1;
  }
  let (p, q) = (spec.p, spec.q);
  match spec.kind {
    GarchKind::Garch => {
      out.push(theta[i].exp());
      let s = sigmoid(theta[i + 1]);
      let w = softmax(&theta[i + 2..i + 1 + p + q]);
      out.extend(w.iter().map(|x| s * x));
    }
    GarchKind::GjrGarch => {
      out.push(theta[i].exp());
      let s = sigmoid(theta[i + 1]);
      let w = softmax(&theta[i + 2..i + 1 + 2 * p + q]);
      out.extend((0..p).map(|k| 2.0 * s * w[k]));
      out.extend((0..p).map(|k| 2.0 * s * w[p + k] - 2.0 * s * w[k]));
      out.extend((0..q).map(|k| s * w[2 * p + k]));
    }
    GarchKind::Egarch => {
      out.extend_from_slice(&theta[i..i + 1 + 2 * p]);
      if q > 0 {
        let s = theta[i + 1 + 2 * p].tanh();
        let w = softmax(&theta[i + 2 + 2 * p..i + 1 + 2 * p + q]);
        out.extend(w.iter().map(|x| s * x));
      }
    }
  }
  out
}

/// Natural parameters (strictly inside the region) → unconstrained
/// coordinates.
pub(super) fn to_unconstrained(spec: &GarchSpec, natural: &[f64]) -> Vec<f64> {
  let split = spec.split(natural);
  let mut theta = Vec::with_capacity(natural.len());
  if spec.mean == MeanSpec::Constant {
    theta.push(split.mu);
  }
  match spec.kind {
    GarchKind::Garch => {
      theta.push(split.omega.ln());
      let s = split.alpha.iter().sum::<f64>() + split.beta.iter().sum::<f64>();
      theta.push(logit(s));
      let weights: Vec<f64> = split
        .alpha
        .iter()
        .chain(split.beta)
        .map(|c| c / s)
        .collect();
      theta.extend(logits(&weights));
    }
    GarchKind::GjrGarch => {
      theta.push(split.omega.ln());
      let s = split.alpha.iter().sum::<f64>()
        + 0.5 * split.gamma.iter().sum::<f64>()
        + split.beta.iter().sum::<f64>();
      theta.push(logit(s));
      let weights: Vec<f64> = split
        .alpha
        .iter()
        .map(|a| a / (2.0 * s))
        .chain(
          split
            .alpha
            .iter()
            .zip(split.gamma)
            .map(|(a, g)| (a + g) / (2.0 * s)),
        )
        .chain(split.beta.iter().map(|b| b / s))
        .collect();
      theta.extend(logits(&weights));
    }
    GarchKind::Egarch => {
      theta.push(split.omega);
      theta.extend_from_slice(split.alpha);
      theta.extend_from_slice(split.gamma);
      if spec.q > 0 {
        let s = split.beta.iter().sum::<f64>();
        theta.push(s.atanh());
        let weights: Vec<f64> = split.beta.iter().map(|b| b / s).collect();
        theta.extend(logits(&weights));
      }
    }
  }
  theta
}

/// Typical magnitude of each natural parameter, used to scale the
/// finite-difference steps: $\sqrt{\bar\sigma^2}$ for $\mu$, $\bar\sigma^2$
/// for a level-recursion $\omega$, one for everything else.
pub(super) fn natural_scales(spec: &GarchSpec, backcast: f64) -> Vec<f64> {
  let mut scales = Vec::with_capacity(spec.n_params());
  if spec.mean == MeanSpec::Constant {
    scales.push(backcast.sqrt());
  }
  scales.push(if spec.kind == GarchKind::Egarch {
    1.0
  } else {
    backcast
  });
  scales.resize(spec.n_params(), 1.0);
  scales
}

/// The best point of a small persistence × shock-share grid of
/// variance-targeted starting values, in unconstrained coordinates.
pub(super) fn best_start<F: Fn(&[f64]) -> f64>(
  spec: &GarchSpec,
  mean: f64,
  backcast: f64,
  objective: F,
) -> Vec<f64> {
  let mut best: Option<(f64, Vec<f64>)> = None;
  for natural in starting_grid(spec, mean, backcast) {
    let theta = to_unconstrained(spec, &natural);
    let cost = objective(&theta);
    let better = match &best {
      Some((c, _)) => cost < *c,
      None => true,
    };
    if better {
      best = Some((cost, theta));
    }
  }
  best.expect("the starting grid is never empty").1
}

fn starting_grid(spec: &GarchSpec, mean: f64, backcast: f64) -> Vec<Vec<f64>> {
  const PERSISTENCE: [f64; 4] = [0.80, 0.90, 0.95, 0.98];
  const LEVEL_SHOCK: [f64; 4] = [0.03, 0.08, 0.15, 0.25];
  const LOG_SHOCK: [f64; 3] = [0.05, 0.15, 0.30];
  const ARCH_ONLY: [f64; 4] = [0.10, 0.30, 0.50, 0.80];
  let mut grid = Vec::new();
  match spec.kind {
    GarchKind::Garch | GarchKind::GjrGarch => {
      if spec.q == 0 {
        for &a in &ARCH_ONLY {
          grid.push(natural_start(spec, mean, backcast * (1.0 - a), a, 0.0));
        }
      } else {
        for &s in &PERSISTENCE {
          for &a in &LEVEL_SHOCK {
            grid.push(natural_start(spec, mean, backcast * (1.0 - s), a, s - a));
          }
        }
      }
    }
    GarchKind::Egarch => {
      let log_backcast = backcast.ln();
      if spec.q == 0 {
        for &a in &LOG_SHOCK {
          grid.push(natural_start(spec, mean, log_backcast, a, 0.0));
        }
      } else {
        for &s in &PERSISTENCE {
          for &a in &LOG_SHOCK {
            grid.push(natural_start(spec, mean, (1.0 - s) * log_backcast, a, s));
          }
        }
      }
    }
  }
  grid
}

/// Natural start with the shock total spread evenly over the $\alpha_i$,
/// the persistence remainder over the $\beta_j$, and every $\gamma_i$ zero.
fn natural_start(
  spec: &GarchSpec,
  mean: f64,
  omega: f64,
  alpha_total: f64,
  beta_total: f64,
) -> Vec<f64> {
  let mut v = Vec::with_capacity(spec.n_params());
  if spec.mean == MeanSpec::Constant {
    v.push(mean);
  }
  v.push(omega);
  v.extend(std::iter::repeat_n(alpha_total / spec.p as f64, spec.p));
  if spec.has_gamma() {
    v.extend(std::iter::repeat_n(0.0, spec.p));
  }
  if spec.q > 0 {
    v.extend(std::iter::repeat_n(beta_total / spec.q as f64, spec.q));
  }
  v
}
