//! Bijection from unconstrained coordinates onto each kind's region, which no step can leave.
//!
//! $$
//! \begin{aligned}
//! \text{GARCH}:&\quad \omega = e^{\theta_\omega},\quad s = \sigma(\theta_s) \in (0, 1),\quad
//! \alpha_i = s w_i,\quad \beta_j = s w_{p+j},\\
//! \text{GJR}:&\quad \alpha_i = 2 s w_i,\quad \alpha_i + \gamma_i = 2 s w_{p+i},\quad
//! \beta_j = s w_{2p+j},\quad \sum\alpha_i + \tfrac12\sum\gamma_i + \sum\beta_j = s,\\
//! \text{EGARCH}:&\quad \alpha_i \pm \gamma_i = e^{\theta_a} v_{2i - 1},\ e^{\theta_a} v_{2i},\quad
//! \sum\beta_j = \sigma(\theta_s),\quad \beta_{j+1} = \beta_j\,\sigma(\theta_j),
//! \end{aligned}
//! $$
//!
//! `w` and `v` are sticks: each logit is the share a weight takes of what the earlier ones leave,
//! the first weight keeping the rest. `μ` and EGARCH's `ω` are free.
//! EGARCH(1,1)'s region is Straumann & Mikosch's (2006, Example 3.8); above it the crate's own.

use super::GarchKind;
use super::GarchSpec;
use super::MeanSpec;
use crate::optim::search::Coordinate;
use crate::optim::search::LOGIT;
use crate::traits::RealExt;

fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}

fn logit(p: f64) -> f64 {
  (p / (1.0 - p)).ln()
}

/// Softmax over `logits.len() + 1` slots, the first logit pinned at zero: a stick's one break.
fn softmax(logits: &[f64]) -> Vec<f64> {
  let max = f64::max_or_nan_fold(std::iter::once(0.0).chain(logits.iter().copied()));
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

/// A stick broken by `logits`: weight `k` takes the share `σ(l_k)` of what the earlier ones leave,
/// and the first keeps the rest. One weight vanishes alone at an end of one logit.
fn stick(logits: &[f64]) -> Vec<f64> {
  let mut weights = vec![1.0];
  for logit in logits {
    let split = softmax(&[*logit]);
    weights.push(weights[0] * split[1]);
    weights[0] *= split[0];
  }
  weights
}

/// The logits that break a stick into positive `weights`.
fn breaks(weights: &[f64]) -> Vec<f64> {
  let rest = |k: usize| weights[k + 1..].iter().fold(weights[0], |sum, w| sum + w);
  (1..weights.len())
    .map(|k| (weights[k] / rest(k)).ln())
    .collect()
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
      let w = stick(&theta[i + 2..i + 1 + p + q]);
      out.extend(w.iter().map(|x| s * x));
    }
    GarchKind::GjrGarch => {
      out.push(theta[i].exp());
      let s = sigmoid(theta[i + 1]);
      let w = stick(&theta[i + 2..i + 1 + 2 * p + q]);
      out.extend((0..p).map(|k| 2.0 * s * w[k]));
      out.extend((0..p).map(|k| 2.0 * s * w[p + k] - 2.0 * s * w[k]));
      out.extend((0..q).map(|k| s * w[2 * p + k]));
    }
    GarchKind::Egarch => {
      out.push(theta[i]);
      let total = theta[i + 1].exp();
      let w = stick(&theta[i + 2..i + 1 + 2 * p]);
      out.extend((0..p).map(|k| 0.5 * total * (w[2 * k] + w[2 * k + 1])));
      out.extend((0..p).map(|k| 0.5 * total * (w[2 * k] - w[2 * k + 1])));
      if q > 0 {
        // `β₁ ≥ β₂ ≥ … ≥ 0`: at a frozen news slope the companion polynomial's margin at `−1`,
        // `1 + β₁ − β₂ + β₃ − …`, is then at least one, as one lag's `1 + β` is.
        let ratios = theta[i + 2 + 2 * p..i + 1 + 2 * p + q].iter();
        let mut shares = vec![1.0];
        for ratio in ratios {
          shares.push(shares[shares.len() - 1] * sigmoid(*ratio));
        }
        let s = sigmoid(theta[i + 1 + 2 * p]) / shares.iter().sum::<f64>();
        out.extend(shares.iter().map(|x| s * x));
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
      theta.extend(breaks(&weights));
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
      theta.extend(breaks(&weights));
    }
    GarchKind::Egarch => {
      theta.push(split.omega);
      let lags = split.alpha.iter().zip(split.gamma);
      let slopes: Vec<f64> = lags.flat_map(|(a, g)| [a + g, a - g]).collect();
      let total = slopes.iter().sum::<f64>();
      theta.push(total.ln());
      theta.extend(breaks(
        &slopes.iter().map(|x| x / total).collect::<Vec<_>>(),
      ));
      if spec.q > 0 {
        let s = split.beta.iter().sum::<f64>();
        theta.push(logit(s));
        theta.extend(split.beta.windows(2).map(|pair| logit(pair[1] / pair[0])));
      }
    }
  }
  theta
}

/// The search's coordinates and their stops' names: the parameters', `persistence`, and the
/// sums and differences a share's ends bound, and the weights that vanish together at an end.
pub(super) fn coordinates(
  spec: &GarchSpec,
  mean: f64,
  backcast: f64,
) -> (Vec<Coordinate>, Vec<String>) {
  let mut names = spec.param_names();
  let mut coordinates = Vec::with_capacity(names.len());
  if spec.mean == MeanSpec::Constant {
    coordinates.push(Coordinate::location(mean, backcast.sqrt()));
  }
  let first = coordinates.len() + 1;
  let persistence = names.len();
  names.push("persistence".to_string());
  let (p, q) = (spec.p, spec.q);
  let weights = match spec.kind {
    GarchKind::Garch => {
      coordinates.push(Coordinate::log(backcast.ln()));
      coordinates.push(Coordinate::share(LOGIT, persistence, persistence));
      (first..first + p + q).collect::<Vec<_>>()
    }
    GarchKind::GjrGarch => {
      coordinates.push(Coordinate::log(backcast.ln()));
      coordinates.push(Coordinate::share(LOGIT, persistence, persistence));
      let sums = names.len()..names.len() + p;
      names.extend((1..=p).map(|i| format!("alpha[{i}] + gamma[{i}]")));
      (first..first + p)
        .chain(sums)
        .chain(first + 2 * p..first + 2 * p + q)
        .collect()
    }
    GarchKind::Egarch => {
      coordinates.push(Coordinate::free());
      let betas = first + 2 * p;
      let slopes = (names.len()..names.len() + 2 * p).collect::<Vec<_>>();
      names.extend(
        (1..=p).flat_map(|i| ['+', '-'].map(|sign| format!("alpha[{i}] {sign} gamma[{i}]"))),
      );
      // With no news curve left the variance is constant, whatever `β` is.
      let size = Coordinate::log(0.0);
      coordinates.push(if q > 0 { size.vanishing(betas) } else { size });
      coordinates.extend(shares(&slopes, &mut names));
      if q > 0 {
        coordinates.push(Coordinate::share(LOGIT, betas, persistence));
      }
      // Each ratio `β_{k+1}/β_k` ends where the later lag vanishes or the two are equal.
      for k in 1..q {
        names.push(format!("beta[{k}] - beta[{}]", k + 1));
        coordinates.push(Coordinate::share(LOGIT, betas + k, names.len() - 1));
      }
      Vec::new()
    }
  };
  coordinates.extend(shares(&weights, &mut names));
  (coordinates, names)
}

/// A stick's logits as coordinates: the low end of logit `k` bounds weight `k`; at its high end
/// the first weight and every later one vanish, and the end is named after each of them.
fn shares(weights: &[usize], names: &mut Vec<String>) -> Vec<Coordinate> {
  let share = |k: usize| {
    let gone = [&weights[..1], &weights[k + 1..]].concat();
    let high = match gone[..] {
      [alone] => alone,
      _ => {
        let each = gone.iter().map(|name| names[*name].as_str());
        names.push(each.collect::<Vec<_>>().join(" and "));
        names.len() - 1
      }
    };
    Coordinate::share(LOGIT, weights[k], high)
  };
  (1..weights.len()).map(share).collect()
}

/// Typical magnitude of each natural parameter, the scale of its finite-difference step:
/// $\sqrt{\bar\sigma^2}$ for $\mu$, $\bar\sigma^2$ for a level-recursion $\omega$, one otherwise.
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

/// Variance-targeted starts in unconstrained coordinates, one group to a persistence level.
pub(super) fn starts(spec: &GarchSpec, mean: f64, backcast: f64) -> Vec<Vec<Vec<f64>>> {
  const PERSISTENCE: [f64; 4] = [0.80, 0.90, 0.95, 0.98];
  const LEVEL_SHOCK: [f64; 4] = [0.03, 0.08, 0.15, 0.25];
  const LOG_SHOCK: [f64; 3] = [0.05, 0.15, 0.30];
  const ARCH_ONLY: [f64; 4] = [0.10, 0.30, 0.50, 0.80];
  let level = |shocks: &[f64], start: &dyn Fn(f64) -> Vec<f64>| {
    let thetas = shocks.iter().map(|&a| to_unconstrained(spec, &start(a)));
    thetas.collect::<Vec<_>>()
  };
  let log_backcast = backcast.ln();
  let levels = |shocks: &[f64], start: &dyn Fn(f64, f64) -> Vec<f64>| {
    let groups = PERSISTENCE.iter().map(|&s| level(shocks, &|a| start(a, s)));
    groups.collect::<Vec<_>>()
  };
  match (spec.kind, spec.q) {
    (GarchKind::Egarch, 0) => {
      vec![level(&LOG_SHOCK, &|a| {
        natural_start(spec, mean, log_backcast, a, 0.0)
      })]
    }
    (GarchKind::Egarch, _) => levels(&LOG_SHOCK, &|a, s| {
      natural_start(spec, mean, (1.0 - s) * log_backcast, a, s)
    }),
    (_, 0) => vec![level(&ARCH_ONLY, &|a| {
      natural_start(spec, mean, backcast * (1.0 - a), a, 0.0)
    })],
    _ => levels(&LEVEL_SHOCK, &|a, s| {
      natural_start(spec, mean, backcast * (1.0 - s), a, s - a)
    }),
  }
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
  // EGARCH's lags start non-increasing, each half the one before; the others start equal.
  let halving = if spec.kind == GarchKind::Egarch {
    0.5_f64
  } else {
    1.0
  };
  let shares = (0..spec.q).map(|k| halving.powi(k as i32));
  let total = shares.clone().sum::<f64>();
  v.extend(shares.map(|share| beta_total * share / total));
  v
}
