//! Conditional-variance recursions and the Gaussian quasi-log-likelihood,
//! shared by the fitter and the finite-difference inference.

use super::GarchKind;
use super::GarchSpec;
use super::MeanSpec;

/// $\mathbb E|z| = \sqrt{2/\pi}$ for a standard normal $z$: the EGARCH
/// centring constant.
const EXPECTED_ABS_NORMAL: f64 = std::f64::consts::FRAC_2_SQRT_PI / std::f64::consts::SQRT_2;

/// $\log 2\pi$.
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Natural parameters split by role, borrowed from the flat
/// `[mu, omega, alpha, gamma, beta]` layout.
pub(super) struct Params<'a> {
  pub mu: f64,
  pub omega: f64,
  pub alpha: &'a [f64],
  pub gamma: &'a [f64],
  pub beta: &'a [f64],
}

impl GarchSpec {
  pub(super) fn split<'a>(&self, natural: &'a [f64]) -> Params<'a> {
    let mut i = 0;
    let mu = match self.mean {
      MeanSpec::Constant => {
        i += 1;
        natural[0]
      }
      MeanSpec::Zero => 0.0,
    };
    let omega = natural[i];
    i += 1;
    let alpha = &natural[i..i + self.p];
    i += self.p;
    let gamma = if self.has_gamma() {
      let g = &natural[i..i + self.p];
      i += self.p;
      g
    } else {
      &natural[0..0]
    };
    let beta = &natural[i..i + self.q];
    Params {
      mu,
      omega,
      alpha,
      gamma,
      beta,
    }
  }
}

/// Fills `resid` with $\varepsilon_t = r_t - \mu$ and `sigma2` with the
/// recursion's conditional variances, `backcast` seeding every pre-sample
/// term.
pub(super) fn variance_path(
  spec: &GarchSpec,
  natural: &[f64],
  returns: &[f64],
  backcast: f64,
  resid: &mut [f64],
  sigma2: &mut [f64],
) {
  let p = spec.split(natural);
  for (e, r) in resid.iter_mut().zip(returns) {
    *e = r - p.mu;
  }
  match spec.kind {
    GarchKind::Garch | GarchKind::GjrGarch => level_recursion(
      spec.kind == GarchKind::GjrGarch,
      &p,
      resid,
      backcast,
      sigma2,
    ),
    GarchKind::Egarch => log_recursion(&p, resid, backcast, sigma2),
  }
}

fn level_recursion(gjr: bool, p: &Params, resid: &[f64], backcast: f64, sigma2: &mut [f64]) {
  for t in 0..resid.len() {
    let mut v = p.omega;
    for (i, a) in p.alpha.iter().enumerate() {
      let lag = i + 1;
      v += a
        * if t >= lag {
          resid[t - lag] * resid[t - lag]
        } else {
          backcast
        };
    }
    if gjr {
      for (i, g) in p.gamma.iter().enumerate() {
        let lag = i + 1;
        v += g
          * if t < lag {
            0.5 * backcast
          } else if resid[t - lag] < 0.0 {
            resid[t - lag] * resid[t - lag]
          } else {
            0.0
          };
      }
    }
    for (j, b) in p.beta.iter().enumerate() {
      let lag = j + 1;
      v += b * if t >= lag { sigma2[t - lag] } else { backcast };
    }
    sigma2[t] = v;
  }
}

fn log_recursion(p: &Params, resid: &[f64], backcast: f64, sigma2: &mut [f64]) {
  let n = resid.len();
  let log_backcast = backcast.ln();
  let mut log_sigma2 = vec![0.0; n];
  for t in 0..n {
    let mut v = p.omega;
    for i in 0..p.alpha.len() {
      let lag = i + 1;
      if t >= lag {
        let z = resid[t - lag] / sigma2[t - lag].sqrt();
        v += p.alpha[i] * (z.abs() - EXPECTED_ABS_NORMAL) + p.gamma[i] * z;
      }
    }
    for (j, b) in p.beta.iter().enumerate() {
      let lag = j + 1;
      v += b
        * if t >= lag {
          log_sigma2[t - lag]
        } else {
          log_backcast
        };
    }
    log_sigma2[t] = v;
    sigma2[t] = v.exp();
  }
}

/// Per-observation terms $\ell_t = -\tfrac12[\log 2\pi + \log\sigma_t^2 +
/// \varepsilon_t^2/\sigma_t^2]$ and their sum; the sum is `-inf` when some
/// variance is not positive and finite.
pub(super) fn log_likelihood_terms(
  spec: &GarchSpec,
  natural: &[f64],
  returns: &[f64],
  backcast: f64,
) -> (f64, Vec<f64>) {
  let n = returns.len();
  let mut resid = vec![0.0; n];
  let mut sigma2 = vec![0.0; n];
  variance_path(spec, natural, returns, backcast, &mut resid, &mut sigma2);
  let mut terms = vec![0.0; n];
  let mut total = 0.0;
  for t in 0..n {
    let v = sigma2[t];
    if !(v > 0.0 && v.is_finite()) {
      return (f64::NEG_INFINITY, terms);
    }
    terms[t] = -0.5 * (LN_2PI + v.ln() + resid[t] * resid[t] / v);
    total += terms[t];
  }
  (total, terms)
}

/// The summed quasi-log-likelihood alone.
pub(super) fn total_log_likelihood(
  spec: &GarchSpec,
  natural: &[f64],
  returns: &[f64],
  backcast: f64,
) -> f64 {
  log_likelihood_terms(spec, natural, returns, backcast).0
}
