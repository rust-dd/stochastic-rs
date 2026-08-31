//! Monte Carlo basket pricer and the average it is struck against.
//!
//! Split out of `basket.rs` when the file crossed the 600-line cap; a pure
//! move, with the public paths preserved by the `pub use` in the parent.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use rayon::prelude::*;

use crate::OptionType;
use crate::mc::McEstimate;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::FloatExt;

/// Which average the basket is struck against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasketAverageType {
  /// $\sum_i w_i S_{i,T}$.
  Arithmetic,
  /// $\prod_i S_{i,T}^{w_i}$.
  Geometric,
}

/// Monte Carlo basket option pricer. Supports arithmetic and geometric
/// payoffs. Uses a Cholesky factorization for the correlation factor and
/// runs on the pure-Rust `faer`, in every build.
///
/// The struct holds **model, contract and method state only** — the
/// per-asset volatilities and their correlation, the basket weights and
/// which average they are applied to, and the Monte Carlo path count. The
/// spot vector, the strike, the rate, the dividend-yield vector and the
/// maturity are the pricing *query* and travel as arguments.
///
/// [`BasketAverageType`] sits beside the weights because it is the same kind
/// of thing: both say what the option is written on, and neither moves when
/// the market does.
///
/// `n_paths` is neither model, contract nor query but a convergence control,
/// and it sits on the struct for the same reason
/// `GbmMalliavinPricer` keeps its own path and step counts there.
///
/// The dimension and positive-definiteness checks stay in
/// [`try_price`](Self::try_price) rather than moving to [`new`](Self::new):
/// `try_price` is the only advertised way to surface either as an `Err`, and
/// a constructor that panicked on them first would leave it nothing to
/// report.
#[derive(Debug, Clone)]
pub struct McBasketPricer {
  /// Weights — a term of the contract, not a market quote.
  pub weights: Array1<f64>,
  /// Average type — likewise a term of the contract.
  pub avg_type: BasketAverageType,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Correlation matrix.
  pub rho: Array2<f64>,
  /// Number of MC paths.
  pub n_paths: usize,
}

impl McBasketPricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if any `sigma[i]` is negative or `NaN` — not a volatility
  /// - if `n_paths` is `0`
  ///
  /// `rho`, the weights and the dimensions are **deliberately unchecked
  /// here**, for the reason given on
  /// [`McRainbowPricer::new`](crate::pricing::rainbow::McRainbowPricer::new):
  /// [`try_price`](Self::try_price) is the only advertised way to surface
  /// a non-SPD correlation or a dimension mismatch as an `Err`, and a
  /// panicking constructor would leave it nothing to report. The two
  /// closed-form basket pricers, which have no `try_price`, do get the
  /// correlation range check and the shape check.
  pub fn new(
    weights: Array1<f64>,
    avg_type: BasketAverageType,
    sigma: Array1<f64>,
    rho: Array2<f64>,
    n_paths: usize,
  ) -> Self {
    for (i, &v) in sigma.iter().enumerate() {
      assert!(
        v >= 0.0,
        "McBasketPricer::new: sigma[{i}] must be a non-negative volatility (got {v})"
      );
    }
    assert!(
      n_paths >= 1,
      "McBasketPricer::new: n_paths must be at least 1 (got {n_paths})"
    );
    Self {
      weights,
      avg_type,
      sigma,
      rho,
      n_paths,
    }
  }

  /// Falliable variant of [`Self::price_option`] that surfaces invalid inputs
  /// (non-SPD correlation, dimension mismatch) as an `Err` instead of a panic.
  pub fn try_price(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
    option_type: OptionType,
  ) -> anyhow::Result<McEstimate<f64>> {
    let n_assets = s.len();
    if self.rho.shape() != [n_assets, n_assets] {
      anyhow::bail!(
        "rho shape {:?} does not match n_assets={n_assets}",
        self.rho.shape()
      );
    }
    if self.weights.len() != n_assets || self.sigma.len() != n_assets || q.len() != n_assets {
      anyhow::bail!(
        "weights/sigma/q lengths must match n_assets={n_assets} (got {}/{}/{})",
        self.weights.len(),
        self.sigma.len(),
        q.len()
      );
    }
    if !crate::linalg::is_spd_t(&self.rho) {
      anyhow::bail!("correlation matrix is not positive definite");
    }
    Ok(self.price_option(s, k, r, q, tau, option_type))
  }

  /// Price either leg at one query point with a single simulation.
  ///
  /// # Panics
  /// If the correlation matrix is not positive definite — call
  /// [`try_price`](Self::try_price) to handle that gracefully.
  pub fn price_option(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
    option_type: OptionType,
  ) -> McEstimate<f64> {
    let n_assets = s.len();
    let l: Array2<f64> = crate::linalg::spd_cholesky_lower(&self.rho).expect(
      "correlation matrix must be positive definite — call try_price() to handle this gracefully",
    );
    let drifts: Vec<f64> = (0..n_assets)
      .map(|i| (r - q[i] - 0.5 * self.sigma[i] * self.sigma[i]) * tau)
      .collect();
    let vols: Vec<f64> = (0..n_assets).map(|i| self.sigma[i] * tau.sqrt()).collect();
    let phi = match option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    let n_paths = self.n_paths;

    // Generate one big block of standard normals using the project's
    // SIMD ziggurat path, then map paths in parallel.
    let mut all_z = vec![0.0_f64; n_paths * n_assets];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let payoff_of = |p: usize| -> f64 {
      let z = &all_z[p * n_assets..(p + 1) * n_assets];
      let mut zc = vec![0.0_f64; n_assets];
      for i in 0..n_assets {
        let mut acc = 0.0;
        for j in 0..=i {
          acc += l[[i, j]] * z[j];
        }
        zc[i] = acc;
      }
      let s_t: Vec<f64> = (0..n_assets)
        .map(|i| s[i] * (drifts[i] + vols[i] * zc[i]).exp())
        .collect();
      let basket = match self.avg_type {
        BasketAverageType::Arithmetic => {
          (0..n_assets).map(|i| self.weights[i] * s_t[i]).sum::<f64>()
        }
        BasketAverageType::Geometric => {
          let mut log_g = 0.0;
          for i in 0..n_assets {
            log_g += self.weights[i] * s_t[i].ln();
          }
          log_g.exp()
        }
      };
      (phi * (basket - k)).max(0.0)
    };
    let sum: f64 = (0..n_paths).into_par_iter().map(&payoff_of).sum();
    let sum_sq: f64 = (0..n_paths)
      .into_par_iter()
      .map(|p| {
        let y = payoff_of(p);
        y * y
      })
      .sum();

    let discount = (-r * tau).exp();
    McEstimate {
      mean: discount * sum / n_paths as f64,
      std_err: discount * std_err_from_sums(sum, sum_sq, n_paths),
      n_samples: n_paths,
    }
  }

  /// Price the basket call at one query point.
  pub fn price_call(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> McEstimate<f64> {
    self.price_option(s, k, r, q, tau, OptionType::Call)
  }

  /// Price the basket put at one query point.
  pub fn price_put(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> McEstimate<f64> {
    self.price_option(s, k, r, q, tau, OptionType::Put)
  }
}
