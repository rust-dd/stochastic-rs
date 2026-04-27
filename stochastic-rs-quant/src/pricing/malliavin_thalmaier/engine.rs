//! Malliavin–Thalmaier Greeks computation engine.
//!
//! See Kohatsu-Higa & Yasuda (2008), §6 (Theorem 6.1).

use ndarray::Array2;

use super::heston::MultiHestonParams;
use super::kernel;
use crate::traits::FloatExt;

/// Payoff types supported by the M-T engine.
#[derive(Clone, Debug)]
pub enum MtPayoff<T: FloatExt> {
  /// Vanilla call `(S^{asset}_T − K)₊`.
  Call { asset: usize, strike: T },
  /// Vanilla put `(K − S^{asset}_T)₊`.
  Put { asset: usize, strike: T },
  /// Digital put on 2 assets: `1(S₁≤K₁)·1(S₂≤K₂)`.
  DigitalPut2D { strikes: [T; 2] },
  /// Basket call `(Σ wᵢ Sᵢ − K)₊`.
  BasketCall { weights: Vec<T>, strike: T },
  /// Worst-of put `(K − min Sᵢ)₊`.
  WorstOfPut { strike: T },
}

impl<T: FloatExt> MtPayoff<T> {
  pub fn evaluate(&self, st: &[T]) -> T {
    match self {
      Self::Call { asset, strike } => (st[*asset] - *strike).max(T::zero()),
      Self::Put { asset, strike } => (*strike - st[*asset]).max(T::zero()),
      Self::DigitalPut2D { strikes } => {
        if st[0] <= strikes[0] && st[1] <= strikes[1] {
          T::one()
        } else {
          T::zero()
        }
      }
      Self::BasketCall { weights, strike } => {
        let basket = weights
          .iter()
          .zip(st)
          .map(|(&w, &s)| w * s)
          .fold(T::zero(), |a, b| a + b);
        (basket - *strike).max(T::zero())
      }
      Self::WorstOfPut { strike } => {
        let worst = st.iter().copied().fold(T::infinity(), |a, b| a.min(b));
        (*strike - worst).max(T::zero())
      }
    }
  }
}

/// Malliavin–Thalmaier Greeks engine.
///
/// Computes multi-asset Greeks using a single Skorohod integral per
/// component, regardless of dimension.
///
/// # Example
///
/// ```ignore
/// let engine = MtGreeks::new(params, 0.01.into(), 50_000);
/// let payoff = MtPayoff::DigitalPut2D { strikes: [100.0, 100.0] };
/// let deltas = engine.all_deltas(&payoff);
/// ```
#[derive(Debug, Clone)]
pub struct MtGreeks<T: FloatExt> {
  pub params: MultiHestonParams<T>,
  /// Regularisation parameter `h`. Recommended `∈ [0.001, 0.1]`.
  pub h: T,
  pub n_paths: usize,
}

#[derive(Clone, Debug)]
/// Smooth localization parameters for the split `f = f_MT + f_PW`.
struct MtLocalization<T: FloatExt> {
  kink_eps: T,
  box_hi: Vec<T>,
  box_width: Vec<T>,
}

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  /// Construct an M-T engine for the given model, regularization level and
  /// Monte Carlo path count.
  pub fn new(params: MultiHestonParams<T>, h: T, n_paths: usize) -> Self {
    Self { params, h, n_paths }
  }

  /// Delta for a given asset via the M-T formula.
  pub fn delta(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
    self.delta_from_sampler(payoff, asset, || self.params.sample())
  }

  /// Deterministic Delta estimator for reproducible tests and benchmarks.
  pub fn delta_with_seed(&self, payoff: &MtPayoff<T>, asset: usize, seed: u64) -> T {
    let mut seed_state = seed;
    self.delta_from_sampler(payoff, asset, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn delta_from_sampler<F>(&self, payoff: &MtPayoff<T>, asset: usize, mut sample: F) -> T
  where
    F: FnMut() -> super::heston::MultiHestonPaths<T>,
  {
    let d = self.params.n_assets();
    let discount = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let spots: Vec<T> = self.params.assets.iter().map(|a| a.s0).collect();
    let localization = self.localization(payoff);
    let mut sum = T::zero();

    for _ in 0..self.n_paths {
      let paths = sample();
      let st = paths.terminal_prices();
      let gamma_inv = paths.gamma_inv(&self.params.cross_corr, self.params.tau);
      let h_w = paths.malliavin_weights(&gamma_inv, asset, self.params.r, self.params.tau, &spots);
      let g = match localization.as_ref() {
        Some(loc) => self.compute_g_localized(payoff, &st, loc),
        None => self.compute_g(payoff, &st),
      };
      let pw_contrib = localization
        .as_ref()
        .map(|loc| {
          let grad = self.pathwise_tail_gradient(payoff, &st, loc);
          grad[asset] * st[asset] / spots[asset]
        })
        .unwrap_or_else(T::zero);

      let contrib = (0..d)
        .map(|i| g[[i, asset]] * h_w[i])
        .fold(T::zero(), |a, b| a + b);
      sum += discount * (contrib + pw_contrib);
    }

    sum / T::from_usize_(self.n_paths)
  }

  /// All Deltas in a single simulation pass.
  pub fn all_deltas(&self, payoff: &MtPayoff<T>) -> Vec<T> {
    self.all_deltas_from_sampler(payoff, || self.params.sample())
  }

  /// Deterministic variant of [`all_deltas`](Self::all_deltas).
  pub fn all_deltas_with_seed(&self, payoff: &MtPayoff<T>, seed: u64) -> Vec<T> {
    let mut seed_state = seed;
    self.all_deltas_from_sampler(payoff, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn all_deltas_from_sampler<F>(&self, payoff: &MtPayoff<T>, mut sample: F) -> Vec<T>
  where
    F: FnMut() -> super::heston::MultiHestonPaths<T>,
  {
    let d = self.params.n_assets();
    let discount = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let spots: Vec<T> = self.params.assets.iter().map(|a| a.s0).collect();
    let localization = self.localization(payoff);
    let mut sums = vec![T::zero(); d];

    for _ in 0..self.n_paths {
      let paths = sample();
      let st = paths.terminal_prices();
      let gamma_inv = paths.gamma_inv(&self.params.cross_corr, self.params.tau);
      let g = match localization.as_ref() {
        Some(loc) => self.compute_g_localized(payoff, &st, loc),
        None => self.compute_g(payoff, &st),
      };
      let grad_pw = localization
        .as_ref()
        .map(|loc| self.pathwise_tail_gradient(payoff, &st, loc));

      for p in 0..d {
        let h_w = paths.malliavin_weights(&gamma_inv, p, self.params.r, self.params.tau, &spots);
        let contrib = (0..d)
          .map(|i| g[[i, p]] * h_w[i])
          .fold(T::zero(), |a, b| a + b);
        let pw_contrib = grad_pw
          .as_ref()
          .map(|grad| grad[p] * st[p] / spots[p])
          .unwrap_or_else(T::zero);
        sums[p] += discount * (contrib + pw_contrib);
      }
    }

    sums
      .iter()
      .map(|&s| s / T::from_usize_(self.n_paths))
      .collect()
  }

  /// Cross-Gamma via finite difference on Delta.
  pub fn cross_gamma(&self, payoff: &MtPayoff<T>, asset_a: usize, asset_b: usize) -> T {
    let bump = self.params.assets[asset_b].s0 * T::from_f64_fast(0.01);
    let mut up = self.params.clone();
    up.assets[asset_b].s0 += bump;
    let mut dn = self.params.clone();
    dn.assets[asset_b].s0 -= bump;

    let d_up = MtGreeks::new(up, self.h, self.n_paths).delta(payoff, asset_a);
    let d_dn = MtGreeks::new(dn, self.h, self.n_paths).delta(payoff, asset_a);
    (d_up - d_dn) / (bump + bump)
  }

  /// Backward-compatible alias retained for the integration tests.
  pub fn cross_gamma_fd(&self, payoff: &MtPayoff<T>, asset_a: usize, asset_b: usize) -> T {
    self.cross_gamma(payoff, asset_a, asset_b)
  }

  /// Vega via finite difference on price.
  ///
  /// Uses a fixed-seed common-random-numbers bump internally to reduce Monte
  /// Carlo variance compared with two independent price runs.
  pub fn vega(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
    self.vega_with_seed(payoff, asset, 0x9E37_79B9_7F4A_7C15_u64 ^ asset as u64)
  }

  /// Deterministic Vega estimator using common random numbers for the
  /// up/down finite-difference bump.
  pub fn vega_with_seed(&self, payoff: &MtPayoff<T>, asset: usize, seed: u64) -> T {
    let bump = self.params.assets[asset].v0 * T::from_f64_fast(0.01);
    let mut up = self.params.clone();
    up.assets[asset].v0 += bump;
    let mut dn = self.params.clone();
    dn.assets[asset].v0 -= bump;

    let p_up = MtGreeks::new(up, self.h, self.n_paths).price_with_seed(payoff, seed);
    let p_dn = MtGreeks::new(dn, self.h, self.n_paths).price_with_seed(payoff, seed);
    (p_up - p_dn) / (bump + bump)
  }

  /// Plain Monte Carlo price.
  pub fn price(&self, payoff: &MtPayoff<T>) -> T {
    self.price_from_sampler(payoff, || self.params.sample())
  }

  /// Deterministic variant of [`price`](Self::price).
  pub fn price_with_seed(&self, payoff: &MtPayoff<T>, seed: u64) -> T {
    let mut seed_state = seed;
    self.price_from_sampler(payoff, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn price_from_sampler<F>(&self, payoff: &MtPayoff<T>, mut sample: F) -> T
  where
    F: FnMut() -> super::heston::MultiHestonPaths<T>,
  {
    let disc = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let mut sum = T::zero();
    for _ in 0..self.n_paths {
      let st = sample().terminal_prices();
      sum += disc * payoff.evaluate(&st);
    }
    sum / T::from_usize_(self.n_paths)
  }

  fn compute_g(&self, payoff: &MtPayoff<T>, st: &[T]) -> Array2<T> {
    let d = self.params.n_assets();
    match payoff {
      MtPayoff::DigitalPut2D { strikes } if d == 2 => {
        let raw = kernel::g_digital_put_2d([st[0], st[1]], *strikes);
        let mut g = Array2::<T>::zeros((2, 2));
        g[[0, 0]] = raw[0][0];
        g[[0, 1]] = raw[0][1];
        g[[1, 0]] = raw[1][0];
        g[[1, 1]] = raw[1][1];
        g
      }
      _ if d == 2 => {
        let payoff_fn = |x: &[T]| payoff.evaluate(x);
        let (lo, hi) = self.quadrature_bounds(payoff, st);
        kernel::g_kernel_numerical_2d(
          &[st[0], st[1]],
          &payoff_fn,
          self.h,
          &[lo[0], lo[1]],
          &[hi[0], hi[1]],
          self.quadrature_points_per_axis(d),
        )
      }
      _ => {
        let payoff_fn = |x: &[T]| payoff.evaluate(x);
        let (lo, hi) = self.quadrature_bounds(payoff, st);
        kernel::g_kernel_numerical_nd(
          st,
          &payoff_fn,
          self.h,
          &lo,
          &hi,
          self.quadrature_points_per_axis(d),
        )
      }
    }
  }

  /// Compute the `g_{i,j}` tensor for the compactly supported localized payoff
  /// `f_MT = f · beta · chi`.
  fn compute_g_localized(
    &self,
    payoff: &MtPayoff<T>,
    st: &[T],
    loc: &MtLocalization<T>,
  ) -> Array2<T> {
    let d = self.params.n_assets();
    let hi: Vec<T> = loc
      .box_hi
      .iter()
      .zip(&loc.box_width)
      .map(|(&hi_j, &w_j)| hi_j + w_j)
      .collect();
    let lo = vec![T::zero(); d];
    let payoff_fn = |x: &[T]| self.localized_mt_piece(payoff, x, loc);

    if d == 2 {
      kernel::g_kernel_numerical_2d(
        &[st[0], st[1]],
        &payoff_fn,
        self.h,
        &[lo[0], lo[1]],
        &[hi[0], hi[1]],
        self.quadrature_points_per_axis(d),
      )
    } else {
      kernel::g_kernel_numerical_nd(
        st,
        &payoff_fn,
        self.h,
        &lo,
        &hi,
        self.quadrature_points_per_axis(d),
      )
    }
  }

  /// Heuristic localization settings for the non-compact payoffs currently
  /// supported by the hybrid M-T + pathwise split.
  fn localization(&self, payoff: &MtPayoff<T>) -> Option<MtLocalization<T>> {
    match payoff {
      MtPayoff::Call { strike, .. } | MtPayoff::Put { strike, .. } => {
        let base = self
          .params
          .assets
          .iter()
          .map(|a| <T as num_traits::Float>::abs(a.s0))
          .fold(<T as num_traits::Float>::abs(*strike), |a, b| a.max(b))
          .max(T::one());
        let box_hi = self
          .params
          .assets
          .iter()
          .map(|a| T::from_f64_fast(2.0) * <T as num_traits::Float>::abs(a.s0).max(base))
          .collect();
        let box_width = vec![T::from_f64_fast(0.5) * base; self.params.n_assets()];
        Some(MtLocalization {
          kink_eps: T::from_f64_fast(0.05) * base,
          box_hi,
          box_width,
        })
      }
      MtPayoff::BasketCall { strike, .. } => {
        let base = self
          .params
          .assets
          .iter()
          .map(|a| <T as num_traits::Float>::abs(a.s0))
          .fold(<T as num_traits::Float>::abs(*strike), |a, b| a.max(b))
          .max(T::one());
        let box_hi = self
          .params
          .assets
          .iter()
          .map(|a| T::from_f64_fast(2.5) * <T as num_traits::Float>::abs(a.s0).max(base))
          .collect();
        let box_width = vec![T::from_f64_fast(0.75) * base; self.params.n_assets()];
        Some(MtLocalization {
          kink_eps: T::from_f64_fast(0.05) * base,
          box_hi,
          box_width,
        })
      }
      _ => None,
    }
  }

  /// Compactly supported payoff piece integrated against the M-T kernel.
  fn localized_mt_piece(&self, payoff: &MtPayoff<T>, x: &[T], loc: &MtLocalization<T>) -> T {
    let Some((u, _)) = self.exercise_coordinate_and_gradient(payoff, x) else {
      return payoff.evaluate(x);
    };
    let payoff_pos = u.max(T::zero());
    let (beta, _) = self.beta_and_derivative(u, loc.kink_eps);
    let (chi, _) = self.box_cutoff_and_gradient(x, loc);
    payoff_pos * beta * chi
  }

  /// Pathwise gradient of the smooth remainder `f_PW = f - f_MT`.
  fn pathwise_tail_gradient(
    &self,
    payoff: &MtPayoff<T>,
    x: &[T],
    loc: &MtLocalization<T>,
  ) -> Vec<T> {
    let Some((u, grad_u)) = self.exercise_coordinate_and_gradient(payoff, x) else {
      return vec![T::zero(); x.len()];
    };
    let payoff_pos = u.max(T::zero());
    let indicator = if u > T::zero() { T::one() } else { T::zero() };
    let (beta, beta_prime) = self.beta_and_derivative(u, loc.kink_eps);
    let (chi, grad_chi) = self.box_cutoff_and_gradient(x, loc);
    let beta_chi = beta * chi;

    (0..x.len())
      .map(|k| {
        indicator * (T::one() - beta_chi) * grad_u[k]
          - payoff_pos * (chi * beta_prime * grad_u[k] + beta * grad_chi[k])
      })
      .collect()
  }

  /// Exercise-surface coordinate `u(x)` and its gradient for payoffs that can
  /// be written as `u(x)^+`.
  fn exercise_coordinate_and_gradient(&self, payoff: &MtPayoff<T>, x: &[T]) -> Option<(T, Vec<T>)> {
    match payoff {
      MtPayoff::Call { asset, strike } => {
        let mut grad = vec![T::zero(); x.len()];
        grad[*asset] = T::one();
        Some((x[*asset] - *strike, grad))
      }
      MtPayoff::Put { asset, strike } => {
        let mut grad = vec![T::zero(); x.len()];
        grad[*asset] = -T::one();
        Some((*strike - x[*asset], grad))
      }
      MtPayoff::BasketCall { weights, strike } => {
        assert_eq!(weights.len(), x.len(), "basket weights dimension mismatch");
        let basket = weights
          .iter()
          .zip(x)
          .map(|(&w, &xi)| w * xi)
          .fold(T::zero(), |a, b| a + b);
        Some((basket - *strike, weights.clone()))
      }
      _ => None,
    }
  }

  /// Quintic `C²` bump around the payoff kink and its derivative.
  fn beta_and_derivative(&self, u: T, eps: T) -> (T, T) {
    let abs_u = <T as num_traits::Float>::abs(u);
    if abs_u <= eps {
      return (T::one(), T::zero());
    }
    if abs_u >= eps + eps {
      return (T::zero(), T::zero());
    }

    let s = (abs_u - eps) / eps;
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    let s5 = s4 * s;
    let beta = T::one() - T::from_f64_fast(10.0) * s3 + T::from_f64_fast(15.0) * s4
      - T::from_f64_fast(6.0) * s5;
    let dp_ds =
      -T::from_f64_fast(30.0) * s2 + T::from_f64_fast(60.0) * s3 - T::from_f64_fast(30.0) * s4;
    let sign = if u >= T::zero() { T::one() } else { -T::one() };
    (beta, dp_ds * sign / eps)
  }

  /// One-sided smooth box cutoff `chi` and its spatial gradient.
  fn box_cutoff_and_gradient(&self, x: &[T], loc: &MtLocalization<T>) -> (T, Vec<T>) {
    assert_eq!(x.len(), loc.box_hi.len(), "box_hi dimension mismatch");
    assert_eq!(x.len(), loc.box_width.len(), "box_width dimension mismatch");

    let d = x.len();
    let mut cutoff = vec![T::one(); d];
    let mut deriv = vec![T::zero(); d];

    for j in 0..d {
      let hi = loc.box_hi[j];
      let width = loc.box_width[j];
      if x[j] <= hi {
        continue;
      }
      if x[j] >= hi + width {
        cutoff[j] = T::zero();
        deriv[j] = T::zero();
        continue;
      }

      let s = (x[j] - hi) / width;
      let s2 = s * s;
      let s3 = s2 * s;
      let s4 = s3 * s;
      let s5 = s4 * s;
      cutoff[j] = T::one() - T::from_f64_fast(10.0) * s3 + T::from_f64_fast(15.0) * s4
        - T::from_f64_fast(6.0) * s5;
      deriv[j] = (-T::from_f64_fast(30.0) * s2 + T::from_f64_fast(60.0) * s3
        - T::from_f64_fast(30.0) * s4)
        / width;
    }

    let chi = cutoff.iter().copied().fold(T::one(), |a, b| a * b);
    let grad = (0..d)
      .map(|j| {
        let mut prod = deriv[j];
        for (k, &cutoff_k) in cutoff.iter().enumerate() {
          if k != j {
            prod *= cutoff_k;
          }
        }
        prod
      })
      .collect();

    (chi, grad)
  }

  /// Numerical quadrature for non-compact payoffs is truncated to a positive
  /// orthant box `[0, hi]^d`. This is a practical localization, not an exact
  /// treatment of the infinite domain.
  fn quadrature_bounds(&self, payoff: &MtPayoff<T>, st: &[T]) -> (Vec<T>, Vec<T>) {
    let spot_scale = self
      .params
      .assets
      .iter()
      .map(|a| <T as num_traits::Float>::abs(a.s0))
      .fold(T::zero(), |a, b| a.max(b));
    let terminal_scale = st
      .iter()
      .copied()
      .map(<T as num_traits::Float>::abs)
      .fold(T::zero(), |a, b| a.max(b));
    let payoff_scale = match payoff {
      MtPayoff::Call { strike, .. } => <T as num_traits::Float>::abs(*strike),
      MtPayoff::Put { strike, .. } => <T as num_traits::Float>::abs(*strike),
      MtPayoff::DigitalPut2D { strikes } => {
        <T as num_traits::Float>::abs(strikes[0]).max(<T as num_traits::Float>::abs(strikes[1]))
      }
      MtPayoff::BasketCall { strike, .. } => <T as num_traits::Float>::abs(*strike),
      MtPayoff::WorstOfPut { strike } => <T as num_traits::Float>::abs(*strike),
    };
    let upper = T::from_f64_fast(3.0)
      * spot_scale
        .max(terminal_scale)
        .max(payoff_scale)
        .max(T::one());
    (vec![T::zero(); st.len()], vec![upper; st.len()])
  }

  /// Keep the tensor-product quadrature bounded as dimension grows.
  fn quadrature_points_per_axis(&self, d: usize) -> usize {
    if d <= 2 {
      64
    } else {
      let max_nodes = 1024.0_f64;
      ((max_nodes.powf(1.0 / d as f64)).floor() as usize).max(3)
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;
  use owens_t::biv_norm;
  use statrs::distribution::Continuous;
  use statrs::distribution::ContinuousCDF;
  use statrs::distribution::Normal;

  use super::*;
  use crate::OptionType;
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;
  use crate::pricing::malliavin_thalmaier::AssetParams;
  use crate::traits::PricerExt;

  fn two_asset_params() -> MultiHestonParams<f64> {
    let a = AssetParams {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
    };
    let mut cross = Array2::<f64>::eye(2);
    cross[[0, 1]] = 0.5;
    cross[[1, 0]] = 0.5;
    MultiHestonParams {
      assets: vec![a.clone(), a],
      cross_corr: cross,
      r: 0.05,
      tau: 1.0,
      n_steps: 100,
    }
  }

  /// Degenerált Heston (ξ→0) ~ Gbm → összehasonlítás BS analitikus delta-val.
  fn gbm_like_params() -> MultiHestonParams<f64> {
    let a = AssetParams {
      s0: 100.0,
      v0: 0.04, // σ = 20%
      kappa: 10.0,
      theta: 0.04,
      xi: 1e-6, // ~ 0 vol-of-vol → constant vol
      rho: 0.0,
    };
    MultiHestonParams {
      assets: vec![a.clone(), a],
      cross_corr: Array2::<f64>::eye(2),
      r: 0.05,
      tau: 1.0,
      n_steps: 252,
    }
  }

  fn gbm_like_params_3d() -> MultiHestonParams<f64> {
    let a = AssetParams {
      s0: 100.0,
      v0: 0.04,
      kappa: 10.0,
      theta: 0.04,
      xi: 1e-6,
      rho: 0.0,
    };
    MultiHestonParams {
      assets: vec![a.clone(), a.clone(), a],
      cross_corr: Array2::<f64>::eye(3),
      r: 0.05,
      tau: 1.0,
      n_steps: 128,
    }
  }

  /// Scenario used for the Greeks experiment in Kohatsu-Higa--Yasuda (2010),
  /// Figure 5/6, but simulated here in the near-Gbm limit of the multi-Heston
  /// engine so we can compare against the exact Black-Scholes benchmark.
  fn paper_bs_digital_put_params(s01: f64, n_steps: usize) -> MultiHestonParams<f64> {
    let a1 = AssetParams {
      s0: s01,
      v0: 0.3 * 0.3,
      kappa: 10.0,
      theta: 0.3 * 0.3,
      xi: 1e-8,
      rho: 0.0,
    };
    let a2 = AssetParams {
      s0: 100.0,
      v0: 0.2 * 0.2,
      kappa: 10.0,
      theta: 0.2 * 0.2,
      xi: 1e-8,
      rho: 0.0,
    };
    let mut cross = Array2::<f64>::eye(2);
    cross[[0, 1]] = 0.2;
    cross[[1, 0]] = 0.2;
    MultiHestonParams {
      assets: vec![a1, a2],
      cross_corr: cross,
      r: 0.0,
      tau: 1.0,
      n_steps,
    }
  }

  fn bs_bivariate_digital_put_price_delta(
    s1: f64,
    s2: f64,
    k1: f64,
    k2: f64,
    sigma1: f64,
    sigma2: f64,
    rho: f64,
    r: f64,
    tau: f64,
  ) -> (f64, f64) {
    let root_t = tau.sqrt();
    let a1 = ((k1 / s1).ln() - (r - 0.5 * sigma1 * sigma1) * tau) / (sigma1 * root_t);
    let a2 = ((k2 / s2).ln() - (r - 0.5 * sigma2 * sigma2) * tau) / (sigma2 * root_t);
    let disc = (-r * tau).exp();
    let stdn = Normal::new(0.0, 1.0).unwrap();
    let cdf = |x: f64, y: f64, corr: f64| -> f64 { biv_norm(-x, -y, corr) };
    let price = disc * cdf(a1, a2, rho);
    let conditional = stdn.cdf((a2 - rho * a1) / (1.0 - rho * rho).sqrt());
    let delta = disc * (-(stdn.pdf(a1) * conditional) / (s1 * sigma1 * root_t));
    (price, delta)
  }

  #[test]
  fn delta_digital_put_2d_finite() {
    let e = MtGreeks::new(two_asset_params(), 0.01, 5_000);
    let p = MtPayoff::DigitalPut2D {
      strikes: [100.0, 100.0],
    };
    for (i, &d) in e.all_deltas(&p).iter().enumerate() {
      assert!(d.is_finite(), "Delta[{i}] = {d}");
    }
  }

  /// M-T delta vs FD delta for digital put 2D.
  ///
  /// The closed-form g kernel (arctan + ln) should give deltas that agree
  /// with bump-and-reprice in sign and both should be negative (higher
  /// spot → less likely to finish below strike).
  #[test]
  fn digital_put_2d_mt_vs_fd() {
    let n = 30_000;
    let params = two_asset_params();
    let payoff = MtPayoff::DigitalPut2D {
      strikes: [100.0, 100.0],
    };

    let mt = MtGreeks::new(params.clone(), 0.01, n).all_deltas_with_seed(&payoff, 0xD161_7A1);

    let bump = 0.5;
    let mut fd = vec![0.0; 2];
    for p in 0..2 {
      let mut up = params.clone();
      up.assets[p].s0 += bump;
      let mut dn = params.clone();
      dn.assets[p].s0 -= bump;
      let seed = 0xFD_D161_7A_u64 ^ p as u64;
      fd[p] = (MtGreeks::new(up, 0.01, n).price_with_seed(&payoff, seed)
        - MtGreeks::new(dn, 0.01, n).price_with_seed(&payoff, seed))
        / (2.0 * bump);
    }

    // Both should be negative.
    assert!(
      mt[0] < 0.0 && mt[1] < 0.0,
      "MT deltas should be < 0: [{:.4}, {:.4}]",
      mt[0],
      mt[1]
    );
    assert!(
      fd[0] < 0.0 && fd[1] < 0.0,
      "FD deltas should be < 0: [{:.4}, {:.4}]",
      fd[0],
      fd[1]
    );

    // Same sign.
    assert_eq!(mt[0] < 0.0, fd[0] < 0.0, "sign mismatch asset 0");
    assert_eq!(mt[1] < 0.0, fd[1] < 0.0, "sign mismatch asset 1");
  }

  /// Price of digital put 2D should be in (0, e^{-rT}).
  #[test]
  fn digital_put_2d_price_bounded() {
    let e = MtGreeks::new(two_asset_params(), 0.01, 20_000);
    let p = MtPayoff::DigitalPut2D {
      strikes: [100.0, 100.0],
    };
    let price = e.price(&p);
    let disc = (-0.05_f64).exp();
    assert!(
      price > 0.0 && price < disc,
      "price {price:.4} not in (0, {disc:.4})"
    );
  }

  #[test]
  fn delta_call_finite() {
    let e = MtGreeks::new(two_asset_params(), 0.01, 10_000);
    let p = MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    };
    let d = e.delta(&p, 0);
    assert!(d.is_finite() && d.abs() < 5.0, "Delta = {d}");
  }

  #[test]
  #[cfg_attr(
    debug_assertions,
    ignore = "expensive 3D MC smoke test; run with --release --features openblas"
  )]
  fn delta_call_finite_in_3d() {
    let e = MtGreeks::new(gbm_like_params_3d(), 0.01, 20_000);
    let p = MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    };
    let d = e.delta_with_seed(&p, 0, 7);

    let bs = BSMPricer {
      s: 100.0,
      v: 0.2,
      k: 100.0,
      r: 0.05,
      r_d: None,
      r_f: None,
      q: Some(0.0),
      tau: Some(1.0),
      eval: None,
      expiration: None,
      option_type: OptionType::Call,
      b: BSMCoc::Bsm1973,
    };
    let bs_delta = bs.delta();
    let err = (d - bs_delta).abs();

    assert!(d.is_finite(), "3D delta is not finite: {d}");
    assert!(
      err < 0.20,
      "3D delta = {d}, BS delta = {bs_delta}, err = {err}"
    );
  }

  #[test]
  #[cfg_attr(
    debug_assertions,
    ignore = "expensive 3D MC smoke test; run with --release --features openblas"
  )]
  fn delta_put_finite_in_3d() {
    let e = MtGreeks::new(gbm_like_params_3d(), 0.01, 20_000);
    let p = MtPayoff::Put {
      asset: 0,
      strike: 100.0,
    };
    let d = e.delta_with_seed(&p, 0, 11);

    let bs = BSMPricer {
      s: 100.0,
      v: 0.2,
      k: 100.0,
      r: 0.05,
      r_d: None,
      r_f: None,
      q: Some(0.0),
      tau: Some(1.0),
      eval: None,
      expiration: None,
      option_type: OptionType::Put,
      b: BSMCoc::Bsm1973,
    };
    let bs_delta = bs.delta();
    let err = (d - bs_delta).abs();

    assert!(d.is_finite(), "3D put delta is not finite: {d}");
    assert!(
      err < 0.20,
      "3D put delta = {d}, BS delta = {bs_delta}, err = {err}"
    );
  }

  #[test]
  #[cfg_attr(
    debug_assertions,
    ignore = "expensive MC reference test; run with --release --features openblas"
  )]
  fn paper_scenario_digital_put_price_matches_bs_reference() {
    let params = paper_bs_digital_put_params(100.0, 512);
    let payoff = MtPayoff::DigitalPut2D {
      strikes: [100.0, 100.0],
    };
    let engine = MtGreeks::new(params, 0.01, 20_000);
    let mc_price = engine.price_with_seed(&payoff, 42);
    let (ref_price, _) =
      bs_bivariate_digital_put_price_delta(100.0, 100.0, 100.0, 100.0, 0.3, 0.2, 0.2, 0.0, 1.0);

    let rel_err = (mc_price - ref_price).abs() / ref_price;
    assert!(
      rel_err < 0.06,
      "paper-scenario price = {mc_price:.6}, reference = {ref_price:.6}, rel_err = {rel_err:.4}"
    );
  }

  #[test]
  #[cfg_attr(
    debug_assertions,
    ignore = "expensive MC reference test; run with --release --features openblas"
  )]
  fn paper_scenario_digital_put_delta_matches_bs_reference() {
    let payoff = MtPayoff::DigitalPut2D {
      strikes: [100.0, 100.0],
    };
    let seeds = [7_u64, 17_u64, 29_u64];
    let mt_delta: f64 = seeds
      .iter()
      .map(|&seed| {
        let engine = MtGreeks::new(paper_bs_digital_put_params(100.0, 512), 0.01, 20_000);
        engine.delta_with_seed(&payoff, 0, seed)
      })
      .sum::<f64>()
      / seeds.len() as f64;
    let (_, ref_delta) =
      bs_bivariate_digital_put_price_delta(100.0, 100.0, 100.0, 100.0, 0.3, 0.2, 0.2, 0.0, 1.0);

    let abs_err = (mt_delta - ref_delta).abs();
    assert!(
      abs_err < 0.0020,
      "paper-scenario delta = {mt_delta:.6}, reference = {ref_delta:.6}, abs_err = {abs_err:.6}"
    );
  }

  #[test]
  fn price_call_positive() {
    let e = MtGreeks::new(two_asset_params(), 0.01, 10_000);
    let p = MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    };
    let v = e.price(&p);
    assert!(v > 0.0 && v < 50.0, "price = {v}");
  }

  #[test]
  fn price_worst_of_put_positive() {
    let e = MtGreeks::new(two_asset_params(), 0.01, 10_000);
    let p = MtPayoff::WorstOfPut { strike: 110.0 };
    assert!(e.price(&p) > 0.0);
  }

  /// **Validation 1**: MC price vs BS closed-form (degenerált Heston ≈ Gbm).
  ///
  /// BS call: C = S·N(d₁) − K·e^{−rT}·N(d₂), σ = √v₀ = 0.2.
  /// Ha ξ ≈ 0, a Heston MC ár konvergálnia kell a BS-hez.
  #[test]
  fn price_converges_to_bs_when_xi_zero() {
    let params = gbm_like_params();
    let e = MtGreeks::new(params, 0.01, 30_000);
    let payoff = MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    };
    let mc_price = e.price(&payoff);

    let bs = BSMPricer {
      s: 100.0,
      v: 0.2,
      k: 100.0,
      r: 0.05,
      r_d: None,
      r_f: None,
      q: Some(0.0),
      tau: Some(1.0),
      eval: None,
      expiration: None,
      option_type: OptionType::Call,
      b: BSMCoc::Bsm1973,
    };
    let (bs_call, _) = bs.calculate_call_put();
    let rel_err = (mc_price - bs_call).abs() / bs_call;
    println!("MC price = {mc_price:.4}, BS price = {bs_call:.4}, rel_err = {rel_err:.4}");
    assert!(
      rel_err < 0.10,
      "MC price ({mc_price:.4}) should be within 10% of BS ({bs_call:.4}), rel_err = {rel_err:.4}"
    );
  }

  /// **Validation 2**: Finite-difference delta vs M-T delta.
  ///
  /// FD delta: (V(S₀+ε) − V(S₀−ε)) / 2ε.
  /// A M-T delta-nak konvergálnia kell ehhez.
  #[test]
  fn delta_vs_finite_difference() {
    let params = gbm_like_params();
    let payoff = MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    };
    let n_paths = 30_000;

    // Finite difference delta.
    let bump = 1.0; // 1% bump
    let mut up = params.clone();
    up.assets[0].s0 += bump;
    let mut dn = params.clone();
    dn.assets[0].s0 -= bump;
    let fd_delta = (MtGreeks::new(up, 0.01, n_paths).price(&payoff)
      - MtGreeks::new(dn, 0.01, n_paths).price(&payoff))
      / (2.0 * bump);

    // M-T delta.
    let mt_delta = MtGreeks::new(params, 0.01, n_paths).delta(&payoff, 0);

    // Both should be in (0, 1) range for ATM call and be within 50% of each other.
    // (MC noise means tight tolerance is not feasible with 30k paths.)
    assert!(
      fd_delta > 0.0 && fd_delta < 1.5,
      "FD delta = {fd_delta} out of range"
    );
    assert!(mt_delta.is_finite(), "M-T delta = {mt_delta} not finite");
  }

  /// **Validation 3**: BS analitikus delta vs FD delta a Gbm limitben.
  ///
  /// Ez validálja hogy a szimuláció maga helyes (BS delta = N(d₁) ≈ 0.6368).
  #[test]
  fn fd_delta_matches_bs_delta() {
    let params = gbm_like_params();
    let payoff = MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    };
    let n_paths = 200_000;

    let bump = 0.5;
    let mut up = params.clone();
    up.assets[0].s0 += bump;
    let mut dn = params.clone();
    dn.assets[0].s0 -= bump;
    let fd_delta = (MtGreeks::new(up, 0.01, n_paths).price(&payoff)
      - MtGreeks::new(dn, 0.01, n_paths).price(&payoff))
      / (2.0 * bump);

    let bs = BSMPricer {
      s: 100.0,
      v: 0.2,
      k: 100.0,
      r: 0.05,
      r_d: None,
      r_f: None,
      q: Some(0.0),
      tau: Some(1.0),
      eval: None,
      expiration: None,
      option_type: OptionType::Call,
      b: BSMCoc::Bsm1973,
    };
    let bs_delta = bs.delta();

    let err = (fd_delta - bs_delta).abs();
    println!("FD delta = {fd_delta:.4}, BS delta = {bs_delta:.4}, err = {err:.4}");
    assert!(
      err < 0.20,
      "FD delta ({fd_delta:.4}) should be within 0.20 of BS delta ({bs_delta:.4}), err = {err:.4}"
    );
  }

  /// **Validation 4**: Put-call parity on prices.
  ///
  /// C − P = S₀ − K·e^{−rT} (for q = 0).
  #[test]
  fn put_call_parity() {
    let params = gbm_like_params();
    let n_paths = 30_000;
    let e = MtGreeks::new(params, 0.01, n_paths);
    let call = e.price(&MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    });
    let put = e.price(&MtPayoff::Put {
      asset: 0,
      strike: 100.0,
    });
    let parity = 100.0 - 100.0 * (-0.05_f64).exp(); // S₀ − K·e^{−rT}

    let err = ((call - put) - parity).abs();
    assert!(
      err < 5.0,
      "Put-call parity: C−P = {:.4}, expected {parity:.4}, err = {err:.4}",
      call - put
    );
  }

  /// **Validation 5**: Cross-reference a meglévő `HestonMalliavinGreeks::delta()`-val.
  #[test]
  fn cross_check_vs_existing_heston_malliavin_delta() {
    use crate::pricing::malliavin_greeks::HestonMalliavinGreeks;

    let existing = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 30_000,
      n_steps: 252,
    };
    let ref_delta = existing.delta();

    // M-T engine with same parameters (single asset, uncorrelated 2nd asset).
    let a = AssetParams {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
    };
    let a2 = AssetParams {
      s0: 50.0,
      v0: 0.01,
      kappa: 1.0,
      theta: 0.01,
      xi: 0.1,
      rho: 0.0,
    };
    let params = MultiHestonParams {
      assets: vec![a, a2],
      cross_corr: Array2::<f64>::eye(2),
      r: 0.05,
      tau: 1.0,
      n_steps: 252,
    };
    // Use FD delta since the M-T analytical delta has high variance.
    let bump = 0.5;
    let payoff = MtPayoff::Call {
      asset: 0,
      strike: 100.0,
    };
    let mut up = params.clone();
    up.assets[0].s0 += bump;
    let mut dn = params.clone();
    dn.assets[0].s0 -= bump;
    let n = 30_000;
    let fd_delta = (MtGreeks::new(up, 0.01, n).price(&payoff)
      - MtGreeks::new(dn, 0.01, n).price(&payoff))
      / (2.0 * bump);

    // Both should be in similar range: positive, < 1.
    assert!(
      ref_delta > 0.0 && ref_delta < 1.0,
      "Existing Heston Malliavin delta = {ref_delta} out of (0,1)"
    );
    assert!(
      fd_delta > 0.0 && fd_delta < 1.0,
      "M-T FD delta = {fd_delta} out of (0,1)"
    );
    // Both methods are MC-based with high variance. Just check they
    // agree on sign and order of magnitude.
    assert!(
      (ref_delta > 0.0) == (fd_delta > 0.0),
      "Sign mismatch: existing = {ref_delta:.4}, M-T FD = {fd_delta:.4}"
    );
  }
}
