use ndarray::Array2;

use super::super::kernel;
use super::greeks::MtGreeks;
use super::payoff::MtPayoff;
use crate::traits::FloatExt;

#[derive(Clone, Debug)]
/// Smooth localization parameters for the split `f = f_MT + f_PW`.
pub(super) struct MtLocalization<T: FloatExt> {
  pub(super) kink_eps: T,
  pub(super) box_hi: Vec<T>,
  pub(super) box_width: Vec<T>,
}

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  pub(super) fn compute_g(&self, payoff: &MtPayoff<T>, st: &[T]) -> Array2<T> {
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
  pub(super) fn compute_g_localized(
    &self,
    payoff: &MtPayoff<T>,
    st: &[T],
    loc: &MtLocalization<T>,
  ) -> Array2<T> {
    let d = self.params.n_assets();
    let hi = loc
      .box_hi
      .iter()
      .zip(&loc.box_width)
      .map(|(&hi_j, &w_j)| hi_j + w_j)
      .collect::<Vec<_>>();
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
  pub(super) fn localization(&self, payoff: &MtPayoff<T>) -> Option<MtLocalization<T>> {
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
  pub(super) fn pathwise_tail_gradient(
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
