//! # Isonormal
//!
//! $$
//! \mathbb{E}[W(h)W(g)] = \langle h, g \rangle_{\mathcal H}
//! $$
//!
use gauss_quad::GaussLegendre;
use ndarray::Array1;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use ndrustfft::ndfft_inplace;
use num_complex::Complex64;
use statrs::function::gamma::gamma;

use crate::traits::FloatExt;

/// Isonormal process
///
/// The Isonormal process is a generalization of the fractional Brownian motion (fBM) process.
/// It represents a Gaussian process defined by an underlying inner product space, where the covariance
/// structure of the process is governed by an inner product on that space. In mathematical terms,
/// an Isonormal process \( X(\varphi) \) is a Gaussian family of random variables indexed by elements
/// \( \varphi \) from a Hilbert space \( \mathcal{H} \), such that for all \( \varphi_1, \varphi_2 \in \mathcal{H} \):
///
/// \[ \mathbb{E}[X(\varphi_1) X(\varphi_2)] = \langle \varphi_1, \varphi_2 \rangle_{\mathcal{H}} \]
///
/// The fractional Brownian motion (fBM) is a special case of the Isonormal process when the inner product
/// represents the covariance structure of the fBM increments.
///
/// # Example
///
/// ```ignore
/// use stochastic_rs::stochastic::isonormal::{fbm_custom_inc_cov, ISONormal};
///
/// let inner_product = |_aux_idx: usize, idx: usize| -> f64 { fbm_custom_inc_cov(idx, 0.7) };
/// let index_functions = vec![0, 1, 2, 3, 4];
/// let mut iso_normal = ISONormal::new(inner_product, index_functions);
/// let _path = iso_normal.get_path();
/// ```
///
/// In this example, an Isonormal process is defined using the fractional Brownian motion covariance increments.
///
pub struct ISONormal<F>
where
  F: Fn(usize, usize) -> f64,
{
  inner_product: F,
  index_functions: Vec<usize>,
  inner_product_structure: Option<Array1<f64>>,
  covariance_matrix_sqrt: Option<Array1<f64>>,
  fft_handler: Option<FftHandler<f64>>,
}

impl<F> ISONormal<F>
where
  F: Fn(usize, usize) -> f64,
{
  pub fn new(inner_product: F, index_functions: Vec<usize>) -> Self {
    ISONormal {
      inner_product,
      index_functions,
      inner_product_structure: None,
      covariance_matrix_sqrt: None,
      fft_handler: None,
    }
  }

  fn set_inner_product_structure(&mut self) {
    if self.inner_product_structure.is_some() {
      return;
    }

    let inner_product_structure = Array1::from(
      (0..self.index_functions.len())
        .map(|k| (self.inner_product)(self.index_functions[0], self.index_functions[k]))
        .collect::<Vec<f64>>(),
    );

    self.inner_product_structure = Some(inner_product_structure);
  }

  fn set_covariance_matrix_sqrt(&mut self) {
    if self.covariance_matrix_sqrt.is_some() {
      return;
    }

    let inner_product_structure_embedding =
      |inner_product_structure: &Array1<f64>| -> Array1<Complex64> {
        let n = inner_product_structure.len();
        let fft = FftHandler::new(n * 2 - 2);
        let input = concatenate(
          Axis(0),
          &[
            inner_product_structure.view(),
            inner_product_structure
              .slice(s![..;-1])
              .slice(s![1..n - 1])
              .view(),
          ],
        )
        .unwrap();

        let input = input.mapv(|v| Complex64::new(v, 0.0));
        let mut embedded_inner_product_structure = Array1::<Complex64>::zeros(n * 2 - 2);
        ndfft(&input, &mut embedded_inner_product_structure, &fft, 0);
        embedded_inner_product_structure
      };

    let embedded_inner_product_structure =
      inner_product_structure_embedding(self.inner_product_structure.as_ref().unwrap());
    let n = self.inner_product_structure.as_ref().unwrap().len();
    let mut covariance_matrix_sqrt = Array1::<f64>::zeros(embedded_inner_product_structure.len());
    let norm = 2.0 * (n - 1) as f64;
    for (dst, x) in covariance_matrix_sqrt
      .iter_mut()
      .zip(embedded_inner_product_structure.iter())
    {
      let lambda = x.re / norm;
      *dst = if lambda > 0.0 { lambda.sqrt() } else { 0.0 };
    }

    self.covariance_matrix_sqrt = Some(covariance_matrix_sqrt);
    self.fft_handler = Some(FftHandler::new(2 * n - 2));
  }

  pub fn get_path(&mut self) -> Array1<f64> {
    self.set_inner_product_structure();
    self.set_covariance_matrix_sqrt();
    let covariance_matrix_sqrt = self.covariance_matrix_sqrt.as_ref().unwrap();
    let fft = self.fft_handler.as_ref().unwrap();
    let n = self.inner_product_structure.as_ref().unwrap().len();
    let mut path = Array1::<f64>::zeros(n - 1);

    <f64 as FloatExt>::with_fgn_complex_scratch(covariance_matrix_sqrt.len(), |workspace| {
      // SAFETY: Complex64 is repr(C) with two f64 fields, matching [f64; 2] layout.
      let flat = unsafe {
        std::slice::from_raw_parts_mut(workspace.as_mut_ptr() as *mut f64, 2 * workspace.len())
      };
      <f64 as FloatExt>::fill_standard_normal_slice(flat);

      for (z, &w) in workspace.iter_mut().zip(covariance_matrix_sqrt.iter()) {
        z.re *= w;
        z.im *= w;
      }

      let mut workspace_view = ArrayViewMut1::from(workspace);
      ndfft_inplace(&mut workspace_view, fft, 0);
      for (dst, z) in path.iter_mut().zip(workspace_view.slice(s![1..n]).iter()) {
        *dst = z.re;
      }
    });

    path
  }
}

/// Ornstein-Uhlenbeck kernel function
fn ker_ou(t: f64, u: f64, alpha: f64) -> f64 {
  if u <= t {
    (-(alpha * (t - u))).exp()
  } else {
    0.0
  }
}

/// Fractional Brownian Motion covariance increments
pub fn fbm_custom_inc_cov(idx: usize, hurst: f64) -> f64 {
  if idx != 0 {
    0.5
      * (((idx + 1) as f64).powf(2.0 * hurst) - 2.0 * (idx as f64).powf(2.0 * hurst)
        + ((idx - 1) as f64).powf(2.0 * hurst))
  } else {
    1.0
  }
}

// ARFIMA autocovariance function
fn arfima_acf(idx: i32, d: f64, sigma: f64) -> f64 {
  if idx == 0 {
    sigma.powi(2) * gamma(1.0 - 2.0 * d) / (gamma(1.0 - d).powi(2))
  } else {
    sigma.powi(2) * (gamma(idx as f64 + d) * gamma(1.0 - 2.0 * d))
      / (gamma(idx as f64 - d + 1.0) * gamma(1.0 - d) * gamma(d))
  }
}

// L2 inner product function using quad for numerical integration
fn l2_unit_inner_product<F1, F2>(function1: F1, function2: F2) -> f64
where
  F1: Fn(f64) -> f64,
  F2: Fn(f64) -> f64,
{
  let integrand = |u: f64| function1(u) * function2(u);
  let quad = GaussLegendre::new(5).unwrap();
  quad.integrate(0.0, 1.0, integrand)
}

// Fractional Lévy Ornstein-Uhlenbeck inner product function (Unstable)
// https://projecteuclid.org/journals/bernoulli/volume-17/issue-1/Fractional-L%C3%A9vy-driven-OrnsteinUhlenbeck-processes-and-stochastic-differential-equations/10.3150/10-BEJ281.pdf
fn cov_ld(t: f64, s: f64, d: f64, e_l1_squared: f64) -> f64 {
  if d <= 0.0 || d >= 1.0 {
    panic!("The 'd' parameter must be in the range (0, 1).");
  }

  let gamma_term = gamma(2.0 * d + 2.0);
  let sin_term = ((std::f64::consts::PI * (d + 0.5)).sin()).abs();
  let denominator = 2.0 * gamma_term * sin_term;
  if denominator == 0.0 {
    panic!("The denominator is zero.");
  }

  let t_term = t.abs().powf(2.0 * d + 1.0);
  let s_term = s.abs().powf(2.0 * d + 1.0);
  let ts_term = (t - s).abs().powf(2.0 * d + 1.0);

  (e_l1_squared / denominator) * (t_term + s_term - ts_term)
}
