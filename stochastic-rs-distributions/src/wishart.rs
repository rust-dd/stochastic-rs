//! # Wishart distribution
//!
//! $$
//! X \sim \mathcal{W}_p(\nu, V)
//! \iff X = \sum_{i=1}^{\nu} Z_i Z_i^\top,\ Z_i \sim \mathcal{N}_p(0, V),
//! $$
//!
//! for $\nu > p - 1$ degrees of freedom and a $p \times p$ positive-definite
//! scale matrix $V$. The PDF on the cone of $p \times p$ SPD matrices is
//!
//! $$
//! f(X) = \frac{|X|^{(\nu - p - 1)/2}\,
//!              \exp(-\tfrac{1}{2} \mathrm{tr}(V^{-1} X))}
//!             {2^{\nu p / 2}\, |V|^{\nu/2}\, \Gamma_p(\nu/2)},
//! $$
//!
//! with the multivariate gamma $\Gamma_p(a) = \pi^{p(p-1)/4} \prod_{j=1}^{p} \Gamma(a - (j-1)/2)$.
//!
//! ## Sampling — Bartlett (1933) decomposition
//!
//! For integer $\nu$, the **Bartlett decomposition** avoids the
//! $\nu$-fold outer product. Let $L$ be the Cholesky factor of $V$. Build
//! a lower-triangular $A$ with
//!
//! - diagonal $A_{jj} \sim \chi(\nu - j + 1)$, i.e. $\sqrt{\chi^2_{\nu - j + 1}}$;
//! - sub-diagonal $A_{ij} \sim \mathcal{N}(0, 1)$ for $i > j$.
//!
//! Then $X = L A A^\top L^\top \sim \mathcal{W}_p(\nu, V)$.
//!
//! ## Inverse-Wishart
//!
//! If $X \sim \mathcal{W}_p(\nu, V)$ then $X^{-1} \sim
//! \mathcal{IW}_p(\nu, V^{-1})$. Use the stream's
//! [`sample_inverse`](crate::Seeded::sample_inverse) for the Bayesian-prior variant.
//!
//! References:
//! - Bartlett, M.S. (1933), "On the theory of statistical regression",
//!   *Proceedings of the Royal Society of Edinburgh* 53, 260-283,
//!   DOI 10.1017/S0370164600015637.
//! - Smith, W.B., Hocking, R.R. (1972), "Algorithm AS 53: Wishart
//!   variate generator", *Applied Statistics* 21(3), 341-345,
//!   DOI 10.2307/2346290.

use ndarray::Array2;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::chi_square::SimdChiSquared;
use crate::gamma::GammaState;
use crate::normal::SimdNormal;
use crate::seeded::Seeded;
use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Wishart law `W_p(nu, scale)`: parameters and the scale's Cholesky factor; a [`Seeded`] stream draws its matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct SimdWishart<T> {
  nu: f64,
  scale: Array2<f64>,
  p: usize,
  /// Lower-triangular Cholesky factor of the scale matrix $V$.
  chol: Array2<f64>,
  /// The $\chi^2_{\nu - j}$ laws of the Bartlett diagonal, one per row.
  diag: Vec<SimdChiSquared<T>>,
}

/// A Wishart stream: one gamma sub-stream per Bartlett diagonal entry, then the below-diagonal normal.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct WishartState<T: SimdFloatExt, R: SimdRngExt> {
  diag: Vec<GammaState<T, R>>,
  normal: StreamState<T, R, 64>,
}

/// The diagonal χ² and the below-diagonal normal of a Bartlett factor, from a stream or the caller's rng.
trait BartlettSource<T: SimdFloatExt> {
  fn chi2(&mut self, law: &SimdChiSquared<T>, j: usize) -> T;

  fn normal(&mut self) -> T;
}

impl<T: SimdFloatExt, R: SimdRngExt> BartlettSource<T> for WishartState<T, R> {
  #[inline]
  fn chi2(&mut self, law: &SimdChiSquared<T>, j: usize) -> T {
    law.next(&mut self.diag[j])
  }

  #[inline]
  fn normal(&mut self) -> T {
    SimdNormal::<T>::standard().next(&mut self.normal)
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> BartlettSource<T> for AnyRng<'_, G> {
  #[inline]
  fn chi2(&mut self, law: &SimdChiSquared<T>, _j: usize) -> T {
    law.draw_with(self.0)
  }

  #[inline]
  fn normal(&mut self) -> T {
    SimdNormal::<T>::standard().draw_with(self.0)
  }
}

impl<T: SimdFloatExt> SimdWishart<T> {
  /// Construct a Wishart$(\nu, V)$ generator.
  ///
  /// `scale` is the $p \times p$ positive-definite scale matrix; the
  /// constructor Cholesky-factorises it eagerly so subsequent draws are
  /// cheap. Returns a panic on a non-SPD scale matrix or on $\nu \le p - 1$.
  pub fn new(nu: f64, scale: Array2<f64>) -> Self {
    let p = scale.nrows();
    assert_eq!(
      scale.ncols(),
      p,
      "scale must satisfy `scale.ncols() == scale.nrows()`, got shape = {:?}",
      scale.shape()
    );
    assert!(
      nu > (p - 1) as f64,
      "nu must satisfy `nu > p - 1`, got nu = {nu:?}, p = {p}"
    );
    let chol = cholesky_lower(&scale).expect("scale matrix must be positive definite");
    let diag = (0..p)
      .map(|j| SimdChiSquared::<T>::new(T::from_f64_fast(nu - j as f64)))
      .collect::<Vec<_>>();
    Self {
      nu,
      scale,
      p,
      chol,
      diag,
    }
  }

  /// The degrees of freedom `ν`.
  pub fn nu(&self) -> f64 {
    self.nu
  }

  /// The scale matrix `V`.
  pub fn scale(&self) -> &Array2<f64> {
    &self.scale
  }

  pub fn dim(&self) -> usize {
    self.p
  }

  /// `L·A·Aᵀ·Lᵀ` for Bartlett's lower-triangular `A`: `√χ²_{ν−j}` on the diagonal, `N(0, 1)` below, row by row.
  fn bartlett<S: BartlettSource<T>>(&self, src: &mut S) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((self.p, self.p));
    for (j, chi2) in self.diag.iter().enumerate() {
      a[[j, j]] = src.chi2(chi2, j).to_f64().unwrap().max(0.0).sqrt();
      for i in (j + 1)..self.p {
        a[[i, j]] = src.normal().to_f64().unwrap();
      }
    }
    let m = self.chol.dot(&a);
    m.dot(&m.t())
  }

  /// Log-density of `X` (must be SPD). NaN if `X` is not positive definite
  /// (Cholesky fails); otherwise the closed-form Wishart log-PDF.
  pub fn log_pdf(&self, x: &Array2<f64>) -> f64 {
    let nu = self.nu;
    let p = self.p as f64;
    let det_x_log = match cholesky_lower(x) {
      Some(l) => 2.0 * (0..self.p).map(|i| l[[i, i]].ln()).sum::<f64>(),
      None => return f64::NAN,
    };
    let det_v_log = 2.0 * (0..self.p).map(|i| self.chol[[i, i]].ln()).sum::<f64>();
    let v_inv = invert_spd(&self.scale).expect("scale matrix should be invertible");
    let tr_vinv_x: f64 = (0..self.p)
      .map(|i| (0..self.p).map(|j| v_inv[[i, j]] * x[[j, i]]).sum::<f64>())
      .sum();
    let log_gamma_p: f64 = (0..self.p)
      .map(|j| crate::special::ln_gamma(0.5 * (nu - j as f64)))
      .sum();
    let log_norm = -0.5 * nu * p * std::f64::consts::LN_2
      - 0.5 * nu * det_v_log
      - 0.25 * p * (p - 1.0) * std::f64::consts::PI.ln()
      - log_gamma_p;
    log_norm + 0.5 * (nu - p - 1.0) * det_x_log - 0.5 * tr_vinv_x
  }
}

impl<T: SimdFloatExt> Sealed for SimdWishart<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdWishart<T> {
  type State<R: SimdRngExt> = WishartState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (WishartState<T, R>, u64) {
    let mut diag = Vec::with_capacity(self.p);
    let mut basis = 0;
    for chi2 in &self.diag {
      let (state, stream_seed) = chi2.init::<R, S>(seed);
      if diag.is_empty() {
        basis = stream_seed;
      }
      diag.push(state);
    }
    let (normal, _) = SimdNormal::<T>::standard().init::<R, S>(seed);
    (WishartState { diag, normal }, basis)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Seeded<SimdWishart<T>, R> {
  /// One Bartlett draw, a `p × p` Wishart matrix.
  pub fn sample(&mut self) -> Array2<f64> {
    let (law, state) = self.parts_mut();
    law.bartlett(state)
  }

  /// The inverse of one Wishart draw, an inverse-Wishart `IW_p(ν, V⁻¹)` draw; `None` if it is not numerically SPD.
  pub fn sample_inverse(&mut self) -> Option<Array2<f64>> {
    invert_spd(&self.sample())
  }
}

impl<T: SimdFloatExt> Distribution<Array2<f64>> for SimdWishart<T> {
  /// One Bartlett draw from scalar χ² and normal draws on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> Array2<f64> {
    self.bartlett(&mut AnyRng(rng))
  }
}

/// Plain in-place Cholesky decomposition (no external linalg dependency
/// needed for this 2.3.0 implementation; dim ≤ 10 in practice for
/// portfolio / factor-model use cases).
fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
  let n = a.nrows();
  let mut l = Array2::<f64>::zeros((n, n));
  for i in 0..n {
    for j in 0..=i {
      let mut sum = 0.0;
      for k in 0..j {
        sum += l[[i, k]] * l[[j, k]];
      }
      if i == j {
        let diag = a[[i, i]] - sum;
        if diag <= 0.0 {
          return None;
        }
        l[[i, i]] = diag.sqrt();
      } else {
        l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
      }
    }
  }
  Some(l)
}

/// Invert a positive-definite matrix via L · Lᵀ Cholesky → forward / back
/// substitution. Returns `None` if `a` is not SPD.
fn invert_spd(a: &Array2<f64>) -> Option<Array2<f64>> {
  let n = a.nrows();
  let l = cholesky_lower(a)?;
  let mut inv = Array2::<f64>::zeros((n, n));
  for col in 0..n {
    let mut y = vec![0.0_f64; n];
    let mut x = vec![0.0_f64; n];
    // L · y = e_col
    for i in 0..n {
      let mut s = if i == col { 1.0 } else { 0.0 };
      for k in 0..i {
        s -= l[[i, k]] * y[k];
      }
      y[i] = s / l[[i, i]];
    }
    // Lᵀ · x = y  (back substitution).
    for i in (0..n).rev() {
      let mut s = y[i];
      for k in (i + 1)..n {
        s -= l[[k, i]] * x[k];
      }
      x[i] = s / l[[i, i]];
    }
    for i in 0..n {
      inv[[i, col]] = x[i];
    }
  }
  Some(inv)
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::SimdRng;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;

  /// Wishart samples are symmetric positive semi-definite.
  #[test]
  fn wishart_samples_are_spd() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let mut w = SimdWishart::<f64>::new(5.0, v).seeded(&Unseeded);
    for _ in 0..200 {
      let x = w.sample();
      // Symmetry.
      for i in 0..2 {
        for j in 0..2 {
          assert!(
            (x[[i, j]] - x[[j, i]]).abs() < 1e-12,
            "asymmetry at ({i}, {j})"
          );
        }
      }
      // SPD via Cholesky.
      assert!(cholesky_lower(&x).is_some(), "non-SPD sample");
    }
  }

  fn assert_mean_is_nu_v(nu: f64, v: &Array2<f64>, mut draw: impl FnMut() -> Array2<f64>) {
    let n = 5_000;
    let mut acc = Array2::<f64>::zeros((2, 2));
    for _ in 0..n {
      acc += &draw();
    }
    let mean = acc.mapv(|x| x / n as f64);
    for i in 0..2 {
      for j in 0..2 {
        let expected = nu * v[[i, j]];
        let err = (mean[[i, j]] - expected).abs();
        assert!(
          err < 0.4,
          "Wishart mean[{i}, {j}] = {} vs expected {} (err {err})",
          mean[[i, j]],
          expected
        );
      }
    }
  }

  /// Sample mean equals $\nu V$ (Wishart first moment).
  #[test]
  fn wishart_sample_mean_matches_nu_v() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let mut w = SimdWishart::<f64>::new(8.0, v.clone()).seeded(&Unseeded);
    assert_mean_is_nu_v(8.0, &v, || w.sample());
  }

  /// The honest draw on the caller's rng has the same first moment.
  #[test]
  fn scalar_sample_mean_matches_nu_v() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let w = SimdWishart::<f64>::new(8.0, v.clone());
    let mut rng = SimdRng::from_seed(2718);
    assert_mean_is_nu_v(8.0, &v, || w.sample(&mut rng));
  }

  /// Inverse-Wishart sampling produces SPD inverses.
  #[test]
  fn inverse_wishart_samples_are_spd() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let mut w = SimdWishart::<f64>::new(6.0, v).seeded(&Unseeded);
    for _ in 0..100 {
      let x_inv = w.sample_inverse().expect("invertible");
      assert!(cholesky_lower(&x_inv).is_some(), "non-SPD inverse");
    }
  }

  /// Log-pdf returns NaN on non-SPD inputs.
  #[test]
  fn wishart_log_pdf_nan_on_non_spd() {
    let v = array![[1.0, 0.0], [0.0, 1.0]];
    let w = SimdWishart::<f64>::new(3.0, v);
    // Non-SPD test matrix (negative eigenvalue).
    let bad = array![[1.0, 2.0], [2.0, 1.0]];
    assert!(w.log_pdf(&bad).is_nan());
  }
}
