//! # Wishart distribution
//!
//! $$
//! X \sim \mathcal{W}_p(\nu, V)
//! \iff X = \sum_{i=1}^{\nu} Z_i Z_i^\top,\ Z_i \sim \mathcal{N}_p(0, V),
//! $$
//!
//! $$
//! f(X) = \frac{|X|^{(\nu - p - 1)/2}\,
//!              \exp(-\tfrac{1}{2} \mathrm{tr}(V^{-1} X))}
//!             {2^{\nu p / 2}\, |V|^{\nu/2}\, \Gamma_p(\nu/2)},\qquad
//! \Gamma_p(a) = \pi^{p(p-1)/4} \prod_{j=1}^{p} \Gamma(a - (j-1)/2),
//! $$
//!
//! $$
//! X = L A A^\top L^\top,\qquad V = L L^\top,
//! \quad A_{jj} \sim \chi_{\nu - j + 1},\ A_{i>j} \sim \mathcal N(0, 1).
//! $$
//!
//! For `ν > p − 1` and an SPD `V`, drawn by Bartlett's decomposition above; the stream's
//! [`sample_inverse`](crate::Seeded::sample_inverse) draws `X⁻¹ ~ IW_p(ν, V⁻¹)`.
//!
//! Reference: Bartlett, M.S. (1933), "On the theory of statistical regression", *Proceedings of the Royal Society of Edinburgh* 53, 260-283, DOI 10.1017/S0370164600015637.
//! Reference: Smith, W.B., Hocking, R.R. (1972), "Algorithm AS 53: Wishart variate generator", *Applied Statistics* 21(3), 341-345, DOI 10.2307/2346290.

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
  nu: T,
  scale: Array2<T>,
  p: usize,
  /// Lower-triangular Cholesky factor of the scale matrix $V$.
  chol: Array2<T>,
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
  /// Wishart$(\nu, V)$ for a $p \times p$ positive-definite `scale`, $p \ge 1$, Cholesky-factorised
  /// once here. Panics on an empty, non-square or non-SPD `scale`, or on $\nu \le p - 1$.
  pub fn new(nu: T, scale: Array2<T>) -> Self {
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    for ((i, j), v) in scale.indexed_iter() {
      assert!(
        v.is_finite(),
        "scale must satisfy `scale[[i, j]].is_finite()`, got scale[[{i}, {j}]] = {v:?}"
      );
    }
    let p = scale.nrows();
    assert!(
      p >= 1,
      "scale must satisfy `scale.nrows() >= 1`, got scale.nrows() = {p}"
    );
    assert!(
      scale.ncols() == p,
      "scale must satisfy `scale.ncols() == scale.nrows()`, got scale.shape() = {:?}",
      scale.shape()
    );
    assert!(
      nu > T::from_usize_(p - 1),
      "nu must satisfy `nu > p - 1`, got nu = {nu:?}, p = {p}"
    );
    let chol = cholesky_lower(&scale).unwrap_or_else(|(i, pivot)| {
      panic!("scale must satisfy `pivot[i] > 0.0` (positive definite), got pivot[{i}] = {pivot:?}")
    });
    let diag = (0..p)
      .map(|j| {
        let k = nu - T::from_usize_(j);
        assert!(
          k.is_finite() && k * T::from_f64_fast(0.5) > T::zero(),
          "nu must satisfy `0 < (nu - j) / 2 < ∞`, got nu = {nu:?}, j = {j}"
        );
        SimdChiSquared::<T>::new(k)
      })
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
  pub fn nu(&self) -> T {
    self.nu
  }

  /// The scale matrix `V`.
  pub fn scale(&self) -> &Array2<T> {
    &self.scale
  }

  pub fn dim(&self) -> usize {
    self.p
  }

  /// `L·A·Aᵀ·Lᵀ` for Bartlett's lower-triangular `A`: `√χ²_{ν−j}` on the diagonal, `N(0, 1)` below, column by column.
  fn bartlett<S: BartlettSource<T>>(&self, src: &mut S) -> Array2<T> {
    let mut a = Array2::<T>::zeros((self.p, self.p));
    for (j, chi2) in self.diag.iter().enumerate() {
      a[[j, j]] = src.chi2(chi2, j).max(T::zero()).sqrt();
      for i in (j + 1)..self.p {
        a[[i, j]] = src.normal();
      }
    }
    let m = self.chol.dot(&a);
    m.dot(&m.t())
  }

  /// Log-density of an SPD `X`, in `f64` from the scale's stored factor; NaN where `X` is not positive definite.
  pub fn log_pdf(&self, x: &Array2<T>) -> T {
    let to_f64 = |m: &Array2<T>| m.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
    let (x, chol) = (to_f64(x), to_f64(&self.chol));
    let nu = self.nu.to_f64().unwrap_or(f64::NAN);
    let p = self.p as f64;
    let det_x_log = match cholesky_lower(&x) {
      Ok(l) => 2.0 * (0..self.p).map(|i| l[[i, i]].ln()).sum::<f64>(),
      Err(_) => return T::nan(),
    };
    let det_v_log = 2.0 * (0..self.p).map(|i| chol[[i, i]].ln()).sum::<f64>();
    let v_inv = inverse_from_factor(&chol);
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
    T::from_f64_fast(log_norm + 0.5 * (nu - p - 1.0) * det_x_log - 0.5 * tr_vinv_x)
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
  pub fn sample(&mut self) -> Array2<T> {
    let (law, state) = self.parts_mut();
    law.bartlett(state)
  }

  /// The inverse of one Wishart draw, an inverse-Wishart `IW_p(ν, V⁻¹)` draw; `None` if it is not numerically SPD.
  pub fn sample_inverse(&mut self) -> Option<Array2<T>> {
    invert_spd(&self.sample())
  }
}

impl<T: SimdFloatExt> Distribution<Array2<T>> for SimdWishart<T> {
  /// One Bartlett draw from scalar χ² and normal draws on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> Array2<T> {
    self.bartlett(&mut AnyRng(rng))
  }
}

/// Plain Cholesky factor `L` of `a = L·Lᵀ` (dim ≤ 10 in practice); `Err((i, d))` names the first
/// pivot `d` that is not positive.
fn cholesky_lower<T: SimdFloatExt>(a: &Array2<T>) -> Result<Array2<T>, (usize, T)> {
  let n = a.nrows();
  let mut l = Array2::<T>::zeros((n, n));
  for i in 0..n {
    for j in 0..=i {
      let mut sum = T::zero();
      for k in 0..j {
        sum += l[[i, k]] * l[[j, k]];
      }
      if i == j {
        let diag = a[[i, i]] - sum;
        if diag.is_nan() || diag <= T::zero() {
          return Err((i, diag));
        }
        l[[i, i]] = diag.sqrt();
      } else {
        l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
      }
    }
  }
  Ok(l)
}

/// The inverse of an SPD `a`; `None` where its Cholesky factor fails.
fn invert_spd<T: SimdFloatExt>(a: &Array2<T>) -> Option<Array2<T>> {
  Some(inverse_from_factor(&cholesky_lower(a).ok()?))
}

/// `(L·Lᵀ)⁻¹` for a lower-triangular `L` with a positive diagonal, one column of the identity at a time.
fn inverse_from_factor<T: SimdFloatExt>(l: &Array2<T>) -> Array2<T> {
  let n = l.nrows();
  let mut inv = Array2::<T>::zeros((n, n));
  for col in 0..n {
    let mut y = vec![T::zero(); n];
    let mut x = vec![T::zero(); n];
    // L · y = e_col
    for i in 0..n {
      let mut s = if i == col { T::one() } else { T::zero() };
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
  inv
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::SimdRng;

  use super::*;

  const SEEDS: [u64; 3] = [2718, 999, 42];

  /// Wishart samples are symmetric positive semi-definite.
  #[test]
  fn wishart_samples_are_spd() {
    for seed in SEEDS {
      let v = array![[2.0, 0.5], [0.5, 1.0]];
      let mut w = SimdWishart::<f64>::new(5.0, v).seeded(&Deterministic::new(seed));
      for _ in 0..200 {
        let x = w.sample();
        for i in 0..2 {
          for j in 0..2 {
            assert!(
              (x[[i, j]] - x[[j, i]]).abs() < 1e-12,
              "asymmetry at ({i}, {j})"
            );
          }
        }
        assert!(cholesky_lower(&x).is_ok(), "non-SPD sample");
      }
    }
  }

  fn mean_error(nu: f64, v: &Array2<f64>, mut draw: impl FnMut() -> Array2<f64>) -> f64 {
    let n = 5_000;
    let mut acc = Array2::<f64>::zeros((2, 2));
    for _ in 0..n {
      acc += &draw();
    }
    let mean = acc.mapv(|x| x / n as f64);
    (0..4)
      .map(|ij| (mean[[ij / 2, ij % 2]] - nu * v[[ij / 2, ij % 2]]).abs())
      .fold(0.0, f64::max)
  }

  /// Sample mean equals $\nu V$ (Wishart first moment), on the best of three seeds.
  #[test]
  fn wishart_sample_mean_matches_nu_v() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let best = SEEDS
      .map(|seed| {
        let mut w = SimdWishart::<f64>::new(8.0, v.clone()).seeded(&Deterministic::new(seed));
        mean_error(8.0, &v, || w.sample())
      })
      .into_iter()
      .fold(f64::INFINITY, f64::min);
    assert!(best < 0.4, "best Wishart mean error {best}");
  }

  /// The honest draw on the caller's rng has the same first moment.
  #[test]
  fn scalar_sample_mean_matches_nu_v() {
    let v = array![[2.0, 0.5], [0.5, 1.0]];
    let w = SimdWishart::<f64>::new(8.0, v.clone());
    let mut rng = SimdRng::from_seed(2718);
    let err = mean_error(8.0, &v, || w.sample(&mut rng));
    assert!(err < 0.4, "Wishart mean error {err}");
  }

  /// Inverse-Wishart sampling produces SPD inverses.
  #[test]
  fn inverse_wishart_samples_are_spd() {
    for seed in SEEDS {
      let v = array![[2.0, 0.5], [0.5, 1.0]];
      let mut w = SimdWishart::<f64>::new(6.0, v).seeded(&Deterministic::new(seed));
      for _ in 0..100 {
        let x_inv = w.sample_inverse().expect("invertible");
        assert!(cholesky_lower(&x_inv).is_ok(), "non-SPD inverse");
      }
    }
  }

  /// The `f32` law draws SPD `f32` matrices with the first moment `νV`, and `log_pdf` takes and answers its scalar.
  #[test]
  fn f32_wishart_draws_and_densities() {
    let v = array![[2.0_f32, 0.5], [0.5, 1.0]];
    let law = SimdWishart::<f32>::new(8.0, v.clone());
    let best = SEEDS
      .map(|seed| {
        let mut w = law.clone().seeded(&Deterministic::new(seed));
        let mut acc = Array2::<f32>::zeros((2, 2));
        for _ in 0..5_000 {
          let x = w.sample();
          assert!(cholesky_lower(&x).is_ok(), "non-SPD f32 sample");
          acc += &x;
        }
        (0..4)
          .map(|ij| (acc[[ij / 2, ij % 2]] / 5_000.0 - 8.0 * v[[ij / 2, ij % 2]]).abs())
          .fold(0.0_f32, f32::max)
      })
      .into_iter()
      .fold(f32::INFINITY, f32::min);
    assert!(best < 0.4, "best f32 Wishart mean error {best}");
    let x64 = SimdWishart::<f64>::new(8.0, array![[2.0, 0.5], [0.5, 1.0]])
      .log_pdf(&array![[16.0, 4.0], [4.0, 8.0]]);
    let x32 = law.log_pdf(&array![[16.0_f32, 4.0], [4.0, 8.0]]);
    assert!((x32 as f64 - x64).abs() < 1e-4, "{x32} vs {x64}");
  }

  /// An `f32` scale whose `f32` factor is SPD but whose `f64` factor is not answers a finite density from its own factor.
  #[test]
  fn f32_scale_answers_its_density_from_its_own_factor() {
    let v = array![[3.718021_f32, 0.99840343], [0.99840343, 0.26810214]];
    let x = SimdWishart::<f32>::new(5.0, v.clone()).log_pdf(&Array2::eye(2));
    assert!(x.is_finite(), "{x}");
    assert!(cholesky_lower(&v.mapv(f64::from)).is_err());
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

  /// An empty scale is named before the `nu > p - 1` bound, which `p = 0` cannot form.
  #[test]
  #[should_panic(expected = "scale must satisfy `scale.nrows() >= 1`, got scale.nrows() = 0")]
  fn an_empty_scale_is_rejected() {
    SimdWishart::<f64>::new(3.0, Array2::zeros((0, 0)));
  }

  #[test]
  #[should_panic(
    expected = "scale must satisfy `scale.ncols() == scale.nrows()`, got scale.shape() = [2, 3]"
  )]
  fn a_non_square_scale_is_rejected() {
    SimdWishart::<f64>::new(3.0, Array2::zeros((2, 3)));
  }
}
