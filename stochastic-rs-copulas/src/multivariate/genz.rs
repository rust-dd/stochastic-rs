//! Genz's separation-of-variables integrand for `P(Z ≤ b)`, `Z ~ N(0, LLᵀ)`, and its t form over the `χ_ν/√ν` scale,
//! with the variables ordered per query by Gibson, Glasbey & Elston's prioritization as Genz's MVSORT applies it.
//! Reference: Genz, A. (1992), "Numerical Computation of Multivariate Normal Probabilities", JCGS 1(2) 141–149, DOI 10.1080/10618600.1992.10477010 (§§2–3)
//! Reference: Genz, A., Bretz, F. (2002), "Comparison of Methods for the Computation of Multivariate t Probabilities", JCGS 11(4) 950–971, DOI 10.1198/106186002394 (§2.3, eqs. 13–14)

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use super::qmc::sobol_points;

/// The coordinates of a cdf row below 1, or `None` when one is 0 and the row's cdf is 0: a unit coordinate drops out.
pub(crate) fn open_coordinates(u: ArrayView1<f64>) -> Option<Vec<usize>> {
  if u.iter().any(|&x| x == 0.0) {
    return None;
  }
  Some((0..u.len()).filter(|&i| u[i] < 1.0).collect())
}

/// Genz (1992) §2 with `a = −∞`: `y_i = Φ⁻¹(w_i e_i)`, `e_{i+1} = Φ((b_{i+1} − Σ_{j≤i} l_{i+1,j} y_j)/l_{i+1,i+1})`,
/// value `∏ e_i` from `e₁ = Φ(b₁/l₁₁)`; a subnormal `e_i` gives 0, as `w_i e_i` could underflow and `Φ⁻¹(0)` meet a zero of `L`.
fn integrand(l: &Array2<f64>, b: &[f64], mut e: f64, w: &[f64], y: &mut [f64]) -> f64 {
  let mut f = e;
  for i in 1..b.len() {
    if e < f64::MIN_POSITIVE {
      return 0.0;
    }
    y[i - 1] = ndtri(w[i - 1] * e);
    let dot: f64 = (0..i).map(|j| l[[i, j]] * y[j]).sum();
    e = norm_cdf((b[i] - dot) / l[[i, i]]);
    f *= e;
  }
  f
}

/// Genz's MVSORT on the open coordinates: each step puts next the variable of least conditional probability given the
/// expected values before it, returning that order's factor and limits; `None` once no conditional variance passes `1e-10·i`.
pub(crate) fn prioritize(
  corr: &Array2<f64>,
  open: &[usize],
  b: &[f64],
) -> Option<(Array2<f64>, Vec<f64>)> {
  let m = open.len();
  let (mut idx, mut lim) = (open.to_vec(), b.to_vec());
  let mut l = Array2::<f64>::zeros((m, m));
  let mut y = vec![0.0; m];
  for i in 0..m {
    let eps = 1e-10 * (i + 1) as f64;
    let (mut pick, mut p_min) = (None, 1.0);
    for j in i..m {
      let var = 1.0 - (0..i).map(|k| l[[j, k]] * l[[j, k]]).sum::<f64>();
      if var > eps {
        let sd = var.sqrt();
        let z = (lim[j] - (0..i).map(|k| l[[j, k]] * y[k]).sum::<f64>()) / sd;
        let p = norm_cdf(z);
        if p_min >= p {
          (pick, p_min) = (Some((j, sd, z)), p);
        }
      }
    }
    let (j, sd, z) = pick?;
    idx.swap(i, j);
    lim.swap(i, j);
    for k in 0..i {
      l.swap([i, k], [j, k]);
    }
    l[[i, i]] = sd;
    for r in i + 1..m {
      let dot: f64 = (0..i).map(|k| l[[r, k]] * l[[i, k]]).sum();
      l[[r, i]] = (corr[[idx[r], idx[i]]] - dot) / sd;
    }
    y[i] = if p_min > eps { -norm_pdf(z) / p_min } else { z };
  }
  Some((l, lim))
}

/// `P(Z ≤ b)` for `Z ~ N(0, LLᵀ)`, `L` lower triangular, averaged over `points` of shape `(N, d − 1)`.
pub(crate) fn mvn_upper(l: &Array2<f64>, b: &[f64], points: ArrayView2<f64>) -> f64 {
  let mut y = vec![0.0; b.len() - 1];
  let e = norm_cdf(b[0] / l[[0, 0]]);
  let total: f64 = points
    .rows()
    .into_iter()
    .map(|w| integrand(l, b, e, w.as_slice().expect("row-major"), &mut y))
    .sum();
  total / points.nrows() as f64
}

/// `√(χ²_ν⁻¹(w₀)/ν)` over the first Sobol coordinate, which every dimension shares: the radial scale of Genz & Bretz
/// (2002) eqs. 13–14.
pub(crate) fn chi_scale(nu: f64) -> Array1<f64> {
  let chi = SimdChiSquared::<f64>::new(nu);
  sobol_points(1).column(0).mapv(|w| {
    let q = chi.quantile(w).expect("SimdChiSquared has a quantile");
    (q / nu).sqrt()
  })
}

/// `P(T ≤ b)` for `T = Z/r`, `r = χ_ν/√ν`: the normal integrand at `r b` over `points[:, 1..]`, the scale from
/// [`chi_scale`].
pub(crate) fn mvt_upper(
  l: &Array2<f64>,
  b: &[f64],
  points: ArrayView2<f64>,
  scale: ArrayView1<f64>,
) -> f64 {
  let mut y = vec![0.0; b.len() - 1];
  let mut scaled = vec![0.0; b.len()];
  let total: f64 = points
    .rows()
    .into_iter()
    .zip(scale)
    .map(|(w, &r)| {
      // An infinite limit stays infinite where a tiny `ν` underflows the χ quantile to 0.
      for (s, &bk) in scaled.iter_mut().zip(b) {
        *s = if bk.is_infinite() { bk } else { bk * r };
      }
      let e = norm_cdf(scaled[0] / l[[0, 0]]);
      integrand(
        l,
        &scaled,
        e,
        &w.as_slice().expect("row-major")[1..],
        &mut y,
      )
    })
    .sum();
  total / points.nrows() as f64
}
