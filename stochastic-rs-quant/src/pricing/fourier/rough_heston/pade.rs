//! The rational approximation of the rough Heston Riccati solution, any diagonal order to six.
//!
//! $$
//! h^{(n,n)}(t) = \frac{\sum_{i=1}^n p_i y^i}{1 + \sum_{j=1}^n q_j y^j},\quad y = t^\alpha,\qquad
//! h = \sum_{j\ge1} b_j y^j\ (t\to0),\qquad h = \sum_{k\ge0} g_k y^{-k}\ (t\to\infty)
//! $$
//!
//! Reference: Gatheral & Radoičić (2019), IJTAF 22(3), 1950010, DOI 10.1142/S0219024919500109
//! Reference: Gatheral & Radoičić (2024), arXiv:2310.09181, §2–§4

use gauss_quad::GaussJacobi;
use ndarray::Array1;
use num_complex::Complex64;
use stochastic_rs_distributions::special::gamma;

use super::LogChf;
use super::Riccati;
use super::RoughHestonParams;
use super::scalar::RiccatiScalar;

pub(super) const MAX_ORDER: usize = 6;

/// Nodes of the time integral: 32 leave the chf within 1e-9 of 64 along Lewis's line.
const NODES: usize = 32;

/// `(p̂₁…p̂_n, q₀…q_n)`, `p = m₀p̂`: the system divided by the moment every `b_k`, `g_k` carries.
pub(super) type Coefficients<S> = ([S; MAX_ORDER], [S; MAX_ORDER + 1]);

/// `1/Γ(x)`, zero at Γ's poles.
fn reciprocal_gamma(x: f64) -> f64 {
  if x <= 0.0 && x.fract() == 0.0 {
    0.0
  } else {
    gamma(x).recip()
  }
}

/// The Gamma ratios of both recursions at one `α`, and the order.
#[derive(Clone, Debug)]
pub(super) struct Pade {
  alpha: f64,
  order: usize,
  /// `Γ(1 + (k−1)α)/Γ(1 + kα)` at `[k − 1]`; `1/Γ(1 + α)` first.
  short: [f64; MAX_ORDER],
  /// `Γ(1 − (k−1)α)/Γ(1 − kα)` at `[k]`, `k ≥ 1`; `1/Γ(1 − α)` at 1.
  long: [f64; MAX_ORDER],
}

impl Pade {
  /// `alpha` in `(½, 1)`, `1 ≤ order ≤ 6`.
  pub(super) fn new(alpha: f64, order: usize) -> Self {
    let step = |k: usize| alpha * k as f64;
    Self {
      alpha,
      order,
      short: std::array::from_fn(|i| gamma(1.0 + step(i)) / gamma(1.0 + step(i + 1))),
      long: std::array::from_fn(|k| match k {
        0 => 0.0,
        1 => reciprocal_gamma(1.0 - alpha),
        k => gamma(1.0 - step(k - 1)) * reciprocal_gamma(1.0 - step(k)),
      }),
    }
  }

  #[cfg(test)]
  pub(super) fn alpha(&self) -> f64 {
    self.alpha
  }

  /// `(b̂₁…b̂_n, ĝ₀…ĝ_{n−1})`, each series' coefficients over `m₀`, in powers of `t^α`.
  pub(super) fn series<S: RiccatiScalar>(
    &self,
    z: S,
    riccati: &Riccati,
  ) -> ([S; MAX_ORDER], [S; MAX_ORDER]) {
    let n = self.order;
    let moment = Riccati::moment(z);
    let half_square = 0.5 * riccati.nu * riccati.nu;
    let reversion = S::real(riccati.kappa) - z * (riccati.rho * riccati.nu);
    let root = (reversion * reversion - moment * (4.0 * half_square)).sqrt();
    let mut short = [S::ZERO; MAX_ORDER];
    short[0] = S::real(self.short[0]);
    for k in 1..n {
      let pairs = (0..k - 1).fold(S::ZERO, |sum, i| sum + short[i] * short[k - 2 - i]);
      short[k] = (pairs * moment * half_square - reversion * short[k - 1]) * self.short[k];
    }
    let mut long = [S::ZERO; MAX_ORDER];
    // `r₋ = 2ν²m₀/(λ̃ν + Aν)`: the difference `λ̃ − A` cancels as `m₀ → 0`, the quotient does not.
    let sum = reversion + root;
    long[0] = if sum.re() > 0.0 {
      S::real(2.0) / sum
    } else {
      (reversion - root) / (moment * (2.0 * half_square))
    };
    for k in 1..n {
      let pairs = (1..k).fold(S::ZERO, |sum, i| sum + long[i] * long[k - i]);
      long[k] = (pairs * moment * half_square - long[k - 1] * self.long[k]) / root;
    }
    (short, long)
  }

  /// The `n` matching conditions `Σ_{j<m} b_{m−j}q_j = Σ_{j≤n−m} g_j q_{m+j}`, solved for `q`.
  pub(super) fn coefficients<S: RiccatiScalar>(&self, z: S, riccati: &Riccati) -> Coefficients<S> {
    let n = self.order;
    let (short, long) = self.series(z, riccati);
    let mut rows = [[S::ZERO; MAX_ORDER + 1]; MAX_ORDER];
    for m in 1..=n {
      let row = &mut rows[m - 1];
      for j in 1..m {
        row[j - 1] = row[j - 1] + short[m - j - 1];
      }
      for j in 0..=n - m {
        row[m + j - 1] = row[m + j - 1] - long[j];
      }
      row[n] = -short[m - 1];
    }
    let solution = solve(rows, n);
    let mut q = [S::ZERO; MAX_ORDER + 1];
    q[0] = S::real(1.0);
    q[1..=n].copy_from_slice(&solution[..n]);
    let mut p = [S::ZERO; MAX_ORDER];
    for m in 1..=n {
      p[m - 1] = (0..m).fold(S::ZERO, |sum, j| sum + short[m - j - 1] * q[j]);
    }
    (p, q)
  }

  /// `h^{(n,n)}` at `y = t^α`.
  pub(super) fn h<S: RiccatiScalar>((p, q): &Coefficients<S>, moment: S, y: f64) -> S {
    let numerator = p.iter().rev().fold(S::ZERO, |sum, &c| (sum + c) * y);
    let denominator = q.iter().rev().fold(S::ZERO, |sum, &c| sum * y + c);
    moment * numerator / denominator
  }
}

/// Gaussian elimination with row pivoting on the `n × (n+1)` head of `rows`.
fn solve<S: RiccatiScalar>(mut rows: [[S; MAX_ORDER + 1]; MAX_ORDER], n: usize) -> [S; MAX_ORDER] {
  for column in 0..n {
    let pivot = (column..n)
      .max_by(|&a, &b| rows[a][column].norm().total_cmp(&rows[b][column].norm()))
      .expect("the range is not empty");
    rows.swap(column, pivot);
    for row in column + 1..n {
      let factor = rows[row][column] / rows[column][column];
      for entry in column..=n {
        rows[row][entry] = rows[row][entry] - factor * rows[column][entry];
      }
    }
  }
  let mut solution = [S::ZERO; MAX_ORDER];
  for row in (0..n).rev() {
    let known = (row + 1..n).fold(S::ZERO, |sum, j| sum + rows[row][j] * solution[j]);
    solution[row] = (rows[row][n] - known) / rows[row][row];
  }
  solution
}

/// `ln φ = ∫₀ᵗ(κθ·h + V₀·F(z, h))ds` of the approximant, by Gauss–Jacobi in `y = s^α`.
#[derive(Clone, Debug)]
pub(super) struct PadeCf {
  params: RoughHestonParams,
  riccati: Riccati,
  pade: Pade,
  /// `y/t^α` at each node.
  fractions: Array1<f64>,
  /// The rule's weights over `α·2^{1/α}`, which sum to 1.
  weights: Array1<f64>,
}

impl PadeCf {
  pub(super) fn new(params: RoughHestonParams, order: usize) -> Self {
    Self::with_nodes(params, order, NODES)
  }

  pub(super) fn with_nodes(params: RoughHestonParams, order: usize, nodes: usize) -> Self {
    let alpha = params.alpha();
    let exponent = alpha.recip() - 1.0;
    let rule = GaussJacobi::new(
      nodes.try_into().expect("a positive node count"),
      0.0.try_into().expect("zero is above −1"),
      exponent.try_into().expect("1/α − 1 lies in (0, 1)"),
    );
    let pairs = rule.as_node_weight_pairs();
    let scale = 0.5_f64.powf(alpha.recip()) / alpha;
    Self {
      params,
      riccati: Riccati::of(&params),
      pade: Pade::new(alpha, order),
      fractions: pairs.iter().map(|&(x, _)| 0.5 * (1.0 + x)).collect(),
      weights: pairs.iter().map(|&(_, w)| w * scale).collect(),
    }
  }

  /// NaN off the strip `−1 ≤ Im ξ ≤ 0` the approximant is built on; `φ(ξ) = conj φ(−ξ̄)` carries
  /// `Re ξ < 0` into it.
  pub(super) fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    if !(-1.0..=0.0).contains(&xi.im) {
      return Complex64::new(f64::NAN, f64::NAN);
    }
    if xi.re < 0.0 {
      return super::chf(self, t, -xi.conj()).conj();
    }
    super::chf(self, t, xi)
  }
}

impl LogChf for PadeCf {
  fn log_chf<S: RiccatiScalar>(&self, t: f64, z: S) -> S {
    let moment = Riccati::moment(z);
    let coefficients = self.pade.coefficients(z, &self.riccati);
    let reach = t.powf(self.pade.alpha);
    let level = self.params.kappa * self.params.theta;
    let nodes = self.fractions.iter().zip(&self.weights);
    let sum = nodes.fold(S::ZERO, |sum, (&fraction, &weight)| {
      let h = Pade::h(&coefficients, moment, reach * fraction);
      sum + (h * level + self.riccati.rhs(moment, z, h) * self.params.v0) * weight
    });
    sum * t
  }
}
