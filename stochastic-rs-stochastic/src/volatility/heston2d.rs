//! $$
//! \begin{aligned}
//! dx^i_t &= \left(\mu_i - \tfrac{1}{2} v^i_t\right) dt + \sqrt{v^i_t}\, dW^i_t \\
//! dv^i_t &= \kappa_i(\theta_i - v^i_t)\, dt + \sigma_i \sqrt{v^i_t}\, dZ^i_t, \quad i=1,2
//! \end{aligned}
//! $$
//!
//! After `Heston2D.m` of the MATLAB FSDA toolbox: the output is log-prices `x_i`, as `FmVol`
//! expects, and a negative Euler variance is reflected or truncated, unlike MATLAB's raw Euler.
//!
//! Reference: Sanfelici & Toscano, arXiv:2402.00172

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Two correlated Heston SDEs under a 4×4 correlation of `(Z_1, Z_2, W_1, W_2)`, given as MATLAB's
/// six-element `rho`; sampling gives the [`Heston2DPaths`] of the two log-prices and variances.
#[derive(Debug, Clone)]
pub struct Heston2D<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial log-prices `[x_1(0), x_2(0)]`.
  pub x0: [T; 2],
  /// Initial variances `[v_1(0), v_2(0)]`, strictly positive as in the MATLAB source.
  pub v0: [T; 2],
  /// Drifts `[μ_1, μ_2]` of the log-price processes.
  pub mu: [T; 2],
  /// Long-run variances `[θ_1, θ_2]`.
  pub theta: [T; 2],
  /// Mean-reversion speeds `[κ_1, κ_2]`.
  pub kappa: [T; 2],
  /// Volatility-of-volatilities `[σ_1, σ_2]`.
  pub sigma: [T; 2],
  /// Correlation vector in MATLAB `Heston2D.m` order: `[ρ(Z_1,Z_2), ρ(Z_1,W_1), ρ(Z_1,W_2),
  /// ρ(Z_2,W_1), ρ(Z_2,W_2), ρ(W_1,W_2)]`.
  pub rho: [T; 6],
  /// Number of points (so `n - 1` time steps).
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// How a negative square-root state is treated after a step; Feller's condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy.
  pub seed: S,
  /// Cholesky factor `L` of the 4×4 correlation matrix, stored as
  /// `[L_11, L_21, L_22, L_31, L_32, L_33, L_41, L_42, L_43, L_44]`.
  chol: [T; 10],
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Log-prices and variances of the two assets.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Heston2DPaths<T> {
  pub x1: Array1<T>,
  pub v1: Array1<T>,
  pub x2: Array1<T>,
  pub v2: Array1<T>,
}

impl<T> From<[Array1<T>; 4]> for Heston2DPaths<T> {
  fn from([x1, v1, x2, v2]: [Array1<T>; 4]) -> Self {
    Self { x1, v1, x2, v2 }
  }
}

/// Lower Cholesky factor of the 4×4 correlation matrix of `rho` over `(Z_1, Z_2, W_1, W_2)`;
/// panics if the matrix is not positive semidefinite.
fn cholesky_4x4<T: FloatExt>(rho: [T; 6]) -> [T; 10] {
  // A valid rank-deficient correlation matrix can produce a tiny negative pivot after rounding.
  let tol = T::from_f64_fast(1e-10).max(T::from_f64_fast(100.0) * T::epsilon());
  let nonneg = |x: T, pivot: usize| {
    assert!(x >= -tol, "correlation matrix not PSD at pivot {}", pivot);
    x.max(T::zero())
  };
  let divide_or_zero = |residual: T, diagonal: T, pivot: usize| {
    if diagonal > T::zero() {
      residual / diagonal
    } else {
      assert!(
        residual.abs() <= tol,
        "correlation matrix not PSD at pivot {}",
        pivot
      );
      T::zero()
    }
  };

  let l11 = T::one();
  let l21 = rho[0];
  let l22 = nonneg(T::one() - l21 * l21, 2).sqrt();
  let l31 = rho[1];
  let l32 = divide_or_zero(rho[3] - l31 * l21, l22, 2);
  let l33 = nonneg(T::one() - l31 * l31 - l32 * l32, 3).sqrt();
  let l41 = rho[2];
  let l42 = divide_or_zero(rho[4] - l41 * l21, l22, 2);
  let l43 = divide_or_zero(rho[5] - l41 * l31 - l42 * l32, l33, 3);
  let l44 = nonneg(T::one() - l41 * l41 - l42 * l42 - l43 * l43, 4).sqrt();
  [l11, l21, l22, l31, l32, l33, l41, l42, l43, l44]
}

/// Validates every hard precondition except the Feller condition, which [`Heston2D::new`]
/// checks separately and accepts.
fn validate_params<T: FloatExt>(
  x0: &[T; 2],
  v0: &[T; 2],
  mu: &[T; 2],
  theta: &[T; 2],
  kappa: &[T; 2],
  sigma: &[T; 2],
  rho: &[T; 6],
  n: usize,
  t: Option<T>,
) {
  assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
  assert!(
    t.is_none_or(|t| t > T::zero() && t.is_finite()),
    "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
  );
  for i in 0..2 {
    assert!(
      x0[i].is_finite(),
      "x0[{i}] must satisfy `x0[i].is_finite()`, got x0[{i}] = {:?}",
      x0[i]
    );
    let value = v0[i];
    assert!(
      value.is_finite(),
      "v0[{i}] must satisfy `v0[i].is_finite()`, got v0[{i}] = {value:?}"
    );
    assert!(
      value > T::zero(),
      "v0[{i}] must satisfy `v0[i] > T::zero()`, got v0[{i}] = {value:?}"
    );
    assert!(
      mu[i].is_finite(),
      "mu[{i}] must satisfy `mu[i].is_finite()`, got mu[{i}] = {:?}",
      mu[i]
    );
    assert!(
      theta[i].is_finite(),
      "theta[{i}] must satisfy `theta[i].is_finite()`, got theta[{i}] = {:?}",
      theta[i]
    );
    assert!(
      kappa[i].is_finite(),
      "kappa[{i}] must satisfy `kappa[i].is_finite()`, got kappa[{i}] = {:?}",
      kappa[i]
    );
    assert!(
      sigma[i].is_finite(),
      "sigma[{i}] must satisfy `sigma[i].is_finite()`, got sigma[{i}] = {:?}",
      sigma[i]
    );
    assert!(
      kappa[i] >= T::zero(),
      "kappa[{i}] must satisfy `kappa[i] >= T::zero()`, got kappa[{i}] = {:?}",
      kappa[i]
    );
    assert!(
      theta[i] >= T::zero(),
      "theta[{i}] must satisfy `theta[i] >= T::zero()`, got theta[{i}] = {:?}",
      theta[i]
    );
    assert!(
      sigma[i] >= T::zero(),
      "sigma[{i}] must satisfy `sigma[i] >= T::zero()`, got sigma[{i}] = {:?}",
      sigma[i]
    );
  }
  for (k, r) in rho.iter().enumerate() {
    assert!(
      r.is_finite(),
      "rho[{k}] must satisfy `rho[k].is_finite()`, got rho[{k}] = {r:?}"
    );
    assert!(
      *r >= -T::one() && *r <= T::one(),
      "rho[{k}] must satisfy `rho[k] >= -T::one() && rho[k] <= T::one()`, got rho[{k}] = {r:?}"
    );
  }
}

/// A `log::warn!` per asset whose variance violates Feller's `2·kappa·theta ≥ sigma²` under
/// [`VarianceBoundary::Floor`], as [`crate::diffusion::cir::Cir::new`] warns.
fn warn_on_feller_violation<T: FloatExt>(
  kappa: &[T; 2],
  theta: &[T; 2],
  sigma: &[T; 2],
  boundary: VarianceBoundary,
) {
  if boundary == VarianceBoundary::Reflect {
    return;
  }
  for i in 0..2 {
    let feller_lhs = T::from_f64_fast(2.0) * kappa[i] * theta[i];
    if feller_lhs < sigma[i] * sigma[i] {
      log::warn!(
        "Heston2D::new: asset {i} does not satisfy the Feller \
         condition (2*kappa*theta < sigma^2) under VarianceBoundary::Floor; \
         its variance floors at zero on every boundary hit instead \
         of reflecting — pass VarianceBoundary::Reflect for the standard \
         sub-Feller mitigation"
      );
    }
  }
}

impl<T: FloatExt, S: SeedExt> Heston2D<T, S> {
  /// Panics unless every entry is finite, `v0 > 0`, `kappa`, `theta`, `sigma` are non-negative,
  /// `rho` is a positive semidefinite correlation, `n >= 2` and a given `t` is positive.
  pub fn new(
    x0: [T; 2],
    v0: [T; 2],
    mu: [T; 2],
    theta: [T; 2],
    kappa: [T; 2],
    sigma: [T; 2],
    rho: [T; 6],
    n: usize,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    validate_params(&x0, &v0, &mu, &theta, &kappa, &sigma, &rho, n, t);
    warn_on_feller_violation(&kappa, &theta, &sigma, boundary);
    let chol = cholesky_4x4::<T>(rho);
    Self {
      backend: Cpu,
      x0,
      v0,
      mu,
      theta,
      kappa,
      sigma,
      rho,
      n,
      t,
      boundary,
      seed,
      chol,
    }
  }
}

/// The Euler engine's view: the Cholesky factor is computed once at construction and travels as
/// parameters, so the kernel combines four independent shocks as the host sampler does.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 4>
  for Heston2D<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let [l11, l21, l22, l31, l32, l33, l41, l42, l43, l44] = self.chol;
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::TwoAssetHeston {
        mu1: self.mu[0],
        mu2: self.mu[1],
        kappa1: self.kappa[0],
        theta1: self.theta[0],
        sigma1: self.sigma[0],
        kappa2: self.kappa[1],
        theta2: self.theta[1],
        sigma2: self.sigma[1],
        l11,
        l21,
        l22,
        l31,
        l32,
        l33,
        l41,
        l42,
        l43,
        l44,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::TwoAssetHestonReflected {
        mu1: self.mu[0],
        mu2: self.mu[1],
        kappa1: self.kappa[0],
        theta1: self.theta[0],
        sigma1: self.sigma[0],
        kappa2: self.kappa[1],
        theta2: self.theta[1],
        sigma2: self.sigma[1],
        l11,
        l21,
        l22,
        l31,
        l32,
        l33,
        l41,
        l42,
        l43,
        l44,
      },
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.x0[0], self.v0[0], self.x0[1], self.v0[1]]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Heston2DPaths<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Heston2D<T, S> { x0, v0, mu, theta, kappa, sigma, rho, n, t, boundary, seed, chol } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Heston2D<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Heston2D<T, S, B>
{
  type Output = Heston2DPaths<T>;
  type Sampler<'s>
    = Heston2DSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> Heston2DSampler<T> {
    let t_total = self.t.unwrap_or(T::one());
    let n_steps = self.n - 1;
    let dt = t_total / T::from_usize_(n_steps);
    let sqrt_dt = dt.sqrt();
    // The derived streams are drawn in the order e1..e4, which seeded reproducibility depends on.
    let normals =
      std::array::from_fn(|_| SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed.derive()));
    Heston2DSampler {
      n: self.n,
      x0: self.x0,
      v0: self.v0,
      mu: self.mu,
      theta: self.theta,
      kappa: self.kappa,
      sigma: self.sigma,
      chol: self.chol,
      dt,
      boundary: self.boundary,
      normals,
    }
  }

  /// Through the Euler engine: a device steps all four components in its kernel, the host runs
  /// this process's own sampler, chunked as `ProcessExt` chunks.
  fn sample(&self) -> Heston2DPaths<T> {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Heston2DPaths<T>) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Heston2DPaths<T>> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<Heston2DPaths<T>, crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Heston2DPaths<T>>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`Heston2D`] sampling state: the four independent Gaussian streams, the Cholesky
/// factor and the step size.
#[doc(hidden)]
#[derive(Debug)]
pub struct Heston2DSampler<T: FloatExt> {
  n: usize,
  x0: [T; 2],
  v0: [T; 2],
  mu: [T; 2],
  theta: [T; 2],
  kappa: [T; 2],
  sigma: [T; 2],
  chol: [T; 10],
  dt: T,
  boundary: VarianceBoundary,
  normals: [Seeded<SimdNormal<T>>; 4],
}

impl<T: FloatExt> Heston2DSampler<T> {
  fn fill_paths(&mut self, x1: &mut [T], v1: &mut [T], x2: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let n_steps = self.n - 1;
    let mut e1 = Array1::<T>::zeros(n_steps);
    let mut e2 = Array1::<T>::zeros(n_steps);
    let mut e3 = Array1::<T>::zeros(n_steps);
    let mut e4 = Array1::<T>::zeros(n_steps);
    let [n1, n2, n3, n4] = &mut self.normals;
    n1.fill_slice(e1.as_slice_mut().expect("noise slice contiguous"));
    n2.fill_slice(e2.as_slice_mut().expect("noise slice contiguous"));
    n3.fill_slice(e3.as_slice_mut().expect("noise slice contiguous"));
    n4.fill_slice(e4.as_slice_mut().expect("noise slice contiguous"));

    let [l11, l21, l22, l31, l32, l33, l41, l42, l43, l44] = self.chol;

    x1[0] = self.x0[0];
    v1[0] = self.v0[0];
    x2[0] = self.x0[1];
    v2[0] = self.v0[1];

    let dt = self.dt;
    let half = T::from_f64_fast(0.5);
    for i in 1..self.n {
      let dz1 = l11 * e1[i - 1];
      let dz2 = l21 * e1[i - 1] + l22 * e2[i - 1];
      let dw1 = l31 * e1[i - 1] + l32 * e2[i - 1] + l33 * e3[i - 1];
      let dw2 = l41 * e1[i - 1] + l42 * e2[i - 1] + l43 * e3[i - 1] + l44 * e4[i - 1];

      let v1_prev = v1[i - 1].max(T::zero());
      let v2_prev = v2[i - 1].max(T::zero());

      let dv1 =
        self.kappa[0] * (self.theta[0] - v1_prev) * dt + self.sigma[0] * v1_prev.sqrt() * dz1;
      let dv2 =
        self.kappa[1] * (self.theta[1] - v2_prev) * dt + self.sigma[1] * v2_prev.sqrt() * dz2;

      (v1[i], v2[i]) = match self.boundary {
        VarianceBoundary::Floor => (
          (v1[i - 1] + dv1).max(T::zero()),
          (v2[i - 1] + dv2).max(T::zero()),
        ),
        VarianceBoundary::Reflect => ((v1[i - 1] + dv1).abs(), (v2[i - 1] + dv2).abs()),
      };

      x1[i] = x1[i - 1] + (self.mu[0] - half * v1_prev) * dt + v1_prev.sqrt() * dw1;
      x2[i] = x2[i - 1] + (self.mu[1] - half * v2_prev) * dt + v2_prev.sqrt() * dw2;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for Heston2DSampler<T> {}

impl<T: FloatExt> PathSampler<T> for Heston2DSampler<T> {
  type Output = Heston2DPaths<T>;

  fn sample_into(&mut self, out: &mut Heston2DPaths<T>) {
    let Heston2DPaths { x1, v1, x2, v2 } = out;
    self.fill_paths(
      x1.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
      v1.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
      x2.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
      v2.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Heston2DPaths<T> {
    let mut x1 = Array1::<T>::zeros(self.n);
    let mut v1 = Array1::<T>::zeros(self.n);
    let mut x2 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      x1.as_slice_mut().expect("contiguous"),
      v1.as_slice_mut().expect("contiguous"),
      x2.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
    Heston2DPaths { x1, v1, x2, v2 }
  }
}

#[cfg(test)]
mod tests;
