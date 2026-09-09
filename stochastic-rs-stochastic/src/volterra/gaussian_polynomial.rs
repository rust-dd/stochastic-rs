//! # Gaussian polynomial volatility
//!
//! $$
//! X_t = \int_0^t K(t-s)\,dW_s, \qquad \sigma_t = p(X_t) = \sum_{k=0}^{d} c_k X_t^k
//! $$
//!
//! Volatility as a polynomial of a *Gaussian Volterra process* — a stochastic
//! convolution of a kernel against a Brownian motion. The family is the one
//! behind the strongest published results on the joint SPX/VIX calibration
//! problem, long considered the hardest fitting exercise in volatility
//! modelling:
//!
//! - **Abi Jaber, Illand & Li (2022)**, *Joint SPX-VIX calibration with
//!   Gaussian polynomial volatility models*, arXiv:2212.08297.
//! - **Abi Jaber, Illand & Li (2022)**, *The quintic Ornstein-Uhlenbeck
//!   volatility model that jointly calibrates SPX & VIX smiles*,
//!   arXiv:2212.10917 — the degree-five case over a single fast-mean-reverting
//!   OU process.
//!
//! ## Why this is cheap to have here
//!
//! Setting $b\equiv 0$ and $\sigma\equiv 1$ in
//! [`VolterraSde`](super::sve::VolterraSde) already produces the Gaussian
//! Volterra process $X$, and the lift makes it $O(n N')$ rather than
//! $O(n^2)$. Everything this type adds on top is a polynomial evaluated
//! pointwise, so the whole family costs one Horner loop over an existing
//! primitive.
//!
//! Choosing [`ExponentialKernel`](super::kernel::ExponentialKernel) makes $X$
//! an Ornstein–Uhlenbeck process **exactly** — the exponential kernel is
//! represented by a single mode with no approximation error — which is the
//! quintic model's own setting.
//!
//! ## Scope
//!
//! This type is the **volatility** process $\sigma_t$. The price leg
//! $dS_t = \sigma_t S_t\,dW^S_t$ with $d\langle W, W^S\rangle = \rho\,dt$ is
//! not included; a correlated two-dimensional output is a separate piece of
//! work, and stating that plainly is better than shipping half of it under a
//! name that implies the whole model.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::noise::gn::Gn;
use crate::rough::markov_lift::RoughSimd;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::kernel::VolterraKernel;
use crate::volterra::lift::VolterraLift;

/// Volatility as a polynomial of a Gaussian Volterra process.
pub struct GaussianPolynomialVolatility<T: FloatExt, K, S: SeedExt = Unseeded, B = Cpu>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Kernel $K$ of the driving Gaussian Volterra process.
  pub kernel: K,
  /// Polynomial coefficients $c_0,\dots,c_d$ in ascending order, so
  /// `coefficients[k]` multiplies $X^k$.
  pub coefficients: Array1<T>,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Simulation horizon $[0, t]$ (defaults to $1$ when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The Markov lift of `kernel` at the grid spacing: the host sampler steps
  /// it and a device replays it node by node, so both run one object. The
  /// grid setters rebuild it; a direct write to `n` or `t` leaves it stale.
  pub(crate) lift: VolterraLift<T, K>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, K, S: SeedExt> Clone for GaussianPolynomialVolatility<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
  S: Clone,
{
  /// Snapshot semantics, matching every other process in this crate.
  fn clone(&self) -> Self {
    Self {
      backend: Cpu,
      kernel: self.kernel.clone(),
      coefficients: self.coefficients.clone(),
      n: self.n,
      t: self.t,
      seed: self.seed.clone(),
      lift: self.lift.clone(),
    }
  }
}

impl<T: FloatExt, K, S: SeedExt> GaussianPolynomialVolatility<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// # Panics
  /// - if `n < 2`
  /// - if `coefficients` is empty (a polynomial needs at least a constant term)
  #[must_use]
  pub fn new(kernel: K, coefficients: Array1<T>, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(
      !coefficients.is_empty(),
      "coefficients must contain at least a constant term"
    );
    let lift = VolterraLift::new(kernel.clone(), grid_spacing(n, t));
    Self {
      backend: Cpu,
      kernel,
      coefficients,
      n,
      t,
      seed,
      lift,
    }
  }

  /// The quintic parameterisation of arXiv:2212.10917.
  ///
  /// The model's polynomial is **sparse**, not a general quintic: the paper
  /// fixes the quadratic and quartic terms at zero, so
  ///
  /// $$ p(x) = \alpha_0 + \alpha_1 x + \alpha_3 x^3 + \alpha_5 x^5 . $$
  ///
  /// This constructor therefore takes those four coefficients and fills
  /// degrees 2 and 4 with zero, rather than accepting six free values —
  /// passing a dense six-coefficient polynomial would produce a strictly
  /// larger family than the one the citation names. (The paper's "six
  /// parameters" are $\{\rho, H, \alpha_0, \alpha_1, \alpha_3, \alpha_5\}$,
  /// counting the correlation and Hurst exponent, neither of which lives on
  /// this type — see the scope note in the module docs.)
  ///
  /// The calibrated instance in the paper's Figure 1 is
  /// $(\alpha_0, \alpha_1, \alpha_3, \alpha_5) = (0.5907, 1, 0.2893, 0.0549)$
  /// at $\rho = -0.6843$, $H = -0.0358$.
  ///
  /// Use [`new`](Self::new) directly for an unrestricted degree-five
  /// polynomial; it is a different model and this crate does not claim the
  /// paper's results for it.
  ///
  /// # Panics
  /// - under the same conditions as [`new`](Self::new)
  #[must_use]
  pub fn quintic(
    kernel: K,
    alpha0: T,
    alpha1: T,
    alpha3: T,
    alpha5: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    let coefficients = Array1::from_vec(vec![alpha0, alpha1, T::zero(), alpha3, T::zero(), alpha5]);
    Self::new(kernel, coefficients, n, t, seed)
  }
}

impl<T: FloatExt, K, S: SeedExt, B> GaussianPolynomialVolatility<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Replace the polynomial, all else unchanged.
  ///
  /// # Panics
  /// - if `coefficients` is empty
  #[must_use]
  pub fn with_coefficients(mut self, coefficients: Array1<T>) -> Self {
    assert!(
      !coefficients.is_empty(),
      "coefficients must contain at least a constant term"
    );
    self.coefficients = coefficients;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  ///
  /// # Panics
  /// - if `n < 2`
  #[must_use]
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self.lift = VolterraLift::new(self.kernel.clone(), grid_spacing(n, self.t));
    self
  }

  /// Replace the horizon, all else unchanged.
  #[must_use]
  pub fn with_horizon(mut self, t: T) -> Self {
    self.t = Some(t);
    self.lift = VolterraLift::new(self.kernel.clone(), grid_spacing(self.n, Some(t)));
    self
  }

  /// Replace the seed strategy, all else unchanged.
  #[must_use]
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Evaluate $p(x)$ by Horner's rule.
  ///
  /// Horner rather than a naive power sum because the quintic case raises $x$
  /// to the fifth, and a fast-mean-reverting driver with large vol-of-vol —
  /// the regime arXiv:2212.10917 calibrates in — makes $|x|$ large enough for
  /// the cancellation to matter.
  #[must_use]
  pub fn evaluate_polynomial(&self, x: T) -> T {
    let mut acc = T::zero();
    for c in self.coefficients.iter().rev() {
      acc = acc * x + *c;
    }
    acc
  }
}

/// The grid spacing of `n` points over `[0, t]`, `t` defaulting to one.
fn grid_spacing<T: FloatExt>(n: usize, t: Option<T>) -> T {
  t.unwrap_or(T::one()) / T::from_usize_(n - 1)
}

/// The most coefficients a device evaluates: the kernels unroll Horner's
/// rule over this many slots, and a longer polynomial samples on the host.
pub const DEVICE_COEFFICIENTS: usize = 8;

impl<T: FloatExt + RoughSimd, K, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for GaussianPolynomialVolatility<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// The coefficients in rising order, padded with zeros to the kernels'
  /// slots. A longer polynomial never reaches a launch — [`ProcessExt::sample`]
  /// keeps it on the host — so asking here is a caller bypassing that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    assert!(
      self.device_ready(),
      "GaussianPolynomialVolatility: a device evaluates at most {DEVICE_COEFFICIENTS} \
       coefficients; sample through `ProcessExt`, which keeps a longer polynomial on the host"
    );
    let mut coefficients = [T::zero(); DEVICE_COEFFICIENTS];
    coefficients[..self.coefficients.len()].copy_from_slice(
      self
        .coefficients
        .as_slice()
        .expect("coefficients must be contiguous"),
    );
    crate::euler::EulerSpec::GaussianPolynomialVolatility { coefficients }
  }

  /// The reported path starts at the polynomial of a zero Gaussian, its
  /// constant term, as on the host.
  fn initial_value(&self) -> T {
    self.coefficients[0]
  }

  /// The state is the lifted Gaussian itself, which starts at zero; the first
  /// point comes out of the report expression, not this slot.
  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The lift's nodes, decays and weights, and the boundary terms of its
  /// first step, exactly as the host's stepper holds them.
  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, T>> {
    Some(crate::euler::LiftSpec {
      decay: self.lift.exp_neg_x_dt.as_slice().expect("contiguous"),
      weight: self.lift.we.as_slice().expect("contiguous"),
      drift_scale: self.lift.one_minus_e_over_x.as_slice().expect("contiguous"),
      drift_boundary: self.lift.drift_boundary,
      diffusion_boundary: self.lift.diffusion_boundary,
      x0: T::zero(),
    })
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt + RoughSimd, K, S: SeedExt] GaussianPolynomialVolatility<T, K, S> { kernel, coefficients, n, t, seed, lift } via euler where  K: VolterraKernel<T> + Send + Sync);

impl<T: FloatExt + RoughSimd, K, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for GaussianPolynomialVolatility<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = GaussianPolynomialVolatilitySampler<T, K, S>
  where
    Self: 's;

  fn sampler(&self) -> GaussianPolynomialVolatilitySampler<T, K, S> {
    GaussianPolynomialVolatilitySampler {
      n: self.n,
      coefficients: self.coefficients.clone(),
      lift: self.lift.clone(),
      gn: Gn::<T, S> {
        backend: Cpu,
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
    }
  }

  /// Through the Euler engine when the polynomial fits the kernels'
  /// coefficient slots; anything else keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_sample(&self.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map_view(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_sample(&self.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_euler_paths(&self.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: the polynomial fits the kernels'
  /// coefficient slots.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.coefficients.len() <= DEVICE_COEFFICIENTS))
      .then_some("more coefficients than a launch carries")
  }
}

/// Reusable [`GaussianPolynomialVolatility`] sampling state.
#[doc(hidden)]
pub struct GaussianPolynomialVolatilitySampler<T: FloatExt + RoughSimd, K, S: SeedExt>
where
  K: VolterraKernel<T> + Send + Sync,
{
  n: usize,
  coefficients: Array1<T>,
  lift: VolterraLift<T, K>,
  gn: Gn<T, S>,
}

impl<T: FloatExt + RoughSimd, K, S: SeedExt> GaussianPolynomialVolatilitySampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let dw = self.gn.sample();
    let path = self.lift.simulate(
      T::zero(),
      |_, _| T::zero(),
      |_, _| T::one(),
      dw.as_slice().expect("dw must be contiguous"),
    );
    for (o, x) in out
      .iter_mut()
      .zip(path.as_slice().expect("lift path must be contiguous"))
    {
      let mut acc = T::zero();
      for c in self.coefficients.iter().rev() {
        acc = acc * *x + *c;
      }
      *o = acc;
    }
  }
}

impl<T: FloatExt + RoughSimd, K, S: SeedExt> PathSampler<T>
  for GaussianPolynomialVolatilitySampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("GaussianPolynomialVolatility output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
#[path = "gaussian_polynomial_tests.rs"]
mod tests;
