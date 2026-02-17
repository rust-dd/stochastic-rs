//! # Traits
//!
//! $$
//! \text{Trait contracts: }\mathcal{A}:\text{inputs}\to\text{samples/prices/statistics}
//! $$
//!
use std::cmp::Ordering;
use std::error::Error;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::SubAssign;

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;
use ndarray::parallel::prelude::*;
use ndarray::stack;
use ndarray::Array1;
#[cfg(feature = "cuda")]
use ndarray::Array2;
use ndarray::Axis;
use ndarray::ScalarOperand;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use ndrustfft::Zero;
use num_complex::Complex;
use num_complex::Complex64;
use rand::Rng;
use rand_distr::Uniform;
use roots::find_root_brent;
use roots::SimpleConvergency;

use crate::copulas::bivariate::CopulaType as BivariateCopulaType;
use crate::copulas::multivariate::CopulaType as MultivariateCopulaType;
use crate::quant::OptionType;
use crate::stochastic::noise::gn::Gn;

pub enum Fn1D<T: FloatExt> {
  Native(fn(T) -> T),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

impl<T: FloatExt> Fn1D<T> {
  pub fn call(&self, t: T) -> T {
    match self {
      Fn1D::Native(f) => f(t),
      #[cfg(feature = "python")]
      Fn1D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(),))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T) -> T> for Fn1D<T> {
  fn from(f: fn(T) -> T) -> Self {
    Fn1D::Native(f)
  }
}

pub enum Fn2D<T: FloatExt> {
  Native(fn(T, T) -> T),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

impl<T: FloatExt> Fn2D<T> {
  pub fn call(&self, t: T, u: T) -> T {
    match self {
      Fn2D::Native(f) => f(t, u),
      #[cfg(feature = "python")]
      Fn2D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(), u.to_f64().unwrap()))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T, T) -> T> for Fn2D<T> {
  fn from(f: fn(T, T) -> T) -> Self {
    Fn2D::Native(f)
  }
}

#[cfg(feature = "python")]
pub struct CallableDist<T: FloatExt> {
  callable: pyo3::Py<pyo3::PyAny>,
  _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "python")]
impl<T: FloatExt> CallableDist<T> {
  pub fn new(callable: pyo3::Py<pyo3::PyAny>) -> Self {
    Self {
      callable,
      _phantom: std::marker::PhantomData,
    }
  }
}

#[cfg(feature = "python")]
impl<T: FloatExt> rand_distr::Distribution<T> for CallableDist<T> {
  fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
    pyo3::Python::attach(|py| {
      let result: f64 = self.callable.call0(py).unwrap().extract::<f64>(py).unwrap();
      T::from_f64_fast(result)
    })
  }
}

pub trait SimdFloatExt: num_traits::Float + Default + Send + Sync + 'static {
  type Simd: Copy
    + std::ops::Mul<Output = Self::Simd>
    + std::ops::Add<Output = Self::Simd>
    + std::ops::Sub<Output = Self::Simd>
    + std::ops::Div<Output = Self::Simd>
    + std::ops::Neg<Output = Self::Simd>;

  fn splat(val: Self) -> Self::Simd;
  fn simd_from_array(arr: [Self; 8]) -> Self::Simd;
  fn simd_to_array(v: Self::Simd) -> [Self; 8];
  fn simd_ln(v: Self::Simd) -> Self::Simd;
  fn simd_sqrt(v: Self::Simd) -> Self::Simd;
  fn simd_cos(v: Self::Simd) -> Self::Simd;
  fn simd_sin(v: Self::Simd) -> Self::Simd;
  fn simd_exp(v: Self::Simd) -> Self::Simd;
  fn simd_tan(v: Self::Simd) -> Self::Simd;
  fn simd_max(a: Self::Simd, b: Self::Simd) -> Self::Simd;
  fn simd_powf(v: Self::Simd, exp: Self) -> Self::Simd;
  fn simd_floor(v: Self::Simd) -> Self::Simd;
  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [Self]);
  fn fill_uniform_simd(rng: &mut crate::simd_rng::SimdRng, out: &mut [Self]);
  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self;
  fn simd_from_i32x8(v: wide::i32x8) -> Self::Simd;
  fn from_f64_fast(v: f64) -> Self;
  fn pi() -> Self;
  fn two_pi() -> Self;
  fn min_positive_val() -> Self;
}

pub trait FloatExt:
  num_traits::Float
  + num_traits::FromPrimitive
  + num_traits::Signed
  + num_traits::FloatConst
  + Sum
  + SimdFloatExt
  + Zero
  + Default
  + Debug
  + Send
  + Sync
  + ScalarOperand
  + AddAssign
  + SubAssign
  + 'static
{
  fn from_usize_(n: usize) -> Self;
  fn fill_standard_normal_slice(out: &mut [Self]);
  fn with_fgn_complex_scratch<R, F: FnOnce(&mut [Complex<Self>]) -> R>(len: usize, f: F) -> R;
  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self>;
}

pub trait ProcessExt<T: FloatExt>: Send + Sync {
  type Output: Send;

  fn sample(&self) -> Self::Output;

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    (0..m).into_par_iter().map(|_| self.sample()).collect()
  }

  #[cfg(feature = "cuda")]
  fn sample_cuda(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("CUDA sampling is not supported for this process")
  }
}

pub trait DistributionExt {
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    Complex64::new(0.0, 0.0)
  }

  fn pdf(&self, _x: f64) -> f64 {
    0.0
  }

  fn cdf(&self, _x: f64) -> f64 {
    0.0
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    0.0
  }

  fn mean(&self) -> f64 {
    0.0
  }

  fn median(&self) -> f64 {
    0.0
  }

  fn mode(&self) -> f64 {
    0.0
  }

  fn variance(&self) -> f64 {
    0.0
  }

  fn skewness(&self) -> f64 {
    0.0
  }

  fn kurtosis(&self) -> f64 {
    0.0
  }

  fn entropy(&self) -> f64 {
    0.0
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    0.0
  }
}

pub trait BivariateExt {
  fn r#type(&self) -> BivariateCopulaType;

  fn tau(&self) -> Option<f64>;

  fn set_tau(&mut self, tau: f64);

  fn theta(&self) -> Option<f64>;

  fn theta_bounds(&self) -> (f64, f64);

  fn invalid_thetas(&self) -> Vec<f64>;

  fn set_theta(&mut self, theta: f64);

  fn check_theta(&self) -> Result<(), String> {
    let (lower, upper) = self.theta_bounds();
    let theta = self.theta().unwrap();
    let invalid = self.invalid_thetas();

    if !(lower <= theta && theta <= upper) || invalid.contains(&theta) {
      return Err(format!(
        "Theta must be in the interval [{}, {}] and not in {:?}",
        lower, upper, invalid
      ));
    }

    Ok(())
  }

  fn compute_theta(&self) -> f64;

  fn _compute_theta(&mut self) {
    self.set_theta(self.compute_theta());
    let _ = self.check_theta();
  }

  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn sample(&mut self, n: usize) -> Result<ndarray::Array2<f64>, Box<dyn Error>> {
    if self.tau().is_none() {
      return Err("Tau is not defined".into());
    }

    let tau = self.tau().unwrap();

    if !(-1.0..1.0).contains(&tau) {
      return Err("Tau must be in the interval (-1, 1)".into());
    }

    let v = Array1::<f64>::random(n, Uniform::new(0.0, 1.0)?);
    let c = Array1::<f64>::random(n, Uniform::new(0.0, 1.0)?);
    let u = self.percent_point(&c, &v)?;

    Ok(stack![Axis(1), u, v])
  }

  fn fit(&mut self, X: &ndarray::Array2<f64>) -> Result<(), Box<dyn Error>> {
    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    self.check_marginal(&U)?;
    self.check_marginal(&V)?;

    let (tau, ..) = kendalls::tau_b_with_comparator(&U.to_vec(), &V.to_vec(), |a, b| {
      a.partial_cmp(b).unwrap_or(Ordering::Greater)
    })?;

    self.set_tau(tau);
    self._compute_theta();

    Ok(())
  }

  fn check_fit(&self) -> Result<(), Box<dyn Error>> {
    if self.theta().is_none() {
      return Err("Fit the copula first".into());
    }

    self.check_theta()?;
    Ok(())
  }

  fn check_marginal(&self, u: &Array1<f64>) -> Result<(), String> {
    if !(0.0..=1.0).contains(u.min().unwrap()) || !(0.0..=1.0).contains(u.max().unwrap()) {
      return Err("Marginal values must be in the interval [0, 1]".into());
    }

    let mut empirical_cdf = u.to_vec();
    empirical_cdf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let empirical_cdf = Array1::from(empirical_cdf);
    let uniform = Array1::linspace(0.0, 1.0, u.len());
    let ks = (empirical_cdf - uniform).mapv(f64::abs);
    let ks = ks.max().unwrap();

    if *ks > 1.627 / (u.len() as f64).sqrt() {
      return Err("Marginal values do not follow a uniform distribution".into());
    }

    Ok(())
  }

  fn pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn log_pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let n = y.len();
    let mut results = Array1::zeros(n);

    for i in 0..n {
      let y_i = y[i];
      let v_i = V[i];

      let f = |u| self.partial_derivative_scalar(u, v_i).unwrap() - y_i;
      let mut convergency = SimpleConvergency {
        eps: f64::EPSILON,
        max_iter: 50,
      };
      let min = find_root_brent(f64::EPSILON, 1.0, f, &mut convergency);
      results[i] = min.unwrap_or(f64::EPSILON);
    }

    Ok(results)
  }

  fn ppf(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.percent_point(y, V)
  }

  fn partial_derivative(
    &self,
    X: &ndarray::Array2<f64>,
  ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let n = X.nrows();
    let mut X_prime = X.clone();
    let mut delta = Array1::zeros(n);
    for i in 0..n {
      delta[i] = if X[[i, 1]] > 0.5 { -0.0001 } else { 0.0001 };
      X_prime[[i, 1]] = X[[i, 1]] + delta[i];
    }

    let f = self.cdf(X).unwrap();
    let f_prime = self.cdf(&X_prime).unwrap();

    let mut deriv = Array1::zeros(n);
    for i in 0..n {
      deriv[i] = (f_prime[i] - f[i]) / delta[i];
    }

    Ok(deriv)
  }

  fn partial_derivative_scalar(&self, U: f64, V: f64) -> Result<f64, Box<dyn Error>> {
    self.check_fit()?;
    let X = stack![Axis(1), Array1::from(vec![U]), Array1::from(vec![V])];
    let out = self.partial_derivative(&X);

    Ok(*out?.get(0).unwrap())
  }
}

pub trait MultivariateExt {
  fn r#type(&self) -> MultivariateCopulaType;

  fn sample(&self, n: usize) -> Result<ndarray::Array2<f64>, Box<dyn Error>>;

  fn fit(&mut self, X: ndarray::Array2<f64>) -> Result<(), Box<dyn Error>>;

  fn check_fit(&self, X: &ndarray::Array2<f64>) -> Result<(), Box<dyn Error>>;

  fn pdf(&self, X: ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn log_pdf(&self, X: ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;
}

pub trait NCopula2DExt {
  fn sample(&self, n: usize) -> ndarray::Array2<f64>;

  fn get_params(&self) -> Vec<f64>;
}

pub trait PricerExt: TimeExt {
  fn calculate_call_put(&self) -> (f64, f64);

  fn calculate_price(&self) -> f64;

  fn derivatives(&self) -> Vec<f64> {
    vec![]
  }

  fn implied_volatility(&self, _c_price: f64, _option_type: OptionType) -> f64 {
    0.0
  }
}

pub trait TimeExt {
  fn tau(&self) -> Option<f64>;

  fn eval(&self) -> Option<chrono::NaiveDate> {
    None
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    None
  }

  fn tau_or_from_dates(&self) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => x.signed_duration_since(e).num_days() as f64 / 365.0,
      _ => panic!("either tau or both eval and expiration must be set"),
    }
  }

  fn calculate_tau_in_days(&self) -> f64 {
    self.tau_or_from_dates() * 365.0
  }

  fn calculate_tau_in_years(&self) -> f64 {
    self.tau_or_from_dates()
  }
}

pub trait MalliavinExt<T: FloatExt> {
  fn sample_with_noise(&self, noise: &Array1<T>) -> Array1<T>;

  fn n(&self) -> usize;

  fn t(&self) -> Option<T>;

  fn malliavin_derivative<F>(&self, f: F, epsilon: T) -> Array1<T>
  where
    F: Fn(&Array1<T>) -> T,
  {
    let gn = Gn::new(self.n() - 1, self.t());
    let mut noise = gn.sample();
    let path = self.sample_with_noise(&noise);
    let f_original = f(&path);
    let mut derivatives = Array1::zeros(noise.len());

    for i in 0..noise.len() {
      let original = noise[i];
      noise[i] += epsilon;
      let path_perturbed = self.sample_with_noise(&noise);
      derivatives[i] = (f(&path_perturbed) - f_original) / epsilon;
      noise[i] = original;
    }

    derivatives
  }

  fn malliavin_derivative_terminal(&self, epsilon: T) -> Array1<T> {
    self.malliavin_derivative(|path| *path.last().unwrap(), epsilon)
  }
}

pub trait Malliavin2DExt<T: FloatExt> {
  fn sample_with_noise(&self, noise: &[Array1<T>; 2]) -> [Array1<T>; 2];

  fn generate_noise(&self) -> [Array1<T>; 2];

  fn malliavin_derivative<F>(&self, f: F, epsilon: T, noise_component: usize) -> Array1<T>
  where
    F: Fn(&[Array1<T>; 2]) -> T,
  {
    let mut noise = self.generate_noise();
    let paths = self.sample_with_noise(&noise);
    let f_original = f(&paths);
    let n = noise[noise_component].len();
    let mut derivatives = Array1::zeros(n);

    for i in 0..n {
      let original = noise[noise_component][i];
      noise[noise_component][i] += epsilon;
      let paths_perturbed = self.sample_with_noise(&noise);
      derivatives[i] = (f(&paths_perturbed) - f_original) / epsilon;
      noise[noise_component][i] = original;
    }

    derivatives
  }

  fn malliavin_derivative_terminal(
    &self,
    epsilon: T,
    path_component: usize,
    noise_component: usize,
  ) -> Array1<T> {
    self.malliavin_derivative(
      |paths| *paths[path_component].last().unwrap(),
      epsilon,
      noise_component,
    )
  }
}