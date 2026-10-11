//! PyO3 wrappers for `stochastic-rs-copulas`.
//!
//! Bivariate Archimedean copulas (Clayton, Gumbel, Frank, Independence) wrapped
//! as `#[pyclass]`. Each exposes `theta` / `tau` setters, `pdf` / `cdf` /
//! `sample` over numpy arrays.
//!
//! The multivariate Gaussian copula is intentionally **not** wrapped because it
//! depended on a system BLAS at the time it was scoped, which was not part of the default
//! Python build.

#![cfg(feature = "python")]
#![allow(clippy::too_many_arguments)]

use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::error::CopulaError;
use crate::traits::BivariateExt;

fn err_to_py(e: CopulaError) -> PyErr {
  PyValueError::new_err(e.to_string())
}

macro_rules! py_bivariate {
  ($py_name:ident, $inner_path:path, $name_str:literal, $try_new:expr, $from_tau:expr) => {
    #[pyclass(name = $name_str, unsendable)]
    pub struct $py_name {
      inner: $inner_path,
      pending_tau: Option<f64>,
    }

    #[pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = (theta=None, tau=None))]
      fn new(theta: Option<f64>, tau: Option<f64>) -> PyResult<Self> {
        let inner = match (theta, tau) {
          (Some(t), _) => ($try_new)(t).map_err(err_to_py)?,
          (None, Some(k)) => ($from_tau)(k).map_err(err_to_py)?,
          (None, None) => <$inner_path>::default(),
        };
        Ok(Self {
          inner,
          pending_tau: tau,
        })
      }

      fn theta(&self) -> Option<f64> {
        BivariateExt::theta(&self.inner)
      }

      fn tau(&self) -> Option<f64> {
        self.pending_tau.or_else(|| BivariateExt::tau(&self.inner))
      }

      fn set_theta(&mut self, theta: f64) -> PyResult<()> {
        BivariateExt::validate_theta(&self.inner, theta).map_err(err_to_py)?;
        BivariateExt::set_theta(&mut self.inner, theta);
        Ok(())
      }

      fn set_tau(&mut self, tau: f64) {
        self.pending_tau = Some(tau);
      }

      /// Solve for theta from the tau set last (Kendall inversion).
      fn compute_theta(&mut self) -> PyResult<f64> {
        let tau = self
          .pending_tau
          .or_else(|| BivariateExt::tau(&self.inner))
          .ok_or_else(|| PyValueError::new_err("set tau first"))?;
        let theta = BivariateExt::theta_from_tau(&self.inner, tau).map_err(err_to_py)?;
        BivariateExt::set_theta(&mut self.inner, theta);
        Ok(theta)
      }

      /// A successful fit replaces the tau set last with the fitted model's.
      fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        BivariateExt::fit(&mut self.inner, x.as_array()).map_err(err_to_py)?;
        self.pending_tau = None;
        Ok(())
      }

      fn pdf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
        let out = BivariateExt::pdf(&self.inner, x.as_array()).map_err(err_to_py)?;
        Ok(out.into_pyarray(py))
      }

      fn cdf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
        let out = BivariateExt::cdf(&self.inner, x.as_array()).map_err(err_to_py)?;
        Ok(out.into_pyarray(py))
      }

      #[pyo3(signature = (n, seed=None))]
      fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
        stochastic_rs_distributions::python::runtime_error_on_panic(
          || -> PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
            let arr = match seed {
              Some(s) => BivariateExt::sample(&self.inner, n, &Deterministic::new(s)),
              None => BivariateExt::sample(&self.inner, n, &Unseeded),
            }
            .map_err(err_to_py)?;
            Ok(arr.into_pyarray(py))
          },
        )?
      }
    }
  };
}

py_bivariate!(
  PyClayton,
  crate::bivariate::clayton::Clayton,
  "Clayton",
  crate::bivariate::clayton::Clayton::try_new,
  crate::bivariate::clayton::Clayton::from_tau
);
py_bivariate!(
  PyGumbel,
  crate::bivariate::gumbel::Gumbel,
  "Gumbel",
  crate::bivariate::gumbel::Gumbel::try_new,
  crate::bivariate::gumbel::Gumbel::from_tau
);
py_bivariate!(
  PyFrank,
  crate::bivariate::frank::Frank,
  "Frank",
  crate::bivariate::frank::Frank::try_new,
  crate::bivariate::frank::Frank::from_tau
);
py_bivariate!(
  PyIndependence,
  crate::bivariate::independence::Independence,
  "Independence",
  |t: f64| {
    let c = crate::bivariate::independence::Independence::new();
    c.validate_theta(t).map(|()| c)
  },
  |k: f64| {
    let c = crate::bivariate::independence::Independence::new();
    c.theta_from_tau(k).map(|_| c)
  }
);

#[pyclass(name = "EmpiricalCopula2D", unsendable)]
pub struct PyEmpiricalCopula2D {
  inner: crate::empirical::EmpiricalCopula2D,
}

#[pymethods]
impl PyEmpiricalCopula2D {
  /// Build a 2D empirical copula from two equal-length series via rank-transform.
  #[new]
  fn new<'py>(
    x: numpy::PyReadonlyArray1<'py, f64>,
    y: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::empirical::EmpiricalCopula2D::new(x.as_array(), y.as_array()),
    })
  }

  fn rank_data<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    self.inner.rank_data().to_owned().into_pyarray(py)
  }

  #[pyo3(signature = (n, seed=None))]
  fn sample<'py>(
    &self,
    py: Python<'py>,
    n: usize,
    seed: Option<u64>,
  ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      match seed {
        Some(s) => self.inner.sample(n, &Deterministic::new(s)),
        None => self.inner.sample(n, &Unseeded),
      }
      .into_pyarray(py)
    })
  }
}

/// Kendall's τ-b pairwise matrix from an `(n, k)` data matrix.
#[pyfunction]
pub fn kendall_tau_matrix<'py>(
  py: Python<'py>,
  data: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
  let out = crate::correlation::kendall_tau(data.as_array()).map_err(err_to_py)?;
  Ok(out.into_pyarray(py))
}

/// Convert a Kendall τ matrix to a Gaussian copula correlation matrix
/// elementwise via $\rho_{ij} = \sin(\pi \tau_{ij} / 2)$.
#[pyfunction]
pub fn tau_matrix_to_corr_matrix<'py>(
  py: Python<'py>,
  tau: numpy::PyReadonlyArray2<'py, f64>,
) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
  let out = crate::correlation::tau_matrix_to_corr_matrix(tau.as_array());
  out.into_pyarray(py)
}

#[pyfunction]
pub fn tau_to_corr(tau: f64) -> f64 {
  crate::correlation::tau_to_corr(tau)
}

#[pyfunction]
pub fn corr_to_tau(rho: f64) -> f64 {
  crate::correlation::corr_to_tau(rho)
}

macro_rules! py_two_parameter {
  ($py_name:ident, $inner_path:path, $name_str:literal) => {
    #[pyclass(name = $name_str, unsendable)]
    pub struct $py_name {
      inner: $inner_path,
      pending_tau: Option<f64>,
    }

    #[pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = (theta=None, delta=None, tau=None))]
      fn new(theta: Option<f64>, delta: Option<f64>, tau: Option<f64>) -> PyResult<Self> {
        let delta = delta.unwrap_or_else(|| <$inner_path>::default().delta());
        let inner = match (theta, tau) {
          (Some(t), _) => <$inner_path>::try_new(t, delta).map_err(err_to_py)?,
          (None, Some(k)) => <$inner_path>::from_tau(k, delta).map_err(err_to_py)?,
          (None, None) => {
            <$inner_path>::check_delta(delta).map_err(err_to_py)?;
            let mut inner = <$inner_path>::default();
            inner.set_delta(delta);
            inner
          }
        };
        Ok(Self {
          inner,
          pending_tau: tau,
        })
      }

      fn theta(&self) -> Option<f64> {
        BivariateExt::theta(&self.inner)
      }

      fn delta(&self) -> f64 {
        self.inner.delta()
      }

      fn tau(&self) -> Option<f64> {
        self.pending_tau.or_else(|| BivariateExt::tau(&self.inner))
      }

      fn set_theta(&mut self, theta: f64) -> PyResult<()> {
        BivariateExt::validate_theta(&self.inner, theta).map_err(err_to_py)?;
        BivariateExt::set_theta(&mut self.inner, theta);
        Ok(())
      }

      fn set_delta(&mut self, delta: f64) -> PyResult<()> {
        <$inner_path>::check_delta(delta).map_err(err_to_py)?;
        self.inner.set_delta(delta);
        Ok(())
      }

      fn set_tau(&mut self, tau: f64) {
        self.pending_tau = Some(tau);
      }

      /// Solve for theta from the tau set last, at the current delta.
      fn compute_theta(&mut self) -> PyResult<f64> {
        let tau = self
          .pending_tau
          .or_else(|| BivariateExt::tau(&self.inner))
          .ok_or_else(|| PyValueError::new_err("set tau first"))?;
        let theta = BivariateExt::theta_from_tau(&self.inner, tau).map_err(err_to_py)?;
        BivariateExt::set_theta(&mut self.inner, theta);
        Ok(theta)
      }

      /// Maximum-likelihood fit of both parameters; a successful fit replaces the tau set last with the fitted model's.
      fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        BivariateExt::fit(&mut self.inner, x.as_array()).map_err(err_to_py)?;
        self.pending_tau = None;
        Ok(())
      }

      fn pdf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
        let out = BivariateExt::pdf(&self.inner, x.as_array()).map_err(err_to_py)?;
        Ok(out.into_pyarray(py))
      }

      fn cdf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
        let out = BivariateExt::cdf(&self.inner, x.as_array()).map_err(err_to_py)?;
        Ok(out.into_pyarray(py))
      }

      /// `(lower, upper)` tail-dependence coefficients.
      fn tail_dependence(&self) -> PyResult<(f64, f64)> {
        let t = BivariateExt::tail_dependence(&self.inner).map_err(err_to_py)?;
        Ok((t.lower, t.upper))
      }

      #[pyo3(signature = (n, seed=None))]
      fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
        stochastic_rs_distributions::python::runtime_error_on_panic(
          || -> PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
            let arr = match seed {
              Some(s) => BivariateExt::sample(&self.inner, n, &Deterministic::new(s)),
              None => BivariateExt::sample(&self.inner, n, &Unseeded),
            }
            .map_err(err_to_py)?;
            Ok(arr.into_pyarray(py))
          },
        )?
      }
    }
  };
}

py_two_parameter!(PyBb1, crate::bivariate::bb1::Bb1, "Bb1");
py_two_parameter!(PyBb7, crate::bivariate::bb7::Bb7, "Bb7");

fn parse_family(name: &str) -> PyResult<crate::bivariate::BivariateFamily> {
  use crate::bivariate::BivariateFamily;
  Ok(match name.to_ascii_lowercase().as_str() {
    "independence" => BivariateFamily::Independence,
    "gaussian" => BivariateFamily::Gaussian,
    "student_t" | "studentt" | "t" => BivariateFamily::TCopula,
    "clayton" => BivariateFamily::Clayton,
    "frank" => BivariateFamily::Frank,
    "bb1" => BivariateFamily::Bb1,
    "bb7" => BivariateFamily::Bb7,
    other => {
      return Err(PyValueError::new_err(format!(
        "unknown pair-copula family {other:?}"
      )));
    }
  })
}

fn pair_description(pair: &crate::multivariate::dvine::PairCopula) -> (String, Vec<f64>) {
  use crate::multivariate::dvine::PairCopula;
  match *pair {
    PairCopula::Independence => ("independence".into(), vec![]),
    PairCopula::Gaussian { rho } => ("gaussian".into(), vec![rho]),
    PairCopula::StudentT { rho, nu } => ("student_t".into(), vec![rho, nu]),
    PairCopula::Clayton { theta } => ("clayton".into(), vec![theta]),
    PairCopula::Frank { theta } => ("frank".into(), vec![theta]),
    PairCopula::Bb1 { theta, delta } => ("bb1".into(), vec![theta, delta]),
    PairCopula::Bb7 { theta, delta } => ("bb7".into(), vec![theta, delta]),
  }
}

/// Fits a D-vine or C-vine to pseudo-observations `u` (rows = observations)
/// with AIC/BIC family selection; returns a dict with `order`, `families`,
/// `parameters` (per tree, per edge), `log_likelihood`, `parameter_count`,
/// `aic` and `bic`.
#[pyfunction]
#[pyo3(signature = (u, structure="dvine", criterion="aic", families=None))]
pub fn fit_vine<'py>(
  py: Python<'py>,
  u: PyReadonlyArray2<'py, f64>,
  structure: &str,
  criterion: &str,
  families: Option<Vec<String>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
  use pyo3::types::PyDict;

  use crate::bivariate::BivariateFamily;
  use crate::multivariate::fit::SelectionCriterion;
  use crate::multivariate::fit::VineStructure;
  use crate::multivariate::fit::fit_vine;
  use crate::multivariate::rvine::RVine;
  let structure = match structure.to_ascii_lowercase().as_str() {
    "dvine" | "d" => VineStructure::DVine,
    "cvine" | "c" => VineStructure::CVine,
    other => {
      return Err(PyValueError::new_err(format!(
        "unknown vine structure {other:?}"
      )));
    }
  };
  let criterion = match criterion.to_ascii_lowercase().as_str() {
    "aic" => SelectionCriterion::Aic,
    "bic" => SelectionCriterion::Bic,
    other => {
      return Err(PyValueError::new_err(format!(
        "unknown criterion {other:?}"
      )));
    }
  };
  let families: Vec<BivariateFamily> = match families {
    Some(names) => names
      .iter()
      .map(|n| parse_family(n))
      .collect::<PyResult<_>>()?,
    None => BivariateFamily::VINE_CANDIDATES.to_vec(),
  };
  let data = u.as_array().to_owned();
  let fit = py
    .detach(|| fit_vine(data.view(), structure, &families, criterion))
    .map_err(err_to_py)?;
  let trees: &[Vec<crate::multivariate::dvine::PairCopula>] = match &fit.vine {
    RVine::D(d) => d.pair_copulas(),
    RVine::C(c) => c.pair_copulas(),
  };
  let described: Vec<Vec<(String, Vec<f64>)>> = trees
    .iter()
    .map(|t| t.iter().map(pair_description).collect())
    .collect();
  let out = PyDict::new(py);
  out.set_item(
    "structure",
    match structure {
      VineStructure::DVine => "dvine",
      VineStructure::CVine => "cvine",
    },
  )?;
  out.set_item("order", fit.order.clone())?;
  out.set_item(
    "families",
    described
      .iter()
      .map(|t| t.iter().map(|(f, _)| f.clone()).collect::<Vec<_>>())
      .collect::<Vec<_>>(),
  )?;
  out.set_item(
    "parameters",
    described
      .iter()
      .map(|t| t.iter().map(|(_, p)| p.clone()).collect::<Vec<_>>())
      .collect::<Vec<_>>(),
  )?;
  out.set_item("log_likelihood", fit.log_likelihood)?;
  out.set_item("parameter_count", fit.parameter_count)?;
  out.set_item("aic", fit.aic)?;
  out.set_item("bic", fit.bic)?;
  Ok(out)
}

/// Parametric-bootstrap Cramér–von Mises test of the `family` (`clayton`, `gumbel`, `frank`, `gaussian`, `bb1`, `bb7`) fitted
/// to the pseudo-observations `u`; returns `(statistic, p_value)`, and `alpha` fixes the rejection the Rust result carries.
#[pyfunction]
#[pyo3(signature = (family, u, replications=200, seed=42, alpha=0.05))]
pub fn copula_gof<'py>(
  py: Python<'py>,
  family: &str,
  u: PyReadonlyArray2<'py, f64>,
  replications: usize,
  seed: u64,
  alpha: f64,
) -> PyResult<(f64, f64)> {
  use crate::gof::GofConfig;
  use crate::gof::gof_cramer_von_mises;
  let data = u.as_array().to_owned();
  macro_rules! run {
    ($ctor:expr) => {{
      let mut copula = $ctor;
      copula.fit(data.view()).map_err(err_to_py)?;
      stochastic_rs_distributions::python::value_error_on_panic(|| {
        py.detach(|| {
          gof_cramer_von_mises(
            &copula,
            data.view(),
            GofConfig {
              replications,
              alpha,
            },
            &Deterministic::new(seed),
            |c, x| c.fit(x),
          )
        })
      })?
      .map_err(err_to_py)?
    }};
  }
  let result = match family.to_ascii_lowercase().as_str() {
    "clayton" => run!(crate::bivariate::clayton::Clayton::default()),
    "gumbel" => run!(crate::bivariate::gumbel::Gumbel::default()),
    "frank" => run!(crate::bivariate::frank::Frank::default()),
    "gaussian" => run!(crate::bivariate::gaussian::GaussianCopula::default()),
    "bb1" => run!(crate::bivariate::bb1::Bb1::default()),
    "bb7" => run!(crate::bivariate::bb7::Bb7::default()),
    other => {
      return Err(PyValueError::new_err(format!(
        "unsupported family {other:?}"
      )));
    }
  };
  Ok((result.statistic, result.p_value))
}

/// Pseudo-observations `rank / (n + 1)` of every column of raw data, ties sharing their mean rank — the input every
/// copula `fit`, `fit_vine` and `copula_gof` expects.
#[pyfunction]
pub fn pseudo_observations<'py>(
  py: Python<'py>,
  x: PyReadonlyArray2<'py, f64>,
) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
  crate::gof::pseudo_observations(x.as_array()).into_pyarray(py)
}
