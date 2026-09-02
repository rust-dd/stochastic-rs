//! Python surface of the surrogates and the surrogate calibration
//! (`--features ai` builds only; the published wheels leave it out because
//! candle is heavy).

use numpy::IntoPyArray;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::calibration::SurrogateCalibrationResult;
use crate::calibration::SurrogateCalibrator;
use crate::calibration::SurrogateModel;
use crate::volatility::common::StochVolNn;
use crate::volatility::common::TrainConfig;
use crate::volatility::common::TrainReport;

fn err(e: anyhow::Error) -> PyErr {
  PyValueError::new_err(e.to_string())
}

fn report_dict<'py>(py: Python<'py>, report: &TrainReport) -> PyResult<Bound<'py, PyDict>> {
  let out = PyDict::new(py);
  out.set_item(
    "train_rmse",
    report
      .epochs
      .iter()
      .map(|e| e.train_rmse as f64)
      .collect::<Vec<_>>(),
  )?;
  out.set_item(
    "val_rmse",
    report
      .epochs
      .iter()
      .map(|e| e.val_rmse as f64)
      .collect::<Vec<_>>(),
  )?;
  Ok(out)
}

fn to_f32(x: PyReadonlyArray2<'_, f64>) -> ndarray::Array2<f32> {
  x.as_array().mapv(|v| v as f32)
}

macro_rules! py_surrogate {
  ($py_name:ident, $inner:ty, $name_str:literal) => {
    #[pyclass(name = $name_str, unsendable)]
    pub struct $py_name {
      inner: $inner,
    }

    #[pymethods]
    impl $py_name {
      /// A fresh network on the best available device (`hidden_dim` widens
      /// the three hidden layers; the paper's width is 30).
      #[new]
      #[pyo3(signature = (hidden_dim=None))]
      fn new(hidden_dim: Option<usize>) -> PyResult<Self> {
        let device = crate::device::best_available().map_err(err)?;
        let inner = match hidden_dim {
          Some(h) => <$inner>::with_hidden(&device, h),
          None => <$inner>::new(&device),
        }
        .map_err(err)?;
        Ok(Self { inner })
      }

      /// Trains on `(params, surfaces)` rows; returns the per-epoch RMSE
      /// history as a dict.
      #[pyo3(signature = (params, surfaces, epochs=200, batch_size=32, learning_rate=1e-3, test_ratio=0.15, seed=42))]
      #[allow(clippy::too_many_arguments)]
      fn train<'py>(
        &mut self,
        py: Python<'py>,
        params: PyReadonlyArray2<'py, f64>,
        surfaces: PyReadonlyArray2<'py, f64>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        test_ratio: f64,
        seed: u64,
      ) -> PyResult<Bound<'py, PyDict>> {
        let (p, s) = (to_f32(params), to_f32(surfaces));
        let cfg = TrainConfig {
          test_ratio: test_ratio as f32,
          batch_size,
          epochs,
          learning_rate,
          random_seed: seed,
          shuffle: true,
        };
        let report = self.inner.train(&p, &s, &cfg).map_err(err)?;
        report_dict(py, &report)
      }

      /// Implied-volatility surface (flat, row-major maturities × strikes) at `params`.
      fn predict_surface<'py>(&self, py: Python<'py>, params: Vec<f64>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let p: Vec<f32> = params.iter().map(|&v| v as f32).collect();
        let out = SurrogateModel::nn(&self.inner).predict_surface(&p).map_err(err)?;
        Ok(ndarray::Array1::from_vec(out.into_iter().map(|v| v as f64).collect()).into_pyarray(py))
      }

      /// Surface and its Jacobian (grid × params) at `params`.
      fn predict_surface_with_jacobian<'py>(
        &self,
        py: Python<'py>,
        params: Vec<f64>,
      ) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray2<f64>>)> {
        let p: Vec<f32> = params.iter().map(|&v| v as f32).collect();
        let (surface, jacobian) = SurrogateModel::nn(&self.inner).predict_surface_with_jacobian(&p).map_err(err)?;
        Ok((
          ndarray::Array1::from_vec(surface.into_iter().map(|v| v as f64).collect()).into_pyarray(py),
          jacobian.mapv(|v| v as f64).into_pyarray(py),
        ))
      }

      fn save(&self, dir: &str) -> PyResult<()> {
        self.inner.save(dir).map_err(err)
      }

      #[staticmethod]
      fn load(dir: &str) -> PyResult<Self> {
        let device = crate::device::best_available().map_err(err)?;
        Ok(Self {
          inner: <$inner>::load(dir, &device).map_err(err)?,
        })
      }

      #[getter]
      fn param_lb(&self) -> Vec<f64> {
        SurrogateModel::nn(&self.inner).spec().param_lb.iter().map(|&v| v as f64).collect()
      }

      #[getter]
      fn param_ub(&self) -> Vec<f64> {
        SurrogateModel::nn(&self.inner).spec().param_ub.iter().map(|&v| v as f64).collect()
      }

      #[getter]
      fn input_dim(&self) -> usize {
        SurrogateModel::nn(&self.inner).spec().input_dim
      }

      #[getter]
      fn output_dim(&self) -> usize {
        SurrogateModel::nn(&self.inner).spec().output_dim
      }
    }
  };
}

py_surrogate!(PyHestonNn, crate::volatility::heston::HestonNn, "HestonNn");
py_surrogate!(
  PyRBergomiNn,
  crate::volatility::rbergomi::RBergomiNn,
  "RBergomiNn"
);
py_surrogate!(
  PyOneFactorNn,
  crate::volatility::one_factor::OneFactorNn,
  "OneFactorNn"
);

fn result_dict<'py>(
  py: Python<'py>,
  fit: &SurrogateCalibrationResult,
) -> PyResult<Bound<'py, PyDict>> {
  let out = PyDict::new(py);
  out.set_item("params", fit.params.clone())?;
  out.set_item("rmse", fit.rmse)?;
  out.set_item("max_error", fit.max_error)?;
  out.set_item("converged", fit.converged)?;
  out.set_item("in_bounds", fit.in_bounds)?;
  out.set_item("evaluations", fit.evaluations)?;
  out.set_item("message", fit.message.clone())?;
  Ok(out)
}

fn run_calibration<'py>(
  py: Python<'py>,
  nn: &StochVolNn,
  market: PyReadonlyArray1<'py, f64>,
  initial: Option<Vec<f64>>,
  weights: Option<Vec<f64>>,
  tolerance: f64,
  patience: usize,
) -> PyResult<Bound<'py, PyDict>> {
  let mut calibrator = SurrogateCalibrator::new(nn, market.as_array().to_vec()).map_err(err)?;
  if let Some(w) = weights {
    calibrator = calibrator.with_weights(w).map_err(err)?;
  }
  let calibrator = calibrator.with_tolerance(tolerance).with_patience(patience);
  let fit = calibrator.run(initial).map_err(err)?;
  result_dict(py, &fit)
}

/// Levenberg–Marquardt calibration of a surrogate (`HestonNn`, `RBergomiNn`
/// or `OneFactorNn`) to a market surface on its grid; returns a dict with the
/// parameters in the surrogate's input order plus the fit diagnostics.
#[pyfunction]
#[pyo3(signature = (model, market, initial=None, weights=None, tolerance=1e-10, patience=200))]
pub fn calibrate_surrogate<'py>(
  py: Python<'py>,
  model: &Bound<'py, PyAny>,
  market: PyReadonlyArray1<'py, f64>,
  initial: Option<Vec<f64>>,
  weights: Option<Vec<f64>>,
  tolerance: f64,
  patience: usize,
) -> PyResult<Bound<'py, PyDict>> {
  if let Ok(m) = model.cast::<PyHestonNn>() {
    let m = m.borrow();
    return run_calibration(
      py,
      SurrogateModel::nn(&m.inner),
      market,
      initial,
      weights,
      tolerance,
      patience,
    );
  }
  if let Ok(m) = model.cast::<PyRBergomiNn>() {
    let m = m.borrow();
    return run_calibration(
      py,
      SurrogateModel::nn(&m.inner),
      market,
      initial,
      weights,
      tolerance,
      patience,
    );
  }
  if let Ok(m) = model.cast::<PyOneFactorNn>() {
    let m = m.borrow();
    return run_calibration(
      py,
      SurrogateModel::nn(&m.inner),
      market,
      initial,
      weights,
      tolerance,
      patience,
    );
  }
  Err(PyValueError::new_err(
    "model must be a HestonNn, RBergomiNn or OneFactorNn",
  ))
}
