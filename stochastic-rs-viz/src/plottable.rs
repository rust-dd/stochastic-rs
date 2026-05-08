//! `Plottable<T>` — trait that lets [`crate::GridPlotter`] consume any
//! `ProcessExt::Output` shape (1D path, 2D matrix, complex path, fixed-arity
//! tuple of paths). Each impl reports its component count and produces a
//! Vec<f64> for the requested component, with all numeric types projected
//! to f64 for plotly.

use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex;
use stochastic_rs_distributions::traits::FloatExt;

pub trait Plottable<T: FloatExt> {
  fn n_components(&self) -> usize;
  fn component_name(&self, idx: usize) -> String;
  fn component(&self, idx: usize) -> Vec<f64>;
  fn len(&self) -> usize;
  fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

impl<T: FloatExt> Plottable<T> for Array1<T> {
  fn n_components(&self) -> usize {
    1
  }

  fn component_name(&self, _idx: usize) -> String {
    String::new()
  }

  fn component(&self, _idx: usize) -> Vec<f64> {
    self.iter().map(|v| v.to_f64().unwrap()).collect()
  }

  fn len(&self) -> usize {
    self.len()
  }
}

impl<T: FloatExt> Plottable<T> for Array1<Complex<T>> {
  fn n_components(&self) -> usize {
    2
  }

  fn component_name(&self, idx: usize) -> String {
    match idx {
      0 => "real".to_string(),
      1 => "imag".to_string(),
      _ => String::new(),
    }
  }

  fn component(&self, idx: usize) -> Vec<f64> {
    match idx {
      0 => self.iter().map(|v| v.re.to_f64().unwrap()).collect(),
      1 => self.iter().map(|v| v.im.to_f64().unwrap()).collect(),
      _ => Vec::new(),
    }
  }

  fn len(&self) -> usize {
    self.len()
  }
}

impl<T: FloatExt, const N: usize> Plottable<T> for [Array1<T>; N] {
  fn n_components(&self) -> usize {
    N
  }

  fn component_name(&self, idx: usize) -> String {
    format!("component {}", idx + 1)
  }

  fn component(&self, idx: usize) -> Vec<f64> {
    self[idx].iter().map(|v| v.to_f64().unwrap()).collect()
  }

  fn len(&self) -> usize {
    self[0].len()
  }
}

impl<T: FloatExt> Plottable<T> for Array2<T> {
  fn n_components(&self) -> usize {
    self.nrows()
  }

  fn component_name(&self, idx: usize) -> String {
    format!("row {}", idx + 1)
  }

  fn component(&self, idx: usize) -> Vec<f64> {
    self.row(idx).iter().map(|v| v.to_f64().unwrap()).collect()
  }

  fn len(&self) -> usize {
    self.ncols()
  }
}
