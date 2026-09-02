use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "DiscountCurve", unsendable)]
pub struct PyDiscountCurve {
  pub(super) inner: crate::curves::discount_curve::DiscountCurve<f64>,
}

#[pymethods]
impl PyDiscountCurve {
  /// Build from `(maturities, zero_rates)` arrays under continuous compounding.
  /// `interp`: "linear" / "log_df" / "cubic" / "monotone_convex".
  #[staticmethod]
  #[pyo3(signature = (maturities, zero_rates, interp="linear"))]
  fn from_zero_rates<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    zero_rates: numpy::PyReadonlyArray1<'py, f64>,
    interp: &str,
  ) -> PyResult<Self> {
    let im = parse_interpolation(interp)?;
    let mat = maturities.as_array().to_owned();
    let zr = zero_rates.as_array().to_owned();
    Ok(Self {
      inner: crate::curves::discount_curve::DiscountCurve::from_zero_rates(&mat, &zr, im),
    })
  }

  fn discount_factor(&self, t: f64) -> f64 {
    self.inner.discount_factor(t)
  }

  fn zero_rate(&self, t: f64) -> f64 {
    self.inner.zero_rate(t)
  }

  fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
    self.inner.forward_rate(t1, t2)
  }

  fn par_rate(&self, maturity: f64, frequency: u32) -> f64 {
    self.inner.par_rate(maturity, frequency)
  }

  /// Vectorised zero rates on a maturity array.
  fn zero_rates<'py>(
    &self,
    py: Python<'py>,
    maturities: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    let mat = maturities.as_array().to_owned();
    self.inner.zero_rates(&mat).into_pyarray(py)
  }

  /// Bootstraps an OIS discount curve from par OIS quotes: one fixed-leg
  /// payment schedule (year fractions, last entry = maturity) per quote, the
  /// swap equation `1 = S Σ δ_i D(t_i) + D(t_n)` solved pillar by pillar.
  #[staticmethod]
  #[pyo3(signature = (payment_schedules, rates, interp="log_df"))]
  fn bootstrap_ois(
    payment_schedules: Vec<Vec<f64>>,
    rates: Vec<f64>,
    interp: &str,
  ) -> PyResult<Self> {
    if payment_schedules.len() != rates.len() {
      return Err(PyValueError::new_err(
        "payment_schedules and rates must have the same length",
      ));
    }
    let method = parse_interpolation(interp)?;
    let instruments: Vec<crate::curves::Instrument<f64>> = payment_schedules
      .into_iter()
      .zip(rates)
      .map(
        |(payment_times, rate)| crate::curves::Instrument::SwapWithSchedule {
          rate,
          payment_times,
        },
      )
      .collect();
    Ok(Self {
      inner: crate::curves::bootstrap(&instruments, method),
    })
  }

  /// Dual-curve bootstrap of a tenor's pseudo-discount curve against this
  /// exogenous OIS `discount` curve: `deposits` as `(maturity, rate)`, `fras`
  /// as `(start, end, rate)`, `swaps` as `(rate, fixed_times, float_times)`.
  #[staticmethod]
  #[pyo3(signature = (discount, deposits, fras, swaps, interp="log_df"))]
  fn bootstrap_forecast(
    discount: &PyDiscountCurve,
    deposits: Vec<(f64, f64)>,
    fras: Vec<(f64, f64, f64)>,
    swaps: Vec<(f64, Vec<f64>, Vec<f64>)>,
    interp: &str,
  ) -> PyResult<Self> {
    use crate::curves::dual_curve::ForecastInstrument;
    let method = parse_interpolation(interp)?;
    let mut instruments: Vec<ForecastInstrument<f64>> = Vec::new();
    instruments.extend(
      deposits
        .into_iter()
        .map(|(maturity, rate)| ForecastInstrument::Deposit { maturity, rate }),
    );
    instruments.extend(
      fras
        .into_iter()
        .map(|(start, end, rate)| ForecastInstrument::Fra { start, end, rate }),
    );
    instruments.extend(swaps.into_iter().map(|(rate, fixed_times, float_times)| {
      ForecastInstrument::Swap {
        rate,
        fixed_times,
        float_times,
      }
    }));
    Ok(Self {
      inner: crate::curves::dual_curve::bootstrap_forecast(&instruments, &discount.inner, method),
    })
  }
}

fn parse_interpolation(interp: &str) -> PyResult<crate::curves::types::InterpolationMethod> {
  use crate::curves::types::InterpolationMethod;
  match interp.to_ascii_lowercase().as_str() {
    "linear" | "linear_zr" => Ok(InterpolationMethod::LinearOnZeroRates),
    "log_df" | "loglinear_df" => Ok(InterpolationMethod::LogLinearOnDiscountFactors),
    "cubic" | "cubic_zr" => Ok(InterpolationMethod::CubicSplineOnZeroRates),
    "monotone_convex" | "mc" => Ok(InterpolationMethod::MonotoneConvex),
    o => Err(PyValueError::new_err(format!(
      "interp must be linear/log_df/cubic/monotone_convex, got '{o}'"
    ))),
  }
}

/// Multi-curve container: an OIS discount curve plus tenor forecast curves.
#[pyclass(name = "MultiCurve", unsendable)]
pub struct PyMultiCurve {
  inner: crate::curves::multi_curve::MultiCurve<f64>,
}

#[pymethods]
impl PyMultiCurve {
  #[new]
  fn new(discount: &PyDiscountCurve) -> Self {
    Self {
      inner: crate::curves::multi_curve::MultiCurve::new(discount.inner.clone()),
    }
  }

  /// Registers `curve` as the forecast curve of `tenor` (e.g. `"3M"`).
  fn add_forecast(&mut self, tenor: &str, curve: &PyDiscountCurve) {
    self.inner.add_forecast(tenor, curve.inner.clone());
  }

  /// OIS discount factor at `t`.
  fn discount_factor(&self, t: f64) -> f64 {
    self.inner.discount.discount_factor(t)
  }

  /// Simple forward projected from the tenor's forecast curve over
  /// `[t1, t2]`; `None` for an unknown tenor.
  fn projected_forward(&self, tenor: &str, t1: f64, t2: f64) -> Option<f64> {
    self.inner.projected_forward(tenor, t1, t2)
  }

  /// Forecast-minus-OIS simple forward spread over `[t1, t2]`.
  fn basis_spread(&self, tenor: &str, t1: f64, t2: f64) -> Option<f64> {
    self.inner.basis_spread(tenor, t1, t2)
  }

  /// Fair fixed rate of a swap against `tenor` on the shared payment
  /// `schedule` (first entry = start).
  fn fair_swap_rate(&self, tenor: &str, schedule: Vec<f64>) -> Option<f64> {
    self
      .inner
      .fair_swap_rate(tenor, &ndarray::Array1::from_vec(schedule))
  }
}

#[pyclass(name = "NelsonSiegel", unsendable)]
pub struct PyNelsonSiegel {
  inner: crate::curves::nelson_siegel::NelsonSiegel<f64>,
}

#[pymethods]
impl PyNelsonSiegel {
  #[new]
  fn new(beta0: f64, beta1: f64, beta2: f64, lambda: f64) -> Self {
    Self {
      inner: crate::curves::nelson_siegel::NelsonSiegel::new(beta0, beta1, beta2, lambda),
    }
  }

  /// Fit Nelson-Siegel parameters to market zero rates.
  #[staticmethod]
  fn fit_curve<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    market_rates: numpy::PyReadonlyArray1<'py, f64>,
  ) -> Self {
    let mat = maturities.as_array().to_owned();
    let mr = market_rates.as_array().to_owned();
    Self {
      inner: <crate::curves::nelson_siegel::NelsonSiegel<f64>>::fit(&mat, &mr),
    }
  }

  fn zero_rate(&self, tau: f64) -> f64 {
    self.inner.zero_rate(tau)
  }

  fn forward_rate(&self, tau: f64) -> f64 {
    self.inner.forward_rate(tau)
  }

  fn discount_factor(&self, tau: f64) -> f64 {
    self.inner.discount_factor(tau)
  }
}

#[pyclass(name = "ZeroCouponInflationCurve", unsendable)]
pub struct PyZeroCouponInflationCurve {
  inner: crate::inflation::curve::ZeroCouponInflationCurve<f64>,
}

#[pymethods]
impl PyZeroCouponInflationCurve {
  /// Build a zero-coupon inflation curve from `(pillars, breakevens)`.
  #[new]
  fn new<'py>(
    pillars: numpy::PyReadonlyArray1<'py, f64>,
    breakevens: numpy::PyReadonlyArray1<'py, f64>,
  ) -> Self {
    Self {
      inner: crate::inflation::curve::ZeroCouponInflationCurve::new(
        pillars.as_array().to_owned(),
        breakevens.as_array().to_owned(),
      ),
    }
  }

  /// Forward CPI index ratio $I(0, T)/I(0, 0) = (1 + b(T))^T$.
  fn forward_index_ratio(&self, t: f64) -> f64 {
    use crate::inflation::curve::InflationCurve;
    self.inner.forward_index_ratio(t)
  }
}
