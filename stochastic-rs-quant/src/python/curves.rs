use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::curves::BootstrapInstrument;
use crate::curves::BootstrapSpec;
use crate::curves::bootstrap;

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
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::curves::discount_curve::DiscountCurve::from_zero_rates(
        maturities.as_array(),
        zero_rates.as_array(),
        im,
      ),
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

  fn par_rate(&self, maturity: f64, frequency: u32) -> PyResult<f64> {
    let frequency =
      crate::calendar::Frequency::from_periods_per_year(frequency).ok_or_else(|| {
        PyValueError::new_err(format!(
          "frequency must be 1, 2, 4 or 12 payments a year, got {frequency}"
        ))
      })?;
    Ok(self.inner.par_rate(maturity, frequency))
  }

  /// Vectorised zero rates on a maturity array.
  fn zero_rates<'py>(
    &self,
    py: Python<'py>,
    maturities: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self
      .inner
      .zero_rates(maturities.as_array())
      .into_pyarray(py)
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
    let instruments: Vec<BootstrapInstrument<f64>> = payment_schedules
      .into_iter()
      .zip(rates)
      .map(
        |(fixed_times, rate)| BootstrapInstrument::SwapWithSchedule {
          rate,
          fixed_times,
          float_times: Vec::new(),
        },
      )
      .collect();
    Ok(Self {
      inner: bootstrap(BootstrapSpec::new(&instruments, method))
        .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
  }

  /// A tenor's forecast curve on `discount` from `(maturity, rate)` deposits, `(start, end, rate)`
  /// FRAs and `(rate, fixed_times, float_times)` swaps, which an error's index counts in turn.
  #[staticmethod]
  #[pyo3(signature = (discount, deposits, fras, swaps, interp="log_df"))]
  fn bootstrap_forecast(
    discount: &PyDiscountCurve,
    deposits: Vec<(f64, f64)>,
    fras: Vec<(f64, f64, f64)>,
    swaps: Vec<(f64, Vec<f64>, Vec<f64>)>,
    interp: &str,
  ) -> PyResult<Self> {
    let method = parse_interpolation(interp)?;
    let mut instruments: Vec<BootstrapInstrument<f64>> = Vec::new();
    instruments.extend(
      deposits
        .into_iter()
        .map(|(maturity, rate)| BootstrapInstrument::Deposit { maturity, rate }),
    );
    instruments.extend(
      fras
        .into_iter()
        .map(|(start, end, rate)| BootstrapInstrument::Fra { start, end, rate }),
    );
    instruments.extend(swaps.into_iter().map(|(rate, fixed_times, float_times)| {
      BootstrapInstrument::SwapWithSchedule {
        rate,
        fixed_times,
        float_times,
      }
    }));
    Ok(Self {
      inner: bootstrap(BootstrapSpec::new(&instruments, method).discounting_with(&discount.inner))
        .map_err(|e| PyValueError::new_err(e.to_string()))?,
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
  fn new(discount: &PyDiscountCurve) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::curves::multi_curve::MultiCurve::new(discount.inner.clone()),
    })
  }

  /// Registers `curve` under `tenor`, read without regard to case: `"O/N"` or `"ON"`, `"1M"`,
  /// `"3M"`, `"6M"`, `"12M"` or `"1Y"`.
  fn add_forecast(&mut self, tenor: &str, curve: &PyDiscountCurve) -> PyResult<()> {
    self
      .inner
      .add_forecast(parse_tenor(tenor)?, curve.inner.clone());
    Ok(())
  }

  /// OIS discount factor at `t`.
  fn discount_factor(&self, t: f64) -> f64 {
    self.inner.discount.discount_factor(t)
  }

  /// Simple forward projected from the tenor's forecast curve over
  /// `[t1, t2]`; `None` for a tenor with no curve.
  fn projected_forward(&self, tenor: &str, t1: f64, t2: f64) -> PyResult<Option<f64>> {
    Ok(self.inner.projected_forward(parse_tenor(tenor)?, t1, t2))
  }

  /// Forecast-minus-OIS simple forward spread over `[t1, t2]`.
  fn basis_spread(&self, tenor: &str, t1: f64, t2: f64) -> PyResult<Option<f64>> {
    Ok(self.inner.basis_spread(parse_tenor(tenor)?, t1, t2))
  }

  /// Fair fixed rate of a swap against `tenor` on the shared payment `schedule` (first entry =
  /// start); `None` for a tenor with no curve or under two dates.
  fn fair_swap_rate(&self, tenor: &str, schedule: Vec<f64>) -> PyResult<Option<f64>> {
    Ok(self.inner.fair_swap_rate(
      parse_tenor(tenor)?,
      ndarray::Array1::from_vec(schedule).view(),
    ))
  }
}

fn parse_tenor(tenor: &str) -> PyResult<crate::cashflows::RateTenor> {
  tenor
    .to_ascii_uppercase()
    .parse()
    .map_err(|e: crate::QuantError| PyValueError::new_err(e.to_string()))
}

#[pyclass(name = "NelsonSiegel", unsendable)]
pub struct PyNelsonSiegel {
  inner: crate::curves::nelson_siegel::NelsonSiegel<f64>,
}

#[pymethods]
impl PyNelsonSiegel {
  #[new]
  fn new(beta0: f64, beta1: f64, beta2: f64, lambda: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::curves::nelson_siegel::NelsonSiegel::new(beta0, beta1, beta2, lambda),
    })
  }

  /// Least squares on the betas at each decay of the grid 0.1, 0.11, …, 5.0, so a decay off it is
  /// not recovered; four quotes at least, one per parameter.
  #[staticmethod]
  fn fit_curve<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    market_rates: numpy::PyReadonlyArray1<'py, f64>,
  ) -> PyResult<Self> {
    <crate::curves::nelson_siegel::NelsonSiegel<f64>>::fit(
      maturities.as_array(),
      market_rates.as_array(),
    )
    .map(|inner| Self { inner })
    .map_err(|e| PyValueError::new_err(e.to_string()))
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

/// Nelson-Siegel-Svensson curve (Svensson 1994): the zero rate, instantaneous forward and discount
/// factor at one maturity or on an array; NaN at a negative or non-finite maturity.
#[pyclass(name = "Svensson", unsendable)]
pub struct PySvensson {
  inner: crate::curves::svensson::Svensson<f64>,
}

#[pymethods]
impl PySvensson {
  #[new]
  fn new(
    beta0: f64,
    beta1: f64,
    beta2: f64,
    beta3: f64,
    lambda1: f64,
    lambda2: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::curves::svensson::Svensson::new(beta0, beta1, beta2, beta3, lambda1, lambda2),
    })
  }

  /// Least squares on the betas at each decay pair of the grid 0.1, 0.15, …, 5.0, so a decay off
  /// it is not recovered; six quotes at least, one per parameter.
  #[staticmethod]
  fn fit_curve<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    market_rates: numpy::PyReadonlyArray1<'py, f64>,
  ) -> PyResult<Self> {
    <crate::curves::svensson::Svensson<f64>>::fit(maturities.as_array(), market_rates.as_array())
      .map(|inner| Self { inner })
      .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  /// Returns `(beta0, beta1, beta2, beta3, lambda1, lambda2)`.
  fn params(&self) -> (f64, f64, f64, f64, f64, f64) {
    let c = &self.inner;
    (c.beta0, c.beta1, c.beta2, c.beta3, c.lambda1, c.lambda2)
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

  fn zero_rates<'py>(
    &self,
    py: Python<'py>,
    maturities: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    maturities
      .as_array()
      .mapv(|tau| self.inner.zero_rate(tau))
      .into_pyarray(py)
  }

  fn forward_rates<'py>(
    &self,
    py: Python<'py>,
    maturities: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    maturities
      .as_array()
      .mapv(|tau| self.inner.forward_rate(tau))
      .into_pyarray(py)
  }

  fn discount_factors<'py>(
    &self,
    py: Python<'py>,
    maturities: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    maturities
      .as_array()
      .mapv(|tau| self.inner.discount_factor(tau))
      .into_pyarray(py)
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
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::inflation::curve::ZeroCouponInflationCurve::new(
        pillars.as_array().to_owned(),
        breakevens.as_array().to_owned(),
      ),
    })
  }

  /// Forward CPI index ratio $I(0, T)/I(0, 0) = (1 + b(T))^T$.
  fn forward_index_ratio(&self, t: f64) -> f64 {
    use crate::inflation::curve::InflationCurve;
    self.inner.forward_index_ratio(t)
  }
}
