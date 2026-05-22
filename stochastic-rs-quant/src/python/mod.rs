//! PyO3 wrappers for `stochastic-rs-quant`.
//!
//! Pricing engines (analytic + Fourier), calibrators, vol-surface parameterisations,
//! and bond-pricing closed forms exposed as `#[pyclass]` types. Registered by the
//! `stochastic-rs-py` cdylib.

#![cfg(feature = "python")]
#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::OptionType;

mod bonds;
mod calibration_advanced;
mod calibration_basic;
mod curves;
mod factors;
mod hscm;
mod microstructure;
mod pricing_analytic;
mod pricing_digital;
mod pricing_fourier;
mod pricing_path_dep;
mod risk;
mod vol_surface;

pub use bonds::PyCIRBond;
pub use bonds::PyHullWhiteBond;
pub use bonds::PyVasicekBond;
pub use calibration_advanced::PyCgmysvCalibrator;
pub use calibration_advanced::PyDoubleHestonCalibrator;
pub use calibration_advanced::PyHKDECalibrator;
pub use calibration_advanced::PyLevyCalibrator;
pub use calibration_advanced::PyRBergomiCalibrator;
pub use calibration_advanced::PySVJCalibrator;
pub use calibration_basic::PyBSMCalibrator;
pub use calibration_basic::PyHestonCalibrator;
pub use calibration_basic::PyMarketSlice;
pub use calibration_basic::PySabrCalibrator;
pub use calibration_basic::PySabrCapletCalibrator;
pub use curves::PyDiscountCurve;
pub use curves::PyNelsonSiegel;
pub use curves::PyZeroCouponInflationCurve;
#[cfg(feature = "openblas")]
pub use factors::PyFamaMacBeth;
#[cfg(feature = "openblas")]
pub use factors::PyPCA;
#[cfg(feature = "openblas")]
pub use factors::PyPairsStrategy;
pub use factors::empirical_cvar;
pub use factors::ledoit_wolf_shrinkage;
pub use factors::sample_covariance;
pub use hscm::PyHscmCalibrator;
pub use hscm::PyHscmMarketOption;
pub use hscm::PyHscmModel;
pub use microstructure::PyAlmgrenChrissPlan;
pub use microstructure::PyKyleEquilibrium;
pub use microstructure::PyOrderBook;
pub use microstructure::corwin_schultz_spread;
pub use microstructure::effective_spread;
pub use microstructure::multi_period_kyle;
pub use microstructure::propagator_price_impact;
pub use microstructure::roll_spread;
pub use pricing_analytic::PyBSMPricer;
pub use pricing_analytic::PyHestonPricer;
pub use pricing_analytic::PyMerton1976Pricer;
pub use pricing_analytic::PySabrPricer;
pub use pricing_digital::PyAssetOrNothingPricer;
pub use pricing_digital::PyCashOrNothingPricer;
pub use pricing_digital::PyGapPricer;
pub use pricing_digital::PySuperSharePricer;
pub use pricing_fourier::PyBSMFourier;
pub use pricing_fourier::PyBatesFourier;
pub use pricing_fourier::PyCGMYFourier;
pub use pricing_fourier::PyCarrMadanPricer;
pub use pricing_fourier::PyDoubleHestonFourier;
pub use pricing_fourier::PyHKDEFourier;
pub use pricing_fourier::PyHestonFourier;
pub use pricing_fourier::PyKouFourier;
pub use pricing_fourier::PyMertonJDFourier;
pub use pricing_fourier::PyNigFourier;
pub use pricing_fourier::PyVarianceGammaFourier;
pub use pricing_path_dep::PyAsianPricer;
pub use pricing_path_dep::PyBarrierPricer;
pub use pricing_path_dep::PyBjerksundStensland2002Pricer;
pub use pricing_path_dep::PyCliquetPricer;
pub use pricing_path_dep::PyCompoundPricer;
pub use pricing_path_dep::PyDoubleBarrierPricer;
pub use pricing_path_dep::PyFixedLookbackPricer;
pub use pricing_path_dep::PyFloatingLookbackPricer;
pub use pricing_path_dep::PyKirkSpreadPricer;
pub use pricing_path_dep::PyMCBarrierPricer;
pub use pricing_path_dep::PySimpleChooserPricer;
pub use pricing_path_dep::PyVarianceSwapPricer;
pub use risk::PyDrawdownStats;
pub use risk::PyExpectedShortfall;
pub use risk::PyVaR;
pub use vol_surface::PyImpliedVolSurface;
pub use vol_surface::PySsviCalibrator;
pub use vol_surface::PySsviParams;
pub use vol_surface::PySviCalibrator;
pub use vol_surface::PySviRawParams;

fn parse_option_type(s: &str) -> PyResult<OptionType> {
  match s.to_ascii_lowercase().as_str() {
    "c" | "call" => Ok(OptionType::Call),
    "p" | "put" => Ok(OptionType::Put),
    other => Err(PyValueError::new_err(format!(
      "option_type must be 'call' or 'put', got '{other}'"
    ))),
  }
}
