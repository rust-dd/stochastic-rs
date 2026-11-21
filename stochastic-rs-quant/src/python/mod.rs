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
mod calendar;
mod calibration_advanced;
mod calibration_basic;
mod calibration_rates;
mod credit_portfolio;
mod curves;
mod factors;
mod hscm;
mod lattice;
mod microstructure;
mod pricing_analytic;
mod pricing_digital;
mod pricing_exotic;
mod pricing_fourier;
mod pricing_path_dep;
mod risk;
mod slv;
mod survival;
mod vol_surface;
mod xva;

pub use bonds::PyCirBond;
pub use bonds::PyHullWhiteBond;
pub use bonds::PyVasicekBond;
pub use calendar::PyBusinessDayConvention;
pub use calendar::PyCalendar;
pub use calendar::PyDayCount;
pub use calendar::PySchedule;
pub use calendar::PyScheduleBuilder;
pub use calendar::py_days_between;
pub use calendar::py_easter_sunday;
pub use calendar::py_imm_date;
pub use calibration_advanced::PyCgmysvCalibrator;
pub use calibration_advanced::PyDoubleHestonCalibrator;
pub use calibration_advanced::PyHkdeCalibrator;
pub use calibration_advanced::PyLevyCalibrator;
pub use calibration_advanced::PyRBergomiCalibrator;
pub use calibration_advanced::PySvjCalibrator;
pub use calibration_basic::PyBsmCalibrator;
pub use calibration_basic::PyHestonCalibrator;
pub use calibration_basic::PyMarketSlice;
pub use calibration_basic::PySabrCalibrator;
pub use calibration_basic::PySabrCapletCalibrator;
pub use calibration_rates::PyBlackKarasinskiSwaptionCalibrator;
pub use calibration_rates::PyG2ppSwaptionCalibrator;
pub use calibration_rates::PyHullWhiteSwaptionCalibrator;
pub use credit_portfolio::PyCdoTranche;
pub use credit_portfolio::PyCdsIndex;
pub use curves::PyDiscountCurve;
pub use curves::PyMultiCurve;
pub use curves::PyNelsonSiegel;
pub use curves::PySvensson;
pub use curves::PyZeroCouponInflationCurve;
pub use factors::PyFamaMacBeth;
pub use factors::PyPairsStrategy;
pub use factors::PyPca;
pub use factors::empirical_cvar;
pub use factors::ledoit_wolf_shrinkage;
pub use factors::sample_covariance;
pub use hscm::PyHscmCalibrator;
pub use hscm::PyHscmMarketOption;
pub use hscm::PyHscmModel;
pub use lattice::PyHullWhiteCallableBond;
pub use microstructure::PyAlmgrenChrissPlan;
pub use microstructure::PyKyleEquilibrium;
pub use microstructure::PyOrderBook;
pub use microstructure::corwin_schultz_spread;
pub use microstructure::effective_spread;
pub use microstructure::multi_period_kyle;
pub use microstructure::propagator_price_impact;
pub use microstructure::roll_spread;
pub use pricing_analytic::PyBachelierPricer;
pub use pricing_analytic::PyBsmPricer;
pub use pricing_analytic::PyHestonAdiPricer;
pub use pricing_analytic::PyHestonPricer;
pub use pricing_analytic::PyMerton1976Pricer;
pub use pricing_analytic::PyQuantoPricer;
pub use pricing_analytic::PySabrPricer;
pub use pricing_digital::PyAssetOrNothingPricer;
pub use pricing_digital::PyCashOrNothingPricer;
pub use pricing_digital::PyGapPricer;
pub use pricing_digital::PySuperSharePricer;
pub use pricing_exotic::PyBarrierPricer;
pub use pricing_exotic::PyCliquetPricer;
pub use pricing_exotic::PyCompoundPricer;
pub use pricing_exotic::PyDoubleBarrierPricer;
pub use pricing_exotic::PyFixedLookbackPricer;
pub use pricing_exotic::PyFloatingLookbackPricer;
pub use pricing_exotic::PySimpleChooserPricer;
pub use pricing_exotic::PyVarianceSwapPricer;
pub use pricing_fourier::PyBatesFourier;
pub use pricing_fourier::PyBsmFourier;
pub use pricing_fourier::PyCarrMadanPricer;
pub use pricing_fourier::PyCgmyFourier;
pub use pricing_fourier::PyCosEngine;
pub use pricing_fourier::PyDoubleHestonFourier;
pub use pricing_fourier::PyHestonFourier;
pub use pricing_fourier::PyHkdeFourier;
pub use pricing_fourier::PyKouFourier;
pub use pricing_fourier::PyMertonJdFourier;
pub use pricing_fourier::PyNigFourier;
pub use pricing_fourier::PyVarianceGammaFourier;
pub use pricing_path_dep::PyAsianPricer;
pub use pricing_path_dep::PyBjerksundStensland2002Pricer;
pub use pricing_path_dep::PyKirkSpreadPricer;
pub use pricing_path_dep::PyMcBarrierPricer;
pub use pricing_path_dep::PyMcEstimate;
pub use risk::PyDrawdownStats;
pub use risk::PyExpectedShortfall;
pub use risk::PyVaR;
pub use slv::PyHestonSlvCalibrationResult;
pub use slv::PyHestonSlvCalibrator;
pub use slv::PyHestonSlvPricer;
pub use slv::PyLeverageSurface;
pub use survival::PySurvivalCurve;
pub use vol_surface::PyEssviSurface;
pub use vol_surface::PyImpliedVolSurface;
pub use vol_surface::PySsviCalibrator;
pub use vol_surface::PySsviParams;
pub use vol_surface::PySviCalibrator;
pub use vol_surface::PySviRawParams;
pub use xva::PyExposureProfile;
pub use xva::PyHullWhiteSwapExposure;

fn parse_option_type(s: &str) -> PyResult<OptionType> {
  match s.to_ascii_lowercase().as_str() {
    "c" | "call" => Ok(OptionType::Call),
    "p" | "put" => Ok(OptionType::Put),
    other => Err(PyValueError::new_err(format!(
      "option_type must be 'call' or 'put', got '{other}'"
    ))),
  }
}
