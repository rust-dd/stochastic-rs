use ndarray::Array1;
use pyo3::prelude::*;

pub(crate) trait IntoF32 {
  type Target;
  fn into_f32(self) -> Self::Target;
}

pub(crate) trait IntoF64 {
  type Target;
  fn into_f64(self) -> Self::Target;
}

impl IntoF32 for f64 {
  type Target = f32;
  fn into_f32(self) -> f32 {
    self as f32
  }
}
impl IntoF64 for f64 {
  type Target = f64;
  fn into_f64(self) -> f64 {
    self
  }
}

impl IntoF32 for Option<f64> {
  type Target = Option<f32>;
  fn into_f32(self) -> Option<f32> {
    self.map(|v| v as f32)
  }
}
impl IntoF64 for Option<f64> {
  type Target = Option<f64>;
  fn into_f64(self) -> Option<f64> {
    self
  }
}

impl IntoF32 for usize {
  type Target = usize;
  fn into_f32(self) -> usize {
    self
  }
}
impl IntoF64 for usize {
  type Target = usize;
  fn into_f64(self) -> usize {
    self
  }
}

impl IntoF32 for Option<usize> {
  type Target = Option<usize>;
  fn into_f32(self) -> Option<usize> {
    self
  }
}
impl IntoF64 for Option<usize> {
  type Target = Option<usize>;
  fn into_f64(self) -> Option<usize> {
    self
  }
}

impl IntoF32 for Option<bool> {
  type Target = Option<bool>;
  fn into_f32(self) -> Option<bool> {
    self
  }
}
impl IntoF64 for Option<bool> {
  type Target = Option<bool>;
  fn into_f64(self) -> Option<bool> {
    self
  }
}

impl IntoF32 for Vec<f64> {
  type Target = Array1<f32>;
  fn into_f32(self) -> Array1<f32> {
    Array1::from_vec(self).mapv(|v| v as f32)
  }
}
impl IntoF64 for Vec<f64> {
  type Target = Array1<f64>;
  fn into_f64(self) -> Array1<f64> {
    Array1::from_vec(self)
  }
}

impl IntoF32 for Option<Vec<f64>> {
  type Target = Option<Array1<f32>>;
  fn into_f32(self) -> Option<Array1<f32>> {
    self.map(|v| Array1::from_vec(v).mapv(|x| x as f32))
  }
}
impl IntoF64 for Option<Vec<f64>> {
  type Target = Option<Array1<f64>>;
  fn into_f64(self) -> Option<Array1<f64>> {
    self.map(Array1::from_vec)
  }
}

impl IntoF32 for u32 {
  type Target = u32;
  fn into_f32(self) -> u32 {
    self
  }
}
impl IntoF64 for u32 {
  type Target = u32;
  fn into_f64(self) -> u32 {
    self
  }
}

#[pymodule]
#[pyo3(name = "stochastic_rs")]
fn stochastic_rs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
  use crate::distributions::alpha_stable::PyAlphaStable;
  use crate::distributions::beta::PyBeta;
  use crate::distributions::binomial::PyBinomial;
  use crate::distributions::cauchy::PyCauchy;
  use crate::distributions::chi_square::PyChiSquared;
  use crate::distributions::exp::PyExp;
  use crate::distributions::gamma::PyGamma;
  use crate::distributions::geometric::PyGeometric;
  use crate::distributions::hypergeometric::PyHypergeometric;
  use crate::distributions::inverse_gauss::PyInverseGauss;
  use crate::distributions::lognormal::PyLogNormal;
  use crate::distributions::normal::PyNormal;
  use crate::distributions::normal_inverse_gauss::PyNormalInverseGauss;
  use crate::distributions::pareto::PyPareto;
  use crate::distributions::poisson::PyPoissonD;
  use crate::distributions::studentt::PyStudentT;
  use crate::distributions::uniform::PyUniform;
  use crate::distributions::weibull::PyWeibull;
  use crate::stochastic::autoregressive::agrach::PyAGARCH;
  use crate::stochastic::autoregressive::ar::PyARp;
  use crate::stochastic::autoregressive::arch::PyARCH;
  use crate::stochastic::autoregressive::arima::PyARIMA;
  use crate::stochastic::autoregressive::egarch::PyEGARCH;
  use crate::stochastic::autoregressive::garch::PyGARCH;
  use crate::stochastic::autoregressive::ma::PyMAq;
  use crate::stochastic::autoregressive::sarima::PySARIMA;
  use crate::stochastic::autoregressive::tgarch::PyTGARCH;
  use crate::stochastic::diffusion::cev::PyCEV;
  use crate::stochastic::diffusion::cir::PyCIR;
  use crate::stochastic::diffusion::fcir::PyFCIR;
  use crate::stochastic::diffusion::feller::PyFellerLogistic;
  use crate::stochastic::diffusion::fgbm::PyFGBM;
  use crate::stochastic::diffusion::fjacobi::PyFJacobi;
  use crate::stochastic::diffusion::fou::PyFOU;
  use crate::stochastic::diffusion::fouque::PyFouqueOU2D;
  use crate::stochastic::diffusion::gbm::PyGBM;
  use crate::stochastic::diffusion::gbm_ih::PyGBMIH;
  use crate::stochastic::diffusion::gompertz::PyGompertz;
  use crate::stochastic::diffusion::jacobi::PyJacobi;
  use crate::stochastic::diffusion::kimura::PyKimura;
  use crate::stochastic::diffusion::ou::PyOU;
  use crate::stochastic::diffusion::quadratic::PyQuadratic;
  use crate::stochastic::diffusion::verhulst::PyVerhulst;
  use crate::stochastic::interest::adg::PyADG;
  use crate::stochastic::interest::bgm::PyBGM;
  use crate::stochastic::interest::duffie_kan::PyDuffieKan;
  use crate::stochastic::interest::fvasicek::PyFVasicek;
  use crate::stochastic::interest::hjm::PyHJM;
  use crate::stochastic::interest::ho_lee::PyHoLee;
  use crate::stochastic::interest::hull_white::PyHullWhite;
  use crate::stochastic::interest::hull_white_2f::PyHullWhite2F;
  use crate::stochastic::interest::mod_duffie_kan::PyDuffieKanJumpExp;
  use crate::stochastic::interest::vasicek::PyVasicek;
  use crate::stochastic::interest::wu_zhang::PyWuZhangD;
  use crate::stochastic::jump::bates::PyBates;
  use crate::stochastic::jump::cgmy::PyCGMY;
  use crate::stochastic::jump::cts::PyCTS;
  use crate::stochastic::jump::ig::PyIG;
  use crate::stochastic::jump::jump_fou::PyJumpFOU;
  use crate::stochastic::jump::jump_fou_custom::PyJumpFOUCustom;
  use crate::stochastic::jump::kou::PyKOU;
  use crate::stochastic::jump::levy_diffusion::PyLevyDiffusion;
  use crate::stochastic::jump::merton::PyMerton;
  use crate::stochastic::jump::nig::PyNIG;
  use crate::stochastic::jump::rdts::PyRDTS;
  use crate::stochastic::jump::vg::PyVG;
  use crate::stochastic::noise::cfgns::PyCFGNS;
  use crate::stochastic::noise::cgns::PyCGNS;
  use crate::stochastic::noise::fgn::PyFGN;
  use crate::stochastic::noise::gn::PyGn;
  use crate::stochastic::noise::wn::PyWn;
  use crate::stochastic::process::bm::PyBM;
  use crate::stochastic::process::cbms::PyCBMS;
  use crate::stochastic::process::ccustom::PyCompoundCustom;
  use crate::stochastic::process::cfbms::PyCFBMS;
  use crate::stochastic::process::cpoisson::PyCompoundPoisson;
  use crate::stochastic::process::customjt::PyCustomJt;
  use crate::stochastic::process::fbm::PyFBM;
  use crate::stochastic::process::lfsm::PyLFSM;
  use crate::stochastic::process::poisson::PyPoisson;
  use crate::stochastic::sheet::fbs::PyFBS;
  use crate::stochastic::volatility::bergomi::PyBergomi;
  use crate::stochastic::volatility::fheston::PyRoughHeston;
  use crate::stochastic::volatility::heston::PyHeston;
  use crate::stochastic::volatility::rbergomi::PyRoughBergomi;
  use crate::stochastic::volatility::sabr::PySABR;
  use crate::stochastic::volatility::svcgmy::PySVCGMY;

  m.add_class::<PyFBM>()?;
  m.add_class::<PyLFSM>()?;
  m.add_class::<PyBM>()?;
  m.add_class::<PyCBMS>()?;
  m.add_class::<PyCFBMS>()?;
  m.add_class::<PyPoisson>()?;
  m.add_class::<PyCustomJt>()?;
  m.add_class::<PyCompoundPoisson>()?;
  m.add_class::<PyCompoundCustom>()?;
  m.add_class::<PyFGN>()?;
  m.add_class::<PyGn>()?;
  m.add_class::<PyWn>()?;
  m.add_class::<PyCGNS>()?;
  m.add_class::<PyCFGNS>()?;
  m.add_class::<PyCEV>()?;
  m.add_class::<PyCIR>()?;
  m.add_class::<PyFCIR>()?;
  m.add_class::<PyFellerLogistic>()?;
  m.add_class::<PyFGBM>()?;
  m.add_class::<PyFJacobi>()?;
  m.add_class::<PyFOU>()?;
  m.add_class::<PyFouqueOU2D>()?;
  m.add_class::<PyGBM>()?;
  m.add_class::<PyGBMIH>()?;
  m.add_class::<PyGompertz>()?;
  m.add_class::<PyJacobi>()?;
  m.add_class::<PyKimura>()?;
  m.add_class::<PyOU>()?;
  m.add_class::<PyQuadratic>()?;
  m.add_class::<PyVerhulst>()?;
  m.add_class::<PyCGMY>()?;
  m.add_class::<PyCTS>()?;
  m.add_class::<PyIG>()?;
  m.add_class::<PyNIG>()?;
  m.add_class::<PyRDTS>()?;
  m.add_class::<PyVG>()?;
  m.add_class::<PyMerton>()?;
  m.add_class::<PyKOU>()?;
  m.add_class::<PyLevyDiffusion>()?;
  m.add_class::<PyBates>()?;
  m.add_class::<PyJumpFOU>()?;
  m.add_class::<PyJumpFOUCustom>()?;
  m.add_class::<PyBergomi>()?;
  m.add_class::<PyHeston>()?;
  m.add_class::<PyRoughBergomi>()?;
  m.add_class::<PySABR>()?;
  m.add_class::<PyRoughHeston>()?;
  m.add_class::<PySVCGMY>()?;
  m.add_class::<PyDuffieKan>()?;
  m.add_class::<PyDuffieKanJumpExp>()?;
  m.add_class::<PyFVasicek>()?;
  m.add_class::<PyVasicek>()?;
  m.add_class::<PyBGM>()?;
  m.add_class::<PyWuZhangD>()?;
  m.add_class::<PyADG>()?;
  m.add_class::<PyHJM>()?;
  m.add_class::<PyHoLee>()?;
  m.add_class::<PyHullWhite>()?;
  m.add_class::<PyHullWhite2F>()?;
  m.add_class::<PyAGARCH>()?;
  m.add_class::<PyARp>()?;
  m.add_class::<PyARCH>()?;
  m.add_class::<PyARIMA>()?;
  m.add_class::<PyEGARCH>()?;
  m.add_class::<PyGARCH>()?;
  m.add_class::<PyMAq>()?;
  m.add_class::<PySARIMA>()?;
  m.add_class::<PyTGARCH>()?;
  m.add_class::<PyFBS>()?;
  m.add_class::<PyBeta>()?;
  m.add_class::<PyAlphaStable>()?;
  m.add_class::<PyCauchy>()?;
  m.add_class::<PyChiSquared>()?;
  m.add_class::<PyExp>()?;
  m.add_class::<PyGamma>()?;
  m.add_class::<PyInverseGauss>()?;
  m.add_class::<PyLogNormal>()?;
  m.add_class::<PyNormal>()?;
  m.add_class::<PyNormalInverseGauss>()?;
  m.add_class::<PyPareto>()?;
  m.add_class::<PyStudentT>()?;
  m.add_class::<PyUniform>()?;
  m.add_class::<PyWeibull>()?;
  m.add_class::<PyBinomial>()?;
  m.add_class::<PyGeometric>()?;
  m.add_class::<PyHypergeometric>()?;
  m.add_class::<PyPoissonD>()?;
  Ok(())
}
