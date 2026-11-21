//! `compile_fail` checks: a field a Monte Carlo constructor checks is set only through it.

/// ```compile_fail,E0616
/// use stochastic_rs_quant::pricing::barrier::McBarrierPricer;
///
/// let mut pricer = McBarrierPricer::new(0.2, 4_000, 32);
/// pricer.sigma = -1.0;
/// pricer.n_paths = 0;
/// pricer.n_steps = 0;
/// ```
pub struct McBarrierPricerIsSetThroughNew;

/// ```compile_fail,E0616
/// use stochastic_rs_quant::pricing::cliquet::McCliquetPricer;
///
/// let mut pricer = McCliquetPricer::new(0.2, 4_000);
/// pricer.sigma = -1.0;
/// pricer.n_paths = 0;
/// ```
pub struct McCliquetPricerIsSetThroughNew;

/// ```compile_fail,E0616
/// use stochastic_rs_quant::pricing::autocallable::AutocallablePricer;
///
/// let mut pricer = AutocallablePricer::new(0.2, 4_000, 4);
/// pricer.sigma = -1.0;
/// pricer.n_paths = 0;
/// pricer.steps_per_period = 0;
/// ```
pub struct AutocallablePricerIsSetThroughNew;

/// ```compile_fail,E0616
/// use ndarray::array;
/// use stochastic_rs_quant::pricing::rainbow::McRainbowPricer;
/// use stochastic_rs_quant::pricing::rainbow::RainbowPayoff;
///
/// let rho = array![[1.0, 0.5], [0.5, 1.0]];
/// let mut pricer = McRainbowPricer::new(RainbowPayoff::CallOnMax, array![0.2, 0.3], rho, 4_000);
/// pricer.sigma = array![-1.0, 0.3];
/// pricer.n_paths = 0;
/// ```
pub struct McRainbowPricerIsSetThroughNew;

/// ```compile_fail,E0616
/// use stochastic_rs_quant::pricing::spread::McSpreadPricer;
///
/// let mut pricer = McSpreadPricer::new(0.3, 0.2, 0.5, 4_000);
/// pricer.sigma1 = -1.0;
/// pricer.sigma2 = -1.0;
/// pricer.rho = f64::NAN;
/// pricer.n_paths = 0;
/// ```
pub struct McSpreadPricerIsSetThroughNew;

/// ```compile_fail,E0616
/// use ndarray::array;
/// use stochastic_rs_quant::pricing::basket::BasketAverageType;
/// use stochastic_rs_quant::pricing::basket::McBasketPricer;
///
/// let rho = array![[1.0, 0.5], [0.5, 1.0]];
/// let weights = array![0.5, 0.5];
/// let average = BasketAverageType::Arithmetic;
/// let mut pricer = McBasketPricer::new(weights, average, array![0.2, 0.3], rho, 4_000);
/// pricer.sigma = array![-1.0, 0.3];
/// pricer.n_paths = 0;
/// ```
pub struct McBasketPricerIsSetThroughNew;

/// ```compile_fail,E0616
/// use stochastic_rs_quant::pricing::malliavin_gbm::GbmMalliavinPricer;
///
/// let mut pricer = GbmMalliavinPricer::new(0.2, 2_000, 64, 0.5);
/// pricer.sigma = -1.0;
/// pricer.n_paths = 0;
/// pricer.n_steps = 1;
/// pricer.t_eval = 0.0;
/// ```
pub struct GbmMalliavinPricerIsSetThroughNew;

/// ```compile_fail,E0616
/// use stochastic_rs_quant::pricing::heston::HestonParams;
/// use stochastic_rs_quant::pricing::malliavin_greeks::HestonMalliavinGreeks;
///
/// let params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 };
/// let mut greeks = HestonMalliavinGreeks::new(params, 100.0, 100.0, 0.05, 1.0, 2_000, 32);
/// greeks.params.sigma = 0.0;
/// greeks.n_paths = 0;
/// greeks.n_steps = 1;
/// ```
pub struct HestonMalliavinGreeksIsSetThroughNew;

/// ```compile_fail,E0616
/// use ndarray::Array2;
/// use stochastic_rs_quant::pricing::malliavin_thalmaier::AssetParams;
/// use stochastic_rs_quant::pricing::malliavin_thalmaier::MtGreeks;
/// use stochastic_rs_quant::pricing::malliavin_thalmaier::MultiHestonParams;
///
/// let asset = AssetParams { s: 100.0, v0: 0.04, kappa: 1.0, theta: 0.04, xi: 0.0, rho: 0.0 };
/// let cross_corr = Array2::<f64>::eye(1);
/// let params = MultiHestonParams { assets: vec![asset], cross_corr, r: 0.0, tau: 1.0, n_steps: 9 };
/// let mut engine = MtGreeks::new(params, 0.01, 16);
/// engine.params.tau = -1.0;
/// engine.h = 0.0;
/// engine.n_paths = 0;
/// ```
pub struct MtGreeksIsSetThroughNew;

/// ```compile_fail,E0616
/// use stochastic_rs_quant::calibration::rbergomi::RBergomiCalibrationConfig;
/// use stochastic_rs_quant::calibration::rbergomi::RBergomiCalibrator;
/// use stochastic_rs_quant::calibration::rbergomi::RBergomiParams;
/// use stochastic_rs_quant::calibration::rbergomi::RBergomiXi0;
///
/// let start = RBergomiParams { hurst: 0.1, rho: -0.7, eta: 2.0, xi0: RBergomiXi0::Constant(0.04) };
/// let config = RBergomiCalibrationConfig::default();
/// let mut calibrator = RBergomiCalibrator::new(100.0, 0.01, 0.0, start, Vec::new(), config);
/// calibrator.config.paths = 0;
/// ```
pub struct RBergomiCalibratorIsSetThroughNew;
