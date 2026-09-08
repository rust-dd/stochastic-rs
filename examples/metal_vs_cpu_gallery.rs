//! The same processes on the CPU and on Metal, side by side.
//!
//! Samples a gallery of processes twice — once through the host sampler, once
//! through the Metal kernels — and writes one Plotly page per process into
//! `target/metal_vs_cpu/`, plus an index that links them. Each page holds
//! three panels: the CPU's paths, Metal's paths on the same scale, and the
//! two terminal laws overlaid.
//!
//! ```text
//! cargo run --release --example metal_vs_cpu_gallery --features metal
//! ```
//!
//! **The paths differ, and must.** The host draws from this crate's SIMD
//! ziggurat and the kernel hashes its own normals from `(path, step, seed)`,
//! so no path is shared. What has to agree is the law, which the third panel
//! shows and the table on stdout measures — in standard errors of the
//! estimate, not in a tolerance anyone picked. The same table prints what
//! each process says about itself first: `device_fallback()` names the reason
//! a configuration stays on the host, so a row that ran on the CPU twice is
//! visible rather than merely slow.
//!
//! Metal computes in single precision, so every process here is `f32`; that
//! is a compile-time property of the backend, not a choice this example
//! makes.
//!
//! Seventy-one processes over ten families, every parameter set one the
//! crate's own device-law suite runs. Four of them — Hull-White,
//! Black-Karasinski, CIR++ and Ho-Lee — take a curve, and reach the device
//! here because a native closure can be folded into a table before the
//! launch; the same four are `f64`-only from Python, where the curve is a
//! Python callable.

use std::fs;
use std::time::Instant;

use ndarray::Array1;
use ndarray::array;
use plotly::Histogram;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::autoregressive::ar::ARp;
use stochastic_rs::stochastic::autoregressive::arch::Arch;
use stochastic_rs::stochastic::autoregressive::arima::Arima;
use stochastic_rs::stochastic::autoregressive::egarch::Egarch;
use stochastic_rs::stochastic::autoregressive::garch::Garch;
use stochastic_rs::stochastic::autoregressive::ma::MAq;
use stochastic_rs::stochastic::autoregressive::tgarch::GjrGarch;
use stochastic_rs::stochastic::correlation::teng::TengSCP;
use stochastic_rs::stochastic::device::Cpu;
use stochastic_rs::stochastic::device::Metal;
use stochastic_rs::stochastic::diffusion::ait_sahalia::AitSahalia;
use stochastic_rs::stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs::stochastic::diffusion::cev::Cev;
use stochastic_rs::stochastic::diffusion::cir::Cir;
use stochastic_rs::stochastic::diffusion::ckls::Ckls;
use stochastic_rs::stochastic::diffusion::displaced_diffusion::DisplacedDiffusion;
use stochastic_rs::stochastic::diffusion::fcir::Fcir;
use stochastic_rs::stochastic::diffusion::feller::FellerLogistic;
use stochastic_rs::stochastic::diffusion::feller_root::FellerRoot;
use stochastic_rs::stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs::stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs::stochastic::diffusion::fou::Fou;
use stochastic_rs::stochastic::diffusion::fouque::FouqueOU2D;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs::stochastic::diffusion::hyperbolic::Hyperbolic;
use stochastic_rs::stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs::stochastic::diffusion::kimura::Kimura;
use stochastic_rs::stochastic::diffusion::linear_sde::LinearSDE;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::stochastic::diffusion::pearson::Pearson;
use stochastic_rs::stochastic::diffusion::radial_ou::RadialOU;
use stochastic_rs::stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs::stochastic::diffusion::verhulst::Verhulst;
use stochastic_rs::stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs::stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs::stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs::stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs::stochastic::interest::fractional_vasicek::FVasicek;
use stochastic_rs::stochastic::interest::ho_lee::HoLee;
use stochastic_rs::stochastic::interest::hull_white::HullWhite;
use stochastic_rs::stochastic::interest::vasicek::Vasicek;
use stochastic_rs::stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs::stochastic::jump::cgmy::Cgmy;
use stochastic_rs::stochastic::jump::cts::Cts;
use stochastic_rs::stochastic::jump::hawkes_jd::HawkesJD;
use stochastic_rs::stochastic::jump::ig::Ig;
use stochastic_rs::stochastic::jump::kobol::KoBoL;
use stochastic_rs::stochastic::jump::mjd_log::MjdLog;
use stochastic_rs::stochastic::jump::nig::Nig;
use stochastic_rs::stochastic::jump::vg::Vg;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::stochastic::process::bm::Bm;
use stochastic_rs::stochastic::process::brownian_bridge::BrownianBridge;
use stochastic_rs::stochastic::process::cbms::Cbms;
use stochastic_rs::stochastic::process::cfbms::Cfbms;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::stochastic::process::hawkes::Hawkes;
use stochastic_rs::stochastic::process::lfsm::Lfsm;
use stochastic_rs::stochastic::process::poisson::Poisson;
use stochastic_rs::stochastic::process::subordinator::alpha_stable::AlphaStableSubordinator;
use stochastic_rs::stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs::stochastic::process::subordinator::ig_subordinator::IGSubordinator;
use stochastic_rs::stochastic::process::subordinator::inverse_alpha_stable::InverseAlphaStableSubordinator;
use stochastic_rs::stochastic::process::subordinator::poisson_subordinator::PoissonSubordinator;
use stochastic_rs::stochastic::process::subordinator::tempered_stable::TemperedStableSubordinator;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::bergomi::Bergomi;
use stochastic_rs::stochastic::volatility::heston::Heston;
use stochastic_rs::stochastic::volatility::sabr::Sabr;
use stochastic_rs::traits::Fn1D;
use stochastic_rs::traits::ProcessExt;

/// Grid points per path.
const N: usize = 512;

/// Series terms the truncation-series Lévy processes take.
const J: usize = 64;

/// Paths per side. Enough that a terminal mean and spread are stable, few
/// enough that the whole gallery runs in seconds.
const PATHS: usize = 4_000;

/// Paths drawn in each panel. More than this and the picture is ink.
const DRAWN: usize = 24;

/// The seed both sides are built from. They still draw different streams —
/// that is the point — but each side is reproducible.
const SEED: u64 = 20_260_907;

/// A rising instantaneous-forward curve, for the models that take one.
///
/// A native closure rather than a Python callable, which is what lets these
/// four run on a device at all: a curve the kernel can evaluate is folded
/// into a table before the launch.
fn rising() -> Fn1D<f32> {
  Fn1D::Native(|t: f32| 0.02 + 0.03 * t)
}

/// One entry of the gallery: how to sample it on either backend, and what it
/// says about itself before it runs.
struct Case {
  family: &'static str,
  label: &'static str,
  slug: &'static str,
  host: fn() -> Vec<Array1<f32>>,
  device: fn() -> Vec<Array1<f32>>,
  fallback: fn() -> Option<&'static str>,
}

/// Builds the three closures a [`Case`] needs from one process constructor.
///
/// The constructor is written twice, once per backend, because the backend is
/// a type parameter: `Gbm<f32, _, Cpu>` and `Gbm<f32, _, Metal>` are
/// different types, which is what makes an unsupported backend a compile
/// error rather than a run-time surprise.
macro_rules! case {
  ($family:literal, $label:literal, $slug:literal, $build:expr) => {
    Case {
      family: $family,
      label: $label,
      slug: $slug,
      host: || $build.on::<Cpu>().sample_par(PATHS),
      device: || $build.on::<Metal>().sample_par(PATHS),
      fallback: || $build.device_fallback(),
    }
  };
}

/// [`case!`] for a process that reports several components: the one at
/// `$c` is the path drawn and measured, the others are its siblings in the
/// same launch.
macro_rules! component {
  ($family:literal, $label:literal, $slug:literal, $build:expr, $c:literal) => {
    Case {
      family: $family,
      label: $label,
      slug: $slug,
      host: || {
        $build
          .on::<Cpu>()
          .sample_par(PATHS)
          .into_iter()
          .map(|p| p[$c].clone())
          .collect()
      },
      device: || {
        $build
          .on::<Metal>()
          .sample_par(PATHS)
          .into_iter()
          .map(|p| p[$c].clone())
          .collect()
      },
      fallback: || $build.device_fallback(),
    }
  };
}

fn gallery() -> Vec<Case> {
  vec![
    case!(
      "brownian",
      "Brownian motion",
      "bm",
      Bm::<f32, _>::new(N, Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "brownian",
      "Brownian bridge  σ=0.3  0 → 1",
      "brownian-bridge",
      BrownianBridge::<f32, _>::new(
        0.3,
        N,
        Some(0.0),
        Some(1.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    component!(
      "brownian",
      "correlated BM pair  ρ=−0.5  (leg 1)",
      "cbms-1",
      Cbms::<f32, _>::new(-0.5, N, Some(1.0), Deterministic::new(SEED)),
      0
    ),
    component!(
      "brownian",
      "correlated BM pair  (leg 2)",
      "cbms-2",
      Cbms::<f32, _>::new(-0.5, N, Some(1.0), Deterministic::new(SEED)),
      1
    ),
    case!(
      "diffusion",
      "GBM  μ=0.05 σ=0.2",
      "gbm",
      Gbm::<f32, _>::new(
        0.05,
        0.2,
        N,
        Some(100.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "Ornstein-Uhlenbeck  θ=2 μ=0.04",
      "ou",
      Ou::<f32, _>::new(
        2.0,
        0.04,
        0.25,
        N,
        Some(0.12),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "CIR  θ=2 μ=0.04",
      "cir",
      Cir::<f32, _>::new(
        2.0,
        0.04,
        0.2,
        N,
        Some(0.04),
        Some(1.0),
        None,
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "CEV  γ=0.8",
      "cev",
      Cev::<f32, _>::new(
        0.05,
        0.2,
        0.8,
        N,
        Some(100.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "CKLS  θ=(0.06,−1.5,0.3,0.5)",
      "ckls",
      Ckls::<f32, _>::new(
        0.06,
        -1.5,
        0.3,
        0.5,
        N,
        Some(0.04),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "3/2 model  κ=2 μ=0.04",
      "three-half",
      ThreeHalf::<f32, _>::new(
        2.0,
        0.04,
        0.3,
        N,
        Some(0.04),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "squared Bessel  δ=3",
      "squared-bessel",
      SquaredBessel::<f32, _>::new(3.0, N, Some(1.0), Some(1.0), None, Deterministic::new(SEED))
    ),
    case!(
      "diffusion",
      "displaced diffusion  β=20",
      "displaced",
      DisplacedDiffusion::<f32, _>::new(
        0.05,
        0.2,
        20.0,
        N,
        Some(100.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "linear SDE  a=0.02 b=0.3",
      "linear-sde",
      LinearSDE::<f32, _>::new(
        0.02,
        0.3,
        0.2,
        N,
        Some(1.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "radial OU  κ=1",
      "radial-ou",
      RadialOU::<f32, _>::new(1.0, 0.3, N, Some(1.0), Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "diffusion",
      "Aït-Sahalia  (nonlinear drift)",
      "ait-sahalia",
      AitSahalia::<f32, _>::new(
        0.0001,
        0.15,
        -3.0,
        0.0,
        0.0004,
        0.0,
        0.05,
        1.5,
        N,
        Some(0.05),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "bounded diffusion",
      "Jacobi  α=0.3 β=0.6  (unit interval)",
      "jacobi",
      Jacobi::<f32, _>::new(
        0.3,
        0.6,
        0.2,
        N,
        Some(0.5),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "bounded diffusion",
      "Kimura  a=0.5",
      "kimura",
      Kimura::<f32, _>::new(0.5, 0.2, N, Some(0.5), Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "bounded diffusion",
      "Verhulst  r=1 K=2  (clamped)",
      "verhulst",
      Verhulst::<f32, _>::new(
        1.0,
        2.0,
        0.3,
        N,
        Some(0.5),
        Some(1.0),
        Some(true),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "bounded diffusion",
      "Feller root  θ=(0.5,0.3,0.2)",
      "feller-root",
      FellerRoot::<f32, _>::new(
        0.5,
        0.3,
        0.2,
        N,
        Some(0.5),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "bounded diffusion",
      "Feller logistic  κ=1 θ=1",
      "feller-logistic",
      FellerLogistic::<f32, _>::new(
        1.0,
        1.0,
        0.3,
        N,
        Some(0.5),
        Some(1.0),
        Some(false),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "bounded diffusion",
      "hyperbolic  κ=1",
      "hyperbolic",
      Hyperbolic::<f32, _>::new(1.0, 0.3, N, Some(0.5), Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "bounded diffusion",
      "Pearson  κ=1 μ=0.3",
      "pearson",
      Pearson::<f32, _>::new(
        1.0,
        0.3,
        0.0,
        0.0,
        0.01,
        N,
        Some(0.3),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "bounded diffusion",
      "Gompertz  a=0.5 b=0.3",
      "gompertz",
      Gompertz::<f32, _>::new(
        0.5,
        0.3,
        0.2,
        N,
        Some(1.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "bounded diffusion",
      "Teng stochastic correlation  (−1,1)",
      "teng",
      TengSCP::<f32, _>::new(1.0, 0.3, 0.4, 0.2, N, Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "short rate",
      "Vasicek  θ=0.5 μ=0.04",
      "vasicek",
      Vasicek::<f32, _>::new(
        0.5,
        0.04,
        0.02,
        N,
        Some(0.03),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "short rate",
      "fractional Vasicek  H=0.7",
      "fvasicek",
      FVasicek::<f32, _>::new(
        0.7,
        2.0,
        0.04,
        0.02,
        N,
        Some(0.03),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "short rate",
      "Ho-Lee  θ=0.03 σ=0.01",
      "ho-lee",
      HoLee::<f32, _>::new(
        None,
        Some(0.03),
        0.01,
        N,
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "short rate",
      "Hull-White  (rising curve)",
      "hull-white",
      HullWhite::<f32, _>::new(
        rising(),
        1.0,
        0.01,
        N,
        Some(0.02),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "short rate",
      "Black-Karasinski  (lognormal rate)",
      "black-karasinski",
      BlackKarasinski::<f32, _>::new(
        rising(),
        1.0,
        0.2,
        N,
        Some(0.03),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "short rate",
      "CIR++  (CIR plus a shift)",
      "cir-pp",
      CirPlusPlus::<f32, _>::new(
        2.0,
        0.04,
        0.2,
        rising(),
        N,
        Some(0.04),
        Some(1.0),
        Some(false),
        Deterministic::new(SEED)
      )
    ),
    component!(
      "short rate",
      "Duffie-Kan  (short rate leg)",
      "duffie-kan",
      DuffieKan::<f32, _>::new(
        0.5,
        0.2,
        0.1,
        -0.3,
        -0.5,
        0.1,
        0.02,
        0.1,
        0.05,
        -0.3,
        0.01,
        0.08,
        N,
        Some(0.03),
        Some(0.01),
        Some(1.0),
        Deterministic::new(SEED)
      ),
      0
    ),
    component!(
      "short rate",
      "Duffie-Kan with exponential jumps",
      "duffie-kan-jump",
      DuffieKanJumpExp::<f32, _>::new(
        0.5,
        0.2,
        0.1,
        -0.3,
        -0.5,
        0.1,
        0.02,
        0.1,
        0.05,
        -0.3,
        0.01,
        0.08,
        3.0,
        0.01,
        N,
        Some(0.03),
        Some(0.01),
        Some(1.0),
        Deterministic::new(SEED)
      ),
      0
    ),
    component!(
      "volatility",
      "Heston  κ=2 ξ=0.3 ρ=−0.7  (price)",
      "heston-price",
      Heston::<f32, _>::new(
        Some(100.0),
        Some(0.04),
        2.0,
        0.04,
        0.3,
        -0.7,
        0.0,
        N,
        Some(1.0),
        HestonPow::Sqrt,
        Some(false),
        Deterministic::new(SEED)
      ),
      0
    ),
    component!(
      "volatility",
      "Heston  (variance)",
      "heston-variance",
      Heston::<f32, _>::new(
        Some(100.0),
        Some(0.04),
        2.0,
        0.04,
        0.3,
        -0.7,
        0.0,
        N,
        Some(1.0),
        HestonPow::Sqrt,
        Some(false),
        Deterministic::new(SEED)
      ),
      1
    ),
    component!(
      "volatility",
      "SABR  α=0.4 β=0.5  (forward)",
      "sabr-forward",
      Sabr::<f32, _>::new(
        0.4,
        0.5,
        -0.4,
        N,
        Some(100.0),
        Some(0.2),
        Some(1.0),
        Deterministic::new(SEED)
      ),
      0
    ),
    component!(
      "volatility",
      "SABR  (volatility)",
      "sabr-vol",
      Sabr::<f32, _>::new(
        0.4,
        0.5,
        -0.4,
        N,
        Some(100.0),
        Some(0.2),
        Some(1.0),
        Deterministic::new(SEED)
      ),
      1
    ),
    component!(
      "volatility",
      "Bergomi  ν=0.5 ρ=−0.6",
      "bergomi",
      Bergomi::<f32, _>::new(
        0.5,
        Some(0.2),
        Some(100.0),
        0.02,
        -0.6,
        N,
        Some(1.0),
        Deterministic::new(SEED)
      ),
      0
    ),
    component!(
      "volatility",
      "Fouque OU 2F  (fast-slow volatility)",
      "fouque",
      FouqueOU2D::<f32, _>::new(
        1.0,
        0.3,
        0.25,
        -0.2,
        N,
        Some(0.0),
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      ),
      0
    ),
    case!(
      "jump",
      "Merton in log space  λ=3",
      "mjd-log",
      MjdLog::<f32, _>::new(
        Some(0.05),
        None,
        None,
        None,
        0.2,
        3.0,
        -0.05,
        0.1,
        N,
        Some(100.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "variance gamma  θ=−0.1 ν=0.5",
      "vg",
      Vg::<f32, _>::new(
        -0.1,
        0.2,
        0.5,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "normal inverse Gaussian  κ=0.5",
      "nig",
      Nig::<f32, _>::new(
        -0.1,
        0.2,
        0.5,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "CGMY  Y=0.5",
      "cgmy",
      Cgmy::<f32, _>::new(
        1.0,
        2.0,
        6.0,
        0.5,
        N,
        J,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "KoBoL  α=0.5",
      "kobol",
      KoBoL::<f32, _>::new(
        1.0,
        2.0,
        1.0,
        3.0,
        3.0,
        0.5,
        N,
        J,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "classical tempered stable  α=0.5",
      "cts",
      Cts::<f32, _>::new(
        2.0,
        6.0,
        0.5,
        N,
        J,
        Some(0.5),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "bilateral gamma motion",
      "bilateral-gamma",
      BilateralGammaMotion::<f32, _>::new(
        0.1,
        1.5,
        10.0,
        1.2,
        12.0,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "Hawkes jump-diffusion",
      "hawkes-jd",
      HawkesJD::<f32, _>::new(
        0.02,
        0.2,
        1.0,
        0.5,
        2.0,
        -0.02,
        0.05,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "inverse Gaussian motion  γ=1",
      "ig",
      Ig::<f32, _>::new(1.0, N, Some(0.0), Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "point process",
      "Poisson  λ=25  (arrival times)",
      "poisson",
      Poisson::<f32, _>::new(25.0, Some(N), None, Deterministic::new(SEED))
    ),
    case!(
      "point process",
      "Hawkes  μ=1 α=0.5 β=1.5  (event times)",
      "hawkes",
      Hawkes::<f32, _>::new(1.0, 0.5, 1.5, Some(64), None, Deterministic::new(SEED))
    ),
    case!(
      "fractional",
      "fBm  H=0.7",
      "fbm",
      Fbm::<f32, _>::new(0.7, N, Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "fractional",
      "fGN  H=0.3  (increments)",
      "fgn",
      Fgn::<f32, _>::new(0.3, N, Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "fractional",
      "fractional OU  H=0.7",
      "fou",
      Fou::<f32, _>::new(
        0.7,
        2.0,
        1.0,
        0.3,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "fractional",
      "fractional GBM  H=0.7",
      "fgbm",
      Fgbm::<f32, _>::new(
        0.7,
        0.05,
        0.2,
        N,
        Some(100.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "fractional",
      "fractional CIR  H=0.7",
      "fcir",
      Fcir::<f32, _>::new(
        0.7,
        2.0,
        0.04,
        0.1,
        N,
        Some(0.04),
        Some(1.0),
        None,
        Deterministic::new(SEED)
      )
    ),
    case!(
      "fractional",
      "fractional Jacobi  H=0.7",
      "fjacobi",
      FJacobi::<f32, _>::new(
        0.7,
        0.3,
        0.6,
        0.2,
        N,
        Some(0.5),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "fractional",
      "Lévy fractional stable motion  α=1.7",
      "lfsm",
      Lfsm::<f32, _>::new(
        1.7,
        0.2,
        0.7,
        0.1,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    component!(
      "fractional",
      "correlated fBm pair  H=0.7 ρ=0.4  (leg 1)",
      "cfbms-1",
      Cfbms::<f32, _>::new(0.7, 0.4, N, Some(1.0), Deterministic::new(SEED)),
      0
    ),
    component!(
      "fractional",
      "correlated fBm pair  (leg 2)",
      "cfbms-2",
      Cfbms::<f32, _>::new(0.7, 0.4, N, Some(1.0), Deterministic::new(SEED)),
      1
    ),
    case!(
      "subordinator",
      "α-stable  α=0.7",
      "alpha-stable",
      AlphaStableSubordinator::<f32, _>::new(
        0.7,
        1.0,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "subordinator",
      "gamma  ν=2 rate=1.5",
      "gamma-sub",
      GammaSubordinator::<f32, _>::new(2.0, 1.5, N, Some(0.0), Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "subordinator",
      "inverse Gaussian  δ=1 γ=2",
      "ig-sub",
      IGSubordinator::<f32, _>::new(1.0, 2.0, N, Some(0.0), Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "subordinator",
      "tempered stable  α=0.6",
      "tempered-stable",
      TemperedStableSubordinator::<f32, _>::new(
        0.6,
        1.0,
        2.0,
        0.05,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "subordinator",
      "Poisson  λ=20  (a counting clock)",
      "poisson-sub",
      PoissonSubordinator::<f32, _>::new(20.0, N, Some(0.0), Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "subordinator",
      "inverse α-stable  α=0.7  (a waiting clock)",
      "inverse-alpha-stable",
      InverseAlphaStableSubordinator::<f32, _>::new(
        0.7,
        1.0,
        N,
        Some(1.0),
        256,
        None,
        Deterministic::new(SEED)
      )
    ),
    case!(
      "time series",
      "AR(1)  φ=0.6",
      "ar",
      ARp::<f32, _>::new(
        array![0.6],
        0.2,
        N,
        Some(array![0.5]),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "time series",
      "MA(1)  θ=0.4",
      "ma",
      MAq::<f32, _>::new(array![0.4], 0.2, N, Deterministic::new(SEED))
    ),
    case!(
      "time series",
      "ARIMA(2,1,1)",
      "arima",
      Arima::<f32, _>::new(
        array![0.6, -0.2],
        array![0.3],
        1,
        0.5,
        N,
        Deterministic::new(SEED)
      )
    ),
    case!(
      "time series",
      "ARCH(1)  α=0.3",
      "arch",
      Arch::<f32, _>::new(0.0002, array![0.3], N, Deterministic::new(SEED))
    ),
    case!(
      "time series",
      "GARCH(1,1)  α=0.1 β=0.85",
      "garch",
      Garch::<f32, _>::new(
        0.00001,
        array![0.1],
        array![0.85],
        N,
        Deterministic::new(SEED)
      )
    ),
    case!(
      "time series",
      "EGARCH(1,1)",
      "egarch",
      Egarch::<f32, _>::new(
        -0.2,
        array![0.1],
        array![-0.05],
        array![0.95],
        N,
        Deterministic::new(SEED)
      )
    ),
    case!(
      "time series",
      "GJR-GARCH(1,1)",
      "gjr-garch",
      GjrGarch::<f32, _>::new(
        0.00001,
        array![0.05],
        array![0.1],
        array![0.85],
        N,
        Deterministic::new(SEED)
      )
    ),
  ]
}

/// Mean, standard deviation and the standard error of each, from the paths'
/// terminal values. The spread's error uses the sample's own kurtosis
/// (`σ√((κ−1)/4m)`), which is what makes a heavy-tailed process comparable at
/// all.
fn terminal_stats(paths: &[Array1<f32>]) -> (f64, f64, f64, f64) {
  let last: Vec<f64> = paths.iter().map(|p| p[p.len() - 1] as f64).collect();
  let n = last.len() as f64;
  let mean = last.iter().sum::<f64>() / n;
  let m2 = last.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
  let m4 = last.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
  let kurtosis = m4 / (m2 * m2);
  let sd = m2.sqrt();
  (
    mean,
    (m2 / n).sqrt(),
    sd,
    sd * ((kurtosis - 1.0) / (4.0 * n)).sqrt(),
  )
}

/// One page: the host's paths, the device's paths on the same scale, and the
/// two terminal laws.
fn page(label: &str, host: &[Array1<f32>], device: &[Array1<f32>], path: &str) {
  let mut plot = Plot::new();
  let xs: Vec<f64> = (0..host[0].len()).map(|i| i as f64).collect();
  for (paths, axis, colour) in [(host, 1usize, "#1f77b4"), (device, 2, "#d62728")] {
    for series in paths.iter().take(DRAWN) {
      let trace = Scatter::new(
        xs.clone(),
        series.iter().map(|v| *v as f64).collect::<Vec<_>>(),
      )
      .mode(Mode::Lines)
      .line(Line::new().width(0.8).color(colour))
      .show_legend(false);
      plot.add_trace(match axis {
        1 => trace,
        _ => trace.x_axis("x2").y_axis("y2"),
      });
    }
  }
  let terminal =
    |paths: &[Array1<f32>]| -> Vec<f64> { paths.iter().map(|p| p[p.len() - 1] as f64).collect() };
  plot.add_trace(
    Histogram::new(terminal(host))
      .name("CPU")
      .opacity(0.6)
      .n_bins_x(60)
      .x_axis("x3")
      .y_axis("y3"),
  );
  plot.add_trace(
    Histogram::new(terminal(device))
      .name("Metal")
      .opacity(0.6)
      .n_bins_x(60)
      .x_axis("x3")
      .y_axis("y3"),
  );
  plot.set_layout(
    Layout::new()
      .title(format!(
        "{label} — CPU and Metal draw different streams; the law is what agrees"
      ))
      .grid(
        LayoutGrid::new()
          .rows(1)
          .columns(3)
          .pattern(GridPattern::Independent),
      )
      .height(420)
      .x_axis(Axis::new().title("step — CPU"))
      .x_axis2(Axis::new().title("step — Metal"))
      .x_axis3(Axis::new().title("terminal value"))
      .y_axis3(Axis::new().title("count")),
  );
  plot.write_html(path);
}

fn main() {
  let dir = "target/metal_vs_cpu";
  fs::create_dir_all(dir).expect("output directory");

  // What the device is, before anything is asked of it. A handle is a value:
  // `Metal::default()` reads the ordinal from the environment, and `probe`
  // opens it.
  match stochastic_rs::stochastic::device::Backend::probe(&Metal::default()) {
    Ok(info) => println!(
      "device: {} ({}), precisions {:?}\n",
      info.name, info.backend, info.precisions
    ),
    Err(e) => {
      eprintln!("no Metal device: {e}");
      return;
    }
  }

  let cases = gallery();

  // One launch before the timings: the first call to a Metal device compiles
  // the kernel and builds the pipeline, a few hundred milliseconds that
  // belong to no process in particular.
  let _ = Gbm::<f32, _>::new(0.05, 0.2, 32, Some(1.0), Some(1.0), Deterministic::new(1))
    .on::<Metal>()
    .sample_par(8);

  println!(
    "{:<38} {:>10} {:>10} {:>7} {:>10} {:>10} {:>7} {:>8} {:>8}",
    "process", "mean cpu", "mean metal", "z", "sd cpu", "sd metal", "z", "cpu ms", "metal ms"
  );
  for case in &cases {
    if let Some(why) = (case.fallback)() {
      println!("{:<38} host on both sides: {why}", case.label);
    }
    let start = Instant::now();
    let host = (case.host)();
    let host_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    let device = (case.device)();
    let device_ms = start.elapsed().as_secs_f64() * 1e3;

    let (hm, hme, hs, hse) = terminal_stats(&host);
    let (dm, dme, ds, dse) = terminal_stats(&device);
    println!(
      "{:<38} {hm:>10.4} {dm:>10.4} {:>7.1} {hs:>10.4} {ds:>10.4} {:>7.1} {host_ms:>8.1} {device_ms:>8.1}",
      case.label,
      (hm - dm) / (hme * hme + dme * dme).sqrt(),
      (hs - ds) / (hse * hse + dse * dse).sqrt(),
    );
    page(
      case.label,
      &host,
      &device,
      &format!("{dir}/{}.html", case.slug),
    );
  }

  let mut index = String::from(
    "<!doctype html><meta charset=utf-8><title>Metal against the CPU</title>\
     <style>body{font:15px system-ui;margin:3rem auto;max-width:44rem}\
     li{margin:.35rem 0}code{color:#666}</style>\
     <h1>The same processes on the CPU and on Metal</h1>\
     <p>Each page holds the host's paths, the device's paths on the same \
     scale, and the two terminal laws overlaid. The paths differ by \
     construction — the two sides draw different streams — so what the \
     pictures compare is the law.</p><ul>",
  );
  for case in &cases {
    index.push_str(&format!(
      "<li><a href=\"{}.html\">{}</a> <code>{}</code></li>",
      case.slug, case.label, case.family
    ));
  }
  index.push_str("</ul>");
  fs::write(format!("{dir}/index.html"), index).expect("index");
  println!("\nwrote {} pages to {dir}/index.html", cases.len());
  println!(
    "z is in standard errors; |z| under 5 is agreement. The α-stable row's \
     means are not comparable — that law has no mean — which is why the z \
     column, scaled by the sample's own spread, is the honest read."
  );
}
