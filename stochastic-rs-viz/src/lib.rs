//! # stochastic-rs-viz
//!
//! Plotly-based visualization for stochastic processes and distributions.
//!
//! Module layout:
//! - [`plottable`] — `Plottable<T>` trait + impls for the canonical
//!   `ProcessExt::Output` shapes (1D path, complex path, fixed-arity tuple,
//!   2D matrix).
//! - [`grid_plotter`] — `GridPlotter` builder for multi-subplot HTML grids.
//! - [`convenience`] — one-shot `plot_process` / `plot_distribution` /
//!   `plot_vol_surface` HTML writers.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

pub mod convenience;
pub mod grid_plotter;
pub mod plottable;

pub use convenience::plot_distribution;
pub use convenience::plot_process;
pub use convenience::plot_vol_surface;
pub use grid_plotter::GridPlotter;
pub use plottable::Plottable;

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use plotly::Layout;
  use plotly::Plot;
  use plotly::Scatter;
  use plotly::Surface;
  use plotly::common::DashType;
  use plotly::common::Mode;
  use rand_distr::Exp;
  use rand_distr::Normal;
  use stochastic_rs_stochastic::autoregressive::agrach::Agarch;
  use stochastic_rs_stochastic::autoregressive::ar::ARp;
  use stochastic_rs_stochastic::autoregressive::arch::Arch;
  use stochastic_rs_stochastic::autoregressive::arima::Arima;
  use stochastic_rs_stochastic::autoregressive::egarch::Egarch;
  use stochastic_rs_stochastic::autoregressive::garch::Garch;
  use stochastic_rs_stochastic::autoregressive::ma::MAq;
  use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
  use stochastic_rs_stochastic::autoregressive::tgarch::Tgarch;
  use stochastic_rs_stochastic::diffusion::cev::Cev;
  use stochastic_rs_stochastic::diffusion::cfou::Cfou;
  use stochastic_rs_stochastic::diffusion::cir::Cir as DiffCIR;
  use stochastic_rs_stochastic::diffusion::fcir::Fcir;
  use stochastic_rs_stochastic::diffusion::feller::FellerLogistic;
  use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
  use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
  use stochastic_rs_stochastic::diffusion::fou::Fou;
  use stochastic_rs_stochastic::diffusion::fouque::FouqueOU2D;
  use stochastic_rs_stochastic::diffusion::gbm::Gbm;
  use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
  use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
  use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
  use stochastic_rs_stochastic::diffusion::kimura::Kimura;
  use stochastic_rs_stochastic::diffusion::ou::Ou;
  use stochastic_rs_stochastic::diffusion::quadratic::Quadratic;
  use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;
  use stochastic_rs_stochastic::interest::adg::Adg;
  use stochastic_rs_stochastic::interest::bgm::Bgm;
  use stochastic_rs_stochastic::interest::cir::Cir as RateCIR;
  use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
  use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
  use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
  use stochastic_rs_stochastic::interest::fractional_vasicek::FVasicek;
  use stochastic_rs_stochastic::interest::hjm::Hjm;
  use stochastic_rs_stochastic::interest::ho_lee::HoLee;
  use stochastic_rs_stochastic::interest::hull_white::HullWhite;
  use stochastic_rs_stochastic::interest::hull_white_2f::HullWhite2F;
  use stochastic_rs_stochastic::interest::vasicek::Vasicek;
  use stochastic_rs_stochastic::interest::wu_zhang::WuZhangD;
  use stochastic_rs_stochastic::isonormal::IsoNormal;
  use stochastic_rs_stochastic::isonormal::fbm_custom_inc_cov;
  use stochastic_rs_stochastic::jump::bates::Bates1996;
  use stochastic_rs_stochastic::jump::cgmy::Cgmy;
  use stochastic_rs_stochastic::jump::cts::Cts;
  use stochastic_rs_stochastic::jump::ig::Ig;
  use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
  use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
  use stochastic_rs_stochastic::jump::kobol::KoBoL;
  use stochastic_rs_stochastic::jump::kou::Kou;
  use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
  use stochastic_rs_stochastic::jump::merton::Merton;
  use stochastic_rs_stochastic::jump::nig::Nig;
  use stochastic_rs_stochastic::jump::rdts::Rdts;
  use stochastic_rs_stochastic::jump::vg::Vg;
  use stochastic_rs_stochastic::noise::cfgns::Cfgns;
  use stochastic_rs_stochastic::noise::cgns::Cgns;
  use stochastic_rs_stochastic::noise::fgn::Fgn;
  use stochastic_rs_stochastic::noise::gn::Gn;
  use stochastic_rs_stochastic::noise::wn::Wn;
  use stochastic_rs_stochastic::process::bm::Bm;
  use stochastic_rs_stochastic::process::cbms::Cbms;
  use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
  use stochastic_rs_stochastic::process::cfbms::Cfbms;
  use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
  use stochastic_rs_stochastic::process::customjt::CustomJt;
  use stochastic_rs_stochastic::process::fbm::Fbm;
  use stochastic_rs_stochastic::process::lfsm::Lfsm;
  use stochastic_rs_stochastic::process::poisson::Poisson;
  use stochastic_rs_stochastic::process::subordinator::AlphaStableSubordinator;
  use stochastic_rs_stochastic::process::subordinator::Ctrw;
  use stochastic_rs_stochastic::process::subordinator::CtrwJumpLaw;
  use stochastic_rs_stochastic::process::subordinator::CtrwWaitingLaw;
  use stochastic_rs_stochastic::process::subordinator::GammaSubordinator;
  use stochastic_rs_stochastic::process::subordinator::IGSubordinator;
  use stochastic_rs_stochastic::process::subordinator::InverseAlphaStableSubordinator;
  use stochastic_rs_stochastic::process::subordinator::PoissonSubordinator;
  use stochastic_rs_stochastic::process::subordinator::TemperedStableSubordinator;
  use stochastic_rs_stochastic::sheet::fbs::Fbs;
  use stochastic_rs_stochastic::traits::ProcessExt;
  use stochastic_rs_stochastic::volatility::HestonPow;
  use stochastic_rs_stochastic::volatility::bergomi::Bergomi;
  use stochastic_rs_stochastic::volatility::fheston::RoughHeston;
  use stochastic_rs_stochastic::volatility::heston::Heston;
  use stochastic_rs_stochastic::volatility::rbergomi::RoughBergomi;
  use stochastic_rs_stochastic::volatility::sabr::Sabr;
  use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

  use super::*;

  fn f_const_001(_: f64) -> f64 {
    0.01
  }

  fn f_const_002(_: f64) -> f64 {
    0.02
  }

  fn f_linear_small(t: f64) -> f64 {
    0.01 + 0.005 * t
  }

  fn f_phi_small(t: f64) -> f64 {
    0.002 * t
  }

  fn f_hjm_p(t: f64, u: f64) -> f64 {
    0.01 + 0.01 * (u - t).max(0.0)
  }

  fn f_hjm_q(_: f64, _: f64) -> f64 {
    0.5
  }

  fn f_hjm_v(_: f64, _: f64) -> f64 {
    0.02
  }

  fn f_hjm_alpha(_: f64, _: f64) -> f64 {
    0.01
  }

  fn f_hjm_sigma(_: f64, _: f64) -> f64 {
    0.015
  }

  fn f_adg_k(t: f64) -> f64 {
    0.02 + 0.002 * t
  }

  fn f_adg_theta(_: f64) -> f64 {
    0.6
  }

  fn f_adg_phi(_: f64) -> f64 {
    0.01
  }

  fn f_adg_b(_: f64) -> f64 {
    0.2
  }

  fn f_adg_c(_: f64) -> f64 {
    0.05
  }

  fn normal_cpoisson(lambda: f64, n: usize, jump_sigma: f64) -> CompoundPoisson<f64, Normal<f64>> {
    CompoundPoisson::new(
      Normal::new(0.0, jump_sigma).expect("valid normal"),
      Poisson::new(lambda, Some(n), Some(1.0)),
    )
  }

  #[test]
  fn plot_grid() {
    let n = 96;
    let traj = 1;
    let j = 64;
    let sheet_m = 3;
    let sheet_n = 64;

    let mut isonormal_fbm = IsoNormal::new(
      |aux_idx, idx| fbm_custom_inc_cov(aux_idx.abs_diff(idx), 0.7),
      (0..n).collect(),
    );
    let mut isonormal_paths = Vec::with_capacity(traj);
    for _ in 0..traj {
      let increments = isonormal_fbm.get_path();
      let mut path = Vec::with_capacity(n);
      path.push(0.0);
      let mut acc = 0.0;
      for &dx in &increments {
        acc += dx;
        path.push(acc);
      }
      isonormal_paths.push(path);
    }

    let mut grid = GridPlotter::new()
      .title("Stochastic Processes (Grid)")
      .cols(4)
      .show_legend(false)
      .line_width(1.2)
      .x_gap(0.80)
      .y_gap(5.00);

    grid = grid.register(
      &ARp::new(Array1::from_vec(vec![0.65, -0.2]), 0.08, n, None),
      "Autoreg: AR(2)",
      traj,
    );
    grid = grid.register(
      &MAq::new(Array1::from_vec(vec![0.5, -0.2]), 0.1, n),
      "Autoreg: MA(2)",
      traj,
    );
    grid = grid.register(
      &Arima::new(
        Array1::from_vec(vec![0.4]),
        Array1::from_vec(vec![0.3]),
        1,
        0.1,
        n,
      ),
      "Autoreg: Arima(1,1,1)",
      traj,
    );
    grid = grid.register(
      &Sarima::new(
        Array1::from_vec(vec![0.3]),
        Array1::from_vec(vec![0.2]),
        Array1::from_vec(vec![0.2]),
        Array1::from_vec(vec![0.1]),
        1,
        1,
        12,
        0.08,
        n,
      ),
      "Autoreg: Sarima",
      traj,
    );
    grid = grid.register(
      &Arch::new(0.05, Array1::from_vec(vec![0.2, 0.1]), n),
      "Autoreg: Arch",
      traj,
    );
    grid = grid.register(
      &Garch::new(
        0.03,
        Array1::from_vec(vec![0.12]),
        Array1::from_vec(vec![0.8]),
        n,
      ),
      "Autoreg: Garch",
      traj,
    );
    grid = grid.register(
      &Tgarch::new(
        0.03,
        Array1::from_vec(vec![0.08]),
        Array1::from_vec(vec![0.05]),
        Array1::from_vec(vec![0.85]),
        n,
      ),
      "Autoreg: Tgarch",
      traj,
    );
    grid = grid.register(
      &Egarch::new(
        -0.1,
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![-0.05]),
        Array1::from_vec(vec![0.9]),
        n,
      ),
      "Autoreg: Egarch",
      traj,
    );
    grid = grid.register(
      &Agarch::new(
        0.03,
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.04]),
        Array1::from_vec(vec![0.84]),
        n,
      ),
      "Autoreg: Agarch",
      traj,
    );

    grid = grid.register(&Wn::new(n, Some(0.0), Some(1.0)), "Noise: White", traj);
    grid = grid.register(&Gn::new(n, Some(1.0)), "Noise: Gaussian", traj);
    grid = grid.register(&Fgn::new(0.7, n, Some(1.0)), "Noise: Fgn", traj);
    grid = grid.register(&Cgns::new(-0.4, n, Some(1.0)), "Noise: Cgns", traj);
    grid = grid.register(&Cfgns::new(0.7, -0.3, n, Some(1.0)), "Noise: Cfgns", traj);

    grid = grid.register(&Bm::new(n, Some(1.0)), "Process: Bm", traj);
    grid = grid.register(&Fbm::new(0.7, n, Some(1.0)), "Process: Fbm", traj);
    grid = grid.register_paths(isonormal_paths, "Process: fBM via IsoNormal (H=0.7)");
    grid = grid.register(
      &Poisson::new(2.0, Some(n), Some(1.0)),
      "Process: Poisson",
      traj,
    );
    grid = grid.register(
      &CustomJt::new(
        Some(n),
        Some(1.0),
        Exp::new(10.0).expect("positive exponential rate"),
      ),
      "Process: CustomJt",
      traj,
    );
    grid = grid.register(
      &CompoundPoisson::new(
        Normal::new(0.0, 0.15).expect("valid normal"),
        Poisson::new(1.2, Some(n), Some(1.0)),
      ),
      "Process: CompoundPoisson",
      traj,
    );
    grid = grid.register(
      &CompoundCustom::new(
        Some(n),
        Some(1.0),
        Normal::new(0.0, 0.1).expect("valid normal"),
        Exp::new(15.0).expect("positive exponential rate"),
        CustomJt::new(
          Some(n),
          Some(1.0),
          Exp::new(15.0).expect("positive exponential rate"),
        ),
      ),
      "Process: CompoundCustom",
      traj,
    );
    grid = grid.register(&Cbms::new(0.35, n, Some(1.0)), "Process: Cbms", traj);
    grid = grid.register(&Cfbms::new(0.7, 0.35, n, Some(1.0)), "Process: Cfbms", traj);
    grid = grid.register(
      &Lfsm::new(1.7, 0.0, 0.8, 1.0, n, Some(0.0), Some(1.0)),
      "Process: Lfsm",
      traj,
    );
    grid = grid.register(
      &AlphaStableSubordinator::new(0.7, 1.0, n, Some(0.0), Some(1.0)),
      "Process: AlphaStable Subordinator",
      traj,
    );
    grid = grid.register(
      &InverseAlphaStableSubordinator::new(0.7, 1.0, n, Some(1.0), 2048, Some(4.0)),
      "Process: Inverse AlphaStable",
      traj,
    );
    grid = grid.register(
      &PoissonSubordinator::new(2.0, n, Some(0.0), Some(1.0)),
      "Process: Poisson Subordinator",
      traj,
    );
    grid = grid.register(
      &GammaSubordinator::new(3.0, 5.0, n, Some(0.0), Some(1.0)),
      "Process: Gamma Subordinator",
      traj,
    );
    grid = grid.register(
      &IGSubordinator::new(1.5, 2.0, n, Some(0.0), Some(1.0)),
      "Process: Ig Subordinator",
      traj,
    );
    grid = grid.register(
      &TemperedStableSubordinator::new(0.7, 1.0, 2.0, 0.05, n, Some(0.0), Some(1.0)),
      "Process: Tempered Stable Subordinator",
      traj,
    );
    grid = grid.register(
      &Ctrw::new(
        CtrwWaitingLaw::Exponential { rate: 2.0 },
        CtrwJumpLaw::Normal {
          mean: 0.0,
          std: 0.3,
        },
        n,
        Some(0.0),
        Some(1.0),
      ),
      "Process: Ctrw",
      traj,
    );

    grid = grid.register(
      &Ou::new(2.0, 0.0, 0.2, n, Some(0.0), Some(1.0)),
      "Diffusion: Ou",
      traj,
    );
    grid = grid.register(
      &Gbm::new(0.05, 0.2, n, Some(100.0), Some(1.0)),
      "Diffusion: Gbm",
      traj,
    );
    grid = grid.register(
      &DiffCIR::new(2.5, 0.04, 0.2, n, Some(0.04), Some(1.0), Some(false)),
      "Diffusion: Cir",
      traj,
    );
    grid = grid.register(
      &Cev::new(0.04, 0.2, 0.8, n, Some(1.0), Some(1.0)),
      "Diffusion: Cev",
      traj,
    );
    grid = grid.register(
      &FellerLogistic::new(2.0, 1.0, 0.3, n, Some(0.5), Some(1.0), Some(false)),
      "Diffusion: Feller Logistic",
      traj,
    );
    grid = grid.register(
      &Verhulst::new(1.2, 2.0, 0.2, n, Some(0.5), Some(1.0), Some(true)),
      "Diffusion: Verhulst",
      traj,
    );
    grid = grid.register(
      &Gompertz::new(1.0, 0.8, 0.2, n, Some(1.0), Some(1.0)),
      "Diffusion: Gompertz",
      traj,
    );
    grid = grid.register(
      &Kimura::new(1.0, 0.3, n, Some(0.4), Some(1.0)),
      "Diffusion: Kimura",
      traj,
    );
    grid = grid.register(
      &Quadratic::new(0.1, -0.2, 0.05, 0.15, n, Some(1.0), Some(1.0)),
      "Diffusion: Quadratic",
      traj,
    );
    grid = grid.register(
      &Jacobi::new(0.8, 1.4, 0.4, n, Some(0.3), Some(1.0)),
      "Diffusion: Jacobi",
      traj,
    );
    grid = grid.register(
      &Fcir::new(0.7, 2.5, 0.04, 0.2, n, Some(0.04), Some(1.0), Some(false)),
      "Diffusion: Fcir",
      traj,
    );
    grid = grid.register(
      &FJacobi::new(0.7, 0.8, 1.4, 0.35, n, Some(0.3), Some(1.0)),
      "Diffusion: FJacobi",
      traj,
    );
    grid = grid.register(
      &Fou::new(0.7, 2.0, 0.0, 0.2, n, Some(0.0), Some(1.0)),
      "Diffusion: Fou",
      traj,
    );
    grid = grid.register(
      &Cfou::new(0.7, 1.8, 3.0, 0.4, n, Some(0.0), Some(0.0), Some(1.0)),
      "Diffusion: Complex fOU",
      traj,
    );
    grid = grid.register(
      &Fgbm::new(0.7, 0.04, 0.2, n, Some(100.0), Some(1.0)),
      "Diffusion: Fgbm",
      traj,
    );
    grid = grid.register(
      &GbmIh::new(0.04, 0.2, n, Some(100.0), Some(1.0), None),
      "Diffusion: GbmIh",
      traj,
    );
    grid = grid.register(
      &FouqueOU2D::new(1.5, 0.0, 0.3, 0.0, n, Some(0.0), Some(0.0), Some(1.0)),
      "Diffusion: Fouque Ou 2D",
      traj,
    );

    grid = grid.register(
      &Vasicek::new(3.0, 0.03, 0.02, n, Some(0.03), Some(1.0)),
      "Interest: Vasicek",
      traj,
    );
    grid = grid.register(
      &FVasicek::new(0.7, 2.0, 0.03, 0.02, n, Some(0.03), Some(1.0)),
      "Interest: Fractional Vasicek",
      traj,
    );
    grid = grid.register(
      &RateCIR::new(2.5, 0.04, 0.2, n, Some(0.04), Some(1.0), Some(false)),
      "Interest: Cir (Alias)",
      traj,
    );
    grid = grid.register(
      &HoLee::new(None, Some(0.01), 0.01, n, Some(1.0)),
      "Interest: Ho-Lee",
      traj,
    );
    grid = grid.register(
      &HullWhite::new(
        f_linear_small as fn(f64) -> f64,
        0.4,
        0.02,
        n,
        Some(0.02),
        Some(1.0),
      ),
      "Interest: Hull-White",
      traj,
    );
    grid = grid.register(
      &HullWhite2F::new(
        f_const_001 as fn(f64) -> f64,
        0.5,
        0.02,
        0.015,
        -0.3,
        0.4,
        Some(0.02),
        Some(1.0),
        n,
      ),
      "Interest: Hull-White 2F",
      traj,
    );
    grid = grid.register(
      &Hjm::new(
        f_const_001 as fn(f64) -> f64,
        f_const_002 as fn(f64) -> f64,
        f_hjm_p as fn(f64, f64) -> f64,
        f_hjm_q as fn(f64, f64) -> f64,
        f_hjm_v as fn(f64, f64) -> f64,
        f_hjm_alpha as fn(f64, f64) -> f64,
        f_hjm_sigma as fn(f64, f64) -> f64,
        n,
        Some(0.01),
        Some(1.0),
        Some(0.01),
        Some(1.0),
      ),
      "Interest: Hjm",
      traj,
    );
    grid = grid.register(
      &Bgm::new(
        Array1::from_vec(vec![0.2, 0.15]),
        Array1::from_vec(vec![0.02, 0.025]),
        2,
        Some(1.0),
        n,
      ),
      "Interest: Bgm",
      traj,
    );
    grid = grid.register(
      &Adg::new(
        f_adg_k as fn(f64) -> f64,
        f_adg_theta as fn(f64) -> f64,
        Array1::from_vec(vec![0.02, 0.018]),
        f_adg_phi as fn(f64) -> f64,
        f_adg_b as fn(f64) -> f64,
        f_adg_c as fn(f64) -> f64,
        n,
        2,
        Array1::from_vec(vec![0.01, 0.015]),
        Some(1.0),
      ),
      "Interest: Adg",
      traj,
    );
    grid = grid.register(
      &DuffieKan::new(
        0.2,
        0.1,
        0.05,
        -0.3,
        -0.1,
        0.2,
        0.01,
        0.1,
        0.15,
        -0.2,
        0.01,
        0.12,
        n,
        Some(0.02),
        Some(0.01),
        Some(1.0),
      ),
      "Interest: Duffie-Kan",
      traj,
    );
    grid = grid.register(
      &DuffieKanJumpExp::new(
        0.2,
        0.1,
        0.05,
        -0.3,
        -0.1,
        0.2,
        0.01,
        0.1,
        0.15,
        -0.2,
        0.01,
        0.12,
        2.0,
        0.02,
        n,
        Some(0.02),
        Some(0.01),
        Some(1.0),
      ),
      "Interest: Duffie-Kan Jump Exp",
      traj,
    );
    grid = grid.register(
      &WuZhangD::new(
        Array1::from_vec(vec![0.05, 0.04]),
        Array1::from_vec(vec![1.2, 1.0]),
        Array1::from_vec(vec![0.3, 0.25]),
        Array1::from_vec(vec![0.4, 0.3]),
        Array1::from_vec(vec![0.02, 0.025]),
        Array1::from_vec(vec![0.04, 0.03]),
        2,
        Some(1.0),
        n,
      ),
      "Interest: Wu-Zhang",
      traj,
    );
    grid = grid.register(
      &Cir2F::new(
        RateCIR::new(2.5, 0.03, 0.12, n, Some(0.03), Some(1.0), Some(false)),
        RateCIR::new(2.0, 0.02, 0.1, n, Some(0.02), Some(1.0), Some(false)),
        f_phi_small as fn(f64) -> f64,
      ),
      "Interest: Cir 2F",
      traj,
    );

    grid = grid.register(
      &Vg::new(0.0, 0.2, 0.15, n, Some(0.0), Some(1.0)),
      "Jump: Vg",
      traj,
    );
    grid = grid.register(
      &Nig::new(0.0, 0.2, 0.3, n, Some(0.0), Some(1.0)),
      "Jump: Nig",
      traj,
    );
    grid = grid.register(&Ig::new(1.0, n, Some(0.0), Some(1.0)), "Jump: Ig", traj);
    grid = grid.register(
      &Rdts::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0)),
      "Jump: Rdts",
      traj,
    );
    grid = grid.register(
      &Cts::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0)),
      "Jump: Cts",
      traj,
    );

    let g = 4.0;
    let m = 5.0;
    let y = 0.7;

    let c = Cgmy::<f64>::c_for_unit_variance(g, m, y);
    // KoBoL: in case of p=q=1 D_for_unit_variance == C_for_unit_variance
    let d = KoBoL::<f64>::d_for_unit_variance(1.0, 1.0, g, m, y);

    grid = grid.register(
      &Cgmy::<f64>::new(c, g, m, y, n, j, Some(0.0), Some(1.0)),
      "Jump: Cgmy (unit var, symmetric)",
      traj,
    );

    grid = grid.register(
      &KoBoL::<f64>::new(d, 1.0, 1.0, g, m, y, n, j, Some(0.0), Some(1.0)),
      "Jump: KoBoL (unit var, p=q=1)",
      traj,
    );

    grid = grid.register(
      &Merton::new(
        0.03,
        0.2,
        1.0,
        0.0,
        n,
        Some(0.0),
        Some(1.0),
        normal_cpoisson(1.0, n, 0.1),
      ),
      "Jump: Merton",
      traj,
    );
    grid = grid.register(
      &Kou::new(
        0.03,
        0.2,
        1.0,
        0.0,
        n,
        Some(0.0),
        Some(1.0),
        normal_cpoisson(1.0, n, 0.12),
      ),
      "Jump: Kou",
      traj,
    );
    grid = grid.register(
      &LevyDiffusion::new(
        0.01,
        0.2,
        n,
        Some(0.0),
        Some(1.0),
        normal_cpoisson(1.0, n, 0.08),
      ),
      "Jump: Levy Diffusion",
      traj,
    );
    grid = grid.register(
      &JumpFou::new(
        0.7,
        2.0,
        0.03,
        0.2,
        n,
        Some(0.03),
        Some(1.0),
        normal_cpoisson(1.0, n, 0.08),
      ),
      "Jump: Jump-Fou",
      traj,
    );
    grid = grid.register(
      &JumpFOUCustom::new(
        0.7,
        2.0,
        0.03,
        0.2,
        n,
        Some(0.03),
        Some(1.0),
        Exp::new(20.0).expect("positive exponential rate"),
        Exp::new(30.0).expect("positive exponential rate"),
      ),
      "Jump: Jump-Fou Custom",
      traj,
    );
    grid = grid.register_with_component_labels(
      &Bates1996::new(
        Some(0.03),
        None,
        None,
        None,
        0.8,
        0.0,
        1.5,
        0.8,
        0.3,
        -0.5,
        n,
        Some(100.0),
        Some(0.04),
        Some(1.0),
        Some(false),
        normal_cpoisson(0.8, n, 0.05),
      ),
      "Jump: Bates 1996 (S: solid, v: dashed)",
      &["S", "v"],
      traj,
    );

    grid = grid.register(
      &Heston::new(
        Some(100.0),
        Some(0.04),
        2.0,
        0.04,
        0.3,
        -0.7,
        0.05,
        n,
        Some(1.0),
        HestonPow::Sqrt,
        Some(false),
      ),
      "Volatility: Heston",
      traj,
    );
    grid = grid.register(
      &Bergomi::new(0.4, Some(0.2), Some(100.0), 0.01, -0.6, n, Some(1.0)),
      "Volatility: Bergomi",
      traj,
    );
    grid = grid.register(
      &RoughBergomi::new(0.1, 0.4, Some(0.2), Some(100.0), 0.01, -0.6, n, Some(1.0)),
      "Volatility: Rough Bergomi",
      traj,
    );
    grid = grid.register(
      &RoughHeston::new(0.8, Some(0.2), 0.04, 1.5, 0.3, None, None, Some(1.0), n),
      "Volatility: Rough Heston",
      traj,
    );
    grid = grid.register(
      &Sabr::new(0.4, 0.7, -0.3, n, Some(1.0), Some(0.3), Some(1.0)),
      "Volatility: Sabr",
      traj,
    );
    grid = grid.register(
      &Svcgmy::new(
        3.0,
        4.0,
        0.7,
        1.5,
        0.04,
        0.3,
        -0.4,
        n,
        j,
        Some(0.0),
        Some(0.04),
        Some(1.0),
      ),
      "Volatility: Svcgmy",
      traj,
    );

    grid.show();

    let fbs_field = Fbs::new(0.7, sheet_m, sheet_n, 2.0).sample();
    let z: Vec<Vec<f64>> = fbs_field.outer_iter().map(|row| row.to_vec()).collect();
    let x: Vec<f64> = (0..sheet_n)
      .map(|i| i as f64 / (sheet_n.saturating_sub(1).max(1) as f64))
      .collect();
    let y: Vec<f64> = (0..sheet_m)
      .map(|i| i as f64 / (sheet_m.saturating_sub(1).max(1) as f64))
      .collect();

    let mut sheet_plot = Plot::new();
    let surface = Surface::new(z).x(x).y(y).name("Sheet: Fbs");
    sheet_plot.add_trace(surface);
    sheet_plot.set_layout(
      Layout::new()
        .title("Sheet: Fbs (3D Surface)")
        .height(900)
        .width(1200),
    );
    sheet_plot.show();
  }

  #[test]
  fn plot_sde_gbm_all_methods() {
    use ndarray::Array2;
    use ndarray::array;
    use plotly::Layout;
    use plotly::common::Line;
    use plotly::layout::Margin;
    use rand::rng;
    use stochastic_rs_stochastic::sde::NoiseModel;
    use stochastic_rs_stochastic::sde::Sde;
    use stochastic_rs_stochastic::sde::SdeMethod;

    let mu = 0.05;
    let sigma = 0.2;
    let x0 = array![100.0];
    let t0: f64 = 0.0;
    let t1: f64 = 1.0;
    let dt: f64 = 0.001;
    let n_paths = 5;
    let steps = ((t1 - t0) / dt).ceil() as usize;

    let t_axis: Vec<f64> = (0..=steps).map(|i| t0 + i as f64 * dt).collect();

    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"];
    let methods = [
      (SdeMethod::Euler, "Euler-Maruyama"),
      (SdeMethod::Milstein, "Milstein"),
      (SdeMethod::SRK2, "Midpoint RK2"),
      (SdeMethod::SRK4, "RK4-style"),
    ];

    let mut plot = Plot::new();
    plot.set_layout(
      Layout::new()
        .title("Gbm: SDE Solver Methods Comparison (dS = 0.05 S dt + 0.2 S dW)")
        .auto_size(true)
        .height(700)
        .margin(Margin::new().left(60).right(30).top(80).bottom(50)),
    );

    for (m_idx, (method, method_name)) in methods.into_iter().enumerate() {
      let sde = Sde::new(
        move |x: &ndarray::Array1<f64>, _t: f64| array![mu * x[0]],
        move |x: &ndarray::Array1<f64>, _t: f64| Array2::from_elem((1, 1), sigma * x[0]),
        NoiseModel::Gaussian,
        None,
      );

      let paths = sde.solve(&x0, t0, t1, dt, n_paths, method, &mut rng());

      for p in 0..n_paths {
        let y: Vec<f64> = (0..=steps).map(|i| paths[[p, i, 0]]).collect();
        let name = if p == 0 {
          method_name.to_string()
        } else {
          format!("{method_name} (path {p})")
        };
        let trace = Scatter::new(t_axis.clone(), y)
          .mode(Mode::Lines)
          .line(
            Line::new()
              .width(if p == 0 { 2.0 } else { 1.0 })
              .color(colors[m_idx])
              .dash(match p {
                0 => DashType::Solid,
                1 => DashType::Dash,
                2 => DashType::Dot,
                3 => DashType::DashDot,
                _ => DashType::LongDash,
              }),
          )
          .name(name.as_str())
          .show_legend(p == 0);
        plot.add_trace(trace);
      }
    }

    let mut path = std::env::temp_dir();
    path.push("stochastic_rs_sde_gbm_methods.html");
    plot.write_html(&path);
    assert!(path.exists(), "expected plot HTML at {}", path.display());
    let _ = std::fs::remove_file(path);
  }

  #[test]
  fn plot_process_writes_html() {
    let bm = Bm::new(64, Some(1.0));
    let path_arr = bm.sample();
    let mut out = std::env::temp_dir();
    out.push("stochastic_rs_test_plot_process.html");
    plot_process(&path_arr, out.to_str().unwrap());
    assert!(out.exists(), "plot_process did not write file");
    let _ = std::fs::remove_file(out);
  }

  #[test]
  fn plot_distribution_writes_html() {
    let samples: Array1<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
    let mut out = std::env::temp_dir();
    out.push("stochastic_rs_test_plot_distribution.html");
    plot_distribution(&samples, out.to_str().unwrap(), "test");
    assert!(out.exists(), "plot_distribution did not write file");
    let _ = std::fs::remove_file(out);
  }

  #[test]
  fn plot_vol_surface_writes_html() {
    use ndarray::Array2;
    let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
    let maturities = vec![0.25, 0.5, 1.0];
    let ivs = Array2::<f64>::from_shape_fn((maturities.len(), strikes.len()), |(j, i)| {
      0.2 + 0.01 * (j as f64) - 0.005 * ((i as f64) - 2.0).abs()
    });
    let mut out = std::env::temp_dir();
    out.push("stochastic_rs_test_plot_vol_surface.html");
    plot_vol_surface(&strikes, &maturities, &ivs, out.to_str().unwrap());
    assert!(out.exists(), "plot_vol_surface did not write file");
    let _ = std::fs::remove_file(out);
  }

  #[test]
  #[should_panic(expected = "ivs shape must be")]
  fn plot_vol_surface_rejects_bad_shape() {
    use ndarray::Array2;
    let strikes = vec![80.0, 90.0, 100.0];
    let maturities = vec![0.25, 0.5];
    let bad = Array2::<f64>::zeros((3, 5));
    plot_vol_surface(&strikes, &maturities, &bad, "/tmp/should_not_exist.html");
  }

  #[test]
  fn grid_plotter_rescale_threshold_disabled_writes_html() {
    let bm = Bm::new(64, Some(1.0));
    let grid = GridPlotter::new()
      .title("rescale-threshold=None smoke")
      .cols(1)
      .rescale_threshold(None)
      .register(&bm, "Bm", 1);
    let plot = grid.plot();
    let mut out = std::env::temp_dir();
    out.push("stochastic_rs_test_rescale_disabled.html");
    plot.write_html(&out);
    assert!(out.exists(), "plot did not write file");
    let _ = std::fs::remove_file(out);
  }
}
