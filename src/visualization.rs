//! # Visualization
//!
//! $$
//! \text{paths }\{X_t^{(k)}\}_{k=1}^m \mapsto \text{diagnostic charts on common grids}
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Anchor;
use plotly::common::DashType;
use plotly::common::Font;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::layout::Annotation;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;
use plotly::layout::Margin;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub trait Plottable<T: FloatExt> {
  fn n_components(&self) -> usize;
  fn component_name(&self, idx: usize) -> String;
  fn component(&self, idx: usize) -> Vec<f64>;
  fn len(&self) -> usize;
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

struct GridEntry {
  title: String,
  n_points: usize,
  series: Vec<GridSeries>,
}

struct GridSeries {
  label: String,
  component_idx: usize,
  values: Vec<f64>,
}

pub struct GridPlotter {
  entries: Vec<GridEntry>,
  cols: usize,
  line_width: f64,
  show_legend: bool,
  title: String,
  x_gap: f64,
  y_gap: f64,
}

impl GridPlotter {
  pub fn new() -> Self {
    Self {
      entries: Vec::new(),
      cols: 3,
      line_width: 1.0,
      show_legend: false,
      title: String::new(),
      x_gap: 0.06,
      y_gap: 0.12,
    }
  }

  pub fn title(mut self, title: &str) -> Self {
    self.title = title.into();
    self
  }

  pub fn cols(mut self, n: usize) -> Self {
    self.cols = n;
    self
  }

  pub fn line_width(mut self, w: f64) -> Self {
    self.line_width = w;
    self
  }

  pub fn show_legend(mut self, show: bool) -> Self {
    self.show_legend = show;
    self
  }

  pub fn x_gap(mut self, x_gap: f64) -> Self {
    self.x_gap = x_gap.max(0.0);
    self
  }

  pub fn y_gap(mut self, y_gap: f64) -> Self {
    self.y_gap = y_gap.max(0.0);
    self
  }

  pub fn register<T, P>(self, process: &P, title: &str, n_traj: usize) -> Self
  where
    T: FloatExt,
    P: ProcessExt<T>,
    P::Output: Plottable<T>,
  {
    self.register_impl::<T, P>(process, title, n_traj, None)
  }

  pub fn register_with_component_labels<T, P>(
    self,
    process: &P,
    title: &str,
    component_labels: &[&str],
    n_traj: usize,
  ) -> Self
  where
    T: FloatExt,
    P: ProcessExt<T>,
    P::Output: Plottable<T>,
  {
    self.register_impl::<T, P>(process, title, n_traj, Some(component_labels))
  }

  fn register_impl<T, P>(
    mut self,
    process: &P,
    title: &str,
    n_traj: usize,
    component_labels: Option<&[&str]>,
  ) -> Self
  where
    T: FloatExt,
    P: ProcessExt<T>,
    P::Output: Plottable<T>,
  {
    if n_traj == 0 {
      return self;
    }

    let samples: Vec<P::Output> = (0..n_traj).map(|_| process.sample()).collect();
    let n_comp = samples[0].n_components();
    if let Some(labels) = component_labels {
      assert_eq!(
        labels.len(),
        n_comp,
        "component_labels length must match number of components"
      );
    }
    let n_points = samples[0].len();
    let mut series = Vec::with_capacity(n_comp * n_traj);
    for c in 0..n_comp {
      let comp_name = if let Some(labels) = component_labels {
        labels[c].to_string()
      } else {
        samples[0].component_name(c)
      };
      for (traj_idx, sample) in samples.iter().enumerate() {
        let base_label = if n_comp > 1 && n_traj > 1 {
          format!("{} - traj {}", comp_name, traj_idx + 1)
        } else if n_comp > 1 {
          comp_name.clone()
        } else if n_traj > 1 {
          format!("traj {}", traj_idx + 1)
        } else {
          title.to_string()
        };
        let label = if base_label == title {
          base_label
        } else {
          format!("{} | {}", title, base_label)
        };
        series.push(GridSeries {
          label,
          component_idx: c,
          values: sample.component(c),
        });
      }
    }

    self.entries.push(GridEntry {
      title: title.into(),
      n_points,
      series,
    });

    self
  }

  pub fn register_paths(mut self, trajectories: Vec<Vec<f64>>, title: &str) -> Self {
    if trajectories.is_empty() {
      return self;
    }

    let n_points = trajectories[0].len();
    for traj in &trajectories {
      assert_eq!(
        traj.len(),
        n_points,
        "All trajectories must have the same length"
      );
    }

    let mut series = Vec::with_capacity(trajectories.len());
    for (i, traj) in trajectories.into_iter().enumerate() {
      series.push(GridSeries {
        label: format!("traj {}", i + 1),
        component_idx: 0,
        values: traj,
      });
    }

    self.entries.push(GridEntry {
      title: title.into(),
      n_points,
      series,
    });

    self
  }

  pub fn plot(self) -> Plot {
    let n = self.entries.len();
    let cols = self.cols;
    let rows = n.div_ceil(cols);
    let plot_height = (rows * 440 + 220).max(1500);
    let x_gap = if cols > 1 {
      let scaled = self.x_gap / cols as f64;
      let max_gap = ((1.0 - 0.08 * cols as f64) / cols.saturating_sub(1) as f64).max(0.0);
      scaled.min(max_gap).max(0.0)
    } else {
      0.0
    };
    let y_gap = if rows > 1 {
      let scaled = self.y_gap / rows as f64;
      let max_gap = ((1.0 - 0.02 * rows as f64) / rows.saturating_sub(1) as f64).max(0.0);
      scaled.min(max_gap).max(0.0)
    } else {
      0.0
    };

    let axis_name = |subplot_idx: usize, axis: &str| -> String {
      if subplot_idx == 1 {
        axis.to_string()
      } else {
        format!("{axis}{subplot_idx}")
      }
    };

    let mut annotations = Vec::with_capacity(self.entries.len());
    for (idx, entry) in self.entries.iter().enumerate() {
      let subplot_idx = idx + 1;
      let xa = axis_name(subplot_idx, "x");
      let ya = axis_name(subplot_idx, "y");

      annotations.push(
        Annotation::new()
          .text(format!("<b>{}</b>", entry.title))
          .x_ref(format!("{xa} domain"))
          .y_ref(format!("{ya} domain"))
          .x(0.5)
          .y(0.985)
          .x_anchor(Anchor::Center)
          .y_anchor(Anchor::Top)
          .font(Font::new().size(12))
          .background_color("rgba(255,255,255,0.92)")
          .border_pad(1.0)
          .show_arrow(false),
      );
    }

    let mut plot = Plot::new();
    plot.set_layout(
      Layout::new()
        .title(self.title.as_str())
        .auto_size(true)
        .height(plot_height)
        .margin(Margin::new().left(56).right(24).top(84).bottom(44))
        .annotations(annotations)
        .grid(
          LayoutGrid::new()
            .rows(rows)
            .columns(cols)
            .x_gap(x_gap)
            .y_gap(y_gap)
            .pattern(GridPattern::Independent),
        ),
    );

    for (idx, entry) in self.entries.iter().enumerate() {
      let subplot_idx = idx + 1;
      let xa = axis_name(subplot_idx, "x");
      let ya = axis_name(subplot_idx, "y");
      let n_components = entry
        .series
        .iter()
        .map(|s| s.component_idx)
        .max()
        .map_or(0, |m| m + 1);
      let mut comp_max = vec![0.0f64; n_components];
      for series in &entry.series {
        let local_max = series
          .values
          .iter()
          .fold(0.0f64, |acc, &v| acc.max(v.abs()));
        comp_max[series.component_idx] = comp_max[series.component_idx].max(local_max);
      }
      let global_max = comp_max.iter().copied().fold(0.0f64, f64::max);
      let min_nonzero = comp_max
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .fold(f64::INFINITY, f64::min);
      let mut comp_scale = vec![1.0f64; n_components];
      if n_components > 1 && min_nonzero.is_finite() && global_max / min_nonzero > 20.0 {
        for (i, &m) in comp_max.iter().enumerate() {
          if m > 0.0 {
            comp_scale[i] = global_max / m;
          }
        }
      }

      let t: Vec<f64> = (0..entry.n_points)
        .map(|i| i as f64 / (entry.n_points - 1).max(1) as f64)
        .collect();

      for series in &entry.series {
        let scale = comp_scale[series.component_idx];
        let y_plot = if scale > 1.01 {
          series
            .values
            .iter()
            .map(|v| v * scale)
            .collect::<Vec<f64>>()
        } else {
          series.values.clone()
        };
        let hover_text = series
          .values
          .iter()
          .zip(y_plot.iter())
          .map(|(raw, plotted)| {
            if scale > 1.01 {
              format!(
                "value (raw): {:.6}<br>value (plotted): {:.6}<br>scale: x{:.3}",
                raw, plotted, scale
              )
            } else {
              format!("value: {:.6}", raw)
            }
          })
          .collect::<Vec<String>>();
        let trace_name = if scale > 1.01 {
          format!("{} (plot x{:.2})", series.label, scale)
        } else {
          series.label.clone()
        };
        let dash = match series.component_idx % 4 {
          0 => DashType::Solid,
          1 => DashType::Dash,
          2 => DashType::Dot,
          _ => DashType::DashDot,
        };
        let trace = Scatter::new(t.clone(), y_plot)
          .mode(Mode::Lines)
          .line(Line::new().width(self.line_width).dash(dash))
          .name(trace_name.as_str())
          .hover_text_array(hover_text)
          .hover_template("%{hovertext}<extra></extra>")
          .show_legend(self.show_legend)
          .x_axis(xa.as_str())
          .y_axis(ya.as_str());
        plot.add_trace(trace);
      }
    }

    plot
  }

  pub fn show(self) {
    self.plot().show();
  }
}

#[cfg(test)]
mod tests {
  #[cfg(feature = "cuda")]
  use either::Either;
  use ndarray::Array1;
  use plotly::Surface;
  use rand_distr::Exp;
  use rand_distr::Normal;

  use super::*;
  use crate::stochastic::autoregressive::agrach::AGARCH;
  use crate::stochastic::autoregressive::ar::ARp;
  use crate::stochastic::autoregressive::arch::ARCH;
  use crate::stochastic::autoregressive::arima::ARIMA;
  use crate::stochastic::autoregressive::egarch::EGARCH;
  use crate::stochastic::autoregressive::garch::GARCH;
  use crate::stochastic::autoregressive::ma::MAq;
  use crate::stochastic::autoregressive::sarima::SARIMA;
  use crate::stochastic::autoregressive::tgarch::TGARCH;
  use crate::stochastic::diffusion::cev::CEV;
  use crate::stochastic::diffusion::cfou::CFOU;
  use crate::stochastic::diffusion::cir::CIR as DiffCIR;
  use crate::stochastic::diffusion::fcir::FCIR;
  use crate::stochastic::diffusion::feller::FellerLogistic;
  use crate::stochastic::diffusion::fgbm::FGBM;
  use crate::stochastic::diffusion::fjacobi::FJacobi;
  use crate::stochastic::diffusion::fou::FOU;
  use crate::stochastic::diffusion::fouque::FouqueOU2D;
  use crate::stochastic::diffusion::gbm::GBM;
  use crate::stochastic::diffusion::gbm_ih::GBMIH;
  use crate::stochastic::diffusion::gompertz::Gompertz;
  use crate::stochastic::diffusion::jacobi::Jacobi;
  use crate::stochastic::diffusion::kimura::Kimura;
  use crate::stochastic::diffusion::ou::OU;
  use crate::stochastic::diffusion::quadratic::Quadratic;
  use crate::stochastic::diffusion::verhulst::Verhulst;
  use crate::stochastic::interest::adg::ADG;
  use crate::stochastic::interest::bgm::BGM;
  use crate::stochastic::interest::cir::CIR as RateCIR;
  use crate::stochastic::interest::cir_2f::CIR2F;
  use crate::stochastic::interest::duffie_kan::DuffieKan;
  use crate::stochastic::interest::fvasicek::FVasicek;
  use crate::stochastic::interest::hjm::HJM;
  use crate::stochastic::interest::ho_lee::HoLee;
  use crate::stochastic::interest::hull_white::HullWhite;
  use crate::stochastic::interest::hull_white_2f::HullWhite2F;
  use crate::stochastic::interest::mod_duffie_kan::DuffieKanJumpExp;
  use crate::stochastic::interest::vasicek::Vasicek;
  use crate::stochastic::interest::wu_zhang::WuZhangD;
  use crate::stochastic::isonormal::ISONormal;
  use crate::stochastic::isonormal::fbm_custom_inc_cov;
  use crate::stochastic::jump::bates::Bates1996;
  use crate::stochastic::jump::cgmy::CGMY;
  use crate::stochastic::jump::cts::CTS;
  use crate::stochastic::jump::ig::IG;
  use crate::stochastic::jump::jump_fou::JumpFOU;
  use crate::stochastic::jump::jump_fou_custom::JumpFOUCustom;
  use crate::stochastic::jump::kobol::KoBoL;
  use crate::stochastic::jump::kou::KOU;
  use crate::stochastic::jump::levy_diffusion::LevyDiffusion;
  use crate::stochastic::jump::merton::Merton;
  use crate::stochastic::jump::nig::NIG;
  use crate::stochastic::jump::rdts::RDTS;
  use crate::stochastic::jump::vg::VG;
  use crate::stochastic::noise::cfgns::CFGNS;
  use crate::stochastic::noise::cgns::CGNS;
  use crate::stochastic::noise::fgn::FGN;
  use crate::stochastic::noise::gn::Gn;
  use crate::stochastic::noise::wn::Wn;
  use crate::stochastic::process::bm::BM;
  use crate::stochastic::process::cbms::CBMS;
  use crate::stochastic::process::ccustom::CompoundCustom;
  use crate::stochastic::process::cfbms::CFBMS;
  use crate::stochastic::process::cpoisson::CompoundPoisson;
  use crate::stochastic::process::customjt::CustomJt;
  use crate::stochastic::process::fbm::FBM;
  use crate::stochastic::process::lfsm::LFSM;
  use crate::stochastic::process::poisson::Poisson;
  use crate::stochastic::process::subordinator::AlphaStableSubordinator;
  use crate::stochastic::process::subordinator::CTRW;
  use crate::stochastic::process::subordinator::CtrwJumpLaw;
  use crate::stochastic::process::subordinator::CtrwWaitingLaw;
  use crate::stochastic::process::subordinator::GammaSubordinator;
  use crate::stochastic::process::subordinator::IGSubordinator;
  use crate::stochastic::process::subordinator::InverseAlphaStableSubordinator;
  use crate::stochastic::process::subordinator::PoissonSubordinator;
  use crate::stochastic::process::subordinator::TemperedStableSubordinator;
  use crate::stochastic::sheet::fbs::FBS;
  use crate::stochastic::volatility::HestonPow;
  use crate::stochastic::volatility::bergomi::Bergomi;
  use crate::stochastic::volatility::fheston::RoughHeston;
  use crate::stochastic::volatility::heston::Heston;
  use crate::stochastic::volatility::rbergomi::RoughBergomi;
  use crate::stochastic::volatility::sabr::SABR;
  use crate::stochastic::volatility::svcgmy::SVCGMY;

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

    let mut isonormal_fbm = ISONormal::new(
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
      &ARIMA::new(
        Array1::from_vec(vec![0.4]),
        Array1::from_vec(vec![0.3]),
        1,
        0.1,
        n,
      ),
      "Autoreg: ARIMA(1,1,1)",
      traj,
    );
    grid = grid.register(
      &SARIMA::new(
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
      "Autoreg: SARIMA",
      traj,
    );
    grid = grid.register(
      &ARCH::new(0.05, Array1::from_vec(vec![0.2, 0.1]), n),
      "Autoreg: ARCH",
      traj,
    );
    grid = grid.register(
      &GARCH::new(
        0.03,
        Array1::from_vec(vec![0.12]),
        Array1::from_vec(vec![0.8]),
        n,
      ),
      "Autoreg: GARCH",
      traj,
    );
    grid = grid.register(
      &TGARCH::new(
        0.03,
        Array1::from_vec(vec![0.08]),
        Array1::from_vec(vec![0.05]),
        Array1::from_vec(vec![0.85]),
        n,
      ),
      "Autoreg: TGARCH",
      traj,
    );
    grid = grid.register(
      &EGARCH::new(
        -0.1,
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![-0.05]),
        Array1::from_vec(vec![0.9]),
        n,
      ),
      "Autoreg: EGARCH",
      traj,
    );
    grid = grid.register(
      &AGARCH::new(
        0.03,
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.04]),
        Array1::from_vec(vec![0.84]),
        n,
      ),
      "Autoreg: AGARCH",
      traj,
    );

    grid = grid.register(&Wn::new(n, Some(0.0), Some(1.0)), "Noise: White", traj);
    grid = grid.register(&Gn::new(n, Some(1.0)), "Noise: Gaussian", traj);
    grid = grid.register(&FGN::new(0.7, n, Some(1.0)), "Noise: FGN", traj);
    grid = grid.register(&CGNS::new(-0.4, n, Some(1.0)), "Noise: CGNS", traj);
    grid = grid.register(&CFGNS::new(0.7, -0.3, n, Some(1.0)), "Noise: CFGNS", traj);

    grid = grid.register(&BM::new(n, Some(1.0)), "Process: BM", traj);
    grid = grid.register(&FBM::new(0.7, n, Some(1.0)), "Process: FBM", traj);
    #[cfg(feature = "cuda")]
    {
      let fbm_cuda = FBM::<f32>::new(0.7, n, Some(1.0));
      match fbm_cuda.sample_cuda(traj) {
        Ok(Either::Left(path)) => {
          let trajectories = vec![path.iter().map(|&x| x as f64).collect()];
          grid = grid.register_paths(trajectories, "Process: FBM (CUDA)");
        }
        Ok(Either::Right(paths)) => {
          let trajectories = paths
            .outer_iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();
          grid = grid.register_paths(trajectories, "Process: FBM (CUDA)");
        }
        Err(err) => {
          eprintln!("Skipping Process: FBM (CUDA): {err}");
        }
      }
    }
    grid = grid.register_paths(isonormal_paths, "Process: fBM via ISONormal (H=0.7)");
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
    grid = grid.register(&CBMS::new(0.35, n, Some(1.0)), "Process: CBMS", traj);
    grid = grid.register(&CFBMS::new(0.7, 0.35, n, Some(1.0)), "Process: CFBMS", traj);
    grid = grid.register(
      &LFSM::new(1.7, 0.0, 0.8, 1.0, n, Some(0.0), Some(1.0)),
      "Process: LFSM",
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
      "Process: IG Subordinator",
      traj,
    );
    grid = grid.register(
      &TemperedStableSubordinator::new(0.7, 1.0, 2.0, 0.05, n, Some(0.0), Some(1.0)),
      "Process: Tempered Stable Subordinator",
      traj,
    );
    grid = grid.register(
      &CTRW::new(
        CtrwWaitingLaw::Exponential { rate: 2.0 },
        CtrwJumpLaw::Normal {
          mean: 0.0,
          std: 0.3,
        },
        n,
        Some(0.0),
        Some(1.0),
      ),
      "Process: CTRW",
      traj,
    );

    grid = grid.register(
      &OU::new(2.0, 0.0, 0.2, n, Some(0.0), Some(1.0)),
      "Diffusion: OU",
      traj,
    );
    grid = grid.register(
      &GBM::new(0.05, 0.2, n, Some(100.0), Some(1.0)),
      "Diffusion: GBM",
      traj,
    );
    grid = grid.register(
      &DiffCIR::new(2.5, 0.04, 0.2, n, Some(0.04), Some(1.0), Some(false)),
      "Diffusion: CIR",
      traj,
    );
    grid = grid.register(
      &CEV::new(0.04, 0.2, 0.8, n, Some(1.0), Some(1.0)),
      "Diffusion: CEV",
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
      &FCIR::new(0.7, 2.5, 0.04, 0.2, n, Some(0.04), Some(1.0), Some(false)),
      "Diffusion: FCIR",
      traj,
    );
    grid = grid.register(
      &FJacobi::new(0.7, 0.8, 1.4, 0.35, n, Some(0.3), Some(1.0)),
      "Diffusion: FJacobi",
      traj,
    );
    grid = grid.register(
      &FOU::new(0.7, 2.0, 0.0, 0.2, n, Some(0.0), Some(1.0)),
      "Diffusion: FOU",
      traj,
    );
    grid = grid.register(
      &CFOU::new(0.7, 1.8, 3.0, 0.4, n, Some(0.0), Some(0.0), Some(1.0)),
      "Diffusion: Complex fOU",
      traj,
    );
    grid = grid.register(
      &FGBM::new(0.7, 0.04, 0.2, n, Some(100.0), Some(1.0)),
      "Diffusion: FGBM",
      traj,
    );
    grid = grid.register(
      &GBMIH::new(0.04, 0.2, n, Some(100.0), Some(1.0), None),
      "Diffusion: GBMIH",
      traj,
    );
    grid = grid.register(
      &FouqueOU2D::new(1.5, 0.0, 0.3, 0.0, n, Some(0.0), Some(0.0), Some(1.0)),
      "Diffusion: Fouque OU 2D",
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
      "Interest: CIR (Alias)",
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
      &HJM::new(
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
      "Interest: HJM",
      traj,
    );
    grid = grid.register(
      &BGM::new(
        Array1::from_vec(vec![0.2, 0.15]),
        Array1::from_vec(vec![0.02, 0.025]),
        2,
        Some(1.0),
        n,
      ),
      "Interest: BGM",
      traj,
    );
    grid = grid.register(
      &ADG::new(
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
      "Interest: ADG",
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
      &CIR2F::new(
        RateCIR::new(2.5, 0.03, 0.12, n, Some(0.03), Some(1.0), Some(false)),
        RateCIR::new(2.0, 0.02, 0.1, n, Some(0.02), Some(1.0), Some(false)),
        f_phi_small as fn(f64) -> f64,
      ),
      "Interest: CIR 2F",
      traj,
    );

    grid = grid.register(
      &VG::new(0.0, 0.2, 0.15, n, Some(0.0), Some(1.0)),
      "Jump: VG",
      traj,
    );
    grid = grid.register(
      &NIG::new(0.0, 0.2, 0.3, n, Some(0.0), Some(1.0)),
      "Jump: NIG",
      traj,
    );
    grid = grid.register(&IG::new(1.0, n, Some(0.0), Some(1.0)), "Jump: IG", traj);
    grid = grid.register(
      &RDTS::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0)),
      "Jump: RDTS",
      traj,
    );
    grid = grid.register(
      &CTS::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0)),
      "Jump: CTS",
      traj,
    );
    grid = grid.register(
      &CGMY::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0)),
      "Jump: CGMY",
      traj,
    );
    grid = grid.register(
      &KoBoL::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0)),
      "Jump: KoBoL",
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
      &KOU::new(
        0.03,
        0.2,
        1.0,
        0.0,
        n,
        Some(0.0),
        Some(1.0),
        normal_cpoisson(1.0, n, 0.12),
      ),
      "Jump: KOU",
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
      &JumpFOU::new(
        0.7,
        2.0,
        0.03,
        0.2,
        n,
        Some(0.03),
        Some(1.0),
        normal_cpoisson(1.0, n, 0.08),
      ),
      "Jump: Jump-FOU",
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
      "Jump: Jump-FOU Custom",
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
      &SABR::new(0.4, 0.7, -0.3, n, Some(1.0), Some(0.3), Some(1.0)),
      "Volatility: SABR",
      traj,
    );
    grid = grid.register(
      &SVCGMY::new(
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
      "Volatility: SVCGMY",
      traj,
    );

    grid.show();

    let fbs_field = FBS::new(0.7, sheet_m, sheet_n, 2.0).sample();
    let z: Vec<Vec<f64>> = fbs_field.outer_iter().map(|row| row.to_vec()).collect();
    let x: Vec<f64> = (0..sheet_n)
      .map(|i| i as f64 / (sheet_n.saturating_sub(1).max(1) as f64))
      .collect();
    let y: Vec<f64> = (0..sheet_m)
      .map(|i| i as f64 / (sheet_m.saturating_sub(1).max(1) as f64))
      .collect();

    let mut sheet_plot = Plot::new();
    let surface = Surface::new(z).x(x).y(y).name("Sheet: FBS");
    sheet_plot.add_trace(surface);
    sheet_plot.set_layout(
      Layout::new()
        .title("Sheet: FBS (3D Surface)")
        .height(900)
        .width(1200),
    );
    sheet_plot.show();
  }

  #[cfg(feature = "cuda")]
  #[test]
  fn plot_fbm_cuda_seed_check() {
    let n = 512usize;
    let m = 8usize;
    let fbm_cuda = FBM::<f32>::new(0.7, n, Some(1.0));

    let sample_paths = |process: &FBM<f32>| -> Vec<Vec<f64>> {
      match process.sample_cuda(m).expect("CUDA FBM sampling failed") {
        Either::Left(path) => vec![path.iter().map(|&x| x as f64).collect()],
        Either::Right(paths) => paths
          .outer_iter()
          .map(|row| row.iter().map(|&x| x as f64).collect())
          .collect(),
      }
    };

    let batch_a = sample_paths(&fbm_cuda);
    let batch_b = sample_paths(&fbm_cuda);

    let all_identical = batch_a.iter().zip(batch_b.iter()).all(|(a, b)| {
      a.iter()
        .zip(b.iter())
        .all(|(x, y)| (*x - *y).abs() <= f64::EPSILON)
    });
    eprintln!("CUDA FBM seed check - consecutive batches identical: {all_identical}");

    let plot = GridPlotter::new()
      .title("CUDA FBM Seed Check")
      .cols(2)
      .line_width(1.2)
      .show_legend(false)
      .register_paths(batch_a, "FBM CUDA batch A")
      .register_paths(batch_b, "FBM CUDA batch B")
      .plot();

    plot.write_html("target/fbm_cuda_seed_check.html");
    plot.show();
  }
}
