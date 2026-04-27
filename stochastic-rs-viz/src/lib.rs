//! # stochastic-rs-viz
//!
//! Plotly-based visualization for stochastic processes and distributions.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

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
use stochastic_rs_distributions::traits::FloatExt;
use stochastic_rs_stochastic::traits::ProcessExt;

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

impl Default for GridPlotter {
  fn default() -> Self {
    Self::new()
  }
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

  /// Write the plot to a self-contained HTML file at `path`.
  ///
  /// PNG / SVG export is **not** built in: Plotly's static-image export
  /// requires the orca / Kaleido binary on PATH. Once that is installed,
  /// the produced [`Plot`] can be saved via Plotly's own
  /// `Plot::write_image` (added by `plot.write_image(path, format, w, h, scale)`).
  pub fn write_html<P: AsRef<std::path::Path>>(self, path: P) {
    self.plot().write_html(path);
  }
}

/// Convenience: plot a single 1D process sample as HTML.
///
/// ```ignore
/// use stochastic_rs_stochastic::process::bm::Bm;
/// use stochastic_rs_stochastic::traits::ProcessExt;
/// use stochastic_rs_viz::plot_process;
///
/// let bm = Bm::new(1000, Some(1.0));
/// plot_process(&bm.sample(), "bm.html");
/// ```
pub fn plot_process<T: FloatExt>(sample: &Array1<T>, path: &str) {
  let xs: Vec<f64> = (0..sample.len()).map(|i| i as f64).collect();
  let ys: Vec<f64> = sample.iter().map(|v| v.to_f64().unwrap()).collect();
  let mut plot = Plot::new();
  plot.add_trace(Scatter::new(xs, ys).mode(Mode::Lines).name("path"));
  plot.write_html(path);
}

/// Convenience: plot a histogram of distribution samples to an HTML file.
pub fn plot_distribution<T: FloatExt>(samples: &Array1<T>, path: &str, title: &str) {
  use plotly::Histogram;
  let mut plot = Plot::new();
  let xs: Vec<f64> = samples.iter().map(|v| v.to_f64().unwrap()).collect();
  plot.add_trace(Histogram::new(xs).name(title));
  plot.write_html(path);
}

/// Convenience: plot a 3D vol-surface (strikes × maturities × IV) to an HTML file.
pub fn plot_vol_surface(
  strikes: &[f64],
  maturities: &[f64],
  ivs: &Array2<f64>,
  path: &str,
) {
  use plotly::Surface;
  assert_eq!(
    ivs.dim(),
    (maturities.len(), strikes.len()),
    "ivs shape must be (N_T, N_K)"
  );
  let z: Vec<Vec<f64>> = (0..maturities.len())
    .map(|j| (0..strikes.len()).map(|i| ivs[[j, i]]).collect())
    .collect();
  let mut plot = Plot::new();
  plot.add_trace(Surface::new(z).x(strikes.to_vec()).y(maturities.to_vec()));
  plot.write_html(path);
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use plotly::Surface;
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
    let t0 = 0.0;
    let t1 = 1.0;
    let dt = 0.001;
    let n_paths = 5;
    let steps = (((t1 - t0) / dt) as f64).ceil() as usize;

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
}
