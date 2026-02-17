use ndarray::Array1;
use ndarray::Array2;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;

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

impl<T: FloatExt> Plottable<T> for [Array1<T>; 2] {
  fn n_components(&self) -> usize {
    2
  }

  fn component_name(&self, idx: usize) -> String {
    match idx {
      0 => "component 1".into(),
      _ => "component 2".into(),
    }
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
  trajectories: Vec<Vec<f64>>,
}

pub struct GridPlotter {
  entries: Vec<GridEntry>,
  cols: usize,
  line_width: f64,
  title: String,
}

impl GridPlotter {
  pub fn new() -> Self {
    Self {
      entries: Vec::new(),
      cols: 3,
      line_width: 1.0,
      title: String::new(),
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

  pub fn register<T, P>(mut self, process: &P, title: &str, n_traj: usize) -> Self
  where
    T: FloatExt,
    P: ProcessExt<T>,
    P::Output: Plottable<T>,
  {
    let samples: Vec<P::Output> = (0..n_traj).map(|_| process.sample()).collect();
    let n_comp = samples[0].n_components();
    let n_points = samples[0].len();

    for c in 0..n_comp {
      let comp_title = if n_comp > 1 {
        format!("{} - {}", title, samples[0].component_name(c))
      } else {
        title.into()
      };

      let trajectories = samples.iter().map(|s| s.component(c)).collect();

      self.entries.push(GridEntry {
        title: comp_title,
        n_points,
        trajectories,
      });
    }

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

    self.entries.push(GridEntry {
      title: title.into(),
      n_points,
      trajectories,
    });

    self
  }

  pub fn plot(self) -> Plot {
    let n = self.entries.len();
    let cols = self.cols;
    let rows = n.div_ceil(cols);

    let mut plot = Plot::new();
    plot.set_layout(
      Layout::new().title(self.title.as_str()).grid(
        LayoutGrid::new()
          .rows(rows)
          .columns(cols)
          .pattern(GridPattern::Independent),
      ),
    );

    for (idx, entry) in self.entries.iter().enumerate() {
      let subplot_idx = idx + 1;
      let xa = format!("x{}", subplot_idx);
      let ya = format!("y{}", subplot_idx);

      let t: Vec<f64> = (0..entry.n_points)
        .map(|i| i as f64 / (entry.n_points - 1).max(1) as f64)
        .collect();

      for (traj_idx, traj) in entry.trajectories.iter().enumerate() {
        let trace = Scatter::new(t.clone(), traj.clone())
          .mode(Mode::Lines)
          .line(Line::new().width(self.line_width))
          .name(if traj_idx == 0 {
            entry.title.clone()
          } else {
            String::new()
          })
          .show_legend(traj_idx == 0)
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
  use super::*;
  use crate::stochastic::diffusion::cir::CIR;
  use crate::stochastic::diffusion::gbm::GBM;
  use crate::stochastic::diffusion::ou::OU;
  use crate::stochastic::isonormal::fbm_custom_inc_cov;
  use crate::stochastic::isonormal::ISONormal;
  use crate::stochastic::process::fbm::FBM;
  use crate::stochastic::volatility::heston::Heston;
  use crate::stochastic::volatility::HestonPow;

  #[test]
  fn plot_grid() {
    let n = 1000;
    let traj = 5;
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

    GridPlotter::new()
      .title("Stochastic Processes Overview")
      .cols(3)
      .register(&FBM::new(0.7, n, Some(1.0f64)), "FBM (H=0.7)", traj)
      .register_paths(isonormal_paths, "fBM via ISONormal (H=0.7)")
      .register(&FBM::new(0.3, n, Some(1.0f64)), "FBM (H=0.3)", traj)
      .register(
        &GBM::new(0.05, 0.2, n, Some(100.0f64), Some(1.0)),
        "GBM",
        traj,
      )
      .register(
        &OU::new(5.0f64, 1.0, 0.3, n, Some(1.0), Some(1.0)),
        "Ornstein-Uhlenbeck",
        traj,
      )
      .register(
        &CIR::new(2.0f64, 0.05, 0.1, n, Some(0.03), Some(1.0), None),
        "CIR",
        traj,
      )
      .register(
        &Heston::new(
          Some(100.0f64),
          Some(0.04),
          2.0,
          0.04,
          0.3,
          -0.7,
          0.05,
          n,
          Some(1.0),
          HestonPow::Sqrt,
          None,
        ),
        "Heston",
        traj,
      )
      .show();
  }
}
