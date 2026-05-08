//! `GridPlotter` — builder for a multi-subplot Plotly grid.
//!
//! Drives all per-series scaling / hover-text / dash-pattern logic. Accepts
//! any `P: ProcessExt<T>` whose `Output` implements [`crate::Plottable<T>`].

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

use crate::plottable::Plottable;

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
