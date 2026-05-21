use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::DashType;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::common::Title;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;

pub fn write_surface_fit_plot_html<P: AsRef<Path>>(
  output_html: P,
  title: &str,
  strikes: &[f64],
  maturities: &[f64],
  actual_surface: &[f32],
  predicted_surface: &[f32],
) -> Result<()> {
  if strikes.is_empty() || maturities.is_empty() {
    bail!("strikes and maturities must be non-empty");
  }
  let expected = strikes.len() * maturities.len();
  if actual_surface.len() != expected || predicted_surface.len() != expected {
    bail!(
      "surface length mismatch: expected {}, got actual={} predicted={}",
      expected,
      actual_surface.len(),
      predicted_surface.len()
    );
  }

  let rows = maturities.len().div_ceil(2);
  let cols = 2usize;
  let mut plot = Plot::new();

  for (i, maturity) in maturities.iter().enumerate() {
    let start = i * strikes.len();
    let end = start + strikes.len();
    let actual = actual_surface[start..end]
      .iter()
      .map(|v| *v as f64)
      .collect::<Vec<f64>>();
    let pred = predicted_surface[start..end]
      .iter()
      .map(|v| *v as f64)
      .collect::<Vec<f64>>();

    let axis = i + 1;
    let tr_actual = Scatter::new(strikes.to_vec(), actual)
      .name(format!("Actual T={:.2}", maturity))
      .mode(Mode::Lines)
      .line(Line::new().color("#1f77b4"))
      .x_axis(format!("x{axis}"))
      .y_axis(format!("y{axis}"))
      .show_legend(i == 0);
    let tr_pred = Scatter::new(strikes.to_vec(), pred)
      .name(format!("Pred T={:.2}", maturity))
      .mode(Mode::Lines)
      .line(Line::new().color("#d62728").dash(DashType::Dash))
      .x_axis(format!("x{axis}"))
      .y_axis(format!("y{axis}"))
      .show_legend(i == 0);

    plot.add_trace(tr_actual);
    plot.add_trace(tr_pred);
  }

  let layout = Layout::new()
    .height(rows * 360)
    .width(cols * 520)
    .title(Title::from(title))
    .grid(
      LayoutGrid::new()
        .rows(rows)
        .columns(cols)
        .pattern(GridPattern::Independent),
    );
  plot.set_layout(layout);

  let output_html = output_html.as_ref();
  if let Some(parent) = output_html.parent() {
    fs::create_dir_all(parent)
      .with_context(|| format!("failed creating plot output directory {:?}", parent))?;
  }
  plot.write_html(output_html);
  Ok(())
}
