//! Plot a distribution: Gamma-sampled histogram with the closed-form pdf
//! overlaid, showing the sampler and `DistributionExt` agree.
//!
//! Writes a Plotly HTML chart to `target/plot_distribution_gamma.html`. Run
//! with `cargo run --release --example plot_distribution`.
use plotly::Histogram;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::histogram::HistNorm;
use plotly::layout::Axis;
use stochastic_rs::distributions::gamma::SimdGamma;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::traits::DistributionExt;
use stochastic_rs::traits::DistributionSampler;

fn main() {
  let alpha = 2.0;
  let scale = 2.0;
  let dist = SimdGamma::<f64>::new(alpha, scale, &Deterministic::new(7));

  let samples = dist.sample_n(50_000);
  let mut plot = Plot::new();
  plot.add_trace(
    Histogram::new(samples.to_vec())
      .name("samples")
      .hist_norm(HistNorm::ProbabilityDensity),
  );

  let n_grid = 200;
  let x_max = dist.mean() + 5.0 * dist.variance().sqrt();
  let xs = (0..=n_grid)
    .map(|i| x_max * i as f64 / n_grid as f64)
    .collect::<Vec<f64>>();
  let ys = xs.iter().map(|&x| dist.pdf(x)).collect::<Vec<f64>>();
  plot.add_trace(
    Scatter::new(xs, ys)
      .name("pdf")
      .mode(Mode::Lines)
      .line(Line::new().width(2.0).color("#c23b27")),
  );

  let layout = Layout::new()
    .title(format!(
      "Gamma(α={alpha}, θ={scale}): samples vs closed-form pdf"
    ))
    .x_axis(Axis::new().title("x"))
    .y_axis(Axis::new().title("density"));
  plot.set_layout(layout);

  let out_path = "target/plot_distribution_gamma.html";
  plot.write_html(out_path);
  println!("Wrote {out_path}");
}
