//! Scatter of Clayton `(u, v)` pairs, clustered toward the origin as an independence copula is not;
//! writes `target/plot_copula_clayton.html`; run with `cargo run --release --example plot_copula`.
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Mode;
use plotly::layout::Axis;
use stochastic_rs::copulas::bivariate::clayton::Clayton;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::traits::BivariateExt;

fn main() {
  let tau = 0.6; // Kendall's tau -> strong lower-tail dependence

  let clayton = Clayton::from_tau(tau).unwrap();

  let uv = clayton
    .sample(2_000, &Deterministic::new(42))
    .expect("Clayton sample");
  let u = uv.column(0).to_vec();
  let v = uv.column(1).to_vec();

  let mut plot = Plot::new();
  plot.add_trace(
    Scatter::new(u, v)
      .mode(Mode::Markers)
      .name(format!("Clayton(τ={tau})")),
  );

  let layout = Layout::new()
    .title(format!(
      "Clayton copula samples (τ={tau}) — lower-tail clustering"
    ))
    .x_axis(Axis::new().title("u").range(vec![0.0, 1.0]))
    .y_axis(Axis::new().title("v").range(vec![0.0, 1.0]));
  plot.set_layout(layout);

  let out_path = "target/plot_copula_clayton.html";
  plot.write_html(out_path);
  println!("Wrote {out_path}");
}
