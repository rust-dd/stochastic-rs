use ndarray::Array2;
use ndarray::array;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::DashType;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::layout::Margin;
use rand::rng;
use stochastic_rs_stochastic::sde::NoiseModel;
use stochastic_rs_stochastic::sde::Sde;
use stochastic_rs_stochastic::sde::SdeMethod;

#[test]
fn plot_sde_gbm_all_methods() {
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
