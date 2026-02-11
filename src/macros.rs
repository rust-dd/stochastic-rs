#[macro_export]
macro_rules! plot_1d {
  ($data:expr, $name:expr) => {
    let mut plot = plotly::Plot::new();
    let trace = plotly::Scatter::new((0..$data.len()).collect::<Vec<_>>(), $data.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        plotly::common::Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Linear),
      )
      .name($name)
      .show_legend(true);
    plot.add_trace(trace);
    plot.show();
  };
}

#[macro_export]
macro_rules! plot_2d {
  ($data1:expr, $name1:expr, $data2:expr, $name2:expr) => {
    let mut plot = plotly::Plot::new();
    let trace1 = plotly::Scatter::new((0..$data1.len()).collect::<Vec<_>>(), $data1.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        plotly::common::Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Linear),
      )
      .name($name1)
      .show_legend(true);

    let trace2 = plotly::Scatter::new((0..$data2.len()).collect::<Vec<_>>(), $data2.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        plotly::common::Line::new()
          .color("blue")
          .shape(plotly::common::LineShape::Linear),
      )
      .name($name2)
      .show_legend(true);

    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.show();
  };
}

#[macro_export]
macro_rules! plot_3d {
  ($data1:expr, $name1:expr, $data2:expr, $name2:expr, $data3:expr, $name3:expr) => {
    let mut plot = plotly::Plot::new();
    let trace1 = plotly::Scatter::new((0..$data1.len()).collect::<Vec<_>>(), $data1.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        plotly::common::Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Linear),
      )
      .name($name1)
      .show_legend(true);

    let trace2 = plotly::Scatter::new((0..$data2.len()).collect::<Vec<_>>(), $data2.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        plotly::common::Line::new()
          .color("blue")
          .shape(plotly::common::LineShape::Linear),
      )
      .name($name2)
      .show_legend(true);

    let trace3 = plotly::Scatter::new((0..$data3.len()).collect::<Vec<_>>(), $data3.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        plotly::common::Line::new()
          .color("green")
          .shape(plotly::common::LineShape::Linear),
      )
      .name($name3)
      .show_legend(true);

    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);
    plot.show();
  };
}
