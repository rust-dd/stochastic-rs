use plotly::{Plot, Scatter};
use stochastic_rs::diffusions::{
    cir,
    gbm::{self, gbm},
    jacobi, ou,
};

fn main() {
    // CIR
    // let mut plot = Plot::new();
    // let path = || cir::cir(1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FCIR
    // let mut plot = Plot::new();
    // let path = || cir::cir(1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // OU
    // let mut plot = Plot::new();
    // let path = || ou::ou(1.0, 1.2, 0.2, 1000, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FOU
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();
    //
    //    // OU
    // let mut plot = Plot::new();
    // let path = || ou::ou(1.0, 1.2, 0.2, 1000, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FOU
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // GBM
    let mut plot = Plot::new();
    let path = || gbm::gbm(1.0, 1.8, 5000, Some(1.0), None);

    plot.add_trace(Scatter::new((0..5000).collect::<Vec<usize>>(), path()));
    plot.show();

    // FGBM
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // Jacobi
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();

    // FJacobi
    // let mut plot = Plot::new();
    // let path = || ou::fou(0.7, 1.0, 1.2, 0.2, 1000, None, None, None);

    // plot.add_trace(Scatter::new((0..1000).collect::<Vec<usize>>(), path()));
    // plot.show();
}