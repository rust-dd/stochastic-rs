use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::DashType;
use plotly::common::Line;
use plotly::common::LineShape;
use plotly::common::Mode;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;
use rand::rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::beta::SimdBeta;
use crate::binomial::SimdBinomial;
use crate::cauchy::SimdCauchy;
use crate::chi_square::SimdChiSquared;
use crate::exp::SimdExp;
use crate::gamma::SimdGamma;
use crate::geometric::SimdGeometric;
use crate::hypergeometric::SimdHypergeometric;
use crate::inverse_gauss::SimdInverseGauss;
use crate::lognormal::SimdLogNormal;
use crate::normal::SimdNormal;
use crate::normal_inverse_gauss::SimdNormalInverseGauss;
use crate::pareto::SimdPareto;
use crate::poisson::SimdPoisson;
use crate::studentt::SimdStudentT;
use crate::uniform::SimdUniform;
use crate::weibull::SimdWeibull;

fn make_histogram(
  samples: &[f32],
  bins_count: usize,
  min_x: f32,
  max_x: f32,
) -> (Vec<f32>, Vec<f32>) {
  let bin_width = (max_x - min_x) / bins_count as f32;
  let mut bins = vec![0.0; bins_count + 1];
  for &val in samples {
    if val >= min_x && val < max_x {
      let idx = ((val - min_x) / bin_width) as usize;
      bins[idx] += 1.0;
    }
  }
  let total = samples.len() as f32;
  for b in bins.iter_mut() {
    *b /= total * bin_width;
  }
  let xs: Vec<f32> = (0..bins_count)
    .map(|i| min_x + (i as f32 + 0.5) * bin_width)
    .collect();
  (xs, bins)
}

fn make_discrete_pmf(samples: &[u32], max_range: u32) -> (Vec<f32>, Vec<f32>) {
  let mut bins = vec![0.0; (max_range + 1) as usize];
  for &val in samples {
    if val <= max_range {
      bins[val as usize] += 1.0;
    }
  }
  let n = samples.len() as f32;
  for b in bins.iter_mut() {
    *b /= n;
  }
  let xs: Vec<f32> = (0..=max_range).map(|i| i as f32).collect();
  (xs, bins)
}

fn subplot_axes(row: usize, col: usize) -> (String, String) {
  let index = (row - 1) * 4 + col;
  (format!("x{}", index), format!("y{}", index))
}

fn add_trace(
  plot: &mut Plot,
  xs: Vec<f32>,
  ys: Vec<f32>,
  name: &str,
  dashed: bool,
  xa: &str,
  ya: &str,
) {
  let mut line = Line::new().shape(LineShape::Linear);
  if dashed {
    line = line.dash(DashType::Dash);
  }
  plot.add_trace(
    Scatter::new(xs, ys)
      .name(name)
      .mode(Mode::Lines)
      .line(line)
      .x_axis(xa)
      .y_axis(ya),
  );
}

fn add_continuous_pair<F1, F2>(
  plot: &mut Plot,
  axes: (String, String),
  label: &str,
  range: (f32, f32),
  bins: usize,
  n: usize,
  mut simd_sample: F1,
  mut rand_sample: F2,
) where
  F1: FnMut() -> f32,
  F2: FnMut() -> f32,
{
  let (xa, ya) = axes;
  let samples: Vec<f32> = (0..n).map(|_| simd_sample()).collect();
  let (xs, ys) = make_histogram(&samples, bins, range.0, range.1);
  add_trace(
    plot,
    xs.clone(),
    ys,
    &format!("{label} - SIMD"),
    false,
    &xa,
    &ya,
  );

  let samples_rd: Vec<f32> = (0..n).map(|_| rand_sample()).collect();
  let (_, ys_rd) = make_histogram(&samples_rd, bins, range.0, range.1);
  add_trace(
    plot,
    xs,
    ys_rd,
    &format!("{label} - rand_distr"),
    true,
    &xa,
    &ya,
  );
}

fn add_discrete_pair<F1, F2>(
  plot: &mut Plot,
  axes: (String, String),
  label: &str,
  max_range: u32,
  n: usize,
  mut simd_sample: F1,
  mut rand_sample: F2,
) where
  F1: FnMut() -> u32,
  F2: FnMut() -> u32,
{
  let (xa, ya) = axes;
  let samples: Vec<u32> = (0..n).map(|_| simd_sample()).collect();
  let (xs, ys) = make_discrete_pmf(&samples, max_range);
  add_trace(
    plot,
    xs.clone(),
    ys,
    &format!("{label} - SIMD"),
    false,
    &xa,
    &ya,
  );

  let samples_rd: Vec<u32> = (0..n).map(|_| rand_sample()).collect();
  let (_, ys_rd) = make_discrete_pmf(&samples_rd, max_range);
  add_trace(
    plot,
    xs,
    ys_rd,
    &format!("{label} - rand_distr"),
    true,
    &xa,
    &ya,
  );
}

#[test]
#[ignore = "interactive: opens a browser via plot.show(); run with --ignored when generating combined plots"]
fn combined_all_distributions() {
  let mut plot = Plot::new();
  plot.set_layout(
    Layout::new().grid(
      LayoutGrid::new()
        .rows(5)
        .columns(4)
        .pattern(GridPattern::Independent),
    ),
  );

  let n = 50_000;
  let mut r1 = rand::rng();
  let mut r2 = rand::rng();

  let d_normal: SimdNormal<f32> = SimdNormal::new(0.0, 1.0, &Unseeded);
  let rd_normal = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(1, 1),
    "Normal(0,1)",
    (-4.0, 4.0),
    100,
    n,
    || d_normal.sample(&mut r1),
    || rd_normal.sample(&mut r2),
  );

  let d_cauchy = SimdCauchy::<f32>::new(0.0, 1.0, &Unseeded);
  let rd_cauchy = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(1, 2),
    "Cauchy(0,1)",
    (-10.0, 10.0),
    100,
    n,
    || d_cauchy.sample(&mut r1),
    || rd_cauchy.sample(&mut r2),
  );

  let d_lognormal = SimdLogNormal::<f32>::new(0.0, 1.0, &Unseeded);
  let rd_lognormal = rand_distr::LogNormal::<f32>::new(0.0, 1.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(1, 3),
    "LogNormal(0,1)",
    (0.0, 8.0),
    100,
    n,
    || d_lognormal.sample(&mut r1),
    || rd_lognormal.sample(&mut r2),
  );

  let d_pareto = SimdPareto::<f32>::new(1.0, 1.5, &Unseeded);
  let rd_pareto = rand_distr::Pareto::<f32>::new(1.0, 1.5).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(1, 4),
    "Pareto(1,1.5)",
    (0.0, 10.0),
    100,
    n,
    || d_pareto.sample(&mut r1),
    || rd_pareto.sample(&mut r2),
  );

  let d_weibull = SimdWeibull::<f32>::new(1.0, 1.5, &Unseeded);
  let rd_weibull = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(2, 1),
    "Weibull(1,1.5)",
    (0.0, 3.0),
    100,
    n,
    || d_weibull.sample(&mut r1),
    || rd_weibull.sample(&mut r2),
  );

  let d_gamma = SimdGamma::<f32>::new(2.0, 2.0, &Unseeded);
  let rd_gamma = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(2, 2),
    "Gamma(2,2)",
    (0.0, 20.0),
    120,
    n,
    || d_gamma.sample(&mut r1),
    || rd_gamma.sample(&mut r2),
  );

  let d_beta = SimdBeta::<f32>::new(2.0, 2.0, &Unseeded);
  let rd_beta = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(2, 3),
    "Beta(2,2)",
    (0.0, 1.0),
    100,
    n,
    || d_beta.sample(&mut r1),
    || rd_beta.sample(&mut r2),
  );

  let d_ig = SimdInverseGauss::<f32>::new(1.0, 2.0, &Unseeded);
  let rd_ig = rand_distr::InverseGaussian::<f32>::new(1.0, 2.0).unwrap();
  let mut r1_local = rng();
  add_continuous_pair(
    &mut plot,
    subplot_axes(2, 4),
    "InverseGauss(1,2)",
    (0.0, 3.0),
    100,
    n,
    || d_ig.sample(&mut r1_local),
    || rd_ig.sample(&mut r2),
  );

  let d_nig = SimdNormalInverseGauss::<f32>::new(2.0, 0.0, 1.0, 0.0, &Unseeded);
  let rd_nig = rand_distr::NormalInverseGaussian::<f32>::new(2.0, 0.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(3, 1),
    "Nig(2,0)",
    (-3.0, 3.0),
    100,
    n,
    || d_nig.sample(&mut r1),
    || rd_nig.sample(&mut r2),
  );

  let d_studentt = SimdStudentT::<f32>::new(5.0, &Unseeded);
  let rd_studentt = rand_distr::StudentT::<f32>::new(5.0).unwrap();
  let mut r1_t = rng();
  add_continuous_pair(
    &mut plot,
    subplot_axes(3, 2),
    "StudentT(5)",
    (-5.0, 5.0),
    120,
    n,
    || d_studentt.sample(&mut r1_t),
    || rd_studentt.sample(&mut r2),
  );

  let d_binomial = SimdBinomial::<u32>::new(10, 0.3, &Unseeded);
  let rd_binomial = rand_distr::Binomial::new(10, 0.3).unwrap();
  add_discrete_pair(
    &mut plot,
    subplot_axes(3, 3),
    "Binomial(10,0.3)",
    10,
    n,
    || d_binomial.sample(&mut r1),
    || rd_binomial.sample(&mut r2) as u32,
  );

  let d_geometric = SimdGeometric::<u32>::new(0.25, &Unseeded);
  let rd_geometric = rand_distr::Geometric::new(0.25).unwrap();
  let mut r1_g = rng();
  add_discrete_pair(
    &mut plot,
    subplot_axes(3, 4),
    "Geometric(0.25)",
    20,
    n,
    || d_geometric.sample(&mut r1_g),
    || rd_geometric.sample(&mut r2) as u32,
  );

  let d_hg = SimdHypergeometric::<u32>::new(20, 5, 6, &Unseeded);
  let rd_hg = rand_distr::Hypergeometric::new(20, 5, 6).unwrap();
  add_discrete_pair(
    &mut plot,
    subplot_axes(4, 1),
    "HyperGeo(20,5,6)",
    6,
    n,
    || d_hg.sample(&mut r1),
    || rd_hg.sample(&mut r2) as u32,
  );

  let d_poisson = SimdPoisson::<u32>::new(4.0, &Unseeded);
  let rd_poisson = rand_distr::Poisson::<f64>::new(4.0).unwrap();
  add_discrete_pair(
    &mut plot,
    subplot_axes(4, 2),
    "Poisson(4)",
    15,
    n,
    || d_poisson.sample(&mut r1),
    || rd_poisson.sample(&mut r2) as u32,
  );

  let d_uniform = SimdUniform::<f32>::new(0.0, 1.0, &Unseeded);
  let rd_uniform = rand_distr::Uniform::<f32>::new(0.0, 1.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(4, 3),
    "Uniform(0,1)",
    (0.0, 1.0),
    100,
    n,
    || d_uniform.sample(&mut r1),
    || rd_uniform.sample(&mut r2),
  );

  let d_exp = SimdExp::<f32>::new(1.5, &Unseeded);
  let rd_exp = rand_distr::Exp::<f32>::new(1.5).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(4, 4),
    "Exp(1.5)",
    (0.0, 4.0),
    100,
    n,
    || d_exp.sample(&mut r1),
    || rd_exp.sample(&mut r2),
  );

  let d_chisq = SimdChiSquared::<f32>::new(5.0, &Unseeded);
  let rd_chisq = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
  add_continuous_pair(
    &mut plot,
    subplot_axes(5, 1),
    "ChiSquared(5)",
    (0.0, 20.0),
    100,
    n,
    || d_chisq.sample(&mut r1),
    || rd_chisq.sample(&mut r2),
  );

  plot.show();
}
