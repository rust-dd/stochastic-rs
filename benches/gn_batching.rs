use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use stochastic_rs::stochastic::diffusion::fouque::FouqueOU2D;
use stochastic_rs::stochastic::interest::hjm::HJM;
use stochastic_rs::stochastic::interest::wu_zhang::WuZhangD;
use stochastic_rs::stochastic::noise::cgns::CGNS;
use stochastic_rs::stochastic::noise::gn::Gn;
use stochastic_rs::traits::ProcessExt;

fn bench_cgns(c: &mut Criterion) {
  let mut group = c.benchmark_group("GnBatching/CGNS");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));
  let n = 100_000usize;
  let rho = 0.6f64;

  group.bench_with_input(BenchmarkId::new("new", n), &n, |b, &n| {
    let proc = CGNS::<f64>::new(rho, n, Some(1.0));
    b.iter(|| black_box(proc.sample()));
  });

  group.bench_with_input(BenchmarkId::new("old_style", n), &n, |b, &n| {
    let gn = Gn::<f64>::new(n, Some(1.0));
    b.iter(|| {
      let gn1 = gn.sample();
      let z = gn.sample();
      let c = (1.0 - rho * rho).sqrt();
      let mut gn2 = Array1::<f64>::zeros(n);
      for i in 0..n {
        gn2[i] = rho * gn1[i] + c * z[i];
      }
      black_box(gn2[n - 1])
    });
  });

  group.finish();
}

fn bench_fouque(c: &mut Criterion) {
  let mut group = c.benchmark_group("GnBatching/FouqueOU2D");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));
  let n = 100_000usize;

  group.bench_with_input(BenchmarkId::new("new", n), &n, |b, &n| {
    let proc = FouqueOU2D::<f64>::new(1.0, 0.5, 0.1, 0.0, n, Some(0.0), Some(0.0), Some(1.0));
    b.iter(|| black_box(proc.sample()));
  });

  group.bench_with_input(BenchmarkId::new("old_style", n), &n, |b, &n| {
    let gn = Gn::<f64>::new(n - 1, Some(1.0));
    let dt = gn.dt();
    b.iter(|| {
      let gn_x = gn.sample();
      let gn_y = gn.sample();
      let mut x = Array1::<f64>::zeros(n);
      let mut y = Array1::<f64>::zeros(n);
      let eps = 0.1f64;
      let sqrt_eps_inv = 1.0 / eps.sqrt();
      let eps_inv = 1.0 / eps;
      for i in 1..n {
        x[i] = x[i - 1] + 1.0 * (0.5 - x[i - 1]) * dt + eps * gn_x[i - 1];
        y[i] = y[i - 1] + eps_inv * (0.0 - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
      }
      black_box(x[n - 1] + y[n - 1])
    });
  });

  group.finish();
}

fn a_fn(t: f64) -> f64 {
  0.02 + 0.01 * t
}
fn b_fn(t: f64) -> f64 {
  0.1 + 0.01 * t
}
fn p_fn(_t: f64, tm: f64) -> f64 {
  1.0 / (1.0 + tm)
}
fn q_fn(t: f64, _tm: f64) -> f64 {
  0.01 + 0.005 * t
}
fn v_fn(_t: f64, _tm: f64) -> f64 {
  0.2
}
fn alpha_fn(t: f64, _tm: f64) -> f64 {
  0.01 + 0.002 * t
}
fn sigma_fn(_t: f64, _tm: f64) -> f64 {
  0.15
}

fn bench_hjm(c: &mut Criterion) {
  let mut group = c.benchmark_group("GnBatching/HJM");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));
  let n = 50_000usize;
  let t_max = 1.0f64;

  group.bench_with_input(BenchmarkId::new("new", n), &n, |b, &n| {
    let proc = HJM::<f64>::new(
      a_fn as fn(f64) -> f64,
      b_fn as fn(f64) -> f64,
      p_fn as fn(f64, f64) -> f64,
      q_fn as fn(f64, f64) -> f64,
      v_fn as fn(f64, f64) -> f64,
      alpha_fn as fn(f64, f64) -> f64,
      sigma_fn as fn(f64, f64) -> f64,
      n,
      Some(0.02),
      Some(1.0),
      Some(0.02),
      Some(t_max),
    );
    b.iter(|| black_box(proc.sample()));
  });

  group.bench_with_input(BenchmarkId::new("old_style", n), &n, |b, &n| {
    let gn = Gn::<f64>::new(n - 1, Some(t_max));
    let dt = gn.dt();
    b.iter(|| {
      let mut r = Array1::<f64>::zeros(n);
      let mut p = Array1::<f64>::zeros(n);
      let mut f = Array1::<f64>::zeros(n);
      r[0] = 0.02;
      p[0] = 1.0;
      f[0] = 0.02;
      let gn1 = gn.sample();
      let gn2 = gn.sample();
      let gn3 = gn.sample();

      for i in 1..n {
        let t = i as f64 * dt;
        r[i] = r[i - 1] + a_fn(t) * dt + b_fn(t) * gn1[i - 1];
        p[i] = p[i - 1] + p_fn(t, t_max) * (q_fn(t, t_max) * dt + v_fn(t, t_max) * gn2[i - 1]);
        f[i] = f[i - 1] + alpha_fn(t, t_max) * dt + sigma_fn(t, t_max) * gn3[i - 1];
      }

      black_box(r[n - 1] + p[n - 1] + f[n - 1])
    });
  });

  group.finish();
}

fn bench_wuzhang(c: &mut Criterion) {
  let mut group = c.benchmark_group("GnBatching/WuZhangD");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));
  let n = 5_000usize;
  let xn = 8usize;

  let alpha = Array1::from_elem(xn, 0.1f64);
  let beta = Array1::from_elem(xn, 0.2f64);
  let nu = Array1::from_elem(xn, 0.3f64);
  let lambda = Array1::from_elem(xn, 0.1f64);
  let x0 = Array1::from_elem(xn, 1.0f64);
  let v0 = Array1::from_elem(xn, 0.04f64);

  group.bench_with_input(BenchmarkId::new("new", n), &n, |b, &_n| {
    let proc = WuZhangD::<f64>::new(
      alpha.clone(),
      beta.clone(),
      nu.clone(),
      lambda.clone(),
      x0.clone(),
      v0.clone(),
      xn,
      Some(1.0),
      n,
    );
    b.iter(|| black_box(proc.sample()));
  });

  group.bench_with_input(BenchmarkId::new("old_style", n), &n, |b, &_n| {
    let gn = Gn::<f64>::new(n - 1, Some(1.0));
    let dt = gn.dt();
    b.iter(|| {
      let mut fv = ndarray::Array2::<f64>::zeros((2 * xn, n));
      for i in 0..xn {
        fv[(i, 0)] = x0[i];
        fv[(i + xn, 0)] = v0[i];
      }

      for i in 0..xn {
        let gn_f = gn.sample();
        let gn_v = gn.sample();
        for j in 1..n {
          let v_old = fv[(i + xn, j - 1)].max(0.0);
          let f_old = fv[(i, j - 1)].max(0.0);
          let dv = (alpha[i] - beta[i] * v_old) * dt + nu[i] * v_old.sqrt() * gn_v[j - 1];
          let v_new = (v_old + dv).max(0.0);
          fv[(i + xn, j)] = v_new;
          let df = f_old * lambda[i] * v_new.sqrt() * gn_f[j - 1];
          fv[(i, j)] = f_old + df;
        }
      }
      black_box(fv[(0, n - 1)] + fv[(xn, n - 1)])
    });
  });

  group.finish();
}

criterion_group!(benches, bench_cgns, bench_fouque, bench_hjm, bench_wuzhang);
criterion_main!(benches);
