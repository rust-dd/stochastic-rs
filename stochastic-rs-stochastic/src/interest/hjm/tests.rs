use super::*;

fn zero_1d(_: f64) -> f64 {
  0.0
}

fn zero_2d(_: f64, _: f64) -> f64 {
  0.0
}

fn one_2d(_: f64, _: f64) -> f64 {
  1.0
}

fn tmax_2d(_: f64, t_max: f64) -> f64 {
  t_max
}

#[test]
fn the_output_names_its_slots() {
  let HjmPaths { r, p, f, .. } = Hjm::new(
    zero_1d as fn(f64) -> f64,
    zero_1d as fn(f64) -> f64,
    one_2d as fn(f64, f64) -> f64,
    zero_2d as fn(f64, f64) -> f64,
    one_2d as fn(f64, f64) -> f64,
    zero_2d as fn(f64, f64) -> f64,
    one_2d as fn(f64, f64) -> f64,
    16,
    Some(0.03),
    Some(1.0),
    Some(0.04),
    None,
    Unseeded,
  )
  .sample();
  assert_eq!((r.len(), p.len(), f.len()), (16, 16, 16));
}

#[test]
fn default_t_max_is_one() {
  let model = Hjm::new(
    zero_1d as fn(f64) -> f64,
    zero_1d as fn(f64) -> f64,
    tmax_2d as fn(f64, f64) -> f64,
    one_2d as fn(f64, f64) -> f64,
    zero_2d as fn(f64, f64) -> f64,
    zero_2d as fn(f64, f64) -> f64,
    zero_2d as fn(f64, f64) -> f64,
    3,
    Some(0.0),
    Some(0.0),
    Some(0.0),
    None,
    Unseeded,
  );

  let p = model.sample().p;
  assert!((p[1] - 0.5).abs() < 1e-12);
  assert!((p[2] - 1.0).abs() < 1e-12);
}
