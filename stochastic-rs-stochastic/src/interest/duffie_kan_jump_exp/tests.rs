use super::*;

#[test]
fn sample_returns_two_paths() {
  let dkj = DuffieKanJumpExp::<f64>::new(
    0.5,
    0.04,
    0.5,
    -0.3,
    0.01,
    0.0,
    0.0,
    0.01,
    0.0,
    0.5,
    0.0,
    0.005,
    0.5,
    0.01,
    64,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    Unseeded,
  );
  let [r, x] = dkj.sample();
  assert_eq!(r.len(), 64);
  assert_eq!(x.len(), 64);
}
