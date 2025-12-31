use ndarray::Array2;
use ndarray::Axis;
use rand::prelude::*;

pub fn train_test_split_for_array2(
  arrays: &[Array2<f64>],
  test_size: f64,
  random_state: Option<u64>,
) -> Vec<(Array2<f64>, Array2<f64>)> {
  assert!(!arrays.is_empty(), "One or more arrays must be provided.");

  let n_samples = arrays[0].shape()[0];
  let n_train = (n_samples as f64 * (1.0 - test_size)).round() as usize;

  for array in arrays {
    assert_eq!(
      array.shape()[0],
      n_samples,
      "All arrays must have the same number of samples."
    );
  }

  let mut indices: Vec<usize> = (0..n_samples).collect();
  let mut rng = match random_state {
    Some(seed) => StdRng::seed_from_u64(seed),
    None => StdRng::from_entropy(),
  };
  indices.shuffle(&mut rng);

  let train_indices = &indices[0..n_train];
  let test_indices = &indices[n_train..];

  arrays
    .iter()
    .map(|array| {
      let train_array = array.select(Axis(0), train_indices);
      let test_array = array.select(Axis(0), test_indices);
      (train_array, test_array)
    })
    .collect()
}
