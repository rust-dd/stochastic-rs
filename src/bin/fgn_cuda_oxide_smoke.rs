#[cfg(feature = "cuda-oxide-experimental")]
fn main() {
  use either::Either;
  use stochastic_rs::stochastic::noise::fgn::Fgn;

  let fgn = Fgn::<f32>::new(0.72, 1024, Some(1.0));
  let paths = fgn
    .sample_cuda_oxide_with_module(16, "fgn_cuda_oxide_smoke")
    .expect("cuda-oxide FGN smoke run failed");

  let shape = match paths {
    Either::Left(path) => vec![1, path.len()],
    Either::Right(batch) => batch.shape().to_vec(),
  };
  println!("cuda-oxide FGN smoke shape={shape:?}");
}

#[cfg(not(feature = "cuda-oxide-experimental"))]
fn main() {
  eprintln!("Enable `--features cuda-oxide-experimental` to run this smoke binary.");
}
