use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;

use super::{pool::with_f32_rows, *};

struct Chunk {
  first: usize,
  len: usize,
}

pub(crate) fn index_rows(rows: usize, streams: usize, n: usize) -> usize {
  crate::device::rows_within_index_limit(
    rows,
    streams.saturating_mul(2).saturating_mul(n),
    u32::MAX as usize,
  )
}

fn chunks(fgn_n: usize, out_size: usize, m: usize, device: &crate::device::Metal) -> Vec<Chunk> {
  let budget = device
    .batch_budget
    .min(crate::device::metal::working_set(device.ordinal));
  let rows = index_rows(
    crate::device::chunk_rows(budget, 2 * fgn_n + out_size, 4),
    1,
    fgn_n,
  );
  let mut first = 0;
  let mut out = Vec::new();
  while first < m {
    let len = rows.min(m - first);
    out.push(Chunk { first, len });
    first += len;
  }
  out
}

impl<S: SeedExt, B> Fgn<f32, S, B> {
  pub(crate) fn sample_metal_impl<S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Metal,
  ) -> Result<Vec<Array1<f32>>> {
    let out_size = self.padded_n - self.offset;
    let mut out = Vec::with_capacity(m);
    self.over_metal_chunks(m, seed_src, device, |chunk, rows| {
      out.extend(
        ndarray::ArrayView2::from_shape((chunk.len, out_size), rows)
          .expect("complete rows")
          .outer_iter()
          .map(|row| row.to_owned()),
      );
      Ok(())
    })?;
    Ok(out)
  }

  pub(crate) fn map_metal_impl<S2: SeedExt, R: Send>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Metal,
    f: impl Fn(ndarray::ArrayView1<f32>) -> R + Sync,
  ) -> Result<Vec<R>> {
    use rayon::prelude::*;
    let out_size = self.padded_n - self.offset;
    let mut out = Vec::with_capacity(m);
    self.over_metal_chunks(m, seed_src, device, |chunk, rows| {
      out.par_extend((0..chunk.len).into_par_iter().map(|row| {
        f(ndarray::ArrayView1::from(
          &rows[row * out_size..(row + 1) * out_size],
        ))
      }));
      Ok(())
    })?;
    Ok(out)
  }

  fn over_metal_chunks<S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Metal,
    mut consume: impl FnMut(&Chunk, &[f32]) -> Result<()>,
  ) -> Result<()> {
    let (n, offset) = (self.padded_n, self.offset);
    let out_size = n - offset;
    let eigs = self.sqrt_eigenvalues();
    let seed = seed_src.next_seed() as u32;
    for chunk in chunks(n, out_size, m, device) {
      let launch = Launch {
        sqrt_eigs: eigs,
        n,
        m: chunk.len,
        offset,
        hurst: self.hurst() as f64,
        t: self.t().unwrap_or(1.0) as f64,
        seed,
        first_cell: (chunk.first * 2 * n) as u64,
        ordinal: device.ordinal,
      };
      with_f32_rows(launch, |rows| consume(&chunk, rows))?;
    }
    Ok(())
  }
}

#[cfg(test)]
mod chunk_tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::device::Metal;
  use crate::traits::ProcessExt;

  #[test]
  fn chunks_are_bit_identical_to_one_launch() {
    let fgn = |device: Metal| {
      Fgn::<f32, _>::new(0.7, 512, Some(1.0), Deterministic::new(5)).with_backend(device)
    };
    let whole = fgn(Metal::default()).sample_par(9);
    let chunked = fgn(Metal::default().with_batch_budget((4 * 512 + 512) * 4 * 2)).sample_par(9);
    assert_eq!(whole, chunked);
    assert_ne!(whole[0], whole[1]);
  }

  #[test]
  fn fbm_honours_its_own_seed_on_the_device() {
    use crate::process::fbm::Fbm;
    let fbm =
      |seed: u64| Fbm::<f32, _>::new(0.7, 256, Some(1.0), Deterministic::new(seed)).on::<Metal>();
    assert_eq!(fbm(3).sample_par(3), fbm(3).sample_par(3));
    assert_ne!(fbm(3).sample_par(1), fbm(4).sample_par(1));
    assert_eq!(fbm(3).sample(), fbm(3).sample());
  }

  #[test]
  fn an_empty_fractional_batch_is_not_a_device_failure() {
    use crate::process::fbm::Fbm;
    let fbm = Fbm::<f32, _>::new(0.7, 256, Some(1.0), Deterministic::new(3)).on::<Metal>();
    let paths = fbm.try_sample_par(0).expect("no paths is not a failure");
    assert!(paths.is_empty());
  }
}
