use crate::euler::{normal, philox4x32_10};

/// The complex normal of `cell` under `site`: the real part from `which = 0`, the imaginary from 1.
pub(crate) fn normal_pair(cell: u64, site: u32, key: [u32; 2]) -> (f32, f32) {
  let (low, high) = (cell as u32, (cell >> 32) as u32);
  (
    normal(philox4x32_10([low, high, 0, site], key)),
    normal(philox4x32_10([low, high, 1, site], key)),
  )
}
