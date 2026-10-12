#![no_std]
#![allow(
  clippy::needless_range_loop,
  reason = "metal-oxide kernels iterate integer ranges, not slices"
)]

mod euler;
mod fft;
mod random;
mod sheet;
mod tile;

pub use euler::*;
pub use fft::{
  fft_columns_1, fft_columns_2, fft_columns_3, fft_columns_extract_1, fft_columns_extract_2,
  fft_columns_extract_3,
};
pub use sheet::{sheet_extract, sheet_rows, sheet_transpose};
pub use tile::{fft_tile, gen_fft_extract, gen_tile_fft};

use metal_oxide_device::{block_dim, block_idx, thread_idx};

fn index() -> u32 {
  block_idx().x * block_dim().x + thread_idx().x
}
