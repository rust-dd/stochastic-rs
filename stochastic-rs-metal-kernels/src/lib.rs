#![no_std]

mod fft;
mod random;
mod tile;

pub use fft::{fft_butterfly_extract, fft_radix4, fft_radix4_extract};
pub use tile::gen_tile_fft;

use metal_oxide_device::{block_dim, block_idx, thread_idx};

fn index() -> u32 {
  block_idx().x * block_dim().x + thread_idx().x
}
