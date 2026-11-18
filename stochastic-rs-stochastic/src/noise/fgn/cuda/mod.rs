//! # cudarc Native CUDA
//!
//! Fgn on cuFFT and NVRTC: a fused Philox draw and eigenvalue scale, no cuRAND, one copy fewer.
//!
mod convert;
mod kernels;
pub(crate) mod sampler;
mod state;

pub(crate) use state::PinnedHost;
pub(crate) use state::frees;
pub(crate) use state::release;

#[cfg(test)]
mod tests;
