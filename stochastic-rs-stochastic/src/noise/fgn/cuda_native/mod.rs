//! # cudarc Native CUDA
//!
//! NVIDIA-optimized Fgn sampling via cudarc (cuFFT + NVRTC).
//! Fused Philox RNG + eigenvalue scaling eliminates cuRAND dependency
//! and one GPU memory round-trip.
//!
mod convert;
mod kernels;
mod sampler;
mod state;

#[cfg(test)]
mod tests;
