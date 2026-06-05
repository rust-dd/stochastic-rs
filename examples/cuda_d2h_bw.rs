//! Isolated device->host strategy probe: full cost of producing an owned
//! `Vec<f32>` on the host from a device buffer.
//! Run: cargo run --example cuda_d2h_bw --features cuda-native --release
use std::time::Instant;

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use rayon::prelude::*;

fn fresh_vec(n: usize) -> Vec<f32> {
  let mut v = Vec::<f32>::with_capacity(n);
  #[allow(clippy::uninit_vec)]
  unsafe {
    v.set_len(n)
  };
  v
}

fn main() {
  let n: usize = 256 * 1024 * 1024;
  let bytes = n * std::mem::size_of::<f32>();
  let iters = 10;
  let gb = bytes as f64 / 1e9;

  let ctx = CudaContext::new(0).unwrap();
  let stream = ctx.new_stream().unwrap();
  let d = stream.alloc_zeros::<f32>(n).unwrap();

  let mut warm = fresh_vec(n);
  for _ in 0..3 {
    stream.memcpy_dtoh(&d, &mut warm).unwrap();
    stream.synchronize().unwrap();
  }

  // Current production path: DMA straight into a fresh pageable Vec.
  let t = Instant::now();
  for _ in 0..iters {
    let v: Vec<f32> = stream.clone_dtoh(&d).unwrap();
    std::hint::black_box(&v);
  }
  let a = t.elapsed().as_secs_f64() / iters as f64;

  // Cached pinned staging buffer, then a serial copy into a fresh output Vec.
  let pin = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }.unwrap() as *mut f32;
  let staging = unsafe { std::slice::from_raw_parts_mut(pin, n) };
  stream.memcpy_dtoh(&d, staging).unwrap();
  stream.synchronize().unwrap();

  let t = Instant::now();
  for _ in 0..iters {
    stream.memcpy_dtoh(&d, staging).unwrap();
    stream.synchronize().unwrap();
    let mut out = fresh_vec(n);
    out.copy_from_slice(staging);
    std::hint::black_box(&out);
  }
  let b = t.elapsed().as_secs_f64() / iters as f64;

  // Cached pinned staging buffer, then a rayon-parallel copy (spreads the
  // first-touch page faults across cores) into a fresh output Vec.
  let t = Instant::now();
  for _ in 0..iters {
    stream.memcpy_dtoh(&d, staging).unwrap();
    stream.synchronize().unwrap();
    let mut out = fresh_vec(n);
    let chunk = 1 << 20;
    out
      .par_chunks_mut(chunk)
      .zip(staging.par_chunks(chunk))
      .for_each(|(o, s)| o.copy_from_slice(s));
    std::hint::black_box(&out);
  }
  let e = t.elapsed().as_secs_f64() / iters as f64;

  unsafe {
    let _ = cudarc::driver::result::free_host(pin as *mut std::ffi::c_void);
  }

  // Lower bound: reuse one registered, already-faulted Vec, DMA only.
  let mut reuse = fresh_vec(n);
  let rc = unsafe { sys::cuMemHostRegister_v2(reuse.as_mut_ptr() as *mut _, bytes, 0) };
  assert_eq!(rc, sys::CUresult::CUDA_SUCCESS);
  let t = Instant::now();
  for _ in 0..iters {
    stream.memcpy_dtoh(&d, reuse.as_mut_slice()).unwrap();
    stream.synchronize().unwrap();
  }
  let dd = t.elapsed().as_secs_f64() / iters as f64;
  unsafe {
    sys::cuMemHostUnregister(reuse.as_mut_ptr() as *mut _);
  }

  println!(
    "buffer = {gb:.3} GB, iters = {iters}, threads = {}\n",
    rayon::current_num_threads()
  );
  println!(
    "A) clone_dtoh -> fresh pageable Vec   : {:7.1} ms  ({:5.2} GB/s)",
    a * 1e3,
    gb / a
  );
  println!(
    "B) pinned staging + serial copy       : {:7.1} ms  ({:5.2} GB/s)",
    b * 1e3,
    gb / b
  );
  println!(
    "E) pinned staging + parallel copy     : {:7.1} ms  ({:5.2} GB/s)",
    e * 1e3,
    gb / e
  );
  println!(
    "D) reuse registered Vec (DMA only)    : {:7.1} ms  ({:5.2} GB/s)",
    dd * 1e3,
    gb / dd
  );
}
