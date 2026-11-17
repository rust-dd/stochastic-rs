//! The device generators against Random123's known answers and the host port.

use crate::euler::philox::{key, philox4x32_10};

/// Random123's three vectors, then counters and keys the port scatters over the whole word range,
/// then the words at the ends of the uniform maps.
fn cases() -> Vec<([u32; 4], [u32; 2])> {
  let mut cases = vec![
    ([0; 4], [0; 2]),
    ([u32::MAX; 4], [u32::MAX; 2]),
    (
      [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
      [0xa409_3822, 0x299f_31d0],
    ),
  ];
  for i in 0..58u32 {
    let w = philox4x32_10([i, 1, 2, 3], key(0x5eed));
    let k = philox4x32_10([i, 4, 5, 6], key(0x5eed));
    cases.push((w, [k[0], k[1]]));
  }
  for edge in [
    [u32::MAX, u32::MAX],
    [0, u32::MAX],
    [1 << 9, 0],
    [0x1ff, 0x1ff],
  ] {
    cases.push(([edge[0], edge[1], 7, 9], [3, 5]));
  }
  cases
}

const RANDOM123: [[u32; 4]; 3] = [
  [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
  [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
  [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
];

fn flat(cases: &[([u32; 4], [u32; 2])]) -> Vec<u32> {
  cases
    .iter()
    .flat_map(|(counter, key)| counter.iter().chain(key).copied())
    .collect()
}

#[cfg(metal_backend)]
#[test]
fn metal_philox_reproduces_random123_and_the_port() {
  use metal_oxide::{Argument, DynamicLaunchConfig, Launch, Operation, Pipeline};

  use crate::euler::philox::u01_f32;
  use crate::euler::tests::philox::{exp1_f32, normal_f32};

  let cases = cases();
  let n = cases.len();
  let context = crate::device::metal::context(0).expect("a Metal device");
  let input = context.device.buffer_from_slice(&flat(&cases)).unwrap();
  let mut words = context.device.buffer_zeroed::<u32>(4 * n).unwrap();
  let mut draws = context.device.buffer_zeroed::<f32>(3 * n).unwrap();
  let pipeline = Pipeline::new(&context.device, &context.module, "philox_kat").unwrap();
  let config = DynamicLaunchConfig::for_elements(n as u32, 64).unwrap();
  // SAFETY: the buffers hold `n` cases of six, four and three values; nothing else holds them.
  unsafe {
    Launch::new(
      Ok(&pipeline),
      config,
      (&input, &mut words, &mut draws, n as u32),
      |(input, words, draws, cases)| {
        Ok([
          Argument::read(input),
          Argument::write(words),
          Argument::write(draws),
          Argument::value(*cases)?,
        ])
      },
    )
    .wait()
  }
  .unwrap();
  let words = words.read().wait().to_vec();
  let draws = draws.read().wait().to_vec();
  for (i, (counter, key)) in cases.iter().enumerate() {
    let gpu = [
      words[4 * i],
      words[4 * i + 1],
      words[4 * i + 2],
      words[4 * i + 3],
    ];
    assert_eq!(gpu, philox4x32_10(*counter, *key), "case {i}");
    if i < 3 {
      assert_eq!(gpu, RANDOM123[i], "Random123 vector {i}");
    }
    assert_eq!(
      draws[3 * i].to_bits(),
      u01_f32(counter[0]).to_bits(),
      "case {i}"
    );
    let (e, host) = (draws[3 * i + 1], exp1_f32(counter[0], counter[1]));
    assert!(
      e >= 0.0 && (e - host).abs() <= 4e-6 * (1.0 + host),
      "case {i}: {e} against {host}"
    );
    let (z, host) = (draws[3 * i + 2], normal_f32(gpu));
    assert!(
      (z - host).abs() <= 2e-3 * (1.0 + host.abs()),
      "case {i}: {z} against {host}"
    );
  }
}

/// A Brownian path's increments are the port's normals, the fast math the only difference.
#[cfg(metal_backend)]
#[test]
fn metal_euler_increments_are_the_ports_normals() {
  use stochastic_rs_core::simd_rng::Deterministic;

  use crate::device::Metal;
  use crate::euler::EulerKernel;
  use crate::euler::philox::SITE_NOISE;
  use crate::euler::tests::philox::normal_f32;
  use crate::process::bm::Bm;

  let (first, m, seed) = (5usize, 16usize, 0x0123_4567_89ab_cdef_u64);
  let bm = Bm::<f32, _>::new(65, Some(1.0), Deterministic::new(3));
  let paths = Metal::default()
    .euler_kernel(&bm, first, m, seed)
    .expect("a Metal launch");
  let sqrt_dt = (1.0f32 / 64.0).sqrt();
  for path in 0..m {
    for step in 1..65 {
      let counter = [(first + path) as u32, step as u32, SITE_NOISE, 0];
      let want = sqrt_dt * normal_f32(philox4x32_10(counter, key(seed)));
      let got = paths[[path, step]] - paths[[path, step - 1]];
      assert!(
        (got - want).abs() <= 2e-3 * sqrt_dt * (1.0 + want.abs() / sqrt_dt) + 4e-6,
        "path {path}, step {step}: {got} against {want}"
      );
    }
  }
}

/// A case's step uniforms are those of the path and step its counter's first two words name.
#[cfg(feature = "cuda")]
const C_KAT: &str = r#"extern "C" __global__ void philox_kat(const unsigned int* input, unsigned int* out,
    REAL* draws, unsigned int cases) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cases) return;
    const unsigned int* in = input + 6u * c;
    unsigned int words[4];
    REAL uniforms[4];
    stoch_philox(in[0], in[1], in[2], in[3], in[4], in[5], words);
    for (unsigned int k = 0u; k < 4u; k++) { out[4u * c + k] = words[k]; }
    draws[7u * c + 4u] = stoch_uniform(in[0], in[1]);
    draws[7u * c + 5u] = stoch_exp1(in[0], in[1]);
    draws[7u * c + 6u] = stoch_normal(words[0], words[1], words[2], words[3]);
    stoch_step_uniforms(in[0], in[1], in[4], in[5], words, uniforms);
    for (unsigned int k = 0u; k < 4u; k++) { draws[7u * c + k] = uniforms[k]; }
}
"#;

/// The words and draws one precision's kernel gives for `cases`.
#[cfg(feature = "cuda")]
fn cuda_kat<R>(real: &'static str, cases: &[([u32; 4], [u32; 2])]) -> (Vec<u32>, Vec<R>)
where
  R: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy + Default,
{
  use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

  let lang = crate::euler::kernel::cuda_language(real);
  let source = format!(
    "{}{}",
    crate::euler::kernel::prelude(&lang),
    C_KAT.replace("REAL", real)
  );
  let context = CudaContext::new(0).expect("a CUDA device");
  let stream = context.default_stream();
  let ptx = cudarc::nvrtc::compile_ptx(source).expect("NVRTC compiles the generator");
  let function = context
    .load_module(ptx)
    .and_then(|module| module.load_function("philox_kat"))
    .expect("the known-answer kernel loads");
  let n = cases.len();
  let input = stream.clone_htod(&flat(cases)).unwrap();
  let mut words = stream.alloc_zeros::<u32>(4 * n).unwrap();
  let mut draws = stream.alloc_zeros::<R>(7 * n).unwrap();
  let count = n as u32;
  // SAFETY: the buffers hold `n` cases of six, four and seven values, as the kernel reads them.
  unsafe {
    stream
      .launch_builder(&function)
      .arg(&input)
      .arg(&mut words)
      .arg(&mut draws)
      .arg(&count)
      .launch(LaunchConfig::for_num_elems(count))
      .unwrap();
  }
  (
    stream.clone_dtoh(&words).unwrap(),
    stream.clone_dtoh(&draws).unwrap(),
  )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_philox_reproduces_random123_and_the_port() {
  use crate::euler::philox::{SITE_UNIFORMS, u01_f32, u01_f64};
  use crate::euler::tests::philox::{exp1_f32, normal_f32, normal_f64};

  let cases = cases();
  let (words32, draws32) = cuda_kat::<f32>("float", &cases);
  let (words64, draws64) = cuda_kat::<f64>("double", &cases);
  for (i, (counter, key)) in cases.iter().enumerate() {
    let port = philox4x32_10(*counter, *key);
    assert_eq!(&words32[4 * i..4 * i + 4], &port[..], "float case {i}");
    assert_eq!(&words64[4 * i..4 * i + 4], &port[..], "double case {i}");
    if i < 3 {
      assert_eq!(port, RANDOM123[i], "Random123 vector {i}");
    }
    let step = |index| philox4x32_10([counter[0], counter[1], SITE_UNIFORMS, index], *key);
    let (high, low) = (step(0), step(1));
    let d = &draws32[7 * i..7 * i + 7];
    assert_eq!(
      d[..4].iter().map(|u| u.to_bits()).collect::<Vec<_>>(),
      high.map(|w| u01_f32(w).to_bits()),
      "float case {i}: the step uniforms"
    );
    assert_eq!(
      d[4].to_bits(),
      u01_f32(counter[0]).to_bits(),
      "float case {i}"
    );
    let host = exp1_f32(counter[0], counter[1]);
    assert!(
      d[5] >= 0.0 && (d[5] - host).abs() <= 4e-6 * (1.0 + host),
      "float case {i}"
    );
    let host = normal_f32(port);
    assert!(
      (d[6] - host).abs() <= 2e-3 * (1.0 + host.abs()),
      "float case {i}"
    );
    let d = &draws64[7 * i..7 * i + 7];
    assert_eq!(
      d[..4].iter().map(|u| u.to_bits()).collect::<Vec<_>>(),
      [0, 1, 2, 3].map(|k| u01_f64(high[k], low[k]).to_bits()),
      "double case {i}: the step uniforms"
    );
    assert_eq!(
      d[4].to_bits(),
      u01_f64(counter[0], counter[1]).to_bits(),
      "double case {i}"
    );
    let host = -u01_f64(counter[0], counter[1]).ln();
    assert!(
      d[5] >= 0.0 && (d[5] - host).abs() <= 1e-12 * (1.0 + host),
      "double case {i}"
    );
    let host = normal_f64(port);
    assert!(
      (d[6] - host).abs() <= 1e-12 * (1.0 + host.abs()),
      "double case {i}"
    );
  }
}
