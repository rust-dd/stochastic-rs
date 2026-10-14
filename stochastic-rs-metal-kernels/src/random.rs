use metal_oxide_device::math;

fn mix(mut value: u64) -> u64 {
  value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
  value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
  value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
  value ^ (value >> 31)
}

fn uniform(word: u32) -> f32 {
  ((word >> 8) as f32 + 0.5) / 16777216.0
}

pub(crate) fn normal_pair(cell: u64, seed: u32) -> (f32, f32) {
  let key = cell ^ (seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
  let a = mix(key);
  let b = mix(key ^ 0xd1b5_4a32_d192_ed03);
  let real = math::sqrt(-2.0 * math::ln(uniform(a as u32) + 1e-10))
    * math::cos(core::f32::consts::TAU * uniform((a >> 32) as u32));
  let imag = math::sqrt(-2.0 * math::ln(uniform(b as u32) + 1e-10))
    * math::cos(core::f32::consts::TAU * uniform((b >> 32) as u32));
  (real, imag)
}
