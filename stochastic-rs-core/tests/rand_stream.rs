//! Pins the seeded engine streams and the rand sampling built on them; a rand release that changes a sampling algorithm fails here first.

use rand::Rng;
use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::Uniform;
use rand::seq::SliceRandom;
use stochastic_rs_core::simd_rng::SimdRng;
#[cfg(feature = "unstable-dual-stream-rng")]
use stochastic_rs_core::simd_rng_dual::SimdRngDual;

#[test]
fn the_word_stream_is_pinned() {
  let mut rng = SimdRng::from_seed(42);
  let words = std::array::from_fn::<u64, 4, _>(|_| rng.next_u64());
  assert_eq!(
    words,
    [
      0xafb2_5193_1059_8017,
      0x472a_c5a0_074e_53e4,
      0x0eb5_a192_d05a_47a6,
      0x9d4a_77bc_b4fa_f888,
    ]
  );
  let mut rng = SimdRng::from_seed(42);
  let halves = std::array::from_fn::<u32, 4, _>(|_| rng.next_u32());
  assert_eq!(halves, [0x1059_8017, 0x074e_53e4, 0xd05a_47a6, 0xb4fa_f888]);
}

#[test]
fn fill_bytes_is_pinned() {
  let mut rng = SimdRng::from_seed(42);
  let mut bytes = [0u8; 20];
  rng.fill_bytes(&mut bytes);
  assert_eq!(
    bytes,
    [
      0x17, 0x80, 0x59, 0x10, 0x93, 0x51, 0xb2, 0xaf, 0xe4, 0x53, 0x4e, 0x07, 0xa0, 0xc5, 0x2a,
      0x47, 0xa6, 0x47, 0x5a, 0xd0,
    ]
  );
}

#[cfg(feature = "unstable-dual-stream-rng")]
#[test]
fn the_dual_word_stream_is_pinned() {
  let mut rng = SimdRngDual::from_seed(42);
  let words = std::array::from_fn::<u64, 4, _>(|_| rng.next_u64());
  assert_eq!(
    words,
    [
      0xafb2_5193_1059_8017,
      0xa736_e762_0a53_29ec,
      0x9ac9_49ea_708c_12d1,
      0xddf1_9eb4_9867_692e,
    ]
  );
  let mut rng = SimdRngDual::from_seed(42);
  let halves = std::array::from_fn::<u32, 4, _>(|_| rng.next_u32());
  assert_eq!(halves, [0x1059_8017, 0x0a53_29ec, 0x708c_12d1, 0x9867_692e]);
}

#[cfg(feature = "unstable-dual-stream-rng")]
#[test]
fn dual_fill_bytes_is_pinned() {
  let mut rng = SimdRngDual::from_seed(42);
  let mut bytes = [0u8; 20];
  rng.fill_bytes(&mut bytes);
  assert_eq!(
    bytes,
    [
      0x17, 0x80, 0x59, 0x10, 0x93, 0x51, 0xb2, 0xaf, 0xec, 0x29, 0x53, 0x0a, 0x62, 0xe7, 0x36,
      0xa7, 0xd1, 0x12, 0x8c, 0x70,
    ]
  );
}

#[test]
fn the_rand_extension_methods_are_pinned() {
  // The fourth word has bit 11 set, so it tells `random()`'s 53 bits from `random_range`'s 52.
  let mut rng = SimdRng::from_seed(42);
  assert_eq!(
    std::array::from_fn::<f64, 4, _>(|_| rng.random()),
    [
      0.6863146766703263,
      0.2779963985151934,
      0.05745897135089084,
      0.614417537280115
    ]
  );

  let mut rng = SimdRng::from_seed(42);
  assert_eq!(
    std::array::from_fn::<f64, 4, _>(|_| rng.random_range(0.0..1.0)),
    [
      0.6863146766703263,
      0.2779963985151934,
      0.05745897135089084,
      0.6144175372801148
    ]
  );

  let mut rng = SimdRng::from_seed(42);
  assert_eq!(
    std::array::from_fn::<u32, 4, _>(|_| rng.random_range(0..1000)),
    [63, 28, 813, 706]
  );

  let mut rng = SimdRng::from_seed(42);
  let mut order = (0..10).collect::<Vec<u32>>();
  order.shuffle(&mut rng);
  assert_eq!(order, [2, 4, 8, 5, 9, 7, 1, 6, 3, 0]);
}

#[test]
fn a_rand_distribution_samples_from_simd_rng() {
  let die = Uniform::new_inclusive(1u32, 6).unwrap();
  let mut rng = SimdRng::from_seed(5);
  let rolls = die.sample_iter(&mut rng).take(256).collect::<Vec<u32>>();
  assert!((1..=6).all(|face| rolls.contains(&face)));
  assert!(rolls.iter().all(|roll| (1..=6).contains(roll)));
}

#[test]
fn simd_rng_is_usable_as_a_dyn_rng() {
  let mut rng = SimdRng::from_seed(1);
  let dyn_rng: &mut dyn Rng = &mut rng;
  assert_ne!(dyn_rng.next_u64(), dyn_rng.next_u64());
}

#[test]
fn fill_bytes_writes_every_byte_at_every_length_and_buffer_offset() {
  for offset in 0..5 {
    for len in 0..=130 {
      let fill = |sentinel: u8| {
        let mut rng = SimdRng::from_seed(99);
        for _ in 0..offset {
          rng.next_u64();
        }
        let mut buf = vec![sentinel; len];
        rng.fill_bytes(&mut buf);
        buf
      };
      assert_eq!(fill(0x00), fill(0xff), "offset {offset}, length {len}");
    }
  }
}
