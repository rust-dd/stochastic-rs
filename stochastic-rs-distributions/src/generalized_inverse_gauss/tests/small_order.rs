use super::*;

/// mpmath at 60 and 120 digits below order 50 where a `K` overflows or the clock `√(χψ)` is subnormal, which a one-part sum
/// of logarithms or a subnormal argument read `8.5e-12` to `310 %` off.
#[test]
fn large_logarithms_and_subnormal_clocks_keep_their_digits() {
  let gig = SimdGig::<f64>::new;
  for (law, x, want) in [
    (
      gig(
        49.688_596_105_767_64,
        1.551_561_086_073_610_3e-300,
        1.084_957_471_743_115e-270,
      ),
      4.738_920_752_280_278e271,
      9.264_515_831_780_311e-276,
    ),
    (
      gig(
        47.851_468_583_913_48,
        5.748_85e-319,
        2.836_895_913_294_404e-301,
      ),
      2.129_110_221_096_858_5e302,
      1.632_938_889_028_722_8e-304,
    ),
  ] {
    let got = law.pdf(x).unwrap();
    assert!(close(got, want, 1e-12), "{got:e}");
  }
  for (law, t, want) in [
    (
      gig(34.670_520_298_591_32, 1.93e-322, 4.801e-320),
      2.4e-320,
      1.777_526_876_400_542_7e138,
    ),
    (
      gig(38.428_988_142_659_16, 1e-322, 2.297_256_346_984e-312),
      -1.023_113_149_472_811_4e-307,
      6.111_550_200_308_796e-191,
    ),
  ] {
    let got = law.moment_generating_function(t).unwrap();
    assert!(close(got, want, 1e-12), "{got:e}");
  }
}
