//! # Non Central Chi Squared
//!
//! $$
//! X\sim\chi^2_\nu(\lambda),\quad f_X(x)=\tfrac12 e^{-(x+\lambda)/2}(x/\lambda)^{\nu/4-1/2}I_{\nu/2-1}(\sqrt{\lambda x})
//! $$
//!
use crate::chi_square::SimdChiSquared;
use crate::normal::SimdNormal;
use crate::traits::FloatExt;

pub fn sample<T: FloatExt>(df: T, lambda: T) -> T {
  // χ²_nc(df, ncp) = χ²(df - 1) + (Z + √ncp)² for df ≥ 1
  let normal = SimdNormal::<T, 64>::new(lambda.sqrt(), T::one(), &crate::simd_rng::Unseeded);
  let z = normal.sample_fast();
  let sq = z * z;

  let one = T::one();
  let rem = df - one;
  if rem > T::from_f64_fast(1e-10) {
    let chi_squared = SimdChiSquared::new(rem, &crate::simd_rng::Unseeded);
    chi_squared.sample_fast() + sq
  } else {
    sq
  }
}
