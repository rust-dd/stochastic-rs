// docs: concepts/distribution-ext#default-behaviour-panic-never-zero
//! Backs the "default behaviour: panic, never zero" example on the
//! DistributionExt concept page. Reproduces the doc's helper verbatim
//! (the crate's own equivalent is private) and proves it panics with the
//! documented message instead of silently returning a number.

fn unimplemented_moment<T>() -> ! {
  let name = std::any::type_name::<T>();
  unimplemented!("moment not implemented for {name}")
}

#[test]
#[should_panic(expected = "moment not implemented for f64")]
fn unimplemented_moment_panics_with_type_name() {
  unimplemented_moment::<f64>();
}
