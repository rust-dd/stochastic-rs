//! A Monte Carlo estimate holds one pair of moments per chunk of paths, never a payoff per path.

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::McBarrierPricer;
use stochastic_rs_quant::pricing::exotic::BarrierOption;

static LIVE: AtomicUsize = AtomicUsize::new(0);

static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    let ptr = unsafe { System.alloc(layout) };
    if !ptr.is_null() {
      let live = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
      PEAK.fetch_max(live, Ordering::SeqCst);
    }
    ptr
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    unsafe { System.dealloc(ptr, layout) };
    LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
  }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn the_barrier_holds_no_payoff_per_path() {
  let option = BarrierOption {
    strike: 100.0,
    barrier: 130.0,
    rebate: 0.0,
    barrier_type: BarrierType::UpAndOut,
    option_type: OptionType::Call,
  };
  let price = |n_paths| {
    McBarrierPricer::new(0.2, n_paths, 1)
      .with_seed(Deterministic::new(1))
      .price(&option, 100.0, 0.03, 0.0, 1.0)
  };
  // The first call starts rayon's pool, whose allocations are not the estimate's.
  price(1_024);
  let n = 1 << 18;
  let base = LIVE.load(Ordering::SeqCst);
  PEAK.store(base, Ordering::SeqCst);
  let est = price(n);
  let grown = PEAK.load(Ordering::SeqCst) - base;
  assert_eq!(est.n_samples, n);
  assert!(
    grown < n,
    "{grown} bytes at the peak for {n} paths, where a payoff per path takes {}",
    8 * n
  );
}
