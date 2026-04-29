//! Stale-tracking cache for values derived from observable market data.
//!
//! [`MarketObserver`] watches one or more [`Observable`]s and flips a stale
//! flag the moment any of them notify. [`Cached<T>`] pairs that flag with a
//! computed value plus a recompute closure, giving calibration results,
//! engine outputs, or any "derived from market" datum a uniform invalidation
//! API.
//!
//! ```ignore
//! use std::sync::Arc;
//! use stochastic_rs_quant::market::{SimpleQuote, Cached, Observable};
//!
//! let spot = Arc::new(SimpleQuote::new(100.0));
//! let s2 = Arc::clone(&spot) as Arc<dyn Observable>;
//! let cache = Cached::new(vec![s2], move || spot.value() * spot.value());
//! assert_eq!(cache.get(), 10_000.0);
//! ```

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use super::observable::Observable;
use super::observable::Observer;

/// Watches one or more [`Observable`]s and exposes a single boolean
/// "anything I'm watching has changed" flag.
///
/// `MarketObserver` itself implements [`Observer`] — register it as
/// observer of any number of quotes, handles, or composite observables and
/// it will set its stale flag the first time any of them fires `update`.
/// Reading the flag through [`is_stale`](Self::is_stale) consumes it
/// (clears back to `false`) so callers can implement edge-triggered cache
/// invalidation.
pub struct MarketObserver {
  stale: AtomicBool,
}

impl std::fmt::Debug for MarketObserver {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("MarketObserver")
      .field("stale", &self.stale.load(Ordering::Acquire))
      .finish()
  }
}

impl MarketObserver {
  /// Build a fresh observer (initially not stale) and subscribe it to every
  /// observable in `sources`. Returns an `Arc` so the observer can also be
  /// stored in caller-side state (the observable list keeps a `Weak`).
  pub fn watching(sources: Vec<Arc<dyn Observable>>) -> Arc<Self> {
    let observer = Arc::new(Self {
      stale: AtomicBool::new(false),
    });
    let weak: Weak<dyn Observer + Send + Sync> =
      Arc::downgrade(&observer) as Weak<dyn Observer + Send + Sync>;
    for src in &sources {
      src.register_observer(Weak::clone(&weak));
    }
    observer
  }

  /// Atomically read the stale flag and clear it back to `false`. Returns
  /// `true` exactly once for each contiguous run of `update` notifications.
  pub fn take_stale(&self) -> bool {
    self.stale.swap(false, Ordering::AcqRel)
  }

  /// Read the stale flag without clearing it.
  pub fn is_stale(&self) -> bool {
    self.stale.load(Ordering::Acquire)
  }

  /// Manually mark the observer stale (without an upstream notification).
  /// Useful when the caller already knows the cached value is invalid.
  pub fn invalidate(&self) {
    self.stale.store(true, Ordering::Release);
  }
}

impl Observer for MarketObserver {
  fn update(&self) {
    self.stale.store(true, Ordering::Release);
  }
}

/// Thin cache wrapping an arbitrary computed value `T` whose freshness is
/// tied to a [`MarketObserver`].
///
/// `T` is computed once eagerly on construction. Subsequent calls to
/// [`get`](Self::get) return the cached value if no upstream observable
/// notified, and recompute via the supplied closure on the first read after
/// any change. The cache is `Sync` and can be shared by clones of `Arc<Cached<T>>`.
pub struct Cached<T> {
  value: Mutex<T>,
  recompute: Box<dyn Fn() -> T + Send + Sync>,
  observer: Arc<MarketObserver>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Cached<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Cached")
      .field("value", &self.value.lock().ok().map(|g| format!("{:?}", *g)))
      .field("observer", &self.observer)
      .finish()
  }
}

impl<T: Clone + Send + Sync + 'static> Cached<T> {
  /// Build a cache. The closure is invoked once eagerly to populate the
  /// initial value; afterwards it is invoked on demand whenever any of the
  /// `sources` observables has notified.
  pub fn new(
    sources: Vec<Arc<dyn Observable>>,
    recompute: impl Fn() -> T + Send + Sync + 'static,
  ) -> Self {
    let observer = MarketObserver::watching(sources);
    let value = Mutex::new(recompute());
    Self {
      value,
      recompute: Box::new(recompute),
      observer,
    }
  }

  /// Current value, recomputing first if any upstream observable changed.
  pub fn get(&self) -> T {
    if self.observer.take_stale() {
      let fresh = (self.recompute)();
      *self.value.lock().expect("cached poisoned") = fresh;
    }
    self.value.lock().expect("cached poisoned").clone()
  }

  /// Force a recompute regardless of stale state.
  pub fn refresh(&self) -> T {
    self.observer.take_stale();
    let fresh = (self.recompute)();
    *self.value.lock().expect("cached poisoned") = fresh.clone();
    fresh
  }

  /// Borrow the underlying [`MarketObserver`] for direct stale checks.
  pub fn observer(&self) -> &Arc<MarketObserver> {
    &self.observer
  }
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::AtomicUsize;

  use super::*;
  use crate::market::SimpleQuote;
  use crate::market::quote::Quote;
  use crate::market::handle::RelinkableHandle;

  #[test]
  fn observer_set_by_underlying_change() {
    let q = Arc::new(SimpleQuote::<f64>::new(0.10));
    let observer = MarketObserver::watching(vec![Arc::clone(&q) as Arc<dyn Observable>]);
    assert!(!observer.is_stale());
    q.set_value(0.20);
    assert!(observer.is_stale());
    assert!(observer.take_stale());
    assert!(!observer.is_stale(), "take_stale should clear");
  }

  #[test]
  fn cached_value_recomputes_after_change() {
    let q = Arc::new(SimpleQuote::<f64>::new(2.0));
    let q_clone = Arc::clone(&q);
    let cache = Cached::new(
      vec![Arc::clone(&q) as Arc<dyn Observable>],
      move || q_clone.value().powi(2),
    );
    assert!((cache.get() - 4.0).abs() < 1e-12);
    q.set_value(3.0);
    assert!((cache.get() - 9.0).abs() < 1e-12);
  }

  #[test]
  fn cached_value_does_not_recompute_without_change() {
    let counter = Arc::new(AtomicUsize::new(0));
    let q = Arc::new(SimpleQuote::<f64>::new(1.0));
    let counter_clone = Arc::clone(&counter);
    let q_clone = Arc::clone(&q);
    let cache = Cached::new(
      vec![Arc::clone(&q) as Arc<dyn Observable>],
      move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        q_clone.value() + 1.0
      },
    );
    assert_eq!(counter.load(Ordering::SeqCst), 1, "eager init counted once");
    let _ = cache.get();
    let _ = cache.get();
    let _ = cache.get();
    assert_eq!(counter.load(Ordering::SeqCst), 1, "no recompute without change");
  }

  #[test]
  fn cached_responds_to_handle_relink() {
    let q1 = Arc::new(SimpleQuote::<f64>::new(1.0));
    let h = RelinkableHandle::new(Arc::clone(&q1));
    let read_handle = h.handle();
    let read_handle_for_observer: Arc<dyn Observable> =
      Arc::new(read_handle.clone()) as Arc<dyn Observable>;
    let read_handle_for_compute = read_handle.clone();
    let cache = Cached::new(
      vec![read_handle_for_observer],
      move || read_handle_for_compute.current().map(|q| q.value()).unwrap_or(f64::NAN),
    );
    assert!((cache.get() - 1.0).abs() < 1e-12);
    let q2 = Arc::new(SimpleQuote::<f64>::new(7.0));
    h.link_to(q2);
    assert!((cache.get() - 7.0).abs() < 1e-12);
  }

  #[test]
  fn manual_invalidate_forces_recompute() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let q = Arc::new(SimpleQuote::<f64>::new(0.0));
    let cache = Cached::new(
      vec![Arc::clone(&q) as Arc<dyn Observable>],
      move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        42.0
      },
    );
    let n0 = counter.load(Ordering::SeqCst);
    cache.observer().invalidate();
    let _ = cache.get();
    assert_eq!(counter.load(Ordering::SeqCst), n0 + 1);
  }
}
