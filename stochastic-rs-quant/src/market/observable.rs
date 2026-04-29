//! Observable / Observer pattern for reactive market data.
//!
//! Market quotes, fixing histories, and handles notify registered observers
//! when their underlying value changes. Observers drive lazy recomputation
//! in downstream pricers via a classic Observer / Observable pattern.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

/// A callback receiver notified when an observable mutates.
///
/// Implementations must be cheap to call — consumers typically flip an
/// "is stale" flag rather than recomputing inside `update`.
pub trait Observer: Send + Sync {
  /// Called when a watched observable changes.
  fn update(&self);
}

/// Something that can be observed for change notifications.
pub trait Observable: Send + Sync {
  /// Register an observer by weak handle. Dead weaks are pruned lazily.
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>);
  /// Notify every live observer.
  fn notify_observers(&self);
}

/// Shared observable state. Embed in a quote, handle, or curve-factory type.
#[derive(Default)]
pub struct ObservableBase {
  observers: Mutex<Vec<Weak<dyn Observer + Send + Sync>>>,
}

impl std::fmt::Debug for ObservableBase {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let n = self
      .observers
      .lock()
      .map(|obs| obs.len())
      .unwrap_or_default();
    f.debug_struct("ObservableBase")
      .field("observer_count", &n)
      .finish()
  }
}

impl Clone for ObservableBase {
  /// Clones start with an empty observer list — subscriptions are tied
  /// to the original observable.
  fn clone(&self) -> Self {
    Self::new()
  }
}

impl ObservableBase {
  /// Create an empty observable.
  pub fn new() -> Self {
    Self {
      observers: Mutex::new(Vec::new()),
    }
  }

  /// Number of registered observers still alive.
  pub fn observer_count(&self) -> usize {
    let mut obs = self.observers.lock().expect("observable poisoned");
    obs.retain(|w| w.strong_count() > 0);
    obs.len()
  }
}

impl Observable for ObservableBase {
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>) {
    let mut obs = self.observers.lock().expect("observable poisoned");
    obs.retain(|w| w.strong_count() > 0);
    obs.push(observer);
  }

  fn notify_observers(&self) {
    let mut alive: Vec<Arc<dyn Observer + Send + Sync>> = Vec::new();
    {
      let mut obs = self.observers.lock().expect("observable poisoned");
      obs.retain(|w| {
        if let Some(strong) = w.upgrade() {
          alive.push(strong);
          true
        } else {
          false
        }
      });
    }
    for obs in alive {
      obs.update();
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::AtomicUsize;
  use std::sync::atomic::Ordering;

  use super::*;

  struct Counter(AtomicUsize);

  impl Observer for Counter {
    fn update(&self) {
      self.0.fetch_add(1, Ordering::SeqCst);
    }
  }

  #[test]
  fn notifications_are_delivered() {
    let obs_base = ObservableBase::new();
    let counter = Arc::new(Counter(AtomicUsize::new(0)));
    obs_base.register_observer(Arc::downgrade(&counter) as Weak<dyn Observer + Send + Sync>);
    obs_base.notify_observers();
    obs_base.notify_observers();
    assert_eq!(counter.0.load(Ordering::SeqCst), 2);
  }

  #[test]
  fn dropped_observers_are_pruned() {
    let obs_base = ObservableBase::new();
    {
      let transient = Arc::new(Counter(AtomicUsize::new(0)));
      obs_base.register_observer(Arc::downgrade(&transient) as Weak<dyn Observer + Send + Sync>);
      assert_eq!(obs_base.observer_count(), 1);
    }
    assert_eq!(obs_base.observer_count(), 0);
  }
}
