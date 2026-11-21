//! Observer pattern for market data: quotes, fixing histories and handles notify registered
//! observers, which mark downstream pricers for lazy recomputation.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

/// Receiver notified when an observable changes. `update` must be cheap: flip a stale flag,
/// do not recompute inside it.
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

  /// Number of registered observers still alive. A registry lock poisoned by a panicking
  /// observer is recovered with a warning, not propagated: a feed outlives one bad observer.
  pub fn observer_count(&self) -> usize {
    let mut obs = match self.observers.lock() {
      Ok(g) => g,
      Err(p) => {
        log::warn!(
          "observable mutex poisoned (observer panicked); recovering and retaining live weaks"
        );
        p.into_inner()
      }
    };
    obs.retain(|w| w.strong_count() > 0);
    obs.len()
  }
}

impl Observable for ObservableBase {
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>) {
    let mut obs = match self.observers.lock() {
      Ok(g) => g,
      Err(p) => {
        log::warn!("observable mutex poisoned during register; recovering");
        p.into_inner()
      }
    };
    obs.retain(|w| w.strong_count() > 0);
    obs.push(observer);
  }

  fn notify_observers(&self) {
    let mut alive: Vec<Arc<dyn Observer + Send + Sync>> = Vec::new();
    {
      let mut obs = match self.observers.lock() {
        Ok(g) => g,
        Err(p) => {
          log::warn!("observable mutex poisoned during notify; recovering");
          p.into_inner()
        }
      };
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
  use std::cell::RefCell;
  use std::sync::atomic::AtomicUsize;
  use std::sync::atomic::Ordering;

  use log::Log;
  use log::Metadata;
  use log::Record;

  use super::*;

  struct Capture;

  impl Log for Capture {
    fn enabled(&self, _: &Metadata) -> bool {
      true
    }

    fn log(&self, record: &Record) {
      let message = record.args().to_string();
      RECORDS.with_borrow_mut(|records| records.push(message));
    }

    fn flush(&self) {}
  }

  // One process-wide logger, installed by whichever capture test runs first;
  // each test thread reads back only the records it logged itself.
  static CAPTURE: Capture = Capture;

  thread_local! {
    static RECORDS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
  }

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

  #[test]
  fn poisoned_registry_recovers_and_warns_through_log() {
    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Warn);
    let obs_base = ObservableBase::new();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _guard = obs_base.observers.lock().unwrap();
      panic!("poison the registry");
    }));
    assert!(panicked.is_err());
    assert_eq!(obs_base.observer_count(), 0);
    let records = RECORDS.take();
    assert!(
      records
        .iter()
        .any(|m| m.contains("observable mutex poisoned")),
      "{records:?}"
    );
  }
}
