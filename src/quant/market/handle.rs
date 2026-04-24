//! Relinkable handles to observable market data.
//!
//! A handle is a stable shared pointer whose target can be swapped at
//! runtime. Pricers hold a [`Handle<T>`] once and automatically pick up
//! updates when the back-pointer is relinked — without touching the
//! dependency graph.
//!
//! Reference: Ballabio, "Implementing QuantLib", Leanpub (2020),
//! Chapter "Handles".

use std::sync::Arc;
use std::sync::RwLock;
use std::sync::Weak;

use super::observable::Observable;
use super::observable::ObservableBase;
use super::observable::Observer;

struct HandleInner<T: ?Sized> {
  link: RwLock<Option<Arc<T>>>,
  observable: ObservableBase,
}

/// Observable handle to a shared object.
///
/// Clones share the same link; relinking via any clone is visible
/// everywhere.
pub struct Handle<T: ?Sized> {
  inner: Arc<HandleInner<T>>,
}

impl<T: ?Sized> Clone for Handle<T> {
  fn clone(&self) -> Self {
    Self {
      inner: Arc::clone(&self.inner),
    }
  }
}

impl<T: ?Sized> std::fmt::Debug for Handle<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Handle")
      .field("is_linked", &self.is_linked())
      .finish()
  }
}

impl<T: ?Sized> Default for Handle<T> {
  fn default() -> Self {
    Self::empty()
  }
}

impl<T: ?Sized> Handle<T> {
  /// Handle initialised with a target.
  pub fn new(target: Arc<T>) -> Self {
    Self {
      inner: Arc::new(HandleInner {
        link: RwLock::new(Some(target)),
        observable: ObservableBase::new(),
      }),
    }
  }

  /// Handle with no target.
  pub fn empty() -> Self {
    Self {
      inner: Arc::new(HandleInner {
        link: RwLock::new(None),
        observable: ObservableBase::new(),
      }),
    }
  }

  /// True iff the handle currently points at something.
  pub fn is_linked(&self) -> bool {
    self
      .inner
      .link
      .read()
      .map(|g| g.is_some())
      .unwrap_or(false)
  }

  /// Borrow a strong pointer to the target (if linked).
  pub fn current(&self) -> Option<Arc<T>> {
    self.inner.link.read().expect("handle poisoned").clone()
  }
}

/// Handle whose target can be swapped at runtime.
///
/// `RelinkableHandle` is the write-facing view of a [`Handle`]; pricers
/// hold [`Handle`] and the caller owning the `RelinkableHandle` decides
/// when to swap the target.
pub struct RelinkableHandle<T: ?Sized> {
  handle: Handle<T>,
}

impl<T: ?Sized> Clone for RelinkableHandle<T> {
  fn clone(&self) -> Self {
    Self {
      handle: self.handle.clone(),
    }
  }
}

impl<T: ?Sized> Default for RelinkableHandle<T> {
  fn default() -> Self {
    Self::empty()
  }
}

impl<T: ?Sized> std::fmt::Debug for RelinkableHandle<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("RelinkableHandle")
      .field("handle", &self.handle)
      .finish()
  }
}

impl<T: ?Sized> RelinkableHandle<T> {
  /// Relinkable handle initialised with a target.
  pub fn new(target: Arc<T>) -> Self {
    Self {
      handle: Handle::new(target),
    }
  }

  /// Relinkable handle with no target.
  pub fn empty() -> Self {
    Self {
      handle: Handle::empty(),
    }
  }

  /// Borrow the observable read-side handle (cheap to clone).
  pub fn handle(&self) -> Handle<T> {
    self.handle.clone()
  }

  /// Replace the target and notify observers.
  pub fn link_to(&self, target: Arc<T>) {
    {
      let mut link = self.handle.inner.link.write().expect("handle poisoned");
      *link = Some(target);
    }
    self.handle.inner.observable.notify_observers();
  }

  /// Clear the target and notify observers.
  pub fn unlink(&self) {
    {
      let mut link = self.handle.inner.link.write().expect("handle poisoned");
      *link = None;
    }
    self.handle.inner.observable.notify_observers();
  }
}

impl<T: ?Sized + Send + Sync> Observable for Handle<T> {
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>) {
    self.inner.observable.register_observer(observer);
  }

  fn notify_observers(&self) {
    self.inner.observable.notify_observers();
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::atomic::AtomicUsize;
  use std::sync::atomic::Ordering;

  struct Counter(AtomicUsize);
  impl Observer for Counter {
    fn update(&self) {
      self.0.fetch_add(1, Ordering::SeqCst);
    }
  }

  #[test]
  fn relinking_notifies_read_handle() {
    let relinkable = RelinkableHandle::<i32>::new(Arc::new(1));
    let handle = relinkable.handle();
    let counter = Arc::new(Counter(AtomicUsize::new(0)));
    handle.register_observer(Arc::downgrade(&counter) as Weak<dyn Observer + Send + Sync>);

    assert_eq!(*handle.current().unwrap(), 1);
    relinkable.link_to(Arc::new(42));
    assert_eq!(*handle.current().unwrap(), 42);
    relinkable.unlink();
    assert!(!handle.is_linked());
    assert_eq!(counter.0.load(Ordering::SeqCst), 2);
  }
}
