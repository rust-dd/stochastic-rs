//! Quote wrappers for reactive market data.
//!
//! A [`Quote`] exposes a numeric value and is observable. The primitive
//! [`SimpleQuote`] is a mutable value with change notifications; the
//! composite [`DerivedQuote`] / [`CompositeQuote`] types build new quotes
//! from existing ones and forward notifications.

use std::sync::Arc;
use std::sync::RwLock;
use std::sync::Weak;

use super::observable::Observable;
use super::observable::ObservableBase;
use super::observable::Observer;
use crate::traits::FloatExt;

/// Observable numeric market quote.
///
/// Typical implementations are [`SimpleQuote`], [`DerivedQuote`], and
/// [`CompositeQuote`]. Custom implementations can expose yields, vols,
/// or any scalar that must propagate changes through a pricing graph.
pub trait Quote<T: FloatExt>: Observable {
  /// Current value of the quote.
  fn value(&self) -> T;
  /// True if the quote has a usable value.
  fn is_valid(&self) -> bool {
    let v = self.value();
    v.is_finite()
  }
}

struct SimpleQuoteInner<T: FloatExt> {
  value: RwLock<Option<T>>,
  observable: ObservableBase,
}

/// Mutable scalar quote. Cheap to clone (shared `Arc` interior).
pub struct SimpleQuote<T: FloatExt> {
  inner: Arc<SimpleQuoteInner<T>>,
}

impl<T: FloatExt> std::fmt::Debug for SimpleQuote<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SimpleQuote")
      .field("value", &self.inner.value.read().ok().and_then(|v| *v))
      .finish()
  }
}

impl<T: FloatExt> Clone for SimpleQuote<T> {
  fn clone(&self) -> Self {
    Self {
      inner: Arc::clone(&self.inner),
    }
  }
}

impl<T: FloatExt> SimpleQuote<T> {
  /// Construct an initialised quote.
  pub fn new(value: T) -> Self {
    Self {
      inner: Arc::new(SimpleQuoteInner {
        value: RwLock::new(Some(value)),
        observable: ObservableBase::new(),
      }),
    }
  }

  /// Construct an empty (invalid) quote, to be set later.
  pub fn empty() -> Self {
    Self {
      inner: Arc::new(SimpleQuoteInner {
        value: RwLock::new(None),
        observable: ObservableBase::new(),
      }),
    }
  }

  /// Overwrite the value and notify observers if the value changed.
  ///
  /// Returns `true` if the stored value actually moved.
  pub fn set_value(&self, value: T) -> bool {
    let changed = {
      let mut v = self.inner.value.write().expect("quote poisoned");
      let changed = match *v {
        Some(existing) => existing != value,
        None => true,
      };
      *v = Some(value);
      changed
    };
    if changed {
      self.inner.observable.notify_observers();
    }
    changed
  }

  /// Clear the quote value and notify observers.
  pub fn reset(&self) {
    {
      let mut v = self.inner.value.write().expect("quote poisoned");
      *v = None;
    }
    self.inner.observable.notify_observers();
  }
}

impl<T: FloatExt> Observable for SimpleQuote<T> {
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>) {
    self.inner.observable.register_observer(observer);
  }

  fn notify_observers(&self) {
    self.inner.observable.notify_observers();
  }
}

impl<T: FloatExt> Quote<T> for SimpleQuote<T> {
  fn value(&self) -> T {
    match *self.inner.value.read().expect("quote poisoned") {
      Some(v) => v,
      None => T::nan(),
    }
  }

  fn is_valid(&self) -> bool {
    self
      .inner
      .value
      .read()
      .map(|v| v.map(|x| x.is_finite()).unwrap_or(false))
      .unwrap_or(false)
  }
}

/// Quote derived from another quote via a pure function of its value.
pub struct DerivedQuote<T: FloatExt> {
  base: Arc<dyn Quote<T>>,
  f: Arc<dyn Fn(T) -> T + Send + Sync>,
  observable: ObservableBase,
}

impl<T: FloatExt> std::fmt::Debug for DerivedQuote<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("DerivedQuote")
      .field("value", &self.value())
      .finish()
  }
}

impl<T: FloatExt> Clone for DerivedQuote<T> {
  fn clone(&self) -> Self {
    Self {
      base: Arc::clone(&self.base),
      f: Arc::clone(&self.f),
      observable: ObservableBase::new(),
    }
  }
}

impl<T: FloatExt> DerivedQuote<T> {
  /// Build a derived quote `f(base)`. The returned `Arc<Self>` also serves
  /// as the observer registered against `base`, so relinking the base quote
  /// propagates here without further wiring.
  pub fn new(base: Arc<dyn Quote<T>>, f: impl Fn(T) -> T + Send + Sync + 'static) -> Arc<Self> {
    let derived = Arc::new(Self {
      base,
      f: Arc::new(f),
      observable: ObservableBase::new(),
    });
    let weak: Weak<dyn Observer + Send + Sync> =
      Arc::downgrade(&derived) as Weak<dyn Observer + Send + Sync>;
    derived.base.register_observer(weak);
    derived
  }

  /// Convenience: `base + spread` with `spread` itself a quote.
  pub fn spread(base: Arc<dyn Quote<T>>, spread: Arc<dyn Quote<T>>) -> Arc<Self> {
    let spread_clone = Arc::clone(&spread);
    Self::new(base, move |x| x + spread_clone.value())
  }
}

impl<T: FloatExt> Observable for DerivedQuote<T> {
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>) {
    self.observable.register_observer(observer);
  }

  fn notify_observers(&self) {
    self.observable.notify_observers();
  }
}

impl<T: FloatExt> Observer for DerivedQuote<T> {
  fn update(&self) {
    self.observable.notify_observers();
  }
}

impl<T: FloatExt> Quote<T> for DerivedQuote<T> {
  fn value(&self) -> T {
    (self.f)(self.base.value())
  }
}

/// Quote computed from a slice of underlying quotes.
pub struct CompositeQuote<T: FloatExt> {
  inputs: Vec<Arc<dyn Quote<T>>>,
  f: Arc<dyn Fn(&[T]) -> T + Send + Sync>,
  observable: ObservableBase,
}

impl<T: FloatExt> std::fmt::Debug for CompositeQuote<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("CompositeQuote")
      .field("inputs", &self.inputs.len())
      .field("value", &self.value())
      .finish()
  }
}

impl<T: FloatExt> CompositeQuote<T> {
  /// Compose a new quote from several inputs.
  pub fn new(
    inputs: Vec<Arc<dyn Quote<T>>>,
    f: impl Fn(&[T]) -> T + Send + Sync + 'static,
  ) -> Arc<Self> {
    let composite = Arc::new(Self {
      inputs,
      f: Arc::new(f),
      observable: ObservableBase::new(),
    });
    let weak: Weak<dyn Observer + Send + Sync> =
      Arc::downgrade(&composite) as Weak<dyn Observer + Send + Sync>;
    for input in &composite.inputs {
      input.register_observer(Weak::clone(&weak));
    }
    composite
  }
}

impl<T: FloatExt> Observable for CompositeQuote<T> {
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>) {
    self.observable.register_observer(observer);
  }

  fn notify_observers(&self) {
    self.observable.notify_observers();
  }
}

impl<T: FloatExt> Observer for CompositeQuote<T> {
  fn update(&self) {
    self.observable.notify_observers();
  }
}

impl<T: FloatExt> Quote<T> for CompositeQuote<T> {
  fn value(&self) -> T {
    let values: Vec<T> = self.inputs.iter().map(|q| q.value()).collect();
    (self.f)(&values)
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
  fn simple_quote_notifies_on_change() {
    let quote = SimpleQuote::<f64>::new(0.01);
    let observer = Arc::new(Counter(AtomicUsize::new(0)));
    quote.register_observer(Arc::downgrade(&observer) as Weak<dyn Observer + Send + Sync>);
    assert!(quote.set_value(0.02));
    assert!(!quote.set_value(0.02));
    assert!(quote.set_value(0.03));
    assert_eq!(observer.0.load(Ordering::SeqCst), 2);
  }

  #[test]
  fn derived_quote_propagates_base() {
    let base = Arc::new(SimpleQuote::<f64>::new(0.01));
    let derived = DerivedQuote::new(Arc::clone(&base) as Arc<dyn Quote<f64>>, |x| x + 0.005);
    assert!((derived.value() - 0.015).abs() < 1e-12);

    let counter = Arc::new(Counter(AtomicUsize::new(0)));
    derived.register_observer(Arc::downgrade(&counter) as Weak<dyn Observer + Send + Sync>);
    base.set_value(0.02);
    assert!((derived.value() - 0.025).abs() < 1e-12);
    assert_eq!(counter.0.load(Ordering::SeqCst), 1);
  }

  #[test]
  fn composite_quote_reacts_to_any_input() {
    let a = Arc::new(SimpleQuote::<f64>::new(1.0));
    let b = Arc::new(SimpleQuote::<f64>::new(2.0));
    let sum = CompositeQuote::new(
      vec![
        Arc::clone(&a) as Arc<dyn Quote<f64>>,
        Arc::clone(&b) as Arc<dyn Quote<f64>>,
      ],
      |xs| xs.iter().copied().sum(),
    );
    assert!((sum.value() - 3.0).abs() < 1e-12);

    let counter = Arc::new(Counter(AtomicUsize::new(0)));
    sum.register_observer(Arc::downgrade(&counter) as Weak<dyn Observer + Send + Sync>);
    a.set_value(10.0);
    b.set_value(20.0);
    assert!((sum.value() - 30.0).abs() < 1e-12);
    assert_eq!(counter.0.load(Ordering::SeqCst), 2);
  }
}
