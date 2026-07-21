---
name: turbofish-over-let-t
description: Use when writing or editing Rust where you'd otherwise add a `let x: T = …` binding-type annotation — prefer turbofish on the call site (`Foo::<T>::method()`, `.collect::<Vec<_>>()`) so the type travels with the expression instead of pinning the binding.
---

# Turbofish over `let x: T = …`

## The Rule

When the same type information can be expressed via turbofish on the call site, **prefer turbofish**. It travels with the expression, is shorter, and survives copy-paste refactors that move the call out of the `let`.

```rust
// Avoid
let x: f64 = 1.0_f64.ln_1p();
let arr: Array1<f64> = Array1::zeros(8);
let v: Vec<f64> = (0..8).map(|i| i as f64).collect();
let mean: T = sum / T::from_usize_(n);

// Prefer
let x = 1.0_f64.ln_1p();              // suffix carries the type already
let arr = Array1::<f64>::zeros(8);
let v = (0..8).map(|i| i as f64).collect::<Vec<_>>();
let mean = sum / T::from_usize_(n);   // sum's type already drives T
```

## Exceptions

Use a binding-type annotation only when one of these holds:

1. **No call-site to attach turbofish to.** A literal RHS like `let p: f64 = 0.5;` inside a generic function where inference would otherwise pick `i32`.
2. **The annotation locks an invariant on the binding itself.** `let weights: [f64; 4] = read_calibration();` documents the array length in the type — turbofish on `read_calibration()` would not.
3. **Inference picks a numerically wrong type.** Common when calling `.collect()` and the iterator's `Item` is generic over a context that loses information.

In all three cases the annotation IS the type information; in every other case, turbofish.

## Why

The convention reflects how Rust APIs are designed: methods like `collect`, `parse`, `from`, `zeros`, `try_into`, `into_iter`, etc. all take a turbofish for the type the function constructs. Binding-type annotations restate that type in a different place from where it's used, doubling the maintenance surface when types change.
