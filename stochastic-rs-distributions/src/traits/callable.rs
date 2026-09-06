//! Type-erased `Fn1D` / `Fn2D` callables, the [`Expr`] coefficient language
//! and its compiled [`Program`], and the Python-feature-gated `CallableDist`
//! adapter.

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

use super::float::FloatExt;

/// A coefficient of time and state written as an expression rather than a
/// closure: `Expr::lit(0.01) + Expr::x().abs() * 0.4`. It evaluates on the
/// host like any [`Fn2D`], and it compiles to a short postfix [`Program`] a
/// device kernel interprets — which is what lets a process whose coefficients
/// are functions of the state, the Cheyette local volatility or a Volterra
/// equation's drift and diffusion, run on a GPU. [`Expr::T`] is the time
/// argument and [`Expr::X`] the state; the vocabulary is the one the device
/// engine's own function table carries, so every node here has one meaning on
/// the host and in every kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
  /// The time argument.
  T,
  /// The state argument.
  X,
  /// A constant.
  Const(f64),
  Add(Box<Expr>, Box<Expr>),
  Sub(Box<Expr>, Box<Expr>),
  Mul(Box<Expr>, Box<Expr>),
  Div(Box<Expr>, Box<Expr>),
  /// `a^b`.
  Pow(Box<Expr>, Box<Expr>),
  Max(Box<Expr>, Box<Expr>),
  Min(Box<Expr>, Box<Expr>),
  Neg(Box<Expr>),
  Sqrt(Box<Expr>),
  Exp(Box<Expr>),
  Ln(Box<Expr>),
  Abs(Box<Expr>),
  Tanh(Box<Expr>),
}

impl Expr {
  /// The time argument.
  pub fn t() -> Self {
    Expr::T
  }

  /// The state argument.
  pub fn x() -> Self {
    Expr::X
  }

  /// A constant.
  pub fn lit(value: f64) -> Self {
    Expr::Const(value)
  }

  /// `√self`.
  pub fn sqrt(self) -> Self {
    Expr::Sqrt(Box::new(self))
  }

  /// `exp self`.
  pub fn exp(self) -> Self {
    Expr::Exp(Box::new(self))
  }

  /// `ln self`.
  pub fn ln(self) -> Self {
    Expr::Ln(Box::new(self))
  }

  /// `|self|`.
  pub fn abs(self) -> Self {
    Expr::Abs(Box::new(self))
  }

  /// `tanh self`.
  pub fn tanh(self) -> Self {
    Expr::Tanh(Box::new(self))
  }

  /// `self^rhs`.
  pub fn powf(self, rhs: impl Into<Expr>) -> Self {
    Expr::Pow(Box::new(self), Box::new(rhs.into()))
  }

  /// The larger of `self` and `rhs`.
  pub fn max(self, rhs: impl Into<Expr>) -> Self {
    Expr::Max(Box::new(self), Box::new(rhs.into()))
  }

  /// The smaller of `self` and `rhs`.
  pub fn min(self, rhs: impl Into<Expr>) -> Self {
    Expr::Min(Box::new(self), Box::new(rhs.into()))
  }

  /// The value at `(t, x)`, by walking the tree.
  pub fn eval<T: FloatExt>(&self, t: T, x: T) -> T {
    match self {
      Expr::T => t,
      Expr::X => x,
      Expr::Const(c) => T::from_f64_fast(*c),
      Expr::Add(a, b) => a.eval(t, x) + b.eval(t, x),
      Expr::Sub(a, b) => a.eval(t, x) - b.eval(t, x),
      Expr::Mul(a, b) => a.eval(t, x) * b.eval(t, x),
      Expr::Div(a, b) => a.eval(t, x) / b.eval(t, x),
      Expr::Pow(a, b) => a.eval(t, x).powf(b.eval(t, x)),
      Expr::Max(a, b) => a.eval(t, x).max(b.eval(t, x)),
      Expr::Min(a, b) => a.eval(t, x).min(b.eval(t, x)),
      Expr::Neg(a) => T::zero() - a.eval(t, x),
      Expr::Sqrt(a) => a.eval(t, x).sqrt(),
      Expr::Exp(a) => a.eval(t, x).exp(),
      Expr::Ln(a) => a.eval(t, x).ln(),
      Expr::Abs(a) => a.eval(t, x).abs(),
      Expr::Tanh(a) => a.eval(t, x).tanh(),
    }
  }

  /// The postfix program of this expression.
  pub fn compile(&self) -> Program {
    Program::compile(self)
  }

  fn emit(&self, ops: &mut Vec<(u32, f64)>, depth: &mut usize, deepest: &mut usize) {
    fn push(
      ops: &mut Vec<(u32, f64)>,
      depth: &mut usize,
      deepest: &mut usize,
      code: u32,
      value: f64,
      delta: isize,
    ) {
      ops.push((code, value));
      *depth = (*depth as isize + delta) as usize;
      *deepest = (*deepest).max(*depth);
    }
    match self {
      Expr::T => push(ops, depth, deepest, 0, 0.0, 1),
      Expr::X => push(ops, depth, deepest, 1, 0.0, 1),
      Expr::Const(c) => push(ops, depth, deepest, 2, *c, 1),
      Expr::Add(a, b)
      | Expr::Sub(a, b)
      | Expr::Mul(a, b)
      | Expr::Div(a, b)
      | Expr::Pow(a, b)
      | Expr::Max(a, b)
      | Expr::Min(a, b) => {
        a.emit(ops, depth, deepest);
        b.emit(ops, depth, deepest);
        let code = match self {
          Expr::Add(..) => 3,
          Expr::Sub(..) => 4,
          Expr::Mul(..) => 5,
          Expr::Div(..) => 6,
          Expr::Pow(..) => 7,
          Expr::Max(..) => 8,
          _ => 9,
        };
        push(ops, depth, deepest, code, 0.0, -1);
      }
      Expr::Neg(a) | Expr::Sqrt(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Abs(a) | Expr::Tanh(a) => {
        a.emit(ops, depth, deepest);
        let code = match self {
          Expr::Neg(_) => 10,
          Expr::Sqrt(_) => 11,
          Expr::Exp(_) => 12,
          Expr::Ln(_) => 13,
          Expr::Abs(_) => 14,
          _ => 15,
        };
        push(ops, depth, deepest, code, 0.0, 0);
      }
    }
  }
}

impl From<f64> for Expr {
  fn from(value: f64) -> Self {
    Expr::Const(value)
  }
}

macro_rules! expr_binary {
  ($trait:ident, $method:ident, $variant:ident) => {
    impl $trait<Expr> for Expr {
      type Output = Expr;

      fn $method(self, rhs: Expr) -> Expr {
        Expr::$variant(Box::new(self), Box::new(rhs))
      }
    }

    impl $trait<f64> for Expr {
      type Output = Expr;

      fn $method(self, rhs: f64) -> Expr {
        Expr::$variant(Box::new(self), Box::new(Expr::Const(rhs)))
      }
    }

    impl $trait<Expr> for f64 {
      type Output = Expr;

      fn $method(self, rhs: Expr) -> Expr {
        Expr::$variant(Box::new(Expr::Const(self)), Box::new(rhs))
      }
    }
  };
}

expr_binary!(Add, add, Add);
expr_binary!(Sub, sub, Sub);
expr_binary!(Mul, mul, Mul);
expr_binary!(Div, div, Div);

impl Neg for Expr {
  type Output = Expr;

  fn neg(self) -> Expr {
    Expr::Neg(Box::new(self))
  }
}

/// An [`Expr`] compiled to postfix code: one `(opcode, constant)` pair per
/// node, run on a stack. Codes `0`–`2` push the time, the state and a
/// constant; `3`–`9` are the binary `+ − × ÷ ^ max min`; `10`–`15` the unary
/// `− √ exp ln |·| tanh`. A program never exceeds [`Program::MAX_OPS`]
/// operations or a stack of [`Program::MAX_DEPTH`], the bounds the kernels'
/// fixed arrays hold, so a program that compiles runs anywhere.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
  ops: Vec<(u32, f64)>,
  depth: usize,
}

impl Program {
  /// The most operations a program may hold.
  pub const MAX_OPS: usize = 62;
  /// The deepest stack a program may need.
  pub const MAX_DEPTH: usize = 8;

  /// The postfix program of `expr`.
  ///
  /// # Panics
  /// If the expression needs more than [`Self::MAX_OPS`] operations or a
  /// stack deeper than [`Self::MAX_DEPTH`].
  pub fn compile(expr: &Expr) -> Self {
    let mut ops = Vec::new();
    let (mut depth, mut deepest) = (0usize, 0usize);
    expr.emit(&mut ops, &mut depth, &mut deepest);
    assert!(
      ops.len() <= Self::MAX_OPS,
      "an expression compiles to at most {} operations, this one needs {}",
      Self::MAX_OPS,
      ops.len()
    );
    assert!(
      deepest <= Self::MAX_DEPTH,
      "an expression may need a stack of at most {}, this one needs {deepest}",
      Self::MAX_DEPTH
    );
    Self { ops, depth: deepest }
  }

  /// The value at `(t, x)`, by running the code on a stack exactly as the
  /// kernels run it.
  pub fn eval<T: FloatExt>(&self, t: T, x: T) -> T {
    let mut stack = [T::zero(); Self::MAX_DEPTH];
    let mut sp = 0usize;
    for &(code, value) in &self.ops {
      match code {
        0..=2 => {
          stack[sp] = match code {
            0 => t,
            1 => x,
            _ => T::from_f64_fast(value),
          };
          sp += 1;
        }
        3..=9 => {
          let (a, b) = (stack[sp - 2], stack[sp - 1]);
          stack[sp - 2] = match code {
            3 => a + b,
            4 => a - b,
            5 => a * b,
            6 => a / b,
            7 => a.powf(b),
            8 => a.max(b),
            _ => a.min(b),
          };
          sp -= 1;
        }
        _ => {
          let a = stack[sp - 1];
          stack[sp - 1] = match code {
            10 => T::zero() - a,
            11 => a.sqrt(),
            12 => a.exp(),
            13 => a.ln(),
            14 => a.abs(),
            _ => a.tanh(),
          };
        }
      }
    }
    stack[0]
  }

  /// The `(opcode, constant)` pairs in order.
  pub fn ops(&self) -> &[(u32, f64)] {
    &self.ops
  }

  /// How many operations the program holds.
  pub fn len(&self) -> usize {
    self.ops.len()
  }

  /// Whether the program holds no operation.
  pub fn is_empty(&self) -> bool {
    self.ops.is_empty()
  }

  /// The deepest stack the program needs.
  pub fn depth(&self) -> usize {
    self.depth
  }
}

pub enum Fn1D<T: FloatExt> {
  Native(fn(T) -> T),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

/// Manual, not `#[derive(Clone)]`: `pyo3::Py<PyAny>` (0.28) has no
/// unconditional `Clone` impl, only `clone_ref(py)`, which needs a GIL
/// token — mirrors the GIL-acquisition pattern [`Fn1D::call`]'s own `Py`
/// arm already uses.
impl<T: FloatExt> Clone for Fn1D<T> {
  fn clone(&self) -> Self {
    match self {
      Fn1D::Native(f) => Fn1D::Native(*f),
      #[cfg(feature = "python")]
      Fn1D::Py(callable) => Fn1D::Py(pyo3::Python::attach(|py| callable.clone_ref(py))),
    }
  }
}

impl<T: FloatExt> Fn1D<T> {
  pub fn call(&self, t: T) -> T {
    match self {
      Fn1D::Native(f) => f(t),
      #[cfg(feature = "python")]
      Fn1D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(),))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T) -> T> for Fn1D<T> {
  fn from(f: fn(T) -> T) -> Self {
    Fn1D::Native(f)
  }
}

pub enum Fn2D<T: FloatExt> {
  Native(fn(T, T) -> T),
  /// A coefficient written as an [`Expr`], compiled: the one form a device
  /// kernel can evaluate.
  Expr(Program),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

/// Manual, not `#[derive(Clone)]`: see [`Fn1D`]'s own `Clone` impl doc.
impl<T: FloatExt> Clone for Fn2D<T> {
  fn clone(&self) -> Self {
    match self {
      Fn2D::Native(f) => Fn2D::Native(*f),
      Fn2D::Expr(program) => Fn2D::Expr(program.clone()),
      #[cfg(feature = "python")]
      Fn2D::Py(callable) => Fn2D::Py(pyo3::Python::attach(|py| callable.clone_ref(py))),
    }
  }
}

impl<T: FloatExt> Fn2D<T> {
  pub fn call(&self, t: T, u: T) -> T {
    match self {
      Fn2D::Native(f) => f(t, u),
      Fn2D::Expr(program) => program.eval(t, u),
      #[cfg(feature = "python")]
      Fn2D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(), u.to_f64().unwrap()))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T, T) -> T> for Fn2D<T> {
  fn from(f: fn(T, T) -> T) -> Self {
    Fn2D::Native(f)
  }
}

impl<T: FloatExt> From<Expr> for Fn2D<T> {
  fn from(expr: Expr) -> Self {
    Fn2D::Expr(expr.compile())
  }
}

impl<T: FloatExt> From<Program> for Fn2D<T> {
  fn from(program: Program) -> Self {
    Fn2D::Expr(program)
  }
}

impl<T: FloatExt> Fn2D<T> {
  /// The compiled program behind an [`Fn2D::Expr`], `None` for a closure or a
  /// Python callable — the test a device path makes before it launches.
  pub fn program(&self) -> Option<&Program> {
    match self {
      Fn2D::Expr(program) => Some(program),
      _ => None,
    }
  }
}

#[cfg(feature = "python")]
pub struct CallableDist<T: FloatExt> {
  callable: pyo3::Py<pyo3::PyAny>,
  _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "python")]
impl<T: FloatExt> CallableDist<T> {
  pub fn new(callable: pyo3::Py<pyo3::PyAny>) -> Self {
    Self {
      callable,
      _phantom: std::marker::PhantomData,
    }
  }
}

#[cfg(feature = "python")]
impl<T: FloatExt> rand_distr::Distribution<T> for CallableDist<T> {
  fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
    pyo3::Python::attach(|py| {
      let result: f64 = self.callable.call0(py).unwrap().extract::<f64>(py).unwrap();
      T::from_f64_fast(result)
    })
  }
}

#[cfg(test)]
mod expr_tests {
  use super::*;

  fn sample() -> Expr {
    (Expr::lit(0.01) + Expr::x().abs() * 0.4 - Expr::t() / 3.0).max(0.0) * (-Expr::x()).exp()
      + Expr::x().powf(2.0).sqrt().min(Expr::lit(1.0)).tanh()
      - Expr::lit(2.0).ln()
  }

  fn closed(t: f64, x: f64) -> f64 {
    (0.01 + x.abs() * 0.4 - t / 3.0).max(0.0) * (-x).exp() + x.powi(2).sqrt().min(1.0).tanh()
      - 2.0_f64.ln()
  }

  /// Walking the tree is the closed form.
  #[test]
  fn an_expression_evaluates_like_its_closure() {
    for (t, x) in [(0.0_f64, 0.0_f64), (0.5, -0.3), (1.0, 2.5), (0.2, 0.7)] {
      assert!((sample().eval(t, x) - closed(t, x)).abs() < 1e-12);
    }
  }

  /// The postfix program computes what the tree computes, on a stack no
  /// deeper than the kernels carry.
  #[test]
  fn a_program_runs_its_expression() {
    let program = sample().compile();
    assert!(program.depth() <= Program::MAX_DEPTH && program.depth() >= 2);
    assert!(program.len() <= Program::MAX_OPS);
    for (t, x) in [(0.0_f64, 0.0_f64), (0.5, -0.3), (1.0, 2.5), (0.2, 0.7)] {
      let (tree, code): (f64, f64) = (sample().eval(t, x), program.eval(t, x));
      assert!((tree - code).abs() < 1e-12, "tree {tree}, program {code}");
      let single: f32 = program.eval(t as f32, x as f32);
      assert!((single as f64 - tree).abs() < 1e-5);
    }
  }

  /// A callable built from an expression runs the program.
  #[test]
  fn a_callable_from_an_expression_runs_the_program() {
    let f: Fn2D<f64> = (Expr::x() * 2.0 + Expr::t()).into();
    assert!(f.program().is_some());
    assert_eq!(f.call(0.5, 1.5), 3.5);
    let g: Fn2D<f64> = Fn2D::Native(|t, x| t + x);
    assert!(g.program().is_none());
  }

  /// Too deep an expression is refused at compile time rather than
  /// overflowing a kernel's stack.
  #[test]
  #[should_panic(expected = "a stack of at most 8")]
  fn a_program_deeper_than_the_kernels_stack_is_refused() {
    let mut deep = Expr::x();
    for _ in 0..9 {
      deep = Expr::x() * (Expr::x() + deep);
    }
    let _ = deep.compile();
  }
}
