//! Fully implicit Euler start-up: sparse LU solves all three spatial operators together.
//!
//! Reference: Wyns & Du Toit (2016), §3, arXiv:1611.02961

use faer::Mat;
use faer::linalg::solvers::Solve;
use faer::sparse::SparseColMat;
use faer::sparse::Triplet;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::linalg::solvers::SymbolicLu;

use super::Direction;
use super::Level;
use super::Mesh;
use crate::QuantError;

#[derive(Default)]
pub(super) struct ImplicitEuler {
  symbolic: Option<SymbolicLu<usize>>,
}

impl ImplicitEuler {
  /// Solve `(I - dt (A0 + A1 + A2)) p_next = p` with every operator
  /// evaluated at the new time level, including the mixed derivative.
  pub(crate) fn solve(
    &mut self,
    mesh: &Mesh,
    along_v: &Direction,
    next: &Level<'_>,
    dt: f64,
    p: &[f64],
    out: &mut [f64],
  ) -> Result<(), QuantError> {
    let (m1, m2) = (mesh.m1(), mesh.m2());
    let n = mesh.len();
    let mut entries = Vec::with_capacity(21 * n);
    for j in 0..m2 {
      for i in 0..m1 {
        let row = j * m1 + i;
        let mut add = |col, val| entries.push(Triplet { row, col, val });
        add(row, 1.0 - dt * (next.x.diag[row] + along_v.diag[row]));
        if i > 0 {
          add(row - 1, -dt * next.x.lower[row]);
        }
        if i + 1 < m1 {
          add(row + 1, -dt * next.x.upper[row]);
        }
        if j > 0 {
          add(row - m1, -dt * along_v.lower[row]);
        }
        if j + 1 < m2 {
          add(row + m1, -dt * along_v.upper[row]);
        }
        let scale = -dt / (mesh.wx[i] * mesh.wv[j]);
        for (ci, cj, sign) in [
          (i + 1, j + 1, 1.0),
          (i + 1, j, -1.0),
          (i, j + 1, -1.0),
          (i, j, 1.0),
        ] {
          for (col, coefficient) in next.mixed.corner_entries(ci, cj) {
            // Retain zeros so the symbolic sparsity pattern is independent
            // of the current leverage, correlation and time-step size.
            add(col, scale * sign * coefficient);
          }
        }
      }
    }
    if !entries.iter().all(|entry| entry.val.is_finite()) {
      return Err(step_failure("its coefficients are not finite".to_string()));
    }
    let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &entries)
      .map_err(|e| step_failure(format!("assembling the matrix failed: {e}")))?;
    let symbolic = match &self.symbolic {
      Some(symbolic) => symbolic.clone(),
      None => {
        let symbolic = SymbolicLu::try_new(matrix.symbolic())
          .map_err(|e| step_failure(format!("the symbolic analysis failed: {e}")))?;
        self.symbolic = Some(symbolic.clone());
        symbolic
      }
    };
    let lu = Lu::try_new_with_symbolic(symbolic, matrix.as_ref())
      .map_err(|e| step_failure(format!("the LU factorisation failed: {e}")))?;
    let mut rhs = Mat::from_fn(n, 1, |i, _| p[i]);
    lu.solve_in_place(&mut rhs);
    for (i, value) in out.iter_mut().enumerate() {
      *value = rhs[(i, 0)];
    }
    if !out.iter().all(|value| value.is_finite()) {
      return Err(step_failure("the density came out non-finite".to_string()));
    }
    Ok(())
  }
}

fn step_failure(reason: String) -> QuantError {
  QuantError::Numerical {
    what: "the Fokker–Planck implicit step",
    reason,
  }
}
