//! The `with_*` setters of the fGN-backed processes and `Cir2F`. Each chain
//! in `chains` moves every parameter of a process and must land exactly where
//! a fresh `new` with the final parameters does, path for path; `cache` moves
//! one cache-feeding parameter at a time, so a setter that forgot to rebuild
//! the cached fGN driver cannot hide behind a later setter's rebuild; `cir_2f`
//! does the same for the factor setters; the `rejects!` table in `rejects`
//! pins that each setter re-applies the assertions `new` makes.

mod fgn_family_setters {
  mod cache;
  mod chains;
  mod cir_2f;
  mod common;
  mod rejects;
}
