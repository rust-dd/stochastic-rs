//! Guards `website/content/docs/copulas.mdx`'s catalog tables (and
//! `CLAUDE.md`'s "13 bivariate + 8 multivariate" line) against the failure
//! mode this crate's docs shipped with for a while: a copula added to the
//! tree without its row added to the docs page, so the table silently
//! listed a shrinking fraction of the real catalog.
//!
//! Each match below has **no wildcard arm** — adding a `CopulaType` variant
//! without adding a line here is a compile error, not a silent gap. When
//! it happens: add the arm, add the type's row to the relevant table in
//! `copulas.mdx`, and bump both counts there and in `CLAUDE.md`. This is
//! the same "one line per type, no silent gaps" shape
//! `reproducibility_all_processes.rs` uses for the 127-process count.

use stochastic_rs_copulas::bivariate::CopulaType as Bivariate;

#[test]
fn bivariate_catalog_matches_docs_count() {
  let count = [
    Bivariate::Amh,
    Bivariate::Clayton,
    Bivariate::Fgm,
    Bivariate::Frank,
    Bivariate::Galambos,
    Bivariate::Gaussian,
    Bivariate::Gumbel,
    Bivariate::HuslerReiss,
    Bivariate::Independence,
    Bivariate::Joe,
    Bivariate::MarshallOlkin,
    Bivariate::Plackett,
    Bivariate::TCopula,
  ]
  .into_iter()
  .map(|t| match t {
    Bivariate::Amh
    | Bivariate::Clayton
    | Bivariate::Fgm
    | Bivariate::Frank
    | Bivariate::Galambos
    | Bivariate::Gaussian
    | Bivariate::Gumbel
    | Bivariate::HuslerReiss
    | Bivariate::Independence
    | Bivariate::Joe
    | Bivariate::MarshallOlkin
    | Bivariate::Plackett
    | Bivariate::TCopula => 1,
  })
  .sum::<usize>();

  assert_eq!(count, 13, "copulas.mdx's bivariate table says 13");
}

#[cfg(feature = "openblas")]
#[test]
fn multivariate_catalog_matches_docs_count() {
  use stochastic_rs_copulas::multivariate::CopulaType as Multivariate;

  let count = [
    Multivariate::CVine,
    Multivariate::DVine,
    Multivariate::Gaussian,
    Multivariate::NestedArchimedean,
    Multivariate::RVine,
    Multivariate::TMultivariate,
    Multivariate::Tree,
    Multivariate::Vine,
  ]
  .into_iter()
  .map(|t| match t {
    Multivariate::CVine
    | Multivariate::DVine
    | Multivariate::Gaussian
    | Multivariate::NestedArchimedean
    | Multivariate::RVine
    | Multivariate::TMultivariate
    | Multivariate::Tree
    | Multivariate::Vine => 1,
  })
  .sum::<usize>();

  assert_eq!(count, 8, "copulas.mdx's multivariate table says 8");
}
