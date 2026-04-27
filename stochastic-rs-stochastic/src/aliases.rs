//! Deprecated v1.x type aliases retained for backward compatibility.
//!
//! Acronym-style names (`GBM`, `OU`, `CIR`, …) were renamed to PascalCase
//! (`Gbm`, `Ou`, `Cir`, …) in v2.0.0 for naming-convention consistency
//! across the workspace. These aliases keep v1.x call-sites compiling with
//! a deprecation warning. They are scheduled for removal in v3.0.0.

#![allow(deprecated)]

#[deprecated(since = "2.0.0", note = "renamed to `Cev` for naming consistency")]
pub use crate::diffusion::cev::Cev as CEV;
#[deprecated(since = "2.0.0", note = "renamed to `Cir` for naming consistency")]
pub use crate::diffusion::cir::Cir as CIR;
#[deprecated(since = "2.0.0", note = "renamed to `Gbm` for naming consistency")]
pub use crate::diffusion::gbm::Gbm as GBM;
#[deprecated(since = "2.0.0", note = "renamed to `Ou` for naming consistency")]
pub use crate::diffusion::ou::Ou as OU;
#[deprecated(since = "2.0.0", note = "renamed to `Bgm` for naming consistency")]
pub use crate::interest::bgm::Bgm as BGM;
#[deprecated(since = "2.0.0", note = "renamed to `Hjm` for naming consistency")]
pub use crate::interest::hjm::Hjm as HJM;
#[deprecated(since = "2.0.0", note = "renamed to `Cgmy` for naming consistency")]
pub use crate::jump::cgmy::Cgmy as CGMY;
#[deprecated(since = "2.0.0", note = "renamed to `Kou` for naming consistency")]
pub use crate::jump::kou::Kou as KOU;
#[deprecated(since = "2.0.0", note = "renamed to `Nig` for naming consistency")]
pub use crate::jump::nig::Nig as NIG;
#[deprecated(since = "2.0.0", note = "renamed to `Vg` for naming consistency")]
pub use crate::jump::vg::Vg as VG;
