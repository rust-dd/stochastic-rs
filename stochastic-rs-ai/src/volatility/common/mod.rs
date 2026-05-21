mod dataset;
mod metadata;
mod model;
mod network;
mod plot;
mod scaler;
mod spec;

pub use dataset::load_trainset_gzip_npy;
pub use dataset::rmse_1d;
pub use model::StochVolNn;
pub use plot::write_surface_fit_plot_html;
pub use spec::EpochMetrics;
pub use spec::StochVolModelSpec;
pub use spec::TrainConfig;
pub use spec::TrainReport;

#[cfg(test)]
pub(crate) use dataset::synthetic_surface_dataset;
