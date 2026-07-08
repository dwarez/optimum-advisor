pub mod advisor;
pub mod app;
pub mod cli;
pub mod domain;
pub mod engines;
pub mod results;
pub mod runtime;
pub mod terminal;

pub use advisor::{hardware, model_memory};
pub use domain::{config, engine, logs, trial};
pub use engines::{params, serve};
pub use runtime::runner;

pub type Result<T> = std::result::Result<T, String>;

pub use app::run;
