pub mod app;
pub mod cli;
pub mod config;
pub mod engine;
pub mod engines;
pub mod logs;
pub mod params;
pub mod runner;
pub mod serve;
pub mod trial;

pub type Result<T> = std::result::Result<T, String>;

pub use app::run;
