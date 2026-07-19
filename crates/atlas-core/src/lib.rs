//! Backend-neutral primitives shared by Atlas crates.

use thiserror::Error;

/// Errors that are independent of a concrete accelerator backend.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
