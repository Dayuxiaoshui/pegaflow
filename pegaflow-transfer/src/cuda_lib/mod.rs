//! CUDA wrapper layer (upstream-derived from `pplx-garden`).
//!
//! Exposes safe wrappers over the raw `cuda_sys` / `cudart_sys` / `gdrapi_sys`
//! FFI modules. Only compiled when the crate-level `v2` feature is on.
#![allow(dead_code, unreachable_pub, unused_imports)]
#![allow(non_snake_case)]

pub mod cumem;
pub mod driver;
pub mod event;
pub mod gdr;
pub mod rt;

mod device;
mod error;
mod mem;

pub use device::{CudaDeviceId, Device};
pub use error::{CudaError, CudaResult};
pub use mem::{CudaDeviceMemory, CudaHostMemory};
