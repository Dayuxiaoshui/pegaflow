//! RDMA Verbs fabric (upstream-derived from `pplx-garden`).
//!
//! Whole module is gated behind the crate-level `v2` feature. Exposes
//! `FabricEngine`, `Worker`, `TransferEngine`, and supporting types.
#![allow(dead_code, unreachable_pub, unused_imports)]

pub mod api;
mod cpu_affinity;
mod domain_group;
mod error;
mod fabric_engine;
mod host_buffer;
mod imm_count;
mod interface;
mod mr;
mod provider;
mod provider_dispatch;
mod rdma_op;
mod topo;
mod transfer_engine;
mod transfer_engine_builder;
mod utils;
mod verbs;
mod worker;

pub use domain_group::DomainGroup;
pub use error::*;
pub use fabric_engine::FabricEngine;
pub use host_buffer::{HostBuffer, HostBufferAllocator};
pub use interface::{
    AsyncTransferEngine, BouncingErrorCallback, BouncingRecvCallback, ErrorCallback,
    MockTestTransferEngine, RdmaEngine, RecvCallback, SendBuffer, SendCallback, SendRecvEngine,
};
pub use provider::{RdmaDomain, RdmaDomainInfo};
pub use provider_dispatch::DomainInfo;
pub use topo::{TopologyGroup, detect_topology};
pub use transfer_engine::{ImmCountCallback, TransferCallback, TransferEngine, UvmWatcherCallback};
pub use transfer_engine_builder::TransferEngineBuilder;
pub use worker::{InitializingWorker, Worker, WorkerHandle};
