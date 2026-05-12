//! Tier-attribution integration tests.
//!
//! Verifies that `pegaflow_cache_tier_block_requests` is emitted at most once
//! per `req_id`, that the sum across tiers equals the request size, and that
//! the post-completion fall-through scan does not double count.
//!
//! Integration tests share global OTel state; we use the `test_hooks::tier_attribution`
//! spy (gated by `feature = "test-utils"`) for stable per-test assertions
//! instead of reading global counter values.

mod common;

use std::time::Duration;

use common::*;
use pegaflow_core::test_hooks::tier_attribution;
use pegaflow_core::{PrefetchStatus, SsdCacheConfig, StorageConfig};

const SSD_BLOCK_SIZE: usize = 4096;
const SSD_NUM_BLOCKS: usize = 4;
const SSD_POOL_SIZE: usize = SSD_NUM_BLOCKS * SSD_BLOCK_SIZE * 2;
const SSD_CAPACITY: u64 = 64 * 1024 * 1024;

fn ssd_env(instance_id: &'static str) -> TestEnv {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    let env = TestEnvBuilder::new(instance_id, "test-ns-tier-ssd")
        .layer("layer_0", SSD_NUM_BLOCKS, SSD_BLOCK_SIZE)
        .pool_size(SSD_POOL_SIZE)
        .storage(StorageConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                cache_path,
                capacity_bytes: SSD_CAPACITY,
                ..SsdCacheConfig::default()
            }),
            ..StorageConfig::default()
        })
        .build();
    // Keep temp_dir alive so the cache file remains available for the test.
    std::mem::forget(temp_dir);
    env
}

/// Pure RAM prefix hit attributes all blocks to `ram`, and the sum equals total.
#[tokio::test]
async fn ram_only_hit_attributes_all_to_ram() {
    tier_attribution::reset();

    let env = TestEnvBuilder::new("test-tier-ram", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(70);
    env.save_and_wait(&hashes).await;

    match env.query_with_req_id("req-ram-1", &hashes).await {
        PrefetchStatus::Done { hit, missing } => {
            assert_eq!(hit, 4);
            assert_eq!(missing, 0);
        }
        other => panic!("expected Done, got {other:?}"),
    }
    env.unpin(&hashes);

    let events = tier_attribution::snapshot_for("req-ram-1");
    assert_eq!(events.len(), 1, "exactly one attribution per req_id");
    let a = events[0];
    assert_eq!(a.ram, 4);
    assert_eq!(a.rdma, 0);
    assert_eq!(a.ssd, 0);
    assert_eq!(a.miss, 0);
    assert_eq!(a.total(), 4);
}

/// Full miss with no backing tier configured attributes everything to `miss`.
#[tokio::test]
async fn full_miss_attributes_all_to_miss() {
    tier_attribution::reset();

    let env = TestEnvBuilder::new("test-tier-miss", "test-ns")
        .layer("layer_0", 1, 1024)
        .build();
    // Hash keys not present in cache.
    let hashes = make_block_hashes(3, 99);

    match env.query_with_req_id("req-miss-1", &hashes).await {
        PrefetchStatus::Done { hit, missing } => {
            assert_eq!(hit, 0);
            assert_eq!(missing, 3);
        }
        other => panic!("expected Done, got {other:?}"),
    }

    let events = tier_attribution::snapshot_for("req-miss-1");
    assert_eq!(events.len(), 1);
    let a = events[0];
    assert_eq!(a.ram, 0);
    assert_eq!(a.rdma, 0);
    assert_eq!(a.ssd, 0);
    assert_eq!(a.miss, 3);
}

/// Same `req_id` queried multiple times must attribute exactly once.
#[tokio::test]
async fn repeated_query_with_same_req_id_attributes_once() {
    tier_attribution::reset();

    let env = TestEnvBuilder::new("test-tier-repeat", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(71);
    env.save_and_wait(&hashes).await;

    for _ in 0..5 {
        match env.query_with_req_id("req-repeat", &hashes).await {
            PrefetchStatus::Done { .. } => {}
            other => panic!("expected Done, got {other:?}"),
        }
        env.unpin(&hashes);
    }

    let events = tier_attribution::snapshot_for("req-repeat");
    assert_eq!(
        events.len(),
        1,
        "same req_id queried 5 times must attribute exactly once, got {events:?}"
    );
}

/// Distinct `req_id`s on the same prefix each attribute once.
#[tokio::test]
async fn distinct_req_ids_each_attribute_once() {
    tier_attribution::reset();

    let env = TestEnvBuilder::new("test-tier-distinct", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(72);
    env.save_and_wait(&hashes).await;

    for i in 0..3 {
        let req_id = format!("req-distinct-{i}");
        env.query_with_req_id(&req_id, &hashes).await;
        env.unpin(&hashes);
    }

    let events: Vec<_> = (0..3)
        .flat_map(|i| tier_attribution::snapshot_for(&format!("req-distinct-{i}")))
        .collect();
    assert_eq!(
        events.len(),
        3,
        "three distinct req_ids => three attributions"
    );
    for a in &events {
        assert_eq!(a.ram, 4);
        assert_eq!(a.total(), 4);
    }
}

/// Empty queries do not produce attributions.
#[tokio::test]
async fn empty_query_does_not_attribute() {
    tier_attribution::reset();

    let env = TestEnvBuilder::new("test-tier-empty", "test-ns")
        .layer("layer_0", 1, 1024)
        .build();

    env.query_with_req_id("req-empty", &[]).await;

    assert!(
        tier_attribution::snapshot_for("req-empty").is_empty(),
        "empty query must not attribute"
    );
}

/// SSD prefetch is attributed on the initial Loading decision, while the
/// post-completion fall-through scan does not add a second RAM attribution.
#[tokio::test]
async fn ssd_loading_then_completed_poll_attributes_once_to_ssd() {
    tier_attribution::reset();

    let env = ssd_env("test-tier-ssd-loading");
    let target = env.hashes(73);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;

    // Save enough new blocks to force the original target out of RAM while
    // keeping it available in SSD.
    let filler_a = make_block_hashes(SSD_NUM_BLOCKS, 74);
    let filler_b = make_block_hashes(SSD_NUM_BLOCKS, 76);
    env.save_layer(0, &filler_a).await;
    env.save_layer(0, &filler_b).await;
    env.engine.flush_saves().await;

    match env.query_with_req_id("req-ssd", &target).await {
        PrefetchStatus::Loading { hit, loading } => {
            assert_eq!(hit, 0);
            assert_eq!(loading, target.len());
        }
        other => panic!("expected Loading, got {other:?}"),
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match env.query_with_req_id("req-ssd", &target).await {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, target.len());
                assert_eq!(missing, 0);
                env.unpin(&target);
                break;
            }
            PrefetchStatus::Loading { .. } => {}
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for SSD prefetch completion"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let events = tier_attribution::snapshot_for("req-ssd");
    assert_eq!(
        events.len(),
        1,
        "completed fall-through must not re-attribute as RAM"
    );
    let a = events[0];
    assert_eq!(a.ram, 0);
    assert_eq!(a.rdma, 0);
    assert_eq!(a.ssd, target.len());
    assert_eq!(a.miss, 0);
    assert_eq!(a.total(), target.len());
}

/// Attribution TTL shares the existing prefetch GC lifecycle: once GC removes
/// an attributed req_id, reusing that req_id is treated as a new logical query.
#[tokio::test]
async fn gc_allows_same_req_id_to_attribute_again() {
    tier_attribution::reset();

    let env = TestEnvBuilder::new("test-tier-gc", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(75);
    env.save_and_wait(&hashes).await;

    match env.query_with_req_id("req-gc", &hashes).await {
        PrefetchStatus::Done { hit, missing } => {
            assert_eq!(hit, hashes.len());
            assert_eq!(missing, 0);
        }
        other => panic!("expected Done, got {other:?}"),
    }
    env.unpin(&hashes);
    assert_eq!(tier_attribution::snapshot_for("req-gc").len(), 1);

    env.engine
        .gc_stale_inflight(Duration::ZERO, Duration::ZERO)
        .await;

    match env.query_with_req_id("req-gc", &hashes).await {
        PrefetchStatus::Done { hit, missing } => {
            assert_eq!(hit, hashes.len());
            assert_eq!(missing, 0);
        }
        other => panic!("expected Done, got {other:?}"),
    }
    env.unpin(&hashes);

    assert_eq!(
        tier_attribution::snapshot_for("req-gc").len(),
        2,
        "same req_id attributes again after attribution GC"
    );
}
