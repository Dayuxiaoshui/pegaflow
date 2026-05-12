// Per-request load reservation state machine. A single Mutex is sufficient
// because reserve-load operations are per-query and not a hot data-copy path.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use log::{info, warn};
use mea::oneshot;
use parking_lot::Mutex;

use crate::backing::{PrefetchResult, RdmaFetchStore, SsdBackingStore};
use crate::block::{BlockKey, ReserveLoadStatus};
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;
use super::tier_attribution::{
    AttributionSource, TierAttribution, record_cache_tier_block_requests,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BackingSource {
    Ssd,
    Rdma,
}

impl BackingSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ssd => "ssd",
            Self::Rdma => "rdma",
        }
    }

    const fn as_attribution(self) -> AttributionSource {
        match self {
            Self::Ssd => AttributionSource::Ssd,
            Self::Rdma => AttributionSource::Rdma,
        }
    }
}

struct BackingFetchEntry {
    blocks_rx: oneshot::Receiver<PrefetchResult>,
    loading_count: usize,
    source: BackingSource,
}

/// Result of a single backing fetch attempt (SSD or RDMA).
struct BackingFetch {
    found: usize,
    rx: oneshot::Receiver<PrefetchResult>,
    source: BackingSource,
}

struct ReserveLoadScan<'a> {
    instance_id: &'a str,
    request_id: &'a str,
    namespace: &'a str,
    hashes: &'a [Vec<u8>],
    num_workers: usize,
    emit_tier_metrics: bool,
}

struct ReservationState {
    active: HashMap<String, BackingFetchEntry>,
    /// Invariant: `inflight_count == active.values().map(|e| e.loading_count).sum()`
    inflight_count: usize,
    /// request_ids where RDMA remote fetch returned zero blocks (remote evicted).
    /// Prevents re-triggering RDMA on every subsequent poll for the same request.
    failed_remote: HashMap<String, Instant>,
}

impl ReservationState {
    fn remove_entry(&mut self, request_id: &str) -> Option<BackingFetchEntry> {
        if let Some(entry) = self.active.remove(request_id) {
            self.inflight_count = self.inflight_count.saturating_sub(entry.loading_count);
            Some(entry)
        } else {
            None
        }
    }
}

pub(super) struct LoadReservationScheduler {
    state: Mutex<ReservationState>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    rdma_fetch: Option<Arc<RdmaFetchStore>>,
    max_prefetch_blocks: usize,
}

impl LoadReservationScheduler {
    pub(super) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        rdma_fetch: Option<Arc<RdmaFetchStore>>,
        max_prefetch_blocks: usize,
    ) -> Self {
        Self {
            state: Mutex::new(ReservationState {
                active: HashMap::new(),
                inflight_count: 0,
                failed_remote: HashMap::new(),
            }),
            ssd_store,
            rdma_fetch,
            max_prefetch_blocks,
        }
    }

    pub(super) async fn reserve_load(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        request_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> ReserveLoadStatus {
        // Default: this call may be the first decision and should attribute.
        let mut emit_tier_metrics = true;
        if let Some(status) = self.poll_existing(read_cache, request_id) {
            match status {
                PollResult::StillLoading => {
                    return ReserveLoadStatus::Loading { hit: 0 };
                }
                PollResult::Completed => {
                    // Backing has just written blocks into read_cache. The
                    // fall-through scan will re-see them as RAM hits; we MUST
                    // NOT attribute again, because we already attributed
                    // them as `rdma`/`ssd` on the first decision.
                    emit_tier_metrics = false;
                }
            }
        }

        self.full_prefix_scan(
            read_cache,
            ReserveLoadScan {
                instance_id,
                request_id,
                namespace,
                hashes,
                num_workers,
                emit_tier_metrics,
            },
        )
        .await
    }

    fn poll_existing(&self, read_cache: &ReadCache, request_id: &str) -> Option<PollResult> {
        let mut state = self.state.lock();
        let entry = state.active.get_mut(request_id)?;

        match entry.blocks_rx.try_recv() {
            Err(oneshot::TryRecvError::Empty) => Some(PollResult::StillLoading),
            Ok(prefetched_blocks) => {
                let expected = entry.loading_count;
                let source = entry.source;
                state.remove_entry(request_id);
                // RDMA remote node can return fewer blocks than MetaServer promised
                // (likely evicted). Don't re-trigger RDMA on subsequent scans.
                if source == BackingSource::Rdma
                    && prefetched_blocks.len() < expected
                    && expected > 0
                {
                    state
                        .failed_remote
                        .insert(request_id.to_string(), Instant::now());
                    info!(
                        "RDMA prefetch returned fewer blocks than expected: request_id={} returned={} expected={}",
                        request_id,
                        prefetched_blocks.len(),
                        expected
                    );
                }
                drop(state);
                read_cache.batch_insert(prefetched_blocks);
                Some(PollResult::Completed)
            }
            Err(oneshot::TryRecvError::Disconnected) => {
                warn!(
                    "Backing prefetch sender dropped for request_id={}, falling back to re-scan",
                    request_id
                );
                state.remove_entry(request_id);
                Some(PollResult::Completed)
            }
        }
    }

    async fn full_prefix_scan(
        &self,
        read_cache: &ReadCache,
        scan: ReserveLoadScan<'_>,
    ) -> ReserveLoadStatus {
        let total_start = Instant::now();

        let key_build_start = Instant::now();
        let keys: Vec<BlockKey> = scan
            .hashes
            .iter()
            .map(|hash| BlockKey::new(scan.namespace.to_string(), hash.clone()))
            .collect();
        let key_build = key_build_start.elapsed();

        let cache_scan_start = Instant::now();
        let (hit, blocks_to_lease) = read_cache.get_prefix_blocks(&keys);
        let cache_scan = cache_scan_start.elapsed();
        let remaining = &keys[hit..];

        let fetch_select_start = Instant::now();
        let load = self
            .try_backing_fetch(scan.request_id, scan.namespace, remaining)
            .await;
        let fetch_select = fetch_select_start.elapsed();
        let loading = load.as_ref().map_or(0, |l| l.found);
        let missing = keys.len() - hit - loading;

        if let Some(load) = load {
            let source = load.source;
            let register_start = Instant::now();
            self.register_backing_fetch(scan.request_id, load);
            let register = register_start.elapsed();

            self.maybe_record_tier_attribution(
                keys.len(),
                hit,
                loading,
                Some(source.as_attribution()),
                scan.emit_tier_metrics,
            );

            info!(
                "Reserve-load backing timing: request_id={} source={} total_keys={} hit={} loading={} missing={} key_build={:?} cache_scan={:?} fetch_select={:?} register_backing_fetch={:?} total={:?}",
                scan.request_id,
                source.as_str(),
                keys.len(),
                hit,
                loading,
                missing,
                key_build,
                cache_scan,
                fetch_select,
                register,
                total_start.elapsed()
            );
            ReserveLoadStatus::Loading { hit }
        } else {
            let lease_start = Instant::now();
            let lease_id = read_cache
                .create_load_lease(scan.instance_id, scan.num_workers, &blocks_to_lease)
                .unwrap_or_default();
            let lease = lease_start.elapsed();

            self.maybe_record_tier_attribution(
                keys.len(),
                hit,
                /* loading = */ 0,
                /* loading_source = */ None,
                scan.emit_tier_metrics,
            );

            info!(
                "Reserve-load local timing: request_id={} total_keys={} hit={} missing={} key_build={:?} cache_scan={:?} fetch_select={:?} lease={:?} total={:?}",
                scan.request_id,
                keys.len(),
                hit,
                missing,
                key_build,
                cache_scan,
                fetch_select,
                lease,
                total_start.elapsed()
            );
            ReserveLoadStatus::Ready {
                hit,
                missing,
                lease_id,
            }
        }
    }

    /// Attribute this `reserve_load` decision. Skips attribution when:
    /// * `emit_tier_metrics == false` (e.g. post-completion fall-through);
    /// * `keys` was empty (no decision to attribute).
    fn maybe_record_tier_attribution(
        &self,
        total: usize,
        hit: usize,
        loading: usize,
        loading_source: Option<AttributionSource>,
        emit_tier_metrics: bool,
    ) {
        if !emit_tier_metrics || total == 0 {
            return;
        }
        let attribution = TierAttribution::classify(total, hit, loading, loading_source);
        record_cache_tier_block_requests(total, attribution);
    }

    /// Priority fallback: RDMA -> SSD. Returns `None` when neither source has blocks.
    async fn try_backing_fetch(
        &self,
        request_id: &str,
        namespace: &str,
        remaining: &[BlockKey],
    ) -> Option<BackingFetch> {
        if remaining.is_empty() {
            return None;
        }

        if let Some(result) = self.try_rdma_fetch(request_id, namespace, remaining).await {
            return Some(result);
        }

        self.try_ssd_fetch(remaining)
    }

    fn try_ssd_fetch(&self, remaining: &[BlockKey]) -> Option<BackingFetch> {
        let ssd = self.ssd_store.as_ref()?;
        let check_keys = self.limit_ssd_prefetch(remaining)?;

        let (found, rx) = ssd.submit_prefix(check_keys);
        if found == 0 {
            return None;
        }

        Some(BackingFetch {
            found,
            rx,
            source: BackingSource::Ssd,
        })
    }

    async fn try_rdma_fetch(
        &self,
        request_id: &str,
        namespace: &str,
        remaining: &[BlockKey],
    ) -> Option<BackingFetch> {
        let rdma = self.rdma_fetch.as_ref()?;

        if self.state.lock().failed_remote.contains_key(request_id) {
            return None;
        }

        let hashes: Vec<Vec<u8>> = remaining.iter().map(|k| k.hash.clone()).collect();
        let (node, found) = rdma.query_prefix(namespace, &hashes).await?;

        let rx = rdma.fetch_blocks(&node, request_id, namespace, hashes[..found].to_vec());

        Some(BackingFetch {
            found,
            rx,
            source: BackingSource::Rdma,
        })
    }

    /// Trim keys to fit inflight capacity, report skipped count to metrics.
    fn limit_ssd_prefetch(&self, remaining: &[BlockKey]) -> Option<Vec<BlockKey>> {
        let available = {
            let state = self.state.lock();
            self.max_prefetch_blocks
                .saturating_sub(state.inflight_count)
        };

        if available == 0 {
            core_metrics()
                .ssd_prefetch_backpressure_blocks
                .add(remaining.len() as u64, &[]);
            return None;
        }

        let check_limit = remaining.len().min(available);
        let skipped = remaining.len() - check_limit;
        if skipped > 0 {
            core_metrics()
                .ssd_prefetch_backpressure_blocks
                .add(skipped as u64, &[]);
        }

        Some(remaining[..check_limit].to_vec())
    }

    fn register_backing_fetch(&self, request_id: &str, fetch: BackingFetch) {
        let mut state = self.state.lock();
        state.inflight_count += fetch.found;
        state.active.insert(
            request_id.to_string(),
            BackingFetchEntry {
                blocks_rx: fetch.rx,
                loading_count: fetch.found,
                source: fetch.source,
            },
        );
    }

    /// Sweep `failed_remote` entries older than `max_age`.
    /// Runs under the single `ReservationState` mutex.
    pub(super) fn gc_failed_remote(&self, max_age: std::time::Duration) -> usize {
        let mut state = self.state.lock();
        let failed_before = state.failed_remote.len();
        state.failed_remote.retain(|_, ts| ts.elapsed() < max_age);
        failed_before - state.failed_remote.len()
    }
}

enum PollResult {
    StillLoading,
    Completed,
}
