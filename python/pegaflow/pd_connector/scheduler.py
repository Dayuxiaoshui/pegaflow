"""Scheduler-side logic for the experimental P/D connector."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.metadata import (
    PdConnectorMetadata,
    PdWorkerMetadata,
    PrefillDispatch,
    PushReqMeta,
    RemoteEndpoint,
    WaitReqMeta,
    handshake_from_dict,
    handshakes_from_dicts,
    normalize_block_ids,
)
from pegaflow.pd_connector.prefill import AsyncPrefillSender, prefill_task_from_dispatch

logger = get_connector_logger()


def _request_id(request: Any) -> str:
    return str(request.request_id)


def _kv_params(request: Any) -> dict[str, Any]:
    return getattr(request, "kv_transfer_params", None) or {}


def _num_prompt_tokens(request: Any) -> int:
    if hasattr(request, "num_prompt_tokens"):
        return int(request.num_prompt_tokens)
    token_ids = getattr(request, "prompt_token_ids", None) or []
    return len(token_ids)


def _prompt_token_ids(request: Any) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in (getattr(request, "prompt_token_ids", None) or ()))


class PdSchedulerConnector:
    def __init__(self, vllm_config: Any, prefill_sender: Any | None = None) -> None:
        self.vllm_config = vllm_config
        self.engine_id = getattr(vllm_config.kv_transfer_config, "engine_id", None) or ""
        self._reqs_to_wait: dict[str, WaitReqMeta] = {}
        self._reqs_to_push: dict[str, PushReqMeta] = {}
        self._prefill_dispatches: dict[str, PrefillDispatch] = {}
        self._reqs_to_release: set[str] = set()
        self._active_wait_reqs: set[str] = set()
        self._active_wait_meta: dict[str, WaitReqMeta] = {}
        self._wait_handshakes: dict[str, dict[int, Any]] = {}
        self._dispatched_prefills: set[str] = set()
        self._completed_wait_reqs: set[str] = set()
        self._pending_producer_reqs: set[str] = set()
        self._active_push_meta: dict[str, PushReqMeta] = {}
        self._wait_alloc_ts_ns: dict[str, int] = {}
        self._wait_finished_ts_ns: dict[str, int] = {}
        self._prefill_sender = prefill_sender or AsyncPrefillSender()

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        params = _kv_params(request)
        if params.get("do_remote_prefill"):
            if _request_id(request) in self._completed_wait_reqs:
                logger.debug(
                    "[PdConnector] scheduler get_num_new_matched_tokens req=%s already completed",
                    _request_id(request),
                )
                return 0, False
            count = _num_prompt_tokens(request) - num_computed_tokens
            logger.debug(
                "[PdConnector] scheduler get_num_new_matched_tokens req=%s prompt=%d computed=%d count=%d async=%s active=%d completed=%d",
                _request_id(request),
                _num_prompt_tokens(request),
                num_computed_tokens,
                count,
                count > 0,
                len(self._active_wait_reqs),
                len(self._completed_wait_reqs),
            )
            if count > 0:
                return count, True
        else:
            logger.debug(
                "[PdConnector] scheduler get_num_new_matched_tokens req=%s remote_prefill=false computed=%d",
                _request_id(request),
                num_computed_tokens,
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        params = _kv_params(request)
        req_id = _request_id(request)
        local_block_ids = normalize_block_ids(blocks)

        if params.get("do_remote_prefill"):
            if req_id in self._active_wait_reqs or req_id in self._completed_wait_reqs:
                logger.debug(
                    "[PdConnector] scheduler wait req=%s already tracked active=%s completed=%s",
                    req_id,
                    req_id in self._active_wait_reqs,
                    req_id in self._completed_wait_reqs,
                )
                return
            now_ns = time.time_ns()
            self._active_wait_reqs.add(req_id)
            wait_req = WaitReqMeta(
                local_block_ids=local_block_ids,
                remote=RemoteEndpoint(
                    engine_id=str(
                        params.get("remote_engine_id") or params.get("prefill_engine_id") or ""
                    ),
                    host=params.get("remote_host"),
                    port=params.get("remote_port"),
                    tp_size=int(params.get("tp_size", 1)),
                ),
                remote_request_id=str(params.get("remote_request_id") or req_id),
                done_request_id=str(
                    params.get("done_request_id") or params.get("decode_request_id") or req_id
                ),
                num_prompt_tokens=_num_prompt_tokens(request),
                prompt_token_ids=_prompt_token_ids(request),
                model=str(params.get("model") or getattr(request, "model", "") or ""),
                prefill_url=params.get("prefill_url"),
                prefill_max_tokens=int(params.get("prefill_max_tokens", 1)),
            )
            self._reqs_to_wait[req_id] = wait_req
            self._active_wait_meta[req_id] = wait_req
            self._wait_alloc_ts_ns[req_id] = now_ns
            logger.info(
                "[PdConnector] scheduler wait req=%s blocks=%d remote_req=%s done_req=%s prefill_url=%s prompt_tokens=%d external_tokens=%d ts_ns=%d",
                req_id,
                _count(local_block_ids),
                self._reqs_to_wait[req_id].remote_request_id,
                self._reqs_to_wait[req_id].done_request_id,
                self._reqs_to_wait[req_id].prefill_url or "<oob>",
                wait_req.num_prompt_tokens,
                num_external_tokens,
                now_ns,
            )
            return

        if params.get("do_remote_prefill_sender") or params.get("pd_push_producer"):
            now_ns = time.time_ns()
            push_req = PushReqMeta(
                local_block_ids=local_block_ids,
                target=RemoteEndpoint(
                    engine_id=str(
                        params.get("target_engine_id") or params.get("decode_engine_id") or ""
                    ),
                    host=params.get("target_host"),
                    port=params.get("target_port"),
                    tp_size=int(params.get("tp_size", 1)),
                ),
                target_request_id=str(params.get("target_request_id") or req_id),
                num_prompt_tokens=_num_prompt_tokens(request),
                handshake=handshake_from_dict(params.get("pd_handshake")),
                handshakes=handshakes_from_dicts(params.get("pd_handshakes")),
            )
            self._reqs_to_push[req_id] = push_req
            self._active_push_meta[req_id] = push_req
            self._pending_producer_reqs.add(req_id)
            logger.info(
                "[PdConnector] scheduler push req=%s blocks=%d target_req=%s prompt_tokens=%d handshakes=%d ts_ns=%d",
                req_id,
                _count(local_block_ids),
                self._reqs_to_push[req_id].target_request_id,
                push_req.num_prompt_tokens,
                len(push_req.handshakes)
                if push_req.handshakes
                else int(push_req.handshake is not None),
                now_ns,
            )

    def build_connector_meta(self, scheduler_output: Any) -> PdConnectorMetadata:
        self._add_cached_producer_chunks(scheduler_output)
        meta = PdConnectorMetadata(
            reqs_to_wait=self._reqs_to_wait,
            reqs_to_push=self._reqs_to_push,
            prefill_dispatches=self._prefill_dispatches,
            reqs_to_release=self._reqs_to_release,
        )
        logger.debug(
            "[PdConnector] scheduler build_connector_meta scheduled_tokens=%s wait=%s push=%s dispatch=%s release=%s active_wait=%s completed_wait=%s pending_push=%s",
            getattr(scheduler_output, "total_num_scheduled_tokens", "<unknown>"),
            sorted(self._reqs_to_wait),
            sorted(self._reqs_to_push),
            sorted(self._prefill_dispatches),
            sorted(self._reqs_to_release),
            sorted(self._active_wait_reqs),
            sorted(self._completed_wait_reqs),
            sorted(self._pending_producer_reqs),
        )
        self._reqs_to_wait = {}
        self._reqs_to_push = {}
        self._prefill_dispatches = {}
        self._reqs_to_release = set()
        return meta

    def update_connector_output(self, connector_output: Any) -> None:
        logger.debug(
            "[PdConnector] scheduler update_connector_output sending=%s recving=%s before_active=%s before_completed=%s before_pending=%s",
            sorted(connector_output.finished_sending or ()),
            sorted(connector_output.finished_recving or ()),
            sorted(self._active_wait_reqs),
            sorted(self._completed_wait_reqs),
            sorted(self._pending_producer_reqs),
        )
        for req_id in connector_output.finished_sending or ():
            self._pending_producer_reqs.discard(req_id)
            self._active_push_meta.pop(req_id, None)
            logger.info("[PdConnector] scheduler finished sending req=%s", req_id)
        for req_id in connector_output.finished_recving or ():
            now_ns = time.time_ns()
            self._active_wait_reqs.discard(req_id)
            self._active_wait_meta.pop(req_id, None)
            self._wait_handshakes.pop(req_id, None)
            self._completed_wait_reqs.add(req_id)
            self._wait_finished_ts_ns[req_id] = now_ns
            alloc_ts_ns = self._wait_alloc_ts_ns.get(req_id)
            wait_ms = _elapsed_ms(alloc_ts_ns, now_ns)
            logger.info(
                "[PdConnector] scheduler finished recving req=%s wait_ms=%s ts_ns=%d",
                req_id,
                _fmt_ms(wait_ms),
                now_ns,
            )
        worker_meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(worker_meta, PdWorkerMetadata):
            self._ingest_worker_handshakes(worker_meta)
        logger.debug(
            "[PdConnector] scheduler update_connector_output after_active=%s after_completed=%s after_pending=%s",
            sorted(self._active_wait_reqs),
            sorted(self._completed_wait_reqs),
            sorted(self._pending_producer_reqs),
        )

    def request_finished(
        self,
        request: Any,
        block_ids: Any,
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = _request_id(request)
        params = _kv_params(request)
        is_producer = bool(params.get("do_remote_prefill_sender") or params.get("pd_push_producer"))
        if params.get("do_remote_prefill") or is_producer:
            self._reqs_to_release.add(req_id)
            self._active_wait_reqs.discard(req_id)
            self._active_wait_meta.pop(req_id, None)
            self._wait_handshakes.pop(req_id, None)
            self._dispatched_prefills.discard(req_id)
            self._completed_wait_reqs.discard(req_id)
            self._wait_alloc_ts_ns.pop(req_id, None)
            self._wait_finished_ts_ns.pop(req_id, None)
            logger.debug(
                "[PdConnector] scheduler request_finished req=%s producer=%s queued_release=%d",
                req_id,
                is_producer,
                len(self._reqs_to_release),
            )
        if is_producer and _has_blocks(block_ids):
            self._pending_producer_reqs.add(req_id)
            logger.debug(
                "[PdConnector] scheduler request_finished delay producer req=%s pending=%d",
                req_id,
                len(self._pending_producer_reqs),
            )
            return True, None
        return False, None

    def _add_cached_producer_chunks(self, scheduler_output: Any) -> None:
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None:
            return
        req_ids = tuple(str(req_id) for req_id in (getattr(cached, "req_ids", None) or ()))
        new_block_ids = tuple(getattr(cached, "new_block_ids", None) or ())
        for req_id, blocks in zip(req_ids, new_block_ids, strict=False):
            if req_id in self._reqs_to_push:
                continue
            push_req = self._active_push_meta.get(req_id)
            if push_req is None:
                continue
            local_block_ids = normalize_block_ids(blocks)
            if not _has_blocks(local_block_ids):
                continue
            self._reqs_to_push[req_id] = replace(push_req, local_block_ids=local_block_ids)
            self._pending_producer_reqs.add(req_id)
            logger.info(
                "[PdConnector] scheduler push cached req=%s blocks=%d target_req=%s ts_ns=%d",
                req_id,
                _count(local_block_ids),
                push_req.target_request_id,
                time.time_ns(),
            )

    def _ingest_worker_handshakes(self, worker_meta: PdWorkerMetadata) -> None:
        for req_id, by_rank in worker_meta.handshakes.items():
            wait_req = self._active_wait_meta.get(req_id)
            if wait_req is None or not wait_req.prefill_url:
                continue
            merged = self._wait_handshakes.setdefault(req_id, {})
            merged.update(by_rank)
            expected = next(iter(merged.values())).tp_size if merged else 1
            if len(merged) < expected:
                logger.info(
                    "[PdConnector] scheduler collected handshakes req=%s ranks=%s/%d ts_ns=%d",
                    req_id,
                    sorted(merged),
                    expected,
                    time.time_ns(),
                )
                continue
            if req_id in self._prefill_dispatches or req_id in self._dispatched_prefills:
                continue
            handshakes = tuple(merged[rank] for rank in sorted(merged))
            dispatch = PrefillDispatch(
                request_id=wait_req.remote_request_id,
                prefill_url=wait_req.prefill_url,
                model=wait_req.model,
                prompt_token_ids=wait_req.prompt_token_ids,
                max_tokens=wait_req.prefill_max_tokens,
                target_engine_id=self.engine_id,
                target_request_id=wait_req.done_request_id,
                handshakes=handshakes,
            )
            self._dispatched_prefills.add(req_id)
            serialize_start_ns = time.time_ns()
            task = prefill_task_from_dispatch(dispatch)
            serialize_done_ns = time.time_ns()
            self._prefill_sender.submit(task)
            submit_ts_ns = time.time_ns()
            alloc_ts_ns = self._wait_alloc_ts_ns.get(req_id)
            dispatch_ms = _elapsed_ms(alloc_ts_ns, submit_ts_ns)
            logger.info(
                "[PdConnector] scheduler submitted prefill dispatch req=%s remote_req=%s ranks=%s dispatch_ms=%s serialize_ms=%.3f ts_ns=%d",
                req_id,
                wait_req.remote_request_id,
                [handshake.tp_rank for handshake in handshakes],
                _fmt_ms(dispatch_ms),
                (serialize_done_ns - serialize_start_ns) / 1_000_000,
                submit_ts_ns,
            )

    def shutdown(self) -> None:
        close = getattr(self._prefill_sender, "close", None)
        if close is not None:
            close()


def _count(block_ids: tuple[list[int], ...]) -> int:
    return sum(len(group) for group in block_ids)


def _has_blocks(block_ids: Any) -> bool:
    return any(len(group) > 0 for group in normalize_block_ids(block_ids))


def _elapsed_ms(start_ns: int | None, end_ns: int) -> float | None:
    if start_ns is None:
        return None
    return (end_ns - start_ns) / 1_000_000


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}"
