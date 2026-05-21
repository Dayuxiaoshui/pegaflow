"""Scheduler-side logic for the experimental P/D connector."""

from __future__ import annotations

from typing import Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.metadata import (
    PdConnectorMetadata,
    PushReqMeta,
    RemoteEndpoint,
    WaitReqMeta,
    handshake_from_dict,
    normalize_block_ids,
)

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
    def __init__(self, vllm_config: Any) -> None:
        self.vllm_config = vllm_config
        self.engine_id = getattr(vllm_config.kv_transfer_config, "engine_id", None) or ""
        self._reqs_to_wait: dict[str, WaitReqMeta] = {}
        self._reqs_to_push: dict[str, PushReqMeta] = {}
        self._reqs_to_release: set[str] = set()
        self._active_wait_reqs: set[str] = set()
        self._completed_wait_reqs: set[str] = set()
        self._pending_producer_reqs: set[str] = set()

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
            self._active_wait_reqs.add(req_id)
            self._reqs_to_wait[req_id] = WaitReqMeta(
                local_block_ids=local_block_ids,
                remote=RemoteEndpoint(
                    engine_id=str(
                        params.get("remote_engine_id") or params.get("prefill_engine_id") or ""
                    ),
                    host=params.get("remote_host"),
                    port=params.get("remote_port"),
                    tp_size=int(params.get("tp_size", 1)),
                    done_endpoint=params.get("done_endpoint"),
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
            logger.info(
                "[PdConnector] scheduler wait req=%s blocks=%d remote_req=%s done_req=%s prefill_url=%s done_endpoint=%s",
                req_id,
                _count(local_block_ids),
                self._reqs_to_wait[req_id].remote_request_id,
                self._reqs_to_wait[req_id].done_request_id,
                self._reqs_to_wait[req_id].prefill_url or "<oob>",
                self._reqs_to_wait[req_id].remote.done_endpoint or "<rdma>",
            )
            return

        if params.get("do_remote_prefill_sender") or params.get("pd_push_producer"):
            self._reqs_to_push[req_id] = PushReqMeta(
                local_block_ids=local_block_ids,
                target=RemoteEndpoint(
                    engine_id=str(
                        params.get("target_engine_id") or params.get("decode_engine_id") or ""
                    ),
                    host=params.get("target_host"),
                    port=params.get("target_port"),
                    tp_size=int(params.get("tp_size", 1)),
                    done_endpoint=params.get("done_endpoint"),
                ),
                target_request_id=str(params.get("target_request_id") or req_id),
                num_prompt_tokens=_num_prompt_tokens(request),
                handshake=handshake_from_dict(params.get("pd_handshake")),
            )
            self._pending_producer_reqs.add(req_id)
            logger.info(
                "[PdConnector] scheduler push req=%s blocks=%d target_req=%s done_endpoint=%s",
                req_id,
                _count(local_block_ids),
                self._reqs_to_push[req_id].target_request_id,
                self._reqs_to_push[req_id].target.done_endpoint or "<rdma>",
            )

    def build_connector_meta(self, scheduler_output: Any) -> PdConnectorMetadata:
        meta = PdConnectorMetadata(
            reqs_to_wait=self._reqs_to_wait,
            reqs_to_push=self._reqs_to_push,
            reqs_to_release=self._reqs_to_release,
        )
        logger.debug(
            "[PdConnector] scheduler build_connector_meta scheduled_tokens=%s wait=%s push=%s release=%s active_wait=%s completed_wait=%s pending_push=%s",
            getattr(scheduler_output, "total_num_scheduled_tokens", "<unknown>"),
            sorted(self._reqs_to_wait),
            sorted(self._reqs_to_push),
            sorted(self._reqs_to_release),
            sorted(self._active_wait_reqs),
            sorted(self._completed_wait_reqs),
            sorted(self._pending_producer_reqs),
        )
        self._reqs_to_wait = {}
        self._reqs_to_push = {}
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
            logger.info("[PdConnector] scheduler finished sending req=%s", req_id)
        for req_id in connector_output.finished_recving or ():
            self._active_wait_reqs.discard(req_id)
            self._completed_wait_reqs.add(req_id)
            logger.info("[PdConnector] scheduler finished recving req=%s", req_id)
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
            self._completed_wait_reqs.discard(req_id)
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


def _count(block_ids: tuple[list[int], ...]) -> int:
    return sum(len(group) for group in block_ids)


def _has_blocks(block_ids: Any) -> bool:
    return any(len(group) > 0 for group in normalize_block_ids(block_ids))
