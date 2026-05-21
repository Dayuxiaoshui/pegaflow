"""Worker-side logic for the experimental P/D connector."""

from __future__ import annotations

from typing import Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.chunk_tracker import ChunkTracker
from pegaflow.pd_connector.layout import (
    FlashAttnHndLayout,
    LayerBlockSlices,
    unique_blocks_from_slot_mapping,
)
from pegaflow.pd_connector.metadata import (
    PdConnectorMetadata,
    PdHandshake,
    PdPrefillRequest,
    PushReqMeta,
    WaitReqMeta,
    flatten_block_ids,
)
from pegaflow.pd_connector.oob import InMemoryOobPort
from pegaflow.pd_connector.rdma import NoopRdmaPort, RdmaPort

logger = get_connector_logger()


class PdWorkerConnector:
    def __init__(
        self,
        vllm_config: Any,
        rdma: RdmaPort | None = None,
        oob: InMemoryOobPort | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.rdma = rdma or NoopRdmaPort()
        self.oob = oob or InMemoryOobPort()
        self.engine_id = (
            getattr(getattr(vllm_config, "kv_transfer_config", None), "engine_id", None) or ""
        )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        self.tp_rank = int(getattr(parallel_config, "tensor_parallel_rank", 0) or 0)
        self.tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        self.layouts: dict[str, FlashAttnHndLayout] = {}
        self.layer_names: list[str] = []
        self._wait_reqs: dict[str, WaitReqMeta] = {}
        self._push_reqs: dict[str, PushReqMeta] = {}
        self._tracker = ChunkTracker()
        self._failed_blocks: set[int] = set()

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.layouts = {
            layer_name: FlashAttnHndLayout.from_tensor(layer_name, tensor)
            for layer_name, tensor in kv_caches.items()
        }
        self.layer_names = list(kv_caches.keys())
        self.rdma.register_local_layers(
            tuple(
                self.layouts[layer_name].remote_layout(layer_idx)
                for layer_idx, layer_name in enumerate(self.layer_names)
            )
        )
        logger.info(
            "[PdConnector] registered %d FlashAttention HND KV cache layers",
            len(self.layouts),
        )

    def start_load_kv(
        self,
        metadata: PdConnectorMetadata,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        for req_id, req in metadata.reqs_to_wait.items():
            self._wait_reqs[req_id] = req
            handshake = self._build_handshake(req_id, flatten_block_ids(req.local_block_ids))
            self.oob.publish_prefill_request(
                PdPrefillRequest(
                    request_id=req.remote_request_id,
                    prompt_token_ids=req.prompt_token_ids,
                    producer_kv_transfer_params={
                        "do_remote_prefill_sender": True,
                        "target_engine_id": self.engine_id,
                        "target_request_id": req_id,
                    },
                    handshake=handshake,
                )
            )
            self.rdma.register_remote(req_id, handshake)

        for req_id, req in metadata.reqs_to_push.items():
            self._push_reqs[req_id] = req
            self._tracker.add_request(req_id)
            prefill_request = self.oob.get_prefill_request(req.target_request_id)
            handshake = prefill_request.handshake if prefill_request is not None else None
            self.rdma.register_remote(req_id, handshake)

        for req_id in metadata.reqs_to_release:
            self._wait_reqs.pop(req_id, None)
            self._push_reqs.pop(req_id, None)
            self._tracker.remove(req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert layer_name in self.layouts, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )
        for req_id in list(self._wait_reqs):
            self.rdma.wait_done(req_id)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        layout = self.layouts.get(layer_name)
        assert layout is not None, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )
        # Re-assert the runtime tensor. CUDA graph or backend changes must not
        # silently swap in a different layout.
        runtime_layout = FlashAttnHndLayout.from_tensor(layer_name, kv_layer)
        assert runtime_layout.shape == layout.shape, (
            f"PdConnector KV shape changed for {layer_name}: "
            f"registered={layout.shape} runtime={runtime_layout.shape}"
        )

        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            return
        touched_blocks = unique_blocks_from_slot_mapping(slot_mapping, layout.block_size)
        if not touched_blocks:
            return

        layer_idx = self._layer_idx(layer_name)
        is_last_layer = layer_idx == len(self.layer_names) - 1
        for req_id, req in list(self._push_reqs.items()):
            req_blocks = flatten_block_ids(req.local_block_ids)
            selected = sorted(touched_blocks & req_blocks)
            if not selected:
                continue
            block_slices: list[LayerBlockSlices] = [
                layout.block_slices(block_id) for block_id in selected
            ]
            self.rdma.push_layer(req_id, layer_idx, block_slices)
            self._tracker.mark_layer_pushed(req_id, layer_idx)
            self._tracker.mark_blocks_pushed(req_id, set(selected))
            if is_last_layer and self._tracker.has_pushed_all_blocks(req_id, req_blocks):
                self.rdma.push_done(req_id)
                self._tracker.mark_done(req_id)

    def wait_for_save(self) -> None:
        return None

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        finished_sending = self.rdma.pop_finished_sending()
        finished_recving = self.rdma.pop_finished_recving()
        for req_id in finished_sending:
            self._push_reqs.pop(req_id, None)
            self._tracker.remove(req_id)
        for req_id in finished_recving:
            self._wait_reqs.pop(req_id, None)
        return finished_sending or None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = self._failed_blocks
        self._failed_blocks = set()
        return failed

    def shutdown(self) -> None:
        self._wait_reqs.clear()
        self._push_reqs.clear()

    def _layer_idx(self, layer_name: str) -> int:
        try:
            return self.layer_names.index(layer_name)
        except ValueError as exc:
            raise AssertionError(f"unknown layer {layer_name}") from exc

    def _build_handshake(self, req_id: str, block_ids: set[int]) -> PdHandshake:
        return PdHandshake(
            request_id=req_id,
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=next(iter(self.layouts.values())).block_size,
            kv_layout="HND",
            layers=tuple(
                self.layouts[layer_name].remote_layout(layer_idx, block_ids)
                for layer_idx, layer_name in enumerate(self.layer_names)
            ),
        )
