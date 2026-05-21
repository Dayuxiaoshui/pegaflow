"""Thin RDMA port abstraction used by the P/D connector skeleton."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Protocol

from pegaflow.pd_connector.layout import BlockSlice, LayerBlockSlices
from pegaflow.pd_connector.metadata import LayerRemoteLayout, PdHandshake


class RdmaPort(Protocol):
    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]: ...

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None: ...

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None: ...

    def push_done(self, req_id: str) -> None: ...

    def wait_done(self, req_id: str) -> None: ...

    def mark_done(self, req_id: str) -> None: ...

    def pop_finished_sending(self) -> set[str]: ...

    def pop_finished_recving(self) -> set[str]: ...


class NoopRdmaPort:
    """A non-blocking RDMA stub that records calls and completes immediately."""

    def __init__(self) -> None:
        self.local_layers: tuple[LayerRemoteLayout, ...] = ()
        self.registered: set[str] = set()
        self.remote_handshakes: dict[str, PdHandshake | None] = {}
        self.pushed_layers: dict[str, list[tuple[int, list[LayerBlockSlices]]]] = defaultdict(list)
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        self.local_layers = layers
        return layers

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None:
        self.registered.add(req_id)
        self.remote_handshakes[req_id] = handshake

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.pushed_layers[req_id].append((layer_idx, blocks))

    def push_done(self, req_id: str) -> None:
        self._finished_sending.add(req_id)

    def wait_done(self, req_id: str) -> None:
        return None

    def mark_done(self, req_id: str) -> None:
        self._finished_recving.add(req_id)

    def pop_finished_sending(self) -> set[str]:
        finished = self._finished_sending
        self._finished_sending = set()
        return finished

    def pop_finished_recving(self) -> set[str]:
        finished = self._finished_recving
        self._finished_recving = set()
        return finished


class MockRdmaPort(NoopRdmaPort):
    """Alias for now; later tests can add stricter copy semantics here."""


def _block_slice_to_native(block: BlockSlice) -> dict[str, int]:
    return {
        "block_id": block.block_id,
        "src_offset_bytes": block.src_offset_bytes,
        "bytes": block.bytes,
    }


def _layer_blocks_to_native(blocks: list[LayerBlockSlices]) -> list[dict[str, Any]]:
    return [
        {
            "k": _block_slice_to_native(block.k),
            "v": _block_slice_to_native(block.v),
        }
        for block in blocks
    ]


def _layer_to_native(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "base_addr": layer.base_addr,
        "block_bytes": layer.block_bytes,
        "block_ids": list(layer.block_ids),
        "k_block_addrs": list(layer.k_block_addrs),
        "v_block_addrs": list(layer.v_block_addrs),
        "mr_desc": layer.mr_desc,
    }


def _layer_from_native(layer: LayerRemoteLayout | dict[str, Any]) -> LayerRemoteLayout:
    if isinstance(layer, LayerRemoteLayout):
        return layer
    block_ids = tuple(int(block_id) for block_id in layer["block_ids"])
    k_block_addrs = tuple(int(addr) for addr in layer["k_block_addrs"])
    v_block_addrs = tuple(int(addr) for addr in layer["v_block_addrs"])
    assert len(block_ids) == len(k_block_addrs) == len(v_block_addrs), (
        "native RDMA layer must preserve a one-to-one block_id/K/V address mapping"
    )
    return LayerRemoteLayout(
        layer_name=str(layer["layer_name"]),
        layer_idx=int(layer["layer_idx"]),
        base_addr=int(layer["base_addr"]),
        block_bytes=int(layer["block_bytes"]),
        block_ids=block_ids,
        k_block_addrs=k_block_addrs,
        v_block_addrs=v_block_addrs,
        mr_desc=layer.get("mr_desc"),
    )


def _handshake_to_native(handshake: PdHandshake | None) -> dict[str, Any] | None:
    if handshake is None:
        return None
    data = asdict(handshake)
    data["layers"] = [_layer_to_native(layer) for layer in handshake.layers]
    return data


class RealRdmaPort:
    """Adapter from connector dataclasses to the native PyO3 RDMA engine.

    The native object is intentionally narrow. It owns v2 TransferEngine state,
    memory registration, peer state, and completion polling. This class only
    converts Python connector metadata to stable dictionaries.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        native_layers = [_layer_to_native(layer) for layer in layers]
        registered = self.engine.register_local_layers(native_layers)
        return tuple(_layer_from_native(layer) for layer in registered)

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None:
        self.engine.register_remote(req_id, _handshake_to_native(handshake))

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.engine.push_layer(req_id, layer_idx, _layer_blocks_to_native(blocks))

    def push_done(self, req_id: str) -> None:
        self.engine.push_done(req_id)

    def wait_done(self, req_id: str) -> None:
        wait_done = getattr(self.engine, "wait_done", None)
        if wait_done is None:
            return None
        return wait_done(req_id)

    def mark_done(self, req_id: str) -> None:
        mark_done = getattr(self.engine, "mark_done", None)
        if mark_done is None:
            return None
        return mark_done(req_id)

    def pop_finished_sending(self) -> set[str]:
        return set(self.engine.pop_finished_sending())

    def pop_finished_recving(self) -> set[str]:
        return set(self.engine.pop_finished_recving())
