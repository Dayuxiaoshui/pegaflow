"""Small request progress tracker for chunked prefill push."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestChunks:
    pushed_layers: set[int] = field(default_factory=set)
    pushed_blocks: set[int] = field(default_factory=set)
    done: bool = False


class ChunkTracker:
    def __init__(self) -> None:
        self._requests: dict[str, RequestChunks] = {}

    def add_request(self, req_id: str) -> None:
        self._requests.setdefault(req_id, RequestChunks())

    def mark_layer_pushed(self, req_id: str, layer_idx: int) -> None:
        self.add_request(req_id)
        self._requests[req_id].pushed_layers.add(layer_idx)

    def mark_blocks_pushed(self, req_id: str, block_ids: set[int]) -> None:
        self.add_request(req_id)
        self._requests[req_id].pushed_blocks.update(block_ids)

    def has_pushed_all_blocks(self, req_id: str, block_ids: set[int]) -> bool:
        self.add_request(req_id)
        return block_ids.issubset(self._requests[req_id].pushed_blocks)

    def mark_done(self, req_id: str) -> None:
        self.add_request(req_id)
        self._requests[req_id].done = True

    def remove(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
