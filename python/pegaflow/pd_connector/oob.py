"""Out-of-band control-plane placeholder for P/D handshake messages."""

from __future__ import annotations

from pegaflow.pd_connector.metadata import PdHandshake, PdPrefillRequest


class InMemoryOobPort:
    def __init__(self) -> None:
        self._handshakes: dict[str, PdHandshake] = {}
        self._prefill_requests: dict[str, PdPrefillRequest] = {}

    def publish_handshake(self, handshake: PdHandshake) -> None:
        self._handshakes[handshake.request_id] = handshake

    def get_handshake(self, request_id: str) -> PdHandshake | None:
        return self._handshakes.get(request_id)

    def publish_prefill_request(self, request: PdPrefillRequest) -> None:
        self._prefill_requests[request.request_id] = request
        self.publish_handshake(request.handshake)

    def get_prefill_request(self, request_id: str) -> PdPrefillRequest | None:
        return self._prefill_requests.get(request_id)


class OobPort(InMemoryOobPort):
    """Protocol-shaped concrete placeholder until ZMQ is wired in."""
