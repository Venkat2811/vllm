# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MyelonKVConnector — single-host KV transfer over Myelon SHM rings.

Status: SKELETON (RFC 0031). All abstract methods implement the
KVConnectorBase_V1 contract but defer the actual byte movement to a
`myelon_kv_transport` Python module that does not yet exist. When that
binding lands (PyO3 crate in myelon-playground), this connector picks it
up via lazy import without further changes here.

What this connector targets (the gap RFC 0031 describes):

  GPU↔CPU on a single host          : SimpleCPUOffloadConnector covers it
  Cross-node / persistent           : LMCache, Mooncake, NIXL cover it
  Across-engine within a node, off-GPU : THIS ONE — single-host
                                          disaggregated prefill / decode
                                          via shared memory ring

Why a separate connector instead of patching SimpleCPUOffloadConnector:
SimpleCPUOffloadConnector assumes the same engine process owns both the
GPU and the CPU buffer. Disaggregated PD splits prefill and decode into
two engines on the same host, and KV blocks need to move between them.
The Myelon SHM ring already solves that substrate; this class is the
adapter to vLLM's KVConnectorBase_V1 contract.

Wiring (configured via kv_transfer block in the engine YAML):

    kv_transfer:
      kv_connector: myelon
      kv_connector_extra_config:
        ring_name: vllm_kv          # POSIX SHM segment name
        capacity_blocks: 4096       # blocks per layer (matches Myelon ring depth)
        pin_memory: true            # cudaHostRegister the ring backing store

The `myelon_kv_transport` module exposed by the PyO3 binding must provide:

    class MyelonRingClient:
        def __init__(self, ring_name: str, block_size: int,
                     capacity_blocks: int, role: str): ...
        def publish(self, layer_name: str, kv_layer_tensor) -> None: ...
        def start_consume(self, layer_name: str, dst_buffer) -> None: ...
        def wait_layer(self, layer_name: str) -> None: ...
        def get_finished(self) -> tuple[set, set]: ...
        # ... (matches the methods this connector calls below)

References:
  - RFC 0031 §"Design"  for the connector lifecycle
  - RFC 0031 §"Crate structure" for the corresponding Rust modules
  - vllm/distributed/kv_transfer/kv_connector/v1/base.py:170
        the KVConnectorBase_V1 contract this skeleton fulfils
  - vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py
        closest in spirit (single-host CPU side); read for the
        scheduler/worker split pattern
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Defaults if kv_connector_extra_config does not override.
_DEFAULT_RING_NAME = "vllm_kv"
_DEFAULT_CAPACITY_BLOCKS = 4096
_DEFAULT_PIN_MEMORY = True


class MyelonKVConnectorMetadata(KVConnectorMetadata):
    """Per-step state passed scheduler → worker.

    Mirrors the shape of `ExampleConnectorMetadata`: a list of per-request
    handles that the worker uses to load / save the right ring slots.
    Concrete shape (block-id mapping, monotonic ids) lands when
    `block_table.rs` solidifies on the Rust side. Until then this is an
    empty payload — the worker reads nothing from it.
    """

    def __init__(self) -> None:
        super().__init__()
        # Populated on the scheduler in `update_state_after_alloc` and
        # `request_finished`; consumed on the worker in `start_load_kv` /
        # `save_kv_layer`. RFC 0031 §"Coordination" has the full schema.


def _import_myelon_transport() -> Any:
    """Lazy import of the PyO3 binding.

    Importing at module load time would force every vLLM install to ship
    the binding, which we do not want — the connector should be selectable
    at config time and only blow up if the user actually picks `myelon`.
    Returns the imported module on success; raises a clear ImportError
    on failure.
    """
    try:
        import myelon_kv_transport  # type: ignore[import-not-found]

        return myelon_kv_transport
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "MyelonKVConnector requires the `myelon_kv_transport` Python "
            "module. Build it from the myelon-playground repo:\n"
            "  cd myelon-playground/crates/myelon-kv-connector\n"
            "  uv venv --python 3.12 && source .venv/bin/activate\n"
            "  uv pip install maturin && maturin develop --release\n"
            "Then re-launch vLLM with --kv-connector myelon."
        ) from exc


class MyelonKVConnector(KVConnectorBase_V1):
    """vLLM ↔ Myelon SHM ring KV transport (RFC 0031 skeleton).

    Both scheduler and worker roles are served by the same class. The
    `role` argument selects which side of the abstract surface is active:
    SCHEDULER calls `get_num_new_matched_tokens` etc., WORKER calls
    `start_load_kv` / `save_kv_layer` etc. Role-inappropriate calls return
    no-op defaults rather than raising — this matches how the existing
    SimpleCPUOffloadConnector behaves.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._ring_name: str = str(extra.get("ring_name", _DEFAULT_RING_NAME))
        self._capacity_blocks: int = int(
            extra.get("capacity_blocks", _DEFAULT_CAPACITY_BLOCKS)
        )
        self._pin_memory: bool = bool(extra.get("pin_memory", _DEFAULT_PIN_MEMORY))

        # Will be initialised lazily once the binding is imported. Keep the
        # client out of __init__ so tests that exercise factory wiring
        # without the binding still pass.
        self._client: Any = None

        logger.info(
            "MyelonKVConnector initialised (skeleton): role=%s ring=%s "
            "capacity_blocks=%d pin_memory=%s",
            role.name,
            self._ring_name,
            self._capacity_blocks,
            self._pin_memory,
        )

    # ----- internal -----

    def _ensure_client(self) -> Any:
        """Initialise the Rust ring client on first hot-path call."""
        if self._client is None:
            transport = _import_myelon_transport()
            block_size_bytes = (
                self._kv_cache_config.block_size_bytes
                if self._kv_cache_config is not None
                else 0
            )
            self._client = transport.MyelonRingClient(
                ring_name=self._ring_name,
                block_size=block_size_bytes,
                capacity_blocks=self._capacity_blocks,
                role=self._role.name,
            )
        return self._client

    # ===============================
    # Worker-side (abstract) methods
    # ===============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register vLLM's GPU KV buffers with the transport.

        The transport needs the (layer_name → tensor) map so it can stage
        DMA between the GPU pages and the SHM ring slot. The actual
        registration is a thin pass-through; the binding will call
        `cudaHostRegister` on the ring backing store at construction
        time, not here.
        """
        if self._role != KVConnectorRole.WORKER:
            return
        client = self._ensure_client()
        # TODO(rfc 0031 P2): teach the binding to accept the dict; for now
        # we just record that registration was requested.
        logger.debug("MyelonKVConnector.register_kv_caches: %d layers", len(kv_caches))
        if hasattr(client, "register_kv_caches"):
            client.register_kv_caches(kv_caches)

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        """Begin async loading from ring → GPU for the layers in this step."""
        if self._role != KVConnectorRole.WORKER:
            return
        # TODO(rfc 0031): walk forward_context.layers, kick off async ring →
        # GPU copies via the binding. start_consume returns immediately;
        # wait_for_layer_load below blocks for completion.

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until layer's load has landed in the GPU paged buffer."""
        if self._role != KVConnectorRole.WORKER:
            return
        # TODO(rfc 0031): self._ensure_client().wait_layer(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Publish layer's KV tensor to the SHM ring."""
        if self._role != KVConnectorRole.WORKER:
            return
        # TODO(rfc 0031): self._ensure_client().publish(layer_name, kv_layer)

    def wait_for_save(self) -> None:
        """Block until all in-flight save operations have committed to the ring."""
        if self._role != KVConnectorRole.WORKER:
            return
        # TODO(rfc 0031): self._ensure_client().wait_all_saves()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report request ids whose KV transfers have settled on this worker.

        Returns (sent_ids, received_ids). Default behaviour is to claim
        nothing has finished asynchronously — the connector synchronously
        completes work in `wait_for_save` and `wait_for_layer_load`. A
        future async path (RFC 0031 §"Async transfer queue") will return
        real ids here.
        """
        if self._role != KVConnectorRole.WORKER:
            return None, None
        return None, None

    # ==================================
    # Scheduler-side (abstract) methods
    # ==================================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """How many extra tokens can be skipped because they're already in the ring?

        Returns (num_external_tokens, can_load_async). For the skeleton we
        return (0, False) — the cache lookup against the ring's block
        index lands in P2.
        """
        if self._role != KVConnectorRole.SCHEDULER:
            return 0, False
        # TODO(rfc 0031 §"Coordination"): query the block_table for hits on
        # request.block_hashes; return the matched-prefix length.
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record allocated blocks against the ring's block table."""
        if self._role != KVConnectorRole.SCHEDULER:
            return
        # TODO(rfc 0031): block_table.note_allocation(request, blocks)

    def update_connector_output(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Inject any cross-worker side-effects into the scheduler output.

        For Myelon this is currently a no-op — the worker side owns its
        own ring cursors and doesn't need scheduler-mediated state. NIXL
        and LMCache use this hook for cross-host coordination.
        """
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Drop the request's blocks from the ring's block table."""
        if self._role != KVConnectorRole.SCHEDULER:
            return False, None
        # TODO(rfc 0031): block_table.evict(request.request_id)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain emitted KV cache events for downstream scheduling."""
        return ()
