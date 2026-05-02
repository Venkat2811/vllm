# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

import os

import pytest

from vllm.config import CacheConfig, KVTransferConfig, ParallelConfig, VllmConfig

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "kv_offloading_backend,kv_offloading_size,tp,pp,expected_backend,expected_bytes",
    [
        ("native", 4.0, 1, 1, "OffloadingConnector", 4.0 * (1 << 30)),
        # bytes per rank: 8.0 GiB / (2 * 2) = 2.0 GiB
        ("native", 8.0, 2, 2, "OffloadingConnector", 8.0 * (1 << 30)),
        ("lmcache", 4.0, 1, 1, "LMCacheConnectorV1", 4.0),
        # size per rank: 8.0 GiB / (2 * 2) = 2.0 GiB
        ("lmcache", 8.0, 2, 2, "LMCacheConnectorV1", 2.0),
        ("tensorpuffer", 4.0, 1, 1, "OffloadingConnector", None),
        ("wombatkv", 4.0, 1, 1, "OffloadingConnector", None),
        # When kv_offloading_size is None, offloading is disabled (backend is ignored)
        ("native", None, 1, 1, None, None),
    ],
)
def test_kv_connector(
    kv_offloading_backend, kv_offloading_size, tp, pp, expected_backend, expected_bytes
):
    kv_transfer_config = (
        KVTransferConfig(kv_connector_extra_config={"existing_key": "existing_value"})
        if expected_backend is not None
        else None
    )

    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            kv_offloading_backend=kv_offloading_backend,
            kv_offloading_size=kv_offloading_size,
        ),
        kv_transfer_config=kv_transfer_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            # This unit test checks KV offload config math, not local GPU
            # availability. Put synthetic multi-rank cases across nodes so the
            # test also runs on single-GPU developer machines.
            nnodes=max(1, tp * pp),
        ),
    )

    # No KV transfer config expected
    if expected_backend is None:
        assert vllm_config.kv_transfer_config is expected_backend
        return

    kv_transfer_config = vllm_config.kv_transfer_config
    kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config

    assert kv_transfer_config.kv_connector == expected_backend
    assert kv_transfer_config.kv_role == "kv_both"

    if kv_offloading_backend == "native":
        assert kv_connector_extra_config["cpu_bytes_to_use"] == expected_bytes
        # Existing config should be preserved
        assert kv_connector_extra_config["existing_key"] == "existing_value"
    elif kv_offloading_backend == "lmcache":
        assert kv_connector_extra_config["lmcache.local_cpu"] is True
        assert kv_connector_extra_config["lmcache.max_local_cpu_size"] == expected_bytes
        # Existing config should be replaced
        assert "existing_key" not in kv_connector_extra_config
    elif kv_offloading_backend in ("tensorpuffer", "wombatkv"):
        assert kv_connector_extra_config["spec_name"] == "WombatKVOffloadingSpec"
        assert kv_connector_extra_config["spec_module_path"] == "wombat_kv.vllm.spec"
        assert kv_connector_extra_config["backend"] == "m1"
        assert kv_connector_extra_config["namespace"] == "vllm"
        assert kv_connector_extra_config["restore_on_start"] is True
        # Existing config should be preserved
        assert kv_connector_extra_config["existing_key"] == "existing_value"


def test_kv_offloading_size_only_uses_native_default():
    """Test that setting only kv_offloading_size enables native offloading."""
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            kv_offloading_size=4.0,
            # kv_offloading_backend not set, should default to "native"
        ),
    )

    kv_transfer_config = vllm_config.kv_transfer_config
    kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config
    assert kv_transfer_config.kv_connector == "OffloadingConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_connector_extra_config["cpu_bytes_to_use"] == 4.0 * (1 << 30)


def test_tensorpuffer_sets_deterministic_block_hash_seed(monkeypatch):
    """Durable offload keys must remain stable across vLLM process restarts."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    VllmConfig(
        cache_config=CacheConfig(
            kv_offloading_backend="tensorpuffer",
            kv_offloading_size=4.0,
        )
    )

    assert os.environ["PYTHONHASHSEED"] == "0"


def test_tensorpuffer_preserves_explicit_block_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "1234")

    VllmConfig(
        cache_config=CacheConfig(
            kv_offloading_backend="wombatkv",
            kv_offloading_size=4.0,
        )
    )

    assert os.environ["PYTHONHASHSEED"] == "1234"
