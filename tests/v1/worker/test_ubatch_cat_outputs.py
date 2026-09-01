# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for concatenating per-ubatch model outputs."""

import torch

from vllm.v1.worker.gpu_ubatch_wrapper import _cat_ubatch_outputs


def _hs(value: float, rows: int = 2, cols: int = 4) -> torch.Tensor:
    return torch.full((rows, cols), value)


def test_plain_tensor_outputs():
    """Most models return a single hidden-states tensor per ubatch."""
    out = _cat_ubatch_outputs([_hs(0.0), _hs(1.0)])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 4)
    assert torch.equal(out, torch.cat([_hs(0.0), _hs(1.0)], dim=0))


def test_tuple_aux_outputs():
    """EAGLE3-style targets return a tuple of (hidden_states, aux)."""
    out = _cat_ubatch_outputs(
        [(_hs(0.0), _hs(10.0)), (_hs(1.0), _hs(11.0))]
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert all(part.shape == (4, 4) for part in out)


def test_list_aux_outputs():
    """A list container must be handled like a tuple, and preserved."""
    out = _cat_ubatch_outputs([[_hs(0.0), _hs(10.0)], [_hs(1.0), _hs(11.0)]])
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(part.shape == (4, 4) for part in out)


def test_nested_aux_outputs():
    """DFlash collects per-layer aux states: [hidden, [aux_0, aux_1]]."""
    out = _cat_ubatch_outputs(
        [
            [_hs(0.0), [_hs(10.0), _hs(20.0)]],
            [_hs(1.0), [_hs(11.0), _hs(21.0)]],
        ]
    )
    assert isinstance(out, list)
    assert isinstance(out[0], torch.Tensor)
    assert out[0].shape == (4, 4)
    assert isinstance(out[1], list)
    assert len(out[1]) == 2
    assert all(aux.shape == (4, 4) for aux in out[1])


def test_single_ubatch_roundtrip():
    """A single ubatch must return the same structure the model produced."""
    out = _cat_ubatch_outputs([[_hs(0.0), [_hs(10.0)]]])
    assert isinstance(out, list)
    assert isinstance(out[1], list)
    assert out[0].shape == (2, 4)
