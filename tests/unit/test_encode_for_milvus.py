import pytest
import torch
from unittest.mock import patch
from configs.settings import HDC_DIM
from encoding_methods.encoding_and_search_milvus import _encode_for_milvus


def test_binary_mode_returns_bytes():
    with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "binary"}):
        hv = torch.randint(0, 2, (HDC_DIM,), dtype=torch.uint8)
        result = _encode_for_milvus(hv)
    assert isinstance(result, bytes)
    assert len(result) == HDC_DIM // 8


def test_float_mode_returns_list_of_floats():
    with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "float"}):
        hv = torch.ones(HDC_DIM, dtype=torch.float32)
        result = _encode_for_milvus(hv)
    assert isinstance(result, list)
    assert len(result) == HDC_DIM
    assert all(isinstance(v, float) for v in result)


def test_binary_mode_wrong_dim_raises():
    with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "binary"}):
        hv = torch.randint(0, 2, (HDC_DIM // 2,), dtype=torch.uint8)
        with pytest.raises(ValueError):
            _encode_for_milvus(hv)


def test_float_mode_wrong_dim_raises():
    with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "float"}):
        hv = torch.ones(HDC_DIM // 2, dtype=torch.float32)
        with pytest.raises(ValueError):
            _encode_for_milvus(hv)


def test_binary_mode_all_ones_gives_0xff_bytes():
    # All +1 bipolar vector → (hv > 0) = all 1s → each byte must be 0xFF
    with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "binary"}):
        hv = torch.ones(HDC_DIM, dtype=torch.int8)
        result = _encode_for_milvus(hv)
    assert all(b == 0xFF for b in result)


def test_binary_mode_all_zeros_gives_0x00_bytes():
    # All 0 vector → (hv > 0) = all 0s → each byte must be 0x00
    with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "binary"}):
        hv = torch.zeros(HDC_DIM, dtype=torch.uint8)
        result = _encode_for_milvus(hv)
    assert all(b == 0x00 for b in result)


def test_float_mode_preserves_values():
    with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "float"}):
        hv = torch.full((HDC_DIM,), -1.0, dtype=torch.float32)
        result = _encode_for_milvus(hv)
    assert all(v == -1.0 for v in result)
