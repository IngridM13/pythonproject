import pytest
import torch
from hdc.hdc_common_operations import flip_inplace


def test_flip_inplace_bipolar_inverts_sign():
    v = torch.tensor([1, -1, 1, -1], dtype=torch.int8)
    result = flip_inplace(v, 0)
    assert torch.equal(result, torch.tensor([-1, -1, 1, -1], dtype=torch.int8))


def test_flip_inplace_modifies_inplace():
    v = torch.tensor([1, -1, 1], dtype=torch.int8)
    result = flip_inplace(v, 1)
    assert result is v


def test_flip_inplace_boundary_indices():
    v = torch.tensor([1, -1, 1, -1, 1], dtype=torch.int8)
    flip_inplace(v, 0)
    flip_inplace(v, 4)
    assert v[0].item() == -1
    assert v[4].item() == -1


def test_flip_inplace_binary_vector_raises():
    v = torch.tensor([0, 1, 0, 1], dtype=torch.int8)
    with pytest.raises(ValueError, match="bipolar vector"):
        flip_inplace(v, 0)


def test_flip_inplace_mixed_values_raises():
    v = torch.tensor([1, -1, 2], dtype=torch.int8)
    with pytest.raises(ValueError, match="bipolar vector"):
        flip_inplace(v, 0)


def test_flip_inplace_converts_list_to_tensor():
    v = [1, -1, 1]
    result = flip_inplace(v, 2)
    assert isinstance(result, torch.Tensor)
    assert result[2].item() == -1
