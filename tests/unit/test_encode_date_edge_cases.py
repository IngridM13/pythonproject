from datetime import date

import pytest
import torch

from hdc.binary_hdc import HyperDimensionalComputingBinary
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar

DIM = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------

class TestEncodeDateBinary:

    def setup_method(self):
        self.hdc = HyperDimensionalComputingBinary(dim=DIM, seed=SEED)

    def test_single_date_shape_and_values(self):
        hv = self.hdc.encode_date_binary(date(1990, 5, 15))
        assert hv.shape == (DIM,)
        assert torch.all((hv == 0) | (hv == 1))

    def test_date_before_range_does_not_raise(self):
        hv = self.hdc.encode_date_binary(date(1850, 1, 1))
        assert hv.shape == (DIM,)
        assert torch.all((hv == 0) | (hv == 1))

    def test_date_after_range_does_not_raise(self):
        hv = self.hdc.encode_date_binary(date(2200, 12, 31))
        assert hv.shape == (DIM,)
        assert torch.all((hv == 0) | (hv == 1))

    def test_date_before_range_clamped_to_boundary(self):
        """A date before 1900 must clamp to the same vector as 1900-01-01."""
        hv_old = self.hdc.encode_date_binary(date(1849, 1, 1))
        hv_min = self.hdc.encode_date_binary(date(1900, 1, 1))
        assert torch.equal(hv_old, hv_min)

    def test_empty_list_returns_correct_shape(self):
        hv = self.hdc.encode_date_binary([])
        assert hv.shape == (0, DIM)

    def test_list_of_dates_shape(self):
        dates = [date(1990, 1, 1), date(1985, 6, 15), date(2000, 12, 31)]
        hv = self.hdc.encode_date_binary(dates)
        assert hv.shape == (3, DIM)
        assert torch.all((hv == 0) | (hv == 1))

    def test_non_date_raises_type_error(self):
        with pytest.raises(TypeError):
            self.hdc.encode_date_binary("1990-01-01")

    def test_list_with_non_date_raises_type_error(self):
        with pytest.raises(TypeError):
            self.hdc.encode_date_binary([date(1990, 1, 1), "not a date"])


# ---------------------------------------------------------------------------
# Bipolar
# ---------------------------------------------------------------------------

class TestEncodeDateBipolar:

    def setup_method(self):
        self.hdc = HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)

    def test_single_date_shape_and_values(self):
        hv = self.hdc.encode_date_bipolar(date(1990, 5, 15))
        assert hv.shape == (DIM,)
        assert torch.all((hv == 1) | (hv == -1))

    def test_none_returns_all_ones(self):
        hv = self.hdc.encode_date_bipolar(None)
        assert hv.shape == (DIM,)
        assert torch.all(hv == 1)

    def test_date_before_range_does_not_raise(self):
        hv = self.hdc.encode_date_bipolar(date(1850, 1, 1))
        assert hv.shape == (DIM,)
        assert torch.all((hv == 1) | (hv == -1))

    def test_date_after_range_does_not_raise(self):
        hv = self.hdc.encode_date_bipolar(date(2200, 12, 31))
        assert hv.shape == (DIM,)
        assert torch.all((hv == 1) | (hv == -1))

    def test_date_before_range_clamped_to_boundary(self):
        """A date before 1900 must clamp to the same vector as 1900-01-01."""
        hv_old = self.hdc.encode_date_bipolar(date(1849, 1, 1))
        hv_min = self.hdc.encode_date_bipolar(date(1900, 1, 1))
        assert torch.equal(hv_old, hv_min)

    def test_empty_list_returns_correct_shape(self):
        hv = self.hdc.encode_date_bipolar([])
        assert hv.shape == (0, DIM)

    def test_list_with_none_does_not_raise(self):
        hv = self.hdc.encode_date_bipolar([None, date(1990, 1, 1)])
        assert hv.shape == (2, DIM)
        assert torch.all((hv == 1) | (hv == -1))

    def test_list_of_dates_shape(self):
        dates = [date(1990, 1, 1), date(1985, 6, 15), date(2000, 12, 31)]
        hv = self.hdc.encode_date_bipolar(dates)
        assert hv.shape == (3, DIM)
        assert torch.all((hv == 1) | (hv == -1))
