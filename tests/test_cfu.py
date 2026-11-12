"""cfu-calculation spec scenarios."""

import pytest

from colonycount.cfu import cfu_per_ml, range_flag


def test_standard_calculation():
    assert cfu_per_ml(150, 1e-4, 0.1) == pytest.approx(1.5e7)


@pytest.mark.parametrize(
    "dilution, volume", [(0, 0.1), (-1e-4, 0.1), (None, 0.1), (1e-4, 0), (1e-4, -1), (1e-4, None)]
)
def test_invalid_parameters(dilution, volume):
    with pytest.raises(ValueError, match="must be a positive number"):
        cfu_per_ml(150, dilution, volume)


def test_too_many_colonies():
    assert range_flag(420) == "TNTC"  # the CSV row keeps count 420 next to it (test_cli)
    assert range_flag(29) == "TFTC"
    assert range_flag(30) == range_flag(300) == "OK"


def test_custom_range():
    assert range_flag(270, 25, 250) == "TNTC"
    assert range_flag(27, 25, 250) == "OK"
    with pytest.raises(ValueError):
        range_flag(100, 300, 30)
