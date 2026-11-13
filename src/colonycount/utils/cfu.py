"""CFU/mL from a plate count, and countable-range flags."""

from colonycount._env import env_tuple


def cfu_per_ml(count: int, dilution: float | None, volume: float | None) -> float:
    """CFU/mL = count / (dilution x volume), e.g. 150 at dilution 1e-4 with 0.1 mL -> 1.5e7."""
    for name, value in (("dilution", dilution), ("volume", volume)):
        if value is None or not value > 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")
    return count / (dilution * volume)


def range_flag(count: int, low: int | None = None, high: int | None = None) -> str:
    """'TFTC' below `low`, 'TNTC' above `high`, 'OK' inside the countable range.

    The range defaults to ``COLONYCOUNT_COUNTABLE_RANGE``.
    """
    if low is None or high is None:
        low, high = env_tuple("COLONYCOUNT_COUNTABLE_RANGE", (30, 300))
    if low > high:
        raise ValueError(f"countable range low ({low}) is above high ({high})")
    return "TFTC" if count < low else "TNTC" if count > high else "OK"
