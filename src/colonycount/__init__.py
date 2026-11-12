"""Offline bacterial colony counter for agar plate photos."""

from importlib.metadata import version

from colonycount.detect import Detection, Result, count

__version__ = version("colonycount")
__all__ = ["Detection", "Result", "count"]
