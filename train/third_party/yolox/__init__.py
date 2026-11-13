"""YOLOX model code vendored from Megvii-BaseDetection/YOLOX (Apache-2.0); see NOTICE.

The model names load lazily so data_augment (numpy + cv2 only) imports without torch."""

import importlib

_MODULES = {
    "YOLOX": ".yolox",
    "YOLOPAFPN": ".yolo_pafpn",
    "YOLOXHead": ".yolo_head",
    "ModelEMA": ".ema",
    "postprocess": ".boxes",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_MODULES[name], __name__), name)
