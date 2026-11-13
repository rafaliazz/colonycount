"""YOLOX model code vendored from Megvii-BaseDetection/YOLOX (Apache-2.0); see NOTICE."""

from .boxes import postprocess
from .ema import ModelEMA
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX

__all__ = ["YOLOX", "YOLOPAFPN", "YOLOXHead", "ModelEMA", "postprocess"]
