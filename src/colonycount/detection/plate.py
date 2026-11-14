"""Plate localisation: Hough circle on a downscaled copy of the photo."""

import cv2
import numpy as np

from colonycount._env import env_float, env_int


def find_plate(img: np.ndarray) -> tuple[float, float, float] | None:
    """Return the plate circle (cx, cy, r) in image pixels, or None if no rim is found.

    The radius is widened by COLONYCOUNT_PLATE_PAD (a fraction), so colonies on the rim stay in.
    """
    search = env_int("COLONYCOUNT_PLATE_SEARCH_SIZE", 512, lo=1)  # longest side of the copy
    scale = min(1.0, search / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = small.max(axis=2)  # brightest channel: a red-lit plate on black has no luma edge
    small = cv2.medianBlur(small, 9)  # wipes out colonies, keeps the rim
    side = min(small.shape)
    # ponytail: tuned on the public val photos (198 plates); the first (strongest) circle wins
    # and a plate's inner vs outer rim isn't told apart.
    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=side,
        param1=50,
        param2=50,
        minRadius=int(0.25 * side),
        maxRadius=int(0.75 * max(small.shape)),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0, 0] / scale
    return float(cx), float(cy), float(r) * (1 + env_float("COLONYCOUNT_PLATE_PAD", 0.05, lo=0))
