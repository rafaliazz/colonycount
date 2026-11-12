"""Plate localisation: Hough circle on a downscaled copy of the photo."""

import cv2
import numpy as np

PLATE_SEARCH_SIZE = 512  # longest side of the downscaled copy used to find the plate


def find_plate(img: np.ndarray) -> tuple[float, float, float] | None:
    """Return the plate circle (cx, cy, r) in image pixels, or None if no rim is found."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, PLATE_SEARCH_SIZE / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 9)  # wipes out colonies, keeps the rim
    side = min(small.shape)
    # ponytail: tuned on synthetic plates only, first (strongest) circle wins and a plate's
    # inner vs outer rim isn't told apart; retune on the public test photos (task 2.3).
    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=side,
        param1=100,
        param2=50,
        minRadius=int(0.25 * side),
        maxRadius=int(0.75 * max(small.shape)),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0, 0] / scale
    return float(cx), float(cy), float(r)
