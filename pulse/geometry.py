from __future__ import annotations

import ast
import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:  # pragma: no cover
    Image = None
    ImageDraw = None


def _require_image_stack() -> None:
    if np is None or Image is None or ImageDraw is None:
        raise ModuleNotFoundError(
            "PULSE image utilities require `numpy` and `pillow`. "
            "Install dependencies from requirements.txt first."
        )


def safe_literal_eval(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def normalize_polygon(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    polygon: List[Tuple[float, float]] = []
    for point in points:
        if len(point) < 2:
            continue
        polygon.append((float(point[0]), float(point[1])))
    return polygon


def polygons_to_mask(
    size: Tuple[int, int],
    polygons_by_class: Dict[int, Sequence[Sequence[Sequence[float]]]],
) -> "np.ndarray":
    _require_image_stack()
    width, height = size
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for class_index in sorted(polygons_by_class):
        for polygon in polygons_by_class[class_index]:
            normalized = normalize_polygon(polygon)
            if len(normalized) >= 3:
                draw.polygon(normalized, fill=int(class_index))
    return np.asarray(canvas, dtype=np.uint8)


def load_polygon_json(path: str) -> List[List[Tuple[float, float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        if payload and isinstance(payload[0], list) and payload[0] and isinstance(payload[0][0], (int, float)):
            return [normalize_polygon(payload)]
        return [normalize_polygon(item) for item in payload if isinstance(item, list)]

    if isinstance(payload, dict):
        if "shapes" in payload:
            polygons: List[List[Tuple[float, float]]] = []
            for shape in payload["shapes"]:
                points = shape.get("points", [])
                normalized = normalize_polygon(points)
                if len(normalized) >= 3:
                    polygons.append(normalized)
            return polygons
        if "points" in payload:
            return [normalize_polygon(payload["points"])]
    return []


def parse_via_polygon(region_shape_attributes: str) -> List[Tuple[float, float]]:
    payload = safe_literal_eval(region_shape_attributes)
    if not isinstance(payload, dict):
        return []
    xs = payload.get("all_points_x", [])
    ys = payload.get("all_points_y", [])
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def polygon_to_bbox(points: Sequence[Sequence[float]]) -> List[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_union(boxes: Iterable[Sequence[float]]) -> List[float]:
    boxes = [list(box) for box in boxes]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def xyxy_to_cxcywh(box: Sequence[float], width: float, height: float) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = max(x2 - x1, 1e-6) / width
    bh = max(y2 - y1, 1e-6) / height
    return [cx, cy, bw, bh]


def cxcywh_to_xyxy(box: Sequence[float]) -> List[float]:
    cx, cy, bw, bh = [float(value) for value in box]
    return [cx - (bw / 2.0), cy - (bh / 2.0), cx + (bw / 2.0), cy + (bh / 2.0)]


def ellipse_circumference(major_axis: float, minor_axis: float) -> float:
    a = max(float(major_axis), float(minor_axis)) / 2.0
    b = min(float(major_axis), float(minor_axis)) / 2.0
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-8)
    return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h + 1e-8)))


def carotid_thickness_from_mask(mask: "np.ndarray", pixel_size: float = 1.0) -> float:
    _require_image_stack()
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = mask > 0
    if not binary.any():
        return 0.0

    thicknesses: List[float] = []
    for column in range(binary.shape[1]):
        rows = np.where(binary[:, column])[0]
        if rows.size >= 2:
            thicknesses.append((rows[-1] - rows[0] + 1) * float(pixel_size))

    if not thicknesses:
        rows = np.where(binary)[0]
        return float((rows.max() - rows.min() + 1) * float(pixel_size))
    return float(np.median(np.asarray(thicknesses, dtype=np.float32)))
