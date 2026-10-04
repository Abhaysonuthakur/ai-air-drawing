"""The virtual canvas (with a real eraser) and the on-screen toolbar."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
) -> None:
    """Draw a filled (thickness=-1) or outlined rounded rectangle in place."""
    radius = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if radius == 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return

    if thickness < 0:  # filled: two rectangles + four corner discs
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:  # outline: four straight edges + four arcs
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def blend_panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple[int, int, int] = config.UI_PANEL,
                alpha: float = config.UI_PANEL_ALPHA) -> None:
    """Translucent panel - lets the camera feed show through the UI."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tool:
    """A single toolbar entry."""

    key: str
    label: str
    kind: str                                   # "color" | "eraser" | "action"
    color: tuple[int, int, int] = (200, 200, 200)

    @property
    def is_paint_tool(self) -> bool:
        return self.kind in ("color", "eraser")


DEFAULT_TOOLS: tuple[Tool, ...] = (
    Tool("red", "RED", "color", config.COLOR_RED),
    Tool("blue", "BLUE", "color", config.COLOR_BLUE),
    Tool("green", "GREEN", "color", config.COLOR_GREEN),
    Tool("yellow", "YELLOW", "color", config.COLOR_YELLOW),
    Tool("white", "WHITE", "color", config.COLOR_WHITE),
    Tool("eraser", "ERASER", "eraser", config.COLOR_ERASER),
    Tool("clear", "CLEAR", "action"),
    Tool("save", "SAVE", "action"),
)


class Toolbar:
    """Bottom strip of selectable tools with finger hover hit-testing."""

    def __init__(self, width: int, height: int, tools: tuple[Tool, ...] = DEFAULT_TOOLS) -> None:
        self.tools = tools
        self.frame_width = width
        self.frame_height = height

        self.height = config.TOOLBAR_HEIGHT
        self.margin = config.TOOLBAR_MARGIN
        self.y1 = height - self.height - self.margin
        self.y2 = height - self.margin
        self.x1 = self.margin
        self.x2 = width - self.margin

        self._rects: list[tuple[int, int, int, int]] = []
        self._layout()

    def _layout(self) -> None:
        count = len(self.tools)
        gap = 10
        usable = (self.x2 - self.x1) - gap * (count - 1)
        button_w = usable // count
        self._rects = []
        for i in range(count):
            bx1 = self.x1 + i * (button_w + gap)
            self._rects.append((bx1, self.y1 + 10, bx1 + button_w, self.y2 - 10))

    def rect_for(self, key: str) -> tuple[int, int, int, int] | None:
        for tool, rect in zip(self.tools, self._rects):
            if tool.key == key:
                return rect
        return None

    def tool_for(self, key: str) -> Tool | None:
        for tool in self.tools:
            if tool.key == key:
                return tool
        return None

    def hit_test(self, point: tuple[int, int] | None) -> str | None:
        """Return the key of the button under ``point``, or None."""
        if point is None:
            return None
        x, y = point
        for tool, (x1, y1, x2, y2) in zip(self.tools, self._rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return tool.key
        return None

    def draw(
        self,
        frame: np.ndarray,
        selected_key: str,
        hovered_key: str | None = None,
        dwell_progress: float = 0.0,
    ) -> None:
        blend_panel(frame, self.x1 - 6, self.y1, self.x2 + 6, self.y2)

        for tool, (x1, y1, x2, y2) in zip(self.tools, self._rects):
            selected = tool.key == selected_key
            hovered = tool.key == hovered_key

            fill = (46, 40, 35) if not selected else (70, 58, 44)
            rounded_rect(frame, x1, y1, x2, y2, fill, radius=12)

            if selected:
                rounded_rect(frame, x1, y1, x2, y2, config.UI_ACCENT, radius=12, thickness=2)
            elif hovered:
                rounded_rect(frame, x1, y1, x2, y2, (140, 130, 120), radius=12, thickness=1)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if tool.is_paint_tool:
                # Colour swatch + label underneath.
                cv2.circle(frame, (cx, cy - 8), 13, tool.color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy - 8), 13, (235, 235, 235), 1, cv2.LINE_AA)
                if tool.kind == "eraser":  # diagonal slash reads as "eraser"
                    cv2.line(frame, (cx - 9, cy + 1), (cx + 9, cy - 17), (250, 250, 250), 3, cv2.LINE_AA)
                self._centered_text(frame, tool.label, cx, cy + 22, 0.42, config.UI_TEXT)
            else:
                label_color = config.UI_ACCENT if tool.key == "save" else config.UI_TEXT
                self._centered_text(frame, tool.label, cx, cy + 6, 0.55, label_color)

            # Dwell indicator: fills the top edge of the hovered button.
            if hovered and dwell_progress > 0.0:
                bar_x2 = int(x1 + (x2 - x1) * min(dwell_progress, 1.0))
                if bar_x2 > x1 + 2:
                    rounded_rect(frame, x1 + 2, y1 + 2, bar_x2, y1 + 7, config.UI_ACCENT, radius=3)

    @staticmethod
    def _centered_text(frame: np.ndarray, text: str, cx: int, cy: int,
                       scale: float, color: tuple[int, int, int]) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
class DrawingCanvas:
    """A transparent paint layer that composites over the camera feed.

    Two parallel buffers are kept:
      * ``color_layer`` - the BGR paint
      * ``mask``        - 0 where transparent, 255 where painted

    Keeping an explicit mask is what makes the eraser *actually* erase instead
    of painting black over the drawing.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.color_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self._stroke_count = 0

    # -- painting ----------------------------------------------------------
    def draw_segment(self, p1: tuple[int, int], p2: tuple[int, int],
                     color: tuple[int, int, int], thickness: int) -> None:
        """Connect the previous point to the current one -> a continuous line."""
        thickness = int(np.clip(thickness, config.MIN_BRUSH_SIZE, config.MAX_BRUSH_SIZE))
        cv2.line(self.color_layer, p1, p2, color, thickness, cv2.LINE_AA)
        # Binary mask (LINE_8) keeps transparency crisp and the eraser exact.
        cv2.line(self.mask, p1, p2, 255, thickness, cv2.LINE_8)
        self._stroke_count += 1

    def erase_segment(self, p1: tuple[int, int], p2: tuple[int, int], thickness: int) -> None:
        thickness = int(np.clip(thickness, config.MIN_BRUSH_SIZE, config.MAX_BRUSH_SIZE))
        cv2.line(self.mask, p1, p2, 0, thickness, cv2.LINE_8)
        cv2.line(self.color_layer, p1, p2, (0, 0, 0), thickness, cv2.LINE_8)
        self._stroke_count += 1

    def clear(self) -> None:
        self.color_layer[:] = 0
        self.mask[:] = 0
        self._stroke_count = 0

    @property
    def is_empty(self) -> bool:
        return self._stroke_count == 0 or not bool(np.any(self.mask))

    # -- output ------------------------------------------------------------
    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Blend the paint layer over the camera frame."""
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        painted = self.mask > 0
        if not painted.any():
            return frame
        out = frame.copy()
        out[painted] = self.color_layer[painted]  # fast boolean indexing
        return out

    def export_bgra(self) -> np.ndarray:
        """Flatten the drawing to BGRA with a transparent background."""
        bgra = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        bgra[:, :, :3] = self.color_layer
        bgra[:, :, 3] = self.mask
        return bgra

    def save(self, path: str) -> bool:
        """Write the drawing as a PNG. Returns False instead of raising."""
        try:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            ok = cv2.imwrite(path, self.export_bgra())
            if not ok:
                print(f"[save] OpenCV could not write to {path}")
            return bool(ok)
        except (OSError, cv2.error) as exc:
            print(f"[save] Failed to save drawing: {exc}")
            return False


class DrawingStore:
    """Handles the drawings/ folder and auto-incrementing filenames."""

    def __init__(self, directory: str = config.DRAWINGS_DIR) -> None:
        self.directory = directory
        self._counter = self._scan_existing()

    def _scan_existing(self) -> int:
        """Continue numbering from the highest existing file (no overwrites)."""
        try:
            os.makedirs(self.directory, exist_ok=True)
        except OSError as exc:
            print(f"[store] Could not create '{self.directory}': {exc}")
            return 1

        highest = 0
        try:
            for name in os.listdir(self.directory):
                if name.startswith(config.DRAWING_PREFIX) and name.endswith(config.DRAWING_EXT):
                    digits = name[len(config.DRAWING_PREFIX):-len(config.DRAWING_EXT)]
                    if digits.isdigit():
                        highest = max(highest, int(digits))
        except OSError as exc:
            print(f"[store] Could not list '{self.directory}': {exc}")
        return highest + 1

    def next_path(self) -> str:
        path = os.path.join(self.directory, f"{config.DRAWING_PREFIX}{self._counter:03d}{config.DRAWING_EXT}")
        self._counter += 1
        return path
