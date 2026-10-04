"""Application orchestration: camera -> hand -> gesture -> canvas -> display."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .camera import Camera, CameraError
from .drawing_canvas import DrawingCanvas, DrawingStore, Tool, Toolbar, blend_panel
from .gesture_detector import Gesture, detect_gesture, gesture_hint
from .hand_tracker import Hand, HandTracker, HandTrackerError


# ---------------------------------------------------------------------------
# Settings (populated from the command line by app.py)
# ---------------------------------------------------------------------------
@dataclass
class AppSettings:
    camera_index: int = config.CAMERA_INDEX
    width: int = config.FRAME_WIDTH
    height: int = config.FRAME_HEIGHT
    mirror: bool = config.MIRROR_FRAME
    show_landmarks: bool = config.SHOW_LANDMARKS
    smoothing: float = config.SMOOTHING_FACTOR
    brush_size: int = config.BRUSH_SIZE
    eraser_size: int = config.ERASER_SIZE
    drawings_dir: str = config.DRAWINGS_DIR
    show_fps: bool = True
    max_frames: int = 0  # >0 = run for that many frames then exit (smoke test)


class Smoother:
    """Exponential smoothing for the fingertip.

    ``smoothed = alpha * current + (1 - alpha) * previous``

    High alpha -> follows the finger closely but keeps the jitter.
    Low alpha  -> silky line, at the cost of a little lag.
    """

    def __init__(self, alpha: float = config.SMOOTHING_FACTOR) -> None:
        self.alpha = float(np.clip(alpha, config.MIN_SMOOTHING, config.MAX_SMOOTHING))
        self._value: tuple[float, float] | None = None

    def update(self, point: tuple[int, int]) -> tuple[int, int]:
        if self._value is None:
            self._value = (float(point[0]), float(point[1]))
        else:
            px, py = self._value
            self._value = (
                self.alpha * point[0] + (1.0 - self.alpha) * px,
                self.alpha * point[1] + (1.0 - self.alpha) * py,
            )
        return int(round(self._value[0])), int(round(self._value[1]))

    def reset(self) -> None:
        self._value = None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class AirDrawingApp:
    """Owns the whole runtime loop."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()
        s = self.settings

        self.camera = Camera(index=s.camera_index, width=s.width, height=s.height, mirror=s.mirror)
        self.tracker = HandTracker()
        self.canvas = DrawingCanvas(s.width, s.height)
        self.toolbar = Toolbar(s.width, s.height)
        self.store = DrawingStore(s.drawings_dir)

        self.draw_smoother = Smoother(s.smoothing)
        # The toolbar pointer needs to feel snappy, so it is smoothed less.
        self.pointer_smoother = Smoother(min(1.0, s.smoothing + 0.35))

        # Drawing state
        self._prev_point: tuple[int, int] | None = None
        self.active_tool: Tool = self.toolbar.tool_for("red")  # type: ignore[assignment]
        self.brush_size = s.brush_size

        # Toolbar hover / dwell state
        self._hover_key: str | None = None
        self._hover_frames = 0
        self._cooldown = 0

        # HUD state
        self._toast: str = ""
        self._toast_frames = 0
        self._fps = 0.0
        self._frame_deltas: deque[float] = deque(maxlen=30)
        self._last_frame_time = 0.0
        self._show_landmarks = s.show_landmarks
        self._running = False

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> int:
        """Main loop. Returns a process exit code."""
        try:
            self.camera.open()
        except CameraError as exc:
            print(f"\n[CAMERA ERROR]\n{exc}\n")
            return 2

        self._print_banner()
        self._running = True
        frames = 0
        try:
            while self._running:
                frame = self.camera.read()
                self._process(frame)
                frames += 1
                if 0 < self.settings.max_frames <= frames:
                    print(f"[test] reached --max-frames ({frames}), stopping.")
                    self.stop()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except CameraError as exc:
            print(f"\n[CAMERA ERROR] {exc}")
            return 3
        except cv2.error as exc:
            print(f"\n[OPENCV ERROR] {exc}")
            print("If this happened while opening the window, make sure OpenCV was installed "
                  "with GUI support (pip install opencv-python).")
            return 4
        finally:
            self._teardown()
        return 0

    def _print_banner(self) -> None:
        print("\n=== AI Air Drawing ===")
        print(f"Camera {self.settings.camera_index} @ {self.settings.width}x{self.settings.height}")
        print(f"Drawings will be saved to: {self.store.directory}")
        print("Index finger = draw | Index + middle = pick tool | Fist = stop")
        print("Keys: r b g y w e = colour/eraser | c = clear | s = save | f = landmarks | ESC/Q = quit\n")

    def _teardown(self) -> None:
        self.camera.close()
        self.tracker.close()
        cv2.destroyAllWindows()
        print("Camera released. Bye.")

    def stop(self) -> None:
        self._running = False

    # -- per-frame pipeline ------------------------------------------------
    def _process(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.settings.width or frame.shape[0] != self.settings.height:
            frame = cv2.resize(frame, (self.settings.width, self.settings.height))

        hand = self.tracker.process(frame)
        gesture = detect_gesture(hand)

        if hand is not None and self._show_landmarks:
            self.tracker.draw_landmarks(frame, hand)

        if gesture is Gesture.DRAW and hand is not None:
            self._handle_drawing(hand.index_tip)
        elif gesture is Gesture.SELECT and hand is not None:
            self._handle_selection(hand.index_tip)
        else:
            # Gesture changed or the hand vanished: drop the previous point so
            # no phantom line is drawn when tracking resumes.
            self._reset_stroke()
            self.pointer_smoother.reset()
            self._reset_hover()

        self._render(frame, hand, gesture)

    def _reset_stroke(self) -> None:
        """End the current stroke.

        Clearing the previous point is what prevents a straight line being
        drawn across the screen when the hand leaves the frame and comes back.
        """
        self._prev_point = None
        self.draw_smoother.reset()

    # -- drawing -----------------------------------------------------------
    def _handle_drawing(self, fingertip: tuple[int, int]) -> None:
        point = self.draw_smoother.update(fingertip)

        # Never paint underneath the toolbar - it would be invisible and would
        # silently end up in the saved file.
        if point[1] > self.toolbar.y1 - 6:
            self._prev_point = None
            return

        if self._prev_point is None:
            self._prev_point = point  # first point of a new stroke
            return

        dx = point[0] - self._prev_point[0]
        dy = point[1] - self._prev_point[1]
        if math.hypot(dx, dy) < config.MIN_DRAW_DISTANCE:
            return  # ignore sub-pixel jitter

        if self.active_tool.kind == "eraser":
            self.canvas.erase_segment(self._prev_point, point, self.settings.eraser_size)
        else:
            self.canvas.draw_segment(self._prev_point, point, self.active_tool.color, self.brush_size)

        self._prev_point = point

    # -- toolbar -----------------------------------------------------------
    def _handle_selection(self, fingertip: tuple[int, int]) -> None:
        self._reset_stroke()

        point = self.pointer_smoother.update(fingertip)
        key = self.toolbar.hit_test(point)

        if key is None:
            self._reset_hover()
            return

        if key != self._hover_key:
            self._hover_key = key
            self._hover_frames = 0
        else:
            self._hover_frames += 1

        if self._cooldown > 0:
            self._cooldown -= 1
            return

        if self._hover_frames >= config.TOOLBAR_DWELL_FRAMES:
            self._apply_tool(key)
            self._hover_frames = 0
            self._cooldown = config.TOOLBAR_DWELL_COOLDOWN

    def _reset_hover(self) -> None:
        self._hover_key = None
        self._hover_frames = 0

    @property
    def _dwell_progress(self) -> float:
        if self._hover_key is None or self._cooldown > 0:
            return 0.0
        return self._hover_frames / float(config.TOOLBAR_DWELL_FRAMES)

    def _apply_tool(self, key: str) -> None:
        tool = self.toolbar.tool_for(key)
        if tool is None:
            return

        if tool.is_paint_tool:
            self.active_tool = tool
            self._show_toast(f"{tool.label} selected")
        elif key == "clear":
            self.canvas.clear()
            self._show_toast("Canvas cleared")
        elif key == "save":
            self._save_drawing()

    def _save_drawing(self) -> None:
        if self.canvas.is_empty:
            self._show_toast("Nothing to save yet")
            return
        path = self.store.next_path()
        if self.canvas.save(path):
            self._show_toast(f"Saved {os.path.basename(path)}")
            print(f"[save] {path}")
        else:
            self._show_toast("Save failed - see console")

    # -- keyboard ----------------------------------------------------------
    def _poll_key(self) -> None:
        """Read one key from the OpenCV window and act on it."""
        key = cv2.waitKey(1)
        if key == -1:
            return
        key &= 0xFF

        if key in config.EXIT_KEYS:
            self.stop()
            return

        shortcuts = {
            ord("r"): "red", ord("b"): "blue", ord("g"): "green",
            ord("y"): "yellow", ord("w"): "white", ord("e"): "eraser",
            ord("c"): "clear", ord("s"): "save",
        }
        if key in shortcuts:
            self._apply_tool(shortcuts[key])
        elif key == ord("f"):
            self._show_landmarks = not self._show_landmarks
        elif key in (ord("+"), ord("=")):
            self.brush_size = min(config.MAX_BRUSH_SIZE, self.brush_size + 2)
            self._show_toast(f"Brush {self.brush_size}px")
        elif key in (ord("-"), ord("_")):
            self.brush_size = max(config.MIN_BRUSH_SIZE, self.brush_size - 2)
            self._show_toast(f"Brush {self.brush_size}px")

    def _show_toast(self, message: str, frames: int = 70) -> None:
        self._toast = message
        self._toast_frames = frames

    # -- rendering ---------------------------------------------------------
    def _render(self, frame: np.ndarray, hand: Hand | None, gesture: Gesture) -> None:
        display = self.canvas.composite(frame)

        if hand is not None:
            self._draw_cursor(display, hand, gesture)

        self.toolbar.draw(display, self.active_tool.key, self._hover_key, self._dwell_progress)
        self._draw_hud(display, gesture)
        self._draw_toast(display)

        if self._toast_frames > 0:
            self._toast_frames -= 1

        cv2.imshow(config.WINDOW_NAME, display)
        self._poll_key()

        now = time.perf_counter()
        if self._last_frame_time:
            self._frame_deltas.append(now - self._last_frame_time)
            if self._frame_deltas:
                self._fps = 1.0 / max(1e-6, sum(self._frame_deltas) / len(self._frame_deltas))
        self._last_frame_time = now

        try:
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.stop()
        except cv2.error:
            self.stop()  # window was closed while we were drawing into it

    def _draw_cursor(self, frame: np.ndarray, hand: Hand, gesture: Gesture) -> None:
        tip = hand.index_tip
        if gesture is Gesture.SELECT:
            cv2.circle(frame, tip, 16, config.UI_ACCENT, 2, cv2.LINE_AA)
            cv2.circle(frame, tip, 3, config.UI_ACCENT, -1, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(frame, (tip[0] + dx * 18, tip[1] + dy * 18),
                         (tip[0] + dx * 26, tip[1] + dy * 26), config.UI_ACCENT, 2, cv2.LINE_AA)
        elif gesture is Gesture.DRAW:
            erasing = self.active_tool.kind == "eraser"
            color = config.COLOR_WHITE if erasing else self.active_tool.color
            radius = max(4, (self.settings.eraser_size if erasing else self.brush_size) // 2)
            cv2.circle(frame, tip, radius, color, 1, cv2.LINE_AA)
            cv2.circle(frame, tip, 3, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, tip, 6, (160, 160, 160), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, gesture: Gesture) -> None:
        x, y, w = 22, 22, 400
        h = 196 if self.settings.show_fps else 172

        blend_panel(frame, x, y, x + w, y + h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), config.UI_BORDER, 1)

        cv2.putText(frame, "AI AIR DRAWING", (x + 18, y + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, config.UI_TEXT, 1, cv2.LINE_AA)

        rows = [
            ("Gesture", gesture.value),
            ("Tool", self.active_tool.label),
            ("Brush", f"{self.brush_size} px"),
        ]
        if self.settings.show_fps:
            rows.append(("FPS", f"{self._fps:.0f}"))

        for i, (label, value) in enumerate(rows):
            ry = y + 78 + i * 26
            cv2.putText(frame, f"{label}:", (x + 18, ry),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, config.UI_TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(frame, value, (x + 118, ry),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, config.UI_TEXT, 1, cv2.LINE_AA)

        chip = config.COLOR_ERASER if self.active_tool.kind == "eraser" else self.active_tool.color
        cv2.circle(frame, (x + w - 34, y + 96), 15, chip, -1, cv2.LINE_AA)
        cv2.circle(frame, (x + w - 34, y + 96), 15, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.putText(frame, gesture_hint(gesture), (x + 18, y + h - 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, config.UI_ACCENT, 1, cv2.LINE_AA)

        quit_text = "ESC/Q quit   C clear   S save   F landmarks   +/- brush"
        (tw, _), _ = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        hx = self.settings.width - tw - 24
        blend_panel(frame, hx - 12, 22, self.settings.width - 12, 56, (12, 10, 8), 0.6)
        cv2.putText(frame, quit_text, (hx, 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, config.UI_TEXT, 1, cv2.LINE_AA)

    def _draw_toast(self, frame: np.ndarray) -> None:
        if self._toast_frames <= 0 or not self._toast:
            return
        text = self._toast
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cx = self.settings.width // 2
        baseline = self.toolbar.y1 - 26
        x1, y1 = cx - tw // 2 - 18, baseline - th - 14
        x2, y2 = cx + tw // 2 + 18, baseline + 14
        blend_panel(frame, x1, y1, x2, y2, (12, 10, 8), 0.82)
        cv2.putText(frame, text, (cx - tw // 2, baseline), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    config.UI_ACCENT, 1, cv2.LINE_AA)
