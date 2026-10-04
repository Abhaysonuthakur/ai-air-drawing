"""Headless self-test: proves the whole pipeline works without a webcam.

Run it with::

    python app.py --self-test

It exercises the same classes the live app uses (canvas, toolbar, gesture
detector, HUD renderer) and prints a pass/fail report. Useful as a smoke test
after changing code, and for checking a machine before the demo.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from . import config
from .camera import Camera, CameraError
from .drawing_app import AirDrawingApp, AppSettings
from .drawing_canvas import DrawingCanvas, DrawingStore, Toolbar
from .gesture_detector import Gesture, detect_gesture, is_fist, is_index_finger_up, is_middle_finger_up
from .hand_tracker import Hand, HandTracker

# Landmark offsets relative to the wrist, for a right hand facing the camera.
_BASE: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (-40, -10), 2: (-70, -40), 3: (-95, -70), 4: (-115, -95),
    5: (-45, -110), 6: (-50, -160), 7: (-52, -190), 8: (-54, -215),
    9: (0, -120), 10: (0, -175), 11: (0, -210), 12: (0, -240),
    13: (45, -110), 14: (50, -160), 15: (52, -190), 16: (54, -215),
    17: (85, -85), 18: (92, -125), 19: (95, -150), 20: (97, -170),
}

# Where the DIP + TIP sit when the finger is folded into the palm.
_CURLED: dict[int, tuple[int, int]] = {
    8: (-42, -105), 7: (-45, -125),
    12: (-2, -112), 11: (-1, -135),
    16: (43, -104), 15: (46, -124),
    20: (83, -80), 19: (86, -98),
}


def make_hand(extended: dict[str, bool], origin: tuple[int, int] = (400, 560)) -> Hand:
    """Build a synthetic 21-landmark hand.

    ``extended`` maps finger names ("index", "middle", "ring", "pinky") to
    True/False. Used to unit-test the geometry without a camera.
    """
    points = dict(_BASE)
    for finger, (dip, tip) in (("index", (7, 8)), ("middle", (11, 12)),
                               ("ring", (15, 16)), ("pinky", (19, 20))):
        if not extended.get(finger, False):
            points[dip] = _CURLED[dip]
            points[tip] = _CURLED[tip]

    landmarks_px = [(origin[0] + dx, origin[1] + dy) for dx, dy in
                    (points[i] for i in range(21))]
    return Hand(landmarks_px=landmarks_px, landmarks_norm=[], handedness="Right")


def _center(toolbar: Toolbar, key: str) -> tuple[int, int]:
    """Centre point of a toolbar button, in frame pixels."""
    x1, y1, x2, y2 = toolbar.rect_for(key)  # type: ignore[misc]
    return (x1 + x2) // 2, (y1 + y2) // 2


# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        mark = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        return condition

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n{'-' * 58}")
        print(f"  {self.passed}/{total} checks passed")
        if self.failed:
            print("  Some checks FAILED - see above.")
            return 1
        print("  Environment is ready. Start the app with:  python app.py")
        return 0


def run_self_test(settings: AppSettings | None = None) -> int:
    settings = settings or AppSettings()
    report = Report()

    print("\n=== AI Air Drawing - self test ===\n")

    # 1. Dependencies ------------------------------------------------------
    print("[1] Dependencies")
    report.check("OpenCV", True, f"v{cv2.__version__}")
    report.check("NumPy", True, f"v{np.__version__}")

    tracker: HandTracker | None = None
    try:
        tracker = HandTracker()
        report.check("MediaPipe Hands initialises", True)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        report.check("MediaPipe processes a frame", tracker.process(blank) is None,
                     "empty frame -> no hand, as expected")
    except Exception as exc:
        report.check("MediaPipe Hands initialises", False, str(exc))
        print("\n  Cannot continue without MediaPipe.")
        return report.summary()

    # 2. Gesture geometry --------------------------------------------------
    print("\n[2] Gesture detection (synthetic landmarks)")
    cases = [
        ("Index finger up        -> DRAW", {"index": True, "middle": False, "ring": False, "pinky": False}, Gesture.DRAW),
        ("Index + middle up      -> SELECT", {"index": True, "middle": True, "ring": False, "pinky": False}, Gesture.SELECT),
        ("Closed fist            -> STOP", {"index": False, "middle": False, "ring": False, "pinky": False}, Gesture.FIST),
        ("Middle only            -> IDLE", {"index": False, "middle": True, "ring": False, "pinky": False}, Gesture.IDLE),
        ("No hand                -> IDLE", None, Gesture.IDLE),
    ]
    for label, fingers, expected in cases:
        hand = make_hand(fingers) if fingers is not None else None
        report.check(label, detect_gesture(hand) is expected, f"got {detect_gesture(hand).value}")

    hand_draw = make_hand({"index": True, "middle": False})
    report.check("is_index_finger_up()", is_index_finger_up(hand_draw))
    report.check("is_middle_finger_up() is False", not is_middle_finger_up(hand_draw))
    report.check("is_fist() on a closed hand", is_fist(make_hand({})))

    # 3. Canvas + eraser ---------------------------------------------------
    print("\n[3] Canvas, brush and eraser")
    canvas = DrawingCanvas(settings.width, settings.height)
    report.check("New canvas is empty", canvas.is_empty)

    for x in range(200, 600, 8):
        canvas.draw_segment((x, 300), (x + 8, 305), config.COLOR_RED, config.BRUSH_SIZE)
    painted = int(np.count_nonzero(canvas.mask))
    report.check("Brush paints pixels", painted > 0, f"{painted} px")

    canvas.erase_segment((300, 290), (450, 315), config.ERASER_SIZE)
    after = int(np.count_nonzero(canvas.mask))
    report.check("Eraser removes pixels", after < painted, f"{painted} -> {after} px")

    frame = np.full((settings.height, settings.width, 3), 40, dtype=np.uint8)
    composited = canvas.composite(frame)
    report.check("Compositing keeps the frame size", composited.shape == frame.shape)
    report.check("Compositing actually draws something",
                 int(composited.sum()) > int(frame.sum()))

    canvas.clear()
    report.check("Clear empties the canvas", canvas.is_empty)

    # 4. Saving ------------------------------------------------------------
    print("\n[4] Saving")
    with tempfile.TemporaryDirectory() as tmp:
        store = DrawingStore(tmp)
        canvas.draw_segment((100, 100), (400, 400), config.COLOR_BLUE, config.BRUSH_SIZE)
        first = store.next_path()
        second = store.next_path()
        ok = canvas.save(first)
        report.check("PNG written", ok and os.path.isfile(first), os.path.basename(first))
        report.check("Filenames auto-increment",
                     os.path.basename(first) == "drawing_001.png"
                     and os.path.basename(second) == "drawing_002.png")
        reloaded = cv2.imread(first, cv2.IMREAD_UNCHANGED)
        report.check("Saved file is readable with an alpha channel",
                     reloaded is not None and reloaded.shape[2] == 4,
                     f"shape={None if reloaded is None else reloaded.shape}")
        bad = canvas.save(os.path.join(tmp, "no_such_dir", "x.png"))
        report.check("Missing folder is created automatically", bad)

    # 5. Toolbar -----------------------------------------------------------
    print("\n[5] Toolbar")
    toolbar = Toolbar(settings.width, settings.height)
    keys = [t.key for t in toolbar.tools]
    report.check("All eight tools exist",
                 keys == ["red", "blue", "green", "yellow", "white", "eraser", "clear", "save"],
                 ", ".join(keys))
    hits_ok = True
    for tool in toolbar.tools:
        x1, y1, x2, y2 = toolbar.rect_for(tool.key)  # type: ignore[misc]
        hits_ok &= toolbar.hit_test(((x1 + x2) // 2, (y1 + y2) // 2)) == tool.key
    report.check("Every button is hit-testable at its centre", hits_ok)
    report.check("Clicking above the toolbar hits nothing", toolbar.hit_test((640, 50)) is None)

    # 6. Full render pass + interaction state machine -----------------------
    print("\n[6] End-to-end render and interaction (no window opened)")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            app = AirDrawingApp(AppSettings(**{**settings.__dict__,
                                               "show_fps": True,
                                               "drawings_dir": tmp}))
            demo = np.full((settings.height, settings.width, 3), 30, dtype=np.uint8)
            cv2.rectangle(demo, (0, 0), (settings.width, settings.height), (70, 60, 50), -1)

            hand = make_hand({"index": True, "middle": False}, origin=(640, 520))
            gesture = detect_gesture(hand)
            for step in range(60):
                app._handle_drawing((400 + step * 6, 300 + int(40 * np.sin(step / 6))))
            app._handle_selection((640, settings.height - 50))  # hover over a button

            out = app.canvas.composite(demo)
            app._draw_cursor(out, hand, gesture)
            app.toolbar.draw(out, app.active_tool.key, app._hover_key, 0.6)
            app._draw_hud(out, gesture)
            report.check("Render pass produced a valid frame",
                         out.shape == (settings.height, settings.width, 3) and out.dtype == np.uint8)
            report.check("Gesture tracked through the app", gesture is Gesture.DRAW, gesture.value)
            report.check("Rendered frame is writable to disk",
                         cv2.imwrite(os.path.join(tmp, "preview.png"), out))

            # --- the "phantom line" regression ---------------------------
            app.canvas.clear()
            app._reset_stroke()
            for x in range(300, 460, 10):
                app._handle_drawing((x, 300))
            stroke_px = int(np.count_nonzero(app.canvas.mask))
            report.check("A drag paints a continuous stroke", stroke_px > 0, f"{stroke_px} px")

            app._reset_stroke()                       # hand left the frame
            app._handle_drawing((1100, 300))          # hand comes back far away
            gap = app.canvas.mask[280:320, 520:900]
            report.check("No phantom line after the hand returns",
                         int(np.count_nonzero(gap)) == 0, "gap stays empty")

            # --- toolbar dwell -------------------------------------------
            app.canvas.clear()
            blue_x, blue_y = _center(app.toolbar, "blue")
            for _ in range(config.TOOLBAR_DWELL_FRAMES - 1):
                app._handle_selection((blue_x, blue_y))
            report.check("Dwell does not fire early",
                         app.active_tool.key == "red", f"still {app.active_tool.key}")
            for _ in range(4):
                app._handle_selection((blue_x, blue_y))
            report.check("Dwell selects BLUE after ~0.6 s",
                         app.active_tool.key == "blue", app.active_tool.key)

            # --- the selected colour is what actually gets painted -------
            app.canvas.clear()
            app._reset_stroke()
            for x in range(300, 460, 10):
                app._handle_drawing((x, 300))
            painted = app.canvas.color_layer[app.canvas.mask > 0]
            mean_b, _, mean_r = painted.mean(axis=0)
            report.check("BLUE paints blue pixels", mean_b > mean_r,
                         f"B={mean_b:.0f} R={mean_r:.0f}")

            # --- eraser tool ---------------------------------------------
            app._apply_tool("eraser")
            before = int(np.count_nonzero(app.canvas.mask))
            app._reset_stroke()
            for x in range(300, 460, 10):
                app._handle_drawing((x, 300))
            after = int(np.count_nonzero(app.canvas.mask))
            report.check("Eraser tool removes the stroke", after < before,
                         f"{before} -> {after} px")
            report.check("Eraser restores the raw camera pixels",
                         np.array_equal(app.canvas.composite(demo), demo))

            # --- clear + save through the app ----------------------------
            app._apply_tool("clear")
            report.check("Toolbar CLEAR empties the canvas", app.canvas.is_empty)
            app._apply_tool("save")
            report.check("Saving an empty canvas is refused, not crashed",
                         os.listdir(tmp).count("drawing_001.png") == 0)

            app._apply_tool("red")
            app._reset_stroke()
            for x in range(300, 460, 10):
                app._handle_drawing((x, 300))
            app._apply_tool("save")
            saved = [f for f in os.listdir(tmp) if f.startswith("drawing_")]
            report.check("Toolbar SAVE writes a PNG", len(saved) == 1, ", ".join(saved))

            app.tracker.close()
    except Exception as exc:
        report.check("End-to-end interaction", False, f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    # 7. Camera probe ------------------------------------------------------
    print("\n[7] Webcam")
    camera = Camera(index=settings.camera_index, width=settings.width, height=settings.height)
    try:
        camera.open()
        frame = camera.read()
        report.check("Webcam opened and delivered a frame", frame is not None,
                     f"{frame.shape[1]}x{frame.shape[0]}")
        camera.close()
    except CameraError as exc:
        report.check("Webcam available", False, "not usable right now")
        print(f"      {str(exc).splitlines()[0]}")
        print("      The app is fine - the self-test simply could not grab the camera.")
    finally:
        camera.close()

    tracker.close()
    return report.summary()
