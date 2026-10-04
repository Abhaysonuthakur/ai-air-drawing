"""Central configuration for AI Air Drawing.

Every tunable value lives here so the app can be adjusted without touching
application logic. Change a number, restart the app, see the effect.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0          # 0 = default laptop webcam, try 1/2 for externals
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
MIRROR_FRAME: bool = True      # flip horizontally so it feels like a mirror
CAMERA_WARMUP_FRAMES: int = 5  # discard the first frames (auto-exposure settles)

# ---------------------------------------------------------------------------
# MediaPipe hand tracking
# ---------------------------------------------------------------------------
MAX_HANDS: int = 1
MODEL_COMPLEXITY: int = 1              # 0 = fastest, 1 = balanced
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.6
SHOW_LANDMARKS: bool = False           # draw the 21-point skeleton (debug)

# ---------------------------------------------------------------------------
# Gesture geometry
# ---------------------------------------------------------------------------
# A finger counts as "extended" when its tip is further from the wrist than
# its middle joint (PIP). This is orientation-tolerant, unlike a raw y-compare.
FINGER_EXTEND_MARGIN: float = 1.05

# ---------------------------------------------------------------------------
# Brush / eraser
# ---------------------------------------------------------------------------
BRUSH_SIZE: int = 10
ERASER_SIZE: int = 40
MIN_BRUSH_SIZE: int = 2
MAX_BRUSH_SIZE: int = 80

# ---------------------------------------------------------------------------
# Smoothing / stabilisation
# ---------------------------------------------------------------------------
# Exponential smoothing: smoothed = a*current + (1-a)*previous
# 1.0 = no smoothing (raw, jittery) | 0.2 = very smooth but slightly laggy
SMOOTHING_FACTOR: float = 0.5
MIN_SMOOTHING: float = 0.05
MAX_SMOOTHING: float = 1.0

# Ignore sub-pixel movement so we don't spam the rasteriser with micro-lines.
MIN_DRAW_DISTANCE: float = 2.0

# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------
TOOLBAR_HEIGHT: int = 84
TOOLBAR_MARGIN: int = 14
TOOLBAR_DWELL_FRAMES: int = 18   # frames the finger must rest on a button (~0.6s)
TOOLBAR_DWELL_COOLDOWN: int = 12  # frames of lockout after a selection fires

# Palette (BGR, because OpenCV works in BGR)
COLOR_RED: tuple[int, int, int] = (60, 60, 235)
COLOR_BLUE: tuple[int, int, int] = (235, 130, 45)
COLOR_GREEN: tuple[int, int, int] = (80, 200, 90)
COLOR_YELLOW: tuple[int, int, int] = (60, 220, 245)
COLOR_WHITE: tuple[int, int, int] = (255, 255, 255)
COLOR_ERASER: tuple[int, int, int] = (110, 110, 110)

# ---------------------------------------------------------------------------
# UI theme (BGR)
# ---------------------------------------------------------------------------
UI_PANEL: tuple[int, int, int] = (26, 22, 18)
UI_PANEL_ALPHA: float = 0.78
UI_BORDER: tuple[int, int, int] = (70, 62, 55)
UI_TEXT: tuple[int, int, int] = (238, 238, 238)
UI_TEXT_DIM: tuple[int, int, int] = (165, 160, 155)
UI_ACCENT: tuple[int, int, int] = (255, 190, 60)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DRAWINGS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drawings")
DRAWING_PREFIX: str = "drawing_"
DRAWING_EXT: str = ".png"

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
WINDOW_NAME: str = "AI Air Drawing  -  Hand Gesture Virtual Paint"
TARGET_FPS: int = 60

# Exit keys
EXIT_KEYS: tuple[int, ...] = (27, ord("q"), ord("Q"))  # ESC and Q
