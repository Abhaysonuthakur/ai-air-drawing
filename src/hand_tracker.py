"""MediaPipe hand detection and landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config

# Landmark indices of the 21-point MediaPipe hand model.
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# The skeleton edges, used only for the optional debug overlay.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class HandTrackerError(RuntimeError):
    """Raised when MediaPipe cannot be imported or initialised."""


@dataclass(frozen=True)
class Hand:
    """One detected hand, expressed in both coordinate spaces we need."""

    landmarks_px: list[tuple[int, int]]   # pixel coordinates, ready for OpenCV
    landmarks_norm: list[tuple[float, float, float]]  # MediaPipe 0..1 values
    handedness: str                       # "Left" / "Right" / "Unknown"

    @property
    def index_tip(self) -> tuple[int, int]:
        return self.landmarks_px[INDEX_TIP]

    @property
    def middle_tip(self) -> tuple[int, int]:
        return self.landmarks_px[MIDDLE_TIP]

    @property
    def wrist(self) -> tuple[int, int]:
        return self.landmarks_px[WRIST]

    def point(self, index: int) -> tuple[int, int]:
        return self.landmarks_px[index]


class HandTracker:
    """Wraps ``mediapipe.solutions.hands`` in a small, typed interface."""

    def __init__(
        self,
        max_hands: int = config.MAX_HANDS,
        model_complexity: int = config.MODEL_COMPLEXITY,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment problem
            raise HandTrackerError(
                "MediaPipe is not installed. Run:\n"
                "    pip install -r requirements.txt"
            ) from exc

        self._mp_hands = mp.solutions.hands
        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,      # video mode -> temporal tracking
                max_num_hands=max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - environment problem
            raise HandTrackerError(f"Failed to initialise MediaPipe Hands: {exc}") from exc

    # -- public API --------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> Hand | None:
        """Detect a hand in a BGR frame.

        Returns ``None`` when no hand is visible - a completely normal state,
        not an error. The caller must treat it as "tracking lost".
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        height, width = frame_bgr.shape[:2]
        # MediaPipe expects contiguous RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

        # Normalised (0..1) -> pixel coordinates, clamped so a fingertip that
        # drifts off-screen can never produce an out-of-bounds index.
        landmarks_px = [
            (int(np.clip(x * width, 0, width - 1)),
             int(np.clip(y * height, 0, height - 1)))
            for x, y, _ in landmarks_norm
        ]

        handedness = "Unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return Hand(landmarks_px=landmarks_px,
                    landmarks_norm=landmarks_norm,
                    handedness=handedness)

    def draw_landmarks(self, frame: np.ndarray, hand: Hand) -> None:
        """Draw the 21-point skeleton in place (debug aid)."""
        pts = hand.landmarks_px
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (0, 200, 255), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            radius = 6 if i in (INDEX_TIP, MIDDLE_TIP) else 3
            color = (0, 0, 255) if i == INDEX_TIP else (255, 255, 255)
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
