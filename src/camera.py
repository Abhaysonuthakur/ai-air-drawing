"""Webcam access with explicit initialisation and error handling."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from . import config


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or a frame cannot be read."""


class Camera:
    """Thin, safe wrapper around ``cv2.VideoCapture``.

    Used as a context manager so the device is always released::

        with Camera() as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        mirror: bool = config.MIRROR_FRAME,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._capture: cv2.VideoCapture | None = None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "Camera":
        """Open the device and verify that a frame can actually be read.

        ``VideoCapture.isOpened()`` alone is not enough: on Windows it can
        report True for a device that never delivers frames (already in use by
        another app, or blocked by privacy settings). So we read a test frame.
        """
        capture = self._create_capture()
        if capture is None or not capture.isOpened():
            raise CameraError(
                f"Could not open the webcam at index {self.index}.\n"
                "  - Close any other app using the camera (Zoom, Teams, Camera app).\n"
                "  - Check Windows Settings > Privacy > Camera and allow desktop apps.\n"
                "  - Try a different --camera index (0, 1, 2 ...)."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always work on the newest frame

        ok, _ = capture.read()
        if not ok:
            capture.release()
            raise CameraError(
                "The webcam opened but returned no frames. "
                "It is probably in use by another application."
            )

        self._capture = capture
        self._warm_up()
        return self

    def _create_capture(self) -> cv2.VideoCapture | None:
        """Try the platform-preferred backend first, then fall back.

        On Windows, DirectShow (CAP_DSHOW) opens far faster than the default
        MSMF backend and prints fewer driver warnings.
        """
        backends: list[int] = []
        if sys.platform == "win32":
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            capture = cv2.VideoCapture(self.index, backend)
            if capture.isOpened():
                return capture
            capture.release()
        return None

    def _warm_up(self) -> None:
        """Throw away the first frames - auto-exposure and white balance settle."""
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            self._capture.read()  # type: ignore[union-attr]
            time.sleep(0.02)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- frames ------------------------------------------------------------
    def read(self) -> np.ndarray:
        """Return the next BGR frame, mirrored if configured."""
        if self._capture is None:
            raise CameraError("Camera.read() called before Camera.open().")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Lost the webcam signal (read() returned an empty frame).")

        if self.mirror:
            frame = cv2.flip(frame, 1)  # horizontal flip -> mirror behaviour
        return frame

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height
