"""Gesture recognition from hand landmark geometry.

No machine-learning model here: every gesture is derived from the positions of
the 21 landmarks, which keeps it fast, dependency-free and easy to tune.
"""

from __future__ import annotations

import math
from enum import Enum

from . import config
from .hand_tracker import (
    Hand,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    WRIST,
)


class Gesture(str, Enum):
    """The gestures the application reacts to."""

    DRAW = "DRAWING"        # index finger up          -> draw
    SELECT = "SELECTING"    # index + middle finger up -> move the pointer
    FIST = "STOP"           # closed hand              -> stop / idle
    IDLE = "IDLE"           # anything else            -> do nothing


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_finger_extended(hand: Hand, pip: int, tip: int) -> bool:
    """True when a finger is straight and pointing away from the palm.

    We compare how far the tip and the middle joint (PIP) sit from the wrist.
    A curled fingertip folds back towards the palm, so it ends up *closer* to
    the wrist than the PIP joint. This works regardless of how the hand is
    rotated, unlike the naive "tip is above PIP" test.
    """
    wrist = hand.wrist
    return _distance(hand.point(tip), wrist) > _distance(hand.point(pip), wrist) * config.FINGER_EXTEND_MARGIN


def is_index_finger_up(hand: Hand) -> bool:
    """Index finger extended (landmark 8 raised)."""
    return _is_finger_extended(hand, INDEX_PIP, INDEX_TIP)


def is_middle_finger_up(hand: Hand) -> bool:
    """Middle finger extended (landmark 12 raised)."""
    return _is_finger_extended(hand, MIDDLE_PIP, MIDDLE_TIP)


def is_ring_finger_up(hand: Hand) -> bool:
    return _is_finger_extended(hand, RING_PIP, RING_TIP)


def is_pinky_up(hand: Hand) -> bool:
    return _is_finger_extended(hand, PINKY_PIP, PINKY_TIP)


def is_fist(hand: Hand) -> bool:
    """A closed hand: none of the four main fingers is extended."""
    return not any((
        is_index_finger_up(hand),
        is_middle_finger_up(hand),
        is_ring_finger_up(hand),
        is_pinky_up(hand),
    ))


def detect_gesture(hand: Hand | None) -> Gesture:
    """Map a detected hand onto one of the four application gestures.

    Order matters: the most specific / most restrictive test wins, so a fist
    is never misread as a drawing gesture.
    """
    if hand is None:
        return Gesture.IDLE

    if is_fist(hand):
        return Gesture.FIST

    index_up = is_index_finger_up(hand)
    middle_up = is_middle_finger_up(hand)

    if index_up and middle_up:
        return Gesture.SELECT
    if index_up and not middle_up:
        return Gesture.DRAW
    return Gesture.IDLE


def gesture_hint(gesture: Gesture) -> str:
    """Short human-readable explanation shown in the on-screen HUD."""
    return {
        Gesture.DRAW: "Index finger up  ->  drawing",
        Gesture.SELECT: "Index + middle up  ->  pick a tool",
        Gesture.FIST: "Fist  ->  stopped",
        Gesture.IDLE: "Show your hand to the camera",
    }[gesture]
