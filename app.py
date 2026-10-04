"""AI Air Drawing - entry point.

    python app.py                 # run normally
    python app.py --camera 1      # pick another webcam
    python app.py --landmarks     # show the 21-point hand skeleton
    python app.py --self-test     # verify the install without a webcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `python app.py` work from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.drawing_app import AirDrawingApp, AppSettings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="AI Air Drawing - draw in the air with your index finger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="webcam index (0 = built-in, try 1 or 2 for external)")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help="capture width")
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="capture height")
    parser.add_argument("--brush", type=int, default=config.BRUSH_SIZE, help="brush thickness in px")
    parser.add_argument("--eraser", type=int, default=config.ERASER_SIZE, help="eraser thickness in px")
    parser.add_argument("--smoothing", type=float, default=config.SMOOTHING_FACTOR,
                        help="0.05 = very smooth (laggy) ... 1.0 = raw (jittery)")
    parser.add_argument("--drawings-dir", type=str, default=config.DRAWINGS_DIR,
                        help="folder where saved PNGs go")
    parser.add_argument("--no-mirror", action="store_true", help="do not flip the camera feed")
    parser.add_argument("--landmarks", action="store_true", help="draw the hand skeleton (debug)")
    parser.add_argument("--no-fps", action="store_true", help="hide the FPS counter")
    parser.add_argument("--self-test", action="store_true",
                        help="check the environment and exercise the pipeline without a webcam")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="run for N frames then exit (smoke test, 0 = run until you quit)")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> AppSettings:
    return AppSettings(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        show_landmarks=args.landmarks,
        smoothing=args.smoothing,
        brush_size=args.brush,
        eraser_size=args.eraser,
        drawings_dir=args.drawings_dir,
        show_fps=not args.no_fps,
        max_frames=args.max_frames,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        from src.self_test import run_self_test
        return run_self_test(build_settings(args))

    try:
        app = AirDrawingApp(build_settings(args))
    except Exception as exc:  # import / init failure (e.g. MediaPipe missing)
        print(f"\n[STARTUP ERROR] {type(exc).__name__}: {exc}\n")
        return 1

    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
