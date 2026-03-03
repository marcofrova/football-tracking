from pathlib import Path
from typing import Optional

import numpy as np

try:
    # Use ultralytics YOLO for keypoint detection
    from ultralytics import YOLO  # type: ignore
except ImportError as e:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore


class PitchKeypointDetector:
    """
    Thin wrapper around a YOLO keypoint model trained on football pitch landmarks.

    Expected behaviour:
      - The model returns keypoints for 4 corners of the pitch (or another
        fixed set of landmarks) on a single frame.
      - We convert those into 4 vertices that can be used to compute a
        perspective transform.
    """

    def __init__(self, model_path: str = "pos_model/best.pt"):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Pitch keypoint model not found at {model_file}")

        if YOLO is None:
            raise ImportError(
                "ultralytics is required for the pitch keypoint model. "
                "Install it with `pip install ultralytics`."
            )

        self.model = YOLO(str(model_file))

    def detect_pitch_vertices(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run keypoint detection on a single frame and return 4 pitch vertices
        in image coordinates (x, y), dtype float32.

        This function makes some simple assumptions about the model output:
          - We take the first detection.
          - We expect at least 4 keypoints.
          - We sort keypoints into: bottom-left, top-left, top-right, bottom-right.
        You can adjust this logic to match your custom keypoint ordering.
        """
        if frame is None:
            return None

        results = self.model(frame, verbose=False)
        if not results:
            return None

        res = results[0]
        if not hasattr(res, "keypoints") or res.keypoints is None:
            return None

        # keypoints.xy shape: (num_dets, num_keypoints, 2)
        kpts = res.keypoints.xy
        if kpts is None or len(kpts) == 0:
            return None

        # Take the first detection
        pts = kpts[0].cpu().numpy()  # (num_kpts, 2)

        # Filter out dummy points at (0, 0) which often represent "no keypoint"
        mask_valid = (pts[:, 0] != 0) | (pts[:, 1] != 0)
        pts = pts[mask_valid]
        if pts.shape[0] < 4:
            return None

        # Use all valid keypoints and pick 4 extreme points as corners.
        # Strategy:
        #   - choose top 2 (smallest y) and bottom 2 (largest y)
        #   - within each row, sort by x (left to right)
        ys = pts[:, 1]

        top_indices = np.argsort(ys)[:2]
        bottom_indices = np.argsort(ys)[-2:]

        top_pts = pts[top_indices]
        bottom_pts = pts[bottom_indices]

        # Within each row, sort by x (left to right)
        top_left, top_right = top_pts[np.argsort(top_pts[:, 0])]
        bottom_left, bottom_right = bottom_pts[np.argsort(bottom_pts[:, 0])]

        ordered = np.array(
            [
                bottom_left,  # bottom-left
                top_left,  # top-left
                top_right,  # top-right
                bottom_right,  # bottom-right
            ],
            dtype=np.float32,
        )

        return ordered


