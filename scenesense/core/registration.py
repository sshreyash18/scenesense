import cv2
import numpy as np
from typing import List, Tuple, Optional


class SceneRegistrar:
    """
    Aligns the coordinate frame of a new scene to the baseline
    so that object positions are comparable across different
    camera positions and angles.

    Uses ORB feature matching to find corresponding points
    between baseline frames and new frames, then estimates
    the homography (2D) or essential matrix (3D) between them.
    """

    def __init__(self, max_features: int = 500, match_ratio: float = 0.75):
        """
        Args:
            max_features: Max ORB keypoints to detect per frame
            match_ratio: Lowe's ratio test threshold for match filtering
        """
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = match_ratio

    def _extract_features(self, frame: np.ndarray) -> Tuple:
        """Extract ORB keypoints and descriptors from a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def _match_features(self, desc1, desc2) -> List:
        """Match descriptors using Lowe's ratio test."""
        if desc1 is None or desc2 is None:
            return []
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.match_ratio * n.distance:
                    good.append(m)
        return good

    def estimate_transform(
        self,
        baseline_frames: List[np.ndarray],
        new_frames: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Estimate the 2D homography transform between baseline and new frames.
        Samples a few representative frames from each set and finds the
        best-matching pair to compute the transform from.

        Args:
            baseline_frames: Frames from the baseline video
            new_frames: Frames from the new video

        Returns:
            3x3 homography matrix, or None if registration failed
        """
        # sample up to 5 frames from each
        b_sample = self._sample_frames(baseline_frames, n=5)
        n_sample = self._sample_frames(new_frames, n=5)

        best_H = None
        best_match_count = 0

        for bf in b_sample:
            kp1, desc1 = self._extract_features(bf)
            for nf in n_sample:
                kp2, desc2 = self._extract_features(nf)
                matches = self._match_features(desc1, desc2)

                if len(matches) < 10:
                    continue

                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    inliers = int(mask.sum()) if mask is not None else 0
                    if inliers > best_match_count:
                        best_match_count = inliers
                        best_H = H

        if best_H is not None:
            print(f"[registration] Transform estimated with {best_match_count} inlier matches.")
        else:
            print("[registration] Warning: could not estimate transform. Scenes may be too different.")

        return best_H

    def apply_transform_to_detections(
        self,
        detections: List[dict],
        H: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> List[dict]:
        """
        Apply a homography transform to the 2D centers of detections,
        then reproject to 3D with corrected pixel positions.

        Args:
            detections: List of detection dicts with 'center' and 'position'
            H: Homography matrix from estimate_transform()
            frame_width, frame_height: Frame dimensions

        Returns:
            Detections with corrected positions
        """
        if H is None:
            return detections

        from scenesense.utils.spatial import project_to_3d

        corrected = []
        for det in detections:
            det_copy = dict(det)
            cx, cy = det["center"]

            # apply homography to the 2D center point
            pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, H)
            new_cx = int(transformed[0][0][0])
            new_cy = int(transformed[0][0][1])

            # clamp to frame bounds
            new_cx = max(0, min(new_cx, frame_width - 1))
            new_cy = max(0, min(new_cy, frame_height - 1))

            # reproject to 3D using original depth (Z unchanged)
            old_z = det["position"][2]
            new_position = project_to_3d(new_cx, new_cy, old_z, frame_width, frame_height)

            det_copy["center"] = [new_cx, new_cy]
            det_copy["position"] = new_position
            corrected.append(det_copy)

        return corrected

    def _sample_frames(self, frames: List[np.ndarray], n: int) -> List[np.ndarray]:
        """Evenly sample n frames from a list."""
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]