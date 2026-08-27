"""Localizare 2D LiDAR-to-PCD pentru executorul local al dashboardului v24.

Nu trimite comenzi robotului. Primește puncte-obstacol în cadrul base_link,
folosește odometria drept predicție și corectează poziția prin ICP robust față
de celulele statice extrase din PCD.
"""

import math
import os
import threading
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


class _GridNearestNeighbor:
    """Index 2D compact, fără SciPy, pentru corespondențele ICP apropiate.

    ICP acceptă numai corespondențe aflate la cel mult 0,42 m. Celulele de
    0,50 m permit o căutare exactă în vecinătatea 3x3 pentru toate punctele
    care pot fi acceptate, fără matricea mare scanare x hartă.
    """

    def __init__(self, points: np.ndarray, cell_size: float = 0.50):
        self.points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
        self.cell_size = float(cell_size)
        buckets = {}
        for index, point in enumerate(self.points):
            cell = self._cell(point)
            buckets.setdefault(cell, []).append(index)
        self.buckets = {
            cell: np.asarray(indices, dtype=np.int64)
            for cell, indices in buckets.items()
        }

    def _cell(self, point: np.ndarray) -> Tuple[int, int]:
        return (
            int(math.floor(float(point[0]) / self.cell_size)),
            int(math.floor(float(point[1]) / self.cell_size)),
        )

    def query(self, query_points: np.ndarray, k: int = 1):
        if int(k) != 1:
            raise ValueError("Indexul ICP local acceptă numai k=1")
        query = np.asarray(query_points, dtype=np.float64).reshape((-1, 2))
        distances = np.full(len(query), np.inf, dtype=np.float64)
        nearest = np.zeros(len(query), dtype=np.int64)
        query_groups = {}
        for index, point in enumerate(query):
            query_groups.setdefault(self._cell(point), []).append(index)

        for (cell_x, cell_y), query_indices_list in query_groups.items():
            nearby = []
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    indices = self.buckets.get(
                        (cell_x + offset_x, cell_y + offset_y)
                    )
                    if indices is not None:
                        nearby.append(indices)
            if not nearby:
                continue
            candidates = np.concatenate(nearby)
            query_indices = np.asarray(query_indices_list, dtype=np.int64)
            delta = (
                query[query_indices, np.newaxis, :]
                - self.points[candidates][np.newaxis, :, :]
            )
            squared = np.einsum("qci,qci->qc", delta, delta)
            local_nearest = np.argmin(squared, axis=1)
            distances[query_indices] = np.sqrt(
                squared[np.arange(len(query_indices)), local_nearest]
            )
            nearest[query_indices] = candidates[local_nearest]
        return distances, nearest


# Pe Orin poate exista NumPy 2.x lângă un SciPy compilat pentru NumPy 1.x.
# Folosim implicit indexul intern sigur. SciPy rămâne o optimizare explicită,
# nu o condiție de funcționare a localizării LiDAR.
cKDTree = _GridNearestNeighbor
ICP_INDEX_BACKEND = "numpy_grid"
if os.environ.get("G1_ENABLE_SCIPY_CKDTREE") == "1":
    try:  # pragma: no cover - depinde de imaginea software a robotului
        from scipy.spatial import cKDTree as _SciPyKDTree
        cKDTree = _SciPyKDTree
        ICP_INDEX_BACKEND = "scipy_ckdtree"
    except Exception:
        cKDTree = _GridNearestNeighbor


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def compose_planar(first: Dict[str, float], second: Dict[str, float]) -> Dict[str, float]:
    cosine, sine = math.cos(first["yaw"]), math.sin(first["yaw"])
    return {
        "x": first["x"] + cosine * second["x"] - sine * second["y"],
        "y": first["y"] + sine * second["x"] + cosine * second["y"],
        "yaw": wrap_angle(first["yaw"] + second["yaw"]),
    }


def inverse_planar(transform: Dict[str, float]) -> Dict[str, float]:
    cosine, sine = math.cos(transform["yaw"]), math.sin(transform["yaw"])
    return {
        "x": -cosine * transform["x"] - sine * transform["y"],
        "y": sine * transform["x"] - cosine * transform["y"],
        "yaw": wrap_angle(-transform["yaw"]),
    }


def map_to_odom(global_base: Dict[str, float], odom_base: Dict[str, float]) -> Dict[str, float]:
    return compose_planar(global_base, inverse_planar(odom_base))


def _transform_xy(points: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
    cosine, sine = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return points @ rotation.T + np.array((pose["x"], pose["y"]), dtype=np.float64)


def _rigid_delta(observed: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    observed_center = observed.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (observed - observed_center).T @ (reference - reference_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1, :] *= -1.0
        rotation = right.T @ left.T
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    translation = reference_center - rotation @ observed_center
    return {"x": float(translation[0]), "y": float(translation[1]), "yaw": yaw}


class LocalLidarLocalizer:
    """ICP limitat de predicția odometrică, cu stare explicită și watchdog."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tree = None
        self._map_points = np.empty((0, 2), dtype=np.float64)
        self._map_path = None
        self._initial_pose = None
        self._accepted_streak = 0
        self._last_accepted_at = 0.0
        self._last_attempt_at = 0.0
        self._last_pose = None
        self._last_score = None
        self._last_inlier_ratio = 0.0
        self._last_inliers = 0
        self._last_scan_points = 0
        self._anchor_correction = None
        self._last_error = "harta locală nu este configurată"

    @property
    def available(self) -> bool:
        return True

    def configure_map(self, cells: Iterable[Tuple[int, int]], resolution: float,
                      map_path: str) -> dict:
        resolution = float(resolution)
        points = np.asarray(
            [(float(x) * resolution, float(y) * resolution) for x, y in cells],
            dtype=np.float64,
        )
        if len(points) < 40:
            raise ValueError("PCD-ul nu conține suficiente celule-obstacol pentru localizare")
        # Un singur punct pe celulă este suficient și evită ponderarea excesivă
        # a mobilierului foarte dens din PCD.
        tree = cKDTree(points)
        with self._lock:
            # Indexarea PCD rulează într-un worker. Dacă operatorul setează
            # poziția cât aceeași hartă este încă reindexată, configure_map
            # poate ajunge aici după reset() și nu trebuie să șteargă ancora
            # tocmai confirmată. Pentru o hartă diferită ancora rămâne invalidă.
            same_map = bool(
                self._map_path
                and os.path.realpath(str(self._map_path))
                    == os.path.realpath(str(map_path))
            )
            preserved_initial_pose = (
                dict(self._initial_pose)
                if same_map and self._initial_pose is not None else None
            )
            self._tree = tree
            self._map_points = points
            self._map_path = str(map_path)
            self._initial_pose = preserved_initial_pose
            self._accepted_streak = 0
            self._last_accepted_at = 0.0
            self._last_pose = (
                dict(preserved_initial_pose) if preserved_initial_pose else None
            )
            self._last_score = None
            self._last_inlier_ratio = 0.0
            self._last_inliers = 0
            self._last_scan_points = 0
            self._anchor_correction = None
            self._last_error = (
                "convergență LiDAR în curs"
                if preserved_initial_pose
                else "aștept poziția inițială și scanări LiDAR"
            )
        return {
            "map_points": int(len(points)),
            "map_path": str(map_path),
            "index_backend": ICP_INDEX_BACKEND,
        }

    def clear(self) -> None:
        with self._lock:
            self._tree = None
            self._map_points = np.empty((0, 2), dtype=np.float64)
            self._map_path = None
            self._initial_pose = None
            self._accepted_streak = 0
            self._last_accepted_at = 0.0
            self._last_pose = None
            self._last_scan_points = 0
            self._anchor_correction = None
            self._last_error = "nicio hartă locală încărcată"

    def reset(self, initial_pose: Dict[str, float]) -> None:
        with self._lock:
            self._initial_pose = {
                "x": float(initial_pose["x"]),
                "y": float(initial_pose["y"]),
                "yaw": wrap_angle(float(initial_pose["yaw"])),
            }
            self._accepted_streak = 0
            self._last_accepted_at = 0.0
            self._last_pose = dict(self._initial_pose)
            self._last_score = None
            self._last_inlier_ratio = 0.0
            self._last_inliers = 0
            self._last_scan_points = 0
            self._anchor_correction = None
            self._last_error = "convergență LiDAR în curs"

    def report_input_error(self, reason: str) -> None:
        """Explică lipsa intrării fără să invalideze ancora manuală."""
        with self._lock:
            if self._initial_pose is not None and self._accepted_streak < 3:
                self._last_error = str(reason)

    @staticmethod
    def _prepare_scan(points: Sequence[dict], voxel: float = 0.08,
                      limit: int = 700) -> np.ndarray:
        cells = {}
        for point in points:
            x, y = float(point["x"]), float(point["y"])
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            distance = math.hypot(x, y)
            if not 0.25 <= distance <= 4.0:
                continue
            cell = (round(x / voxel), round(y / voxel))
            cells.setdefault(cell, (x, y))
        values = list(cells.values())
        if len(values) > limit:
            step = len(values) / float(limit)
            values = [values[min(len(values) - 1, int(index * step))] for index in range(limit)]
        return np.asarray(values, dtype=np.float64).reshape((-1, 2))

    def match(self, points_base: Sequence[dict], predicted_pose: Dict[str, float]) -> dict:
        scan = self._prepare_scan(points_base)
        with self._lock:
            tree = self._tree
            initialized = self._initial_pose is not None
            tracking = self._accepted_streak >= 3
        attempted_at = time.monotonic()
        if tree is None:
            return self._reject(attempted_at, "harta locală nu este configurată", len(scan))
        if not initialized:
            return self._reject(attempted_at, "poziția inițială nu este setată", len(scan))
        if len(scan) < 35:
            return self._reject(attempted_at, "prea puține puncte-obstacol LiDAR", len(scan))

        pose = {
            "x": float(predicted_pose["x"]),
            "y": float(predicted_pose["y"]),
            "yaw": wrap_angle(float(predicted_pose["yaw"])),
        }
        max_correspondence = 0.42 if not tracking else 0.30
        for _ in range(9):
            world = _transform_xy(scan, pose)
            distances, indices = tree.query(world, k=1)
            mask = distances <= max_correspondence
            if int(mask.sum()) < 30:
                return self._reject(
                    attempted_at,
                    "scanarea nu se suprapune suficient peste obstacolele PCD",
                    len(scan),
                    int(mask.sum()),
                )
            # Eliminăm coada de outlieri/dinamice chiar în interiorul pragului.
            kept_distances = distances[mask]
            robust_limit = min(
                max_correspondence,
                float(np.quantile(kept_distances, 0.75)) + 0.05,
            )
            mask &= distances <= robust_limit
            if int(mask.sum()) < 25:
                return self._reject(attempted_at, "corespondențe ICP instabile", len(scan), int(mask.sum()))
            delta = _rigid_delta(world[mask], self._map_points[indices[mask]])
            # Un singur pas nu poate muta brutal frame-ul robotului.
            delta_distance = math.hypot(delta["x"], delta["y"])
            if delta_distance > 0.18:
                scale = 0.18 / delta_distance
                delta["x"] *= scale
                delta["y"] *= scale
            delta["yaw"] = max(-0.07, min(0.07, delta["yaw"]))
            pose = compose_planar(delta, pose)
            if math.hypot(delta["x"], delta["y"]) < 0.004 and abs(delta["yaw"]) < 0.003:
                break

        final_world = _transform_xy(scan, pose)
        distances, _ = tree.query(final_world, k=1)
        inlier_mask = distances <= 0.22
        inliers = int(inlier_mask.sum())
        ratio = inliers / float(len(scan))
        score = float(np.median(distances[inlier_mask])) if inliers else float("inf")
        correction_distance = math.hypot(
            pose["x"] - predicted_pose["x"], pose["y"] - predicted_pose["y"]
        )
        correction_yaw = abs(wrap_angle(pose["yaw"] - predicted_pose["yaw"]))
        max_translation = 0.55 if not tracking else 0.28
        max_yaw = 0.28 if not tracking else 0.14
        accepted = bool(
            inliers >= 35 and ratio >= 0.16 and score <= 0.14
            and correction_distance <= max_translation and correction_yaw <= max_yaw
        )
        if not accepted:
            return self._reject(
                attempted_at,
                (
                    f"ICP respins: inliers={inliers}/{len(scan)} ratio={ratio:.2f} "
                    f"median={score:.3f} corecție={correction_distance:.2f}m/{math.degrees(correction_yaw):.1f}°"
                ),
                len(scan), inliers, score, ratio,
            )

        # Filtrare moderată: ICP corectează deriva fără salturi de frame.
        alpha = 0.45 if not tracking else 0.25
        filtered = {
            "x": float(predicted_pose["x"]) + alpha * (pose["x"] - float(predicted_pose["x"])),
            "y": float(predicted_pose["y"]) + alpha * (pose["y"] - float(predicted_pose["y"])),
            "yaw": wrap_angle(
                float(predicted_pose["yaw"])
                + alpha * wrap_angle(pose["yaw"] - float(predicted_pose["yaw"]))
            ),
        }
        with self._lock:
            self._accepted_streak = min(1000, self._accepted_streak + 1)
            self._last_accepted_at = attempted_at
            self._last_attempt_at = attempted_at
            self._last_pose = dict(filtered)
            self._last_score = score
            self._last_inlier_ratio = ratio
            self._last_inliers = inliers
            self._last_scan_points = int(len(scan))
            self._last_error = ""
            if self._accepted_streak == 3 and self._initial_pose is not None:
                self._anchor_correction = {
                    "dx": filtered["x"] - self._initial_pose["x"],
                    "dy": filtered["y"] - self._initial_pose["y"],
                    "distance": math.hypot(
                        filtered["x"] - self._initial_pose["x"],
                        filtered["y"] - self._initial_pose["y"],
                    ),
                    "dyaw": wrap_angle(
                        filtered["yaw"] - self._initial_pose["yaw"]
                    ),
                }
            ready = self._accepted_streak >= 3
        return {
            "ok": True, "ready": ready, "pose": filtered,
            "score": score, "inlier_ratio": ratio, "inliers": inliers,
            "scan_points": int(len(scan)),
        }

    def _reject(self, attempted_at: float, reason: str, scan_points: int,
                inliers: int = 0, score: Optional[float] = None,
                ratio: float = 0.0) -> dict:
        with self._lock:
            self._last_attempt_at = attempted_at
            self._accepted_streak = max(0, self._accepted_streak - 1)
            self._last_error = str(reason)
            self._last_score = score
            self._last_inlier_ratio = float(ratio)
            self._last_inliers = int(inliers)
            self._last_scan_points = int(scan_points)
        return {
            "ok": False, "ready": False, "reason": str(reason),
            "scan_points": int(scan_points), "inliers": int(inliers),
            "score": score, "inlier_ratio": float(ratio),
        }

    def status(self, max_age: float = 1.5) -> dict:
        with self._lock:
            now = time.monotonic()
            age = now - self._last_accepted_at if self._last_accepted_at else None
            ready = bool(
                self._tree is not None and self._accepted_streak >= 3
                and age is not None and age <= float(max_age)
            )
            return {
                "available": self.available,
                "map_configured": self._tree is not None,
                "map_path": self._map_path,
                "map_points": int(len(self._map_points)),
                "initial_pose_set": self._initial_pose is not None,
                "ready": ready,
                "accepted_streak": self._accepted_streak,
                "age": age,
                "pose": dict(self._last_pose) if self._last_pose else None,
                "anchor_pose": dict(self._initial_pose) if self._initial_pose else None,
                "anchor_correction": (
                    dict(self._anchor_correction)
                    if self._anchor_correction else None
                ),
                "score": self._last_score,
                "inlier_ratio": self._last_inlier_ratio,
                "inliers": self._last_inliers,
                "scan_points": self._last_scan_points,
                "error": self._last_error,
                "source": "/utlidar/cloud_livox_mid360 + pelvis odom + PCD ICP",
                "index_backend": ICP_INDEX_BACKEND,
            }
