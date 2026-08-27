import math
from typing import Dict, Optional, Sequence, Tuple


def _normalize_yaw(yaw: float) -> float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _sample_points(points: Sequence[Dict[str, float]], target: int = 120) -> list[Dict[str, float]]:
    if len(points) <= target:
        return list(points)
    step = max(1, len(points) // target)
    return [points[i] for i in range(0, len(points), step)][:target]


def _estimate_transform(reference: Sequence[Dict[str, float]], observed: Sequence[Dict[str, float]]) -> Tuple[Optional[Dict[str, float]], float]:
    ref_pts = _sample_points(reference, 120)
    obs_pts = _sample_points(observed, 120)
    if len(ref_pts) < 3 or len(obs_pts) < 3:
        return None, float("inf")

    ref_cx = sum(p["x"] for p in ref_pts) / len(ref_pts)
    ref_cy = sum(p["y"] for p in ref_pts) / len(ref_pts)
    obs_cx = sum(p["x"] for p in obs_pts) / len(obs_pts)
    obs_cy = sum(p["y"] for p in obs_pts) / len(obs_pts)

    ref_vecs = [(p["x"] - ref_cx, p["y"] - ref_cy) for p in ref_pts]
    obs_vecs = [(p["x"] - obs_cx, p["y"] - obs_cy) for p in obs_pts]

    # Căutăm transformarea observat -> referință. Ordinea veche calcula
    # unghiul invers (referință -> observat) și dubla eroarea de yaw.
    cross = sum(ov[0] * rv[1] - ov[1] * rv[0] for rv, ov in zip(ref_vecs, obs_vecs))
    dot = sum(rv[0] * ov[0] + rv[1] * ov[1] for rv, ov in zip(ref_vecs, obs_vecs))
    yaw = math.atan2(cross, dot)

    c = math.cos(yaw)
    s = math.sin(yaw)

    tx = ref_cx - (c * obs_cx - s * obs_cy)
    ty = ref_cy - (s * obs_cx + c * obs_cy)

    residuals = []
    for ref_pt, obs_pt in zip(ref_pts, obs_pts):
        pred_x = c * obs_pt["x"] - s * obs_pt["y"] + tx
        pred_y = s * obs_pt["x"] + c * obs_pt["y"] + ty
        residuals.append((pred_x - ref_pt["x"]) ** 2 + (pred_y - ref_pt["y"]) ** 2)

    if not residuals:
        return None, float("inf")

    score = sum(residuals) / len(residuals)
    return {"x": tx, "y": ty, "yaw": yaw}, score


def estimate_pose_correction(reference: Sequence[Dict[str, float]], observed: Sequence[Dict[str, float]], current_pose: Dict[str, float]) -> Dict[str, object]:
    if len(reference) < 3 or len(observed) < 3:
        return {"ok": False, "reason": "insufficient_points", "score": float("inf")}

    transform, score = _estimate_transform(reference, observed)
    if transform is None:
        return {"ok": False, "reason": "cannot_estimate", "score": float("inf")}

    corrected = {
        "x": current_pose.get("x", 0.0) + transform["x"],
        "y": current_pose.get("y", 0.0) + transform["y"],
        "yaw": _normalize_yaw(current_pose.get("yaw", 0.0) + transform["yaw"]),
    }

    return {"ok": score < 0.35, "reason": "ok" if score < 0.35 else "high_residual", "score": score, **corrected}
