import math

import numpy as np
import pytest

from local_lidar_localization import (
    ICP_INDEX_BACKEND,
    LocalLidarLocalizer,
    _GridNearestNeighbor,
    inverse_planar,
)


def test_numpy_grid_index_finds_every_correspondence_inside_icp_radius():
    points = np.asarray([
        (-0.51, -0.49), (-0.01, 0.01), (0.49, 0.51), (1.01, -0.01),
    ], dtype=np.float64)
    query = np.asarray([
        (-0.48, -0.45), (0.03, 0.02), (0.52, 0.48), (0.98, 0.02),
    ], dtype=np.float64)

    distances, indices = _GridNearestNeighbor(points).query(query)

    expected = np.linalg.norm(query - points, axis=1)
    assert indices.tolist() == [0, 1, 2, 3]
    assert distances == pytest.approx(expected)
    assert ICP_INDEX_BACKEND in {"numpy_grid", "scipy_ckdtree"}


def _scan_from_map(map_points, pose, radius=4.0):
    inverse = inverse_planar(pose)
    cosine, sine = math.cos(inverse["yaw"]), math.sin(inverse["yaw"])
    scan = []
    for world_x, world_y in map_points:
        local_x = inverse["x"] + cosine * world_x - sine * world_y
        local_y = inverse["y"] + sine * world_x + cosine * world_y
        if 0.25 <= math.hypot(local_x, local_y) <= radius:
            scan.append({"x": local_x, "y": local_y, "z": 0.6})
    return scan


def test_icp_localizer_converges_from_manual_pose_and_becomes_ready():
    resolution = 0.05
    cells = set()
    # Geometrie asimetrică: două ziduri, o nișă și un obiect interior.
    for index in range(0, 121):
        cells.add((index, 0))
        cells.add((0, index))
    for index in range(20, 86):
        cells.add((index, 90))
    for index in range(28, 42):
        for offset in range(3):
            cells.add((index, 42 + offset))
    map_points = [(x * resolution, y * resolution) for x, y in cells]
    localizer = LocalLidarLocalizer()
    localizer.configure_map(cells, resolution, "/maps/test.pcd")

    truth = {"x": 1.35, "y": 1.10, "yaw": 0.22}
    scan = _scan_from_map(map_points, truth)
    predicted = {"x": 1.49, "y": 1.02, "yaw": 0.28}
    localizer.reset(predicted)
    result = None
    for _ in range(6):
        result = localizer.match(scan, predicted)
        assert result["ok"], result
        predicted = result["pose"]

    assert result["ready"] is True
    assert predicted["x"] == pytest.approx(truth["x"], abs=0.08)
    assert predicted["y"] == pytest.approx(truth["y"], abs=0.08)
    assert predicted["yaw"] == pytest.approx(truth["yaw"], abs=0.06)
    status = localizer.status()
    assert status["ready"] is True
    assert status["accepted_streak"] >= 3
    assert status["inliers"] >= 35
    assert status["anchor_pose"] == pytest.approx({
        "x": 1.49, "y": 1.02, "yaw": 0.28,
    })
    assert status["anchor_correction"] is not None
    assert status["anchor_correction"]["distance"] > 0.0


def test_icp_localizer_rejects_scan_without_map_overlap():
    cells = {(x, 0) for x in range(100)} | {(0, y) for y in range(100)}
    localizer = LocalLidarLocalizer()
    localizer.configure_map(cells, 0.05, "/maps/test.pcd")
    localizer.reset({"x": 20.0, "y": 20.0, "yaw": 0.0})
    scan = [{"x": 0.5 + index * 0.02, "y": 0.4, "z": 0.5} for index in range(80)]

    result = localizer.match(scan, {"x": 20.0, "y": 20.0, "yaw": 0.0})

    assert result["ok"] is False
    assert localizer.status()["ready"] is False


def test_reindexing_same_map_does_not_erase_manual_initial_pose():
    cells = {(x, 0) for x in range(100)} | {(0, y) for y in range(100)}
    localizer = LocalLidarLocalizer()
    localizer.configure_map(cells, 0.05, "/maps/test.pcd")
    anchor = {"x": 1.2, "y": 0.8, "yaw": 0.25}
    localizer.reset(anchor)

    # Reproduce ordinea finală a cursei load_robot/relocalize: workerul de
    # indexare pentru aceeași hartă termină după ce ancora a fost instalată.
    localizer.configure_map(cells, 0.05, "/maps/../maps/test.pcd")

    status = localizer.status()
    assert status["initial_pose_set"] is True
    assert status["anchor_pose"] == pytest.approx(anchor)
    assert status["error"] == "convergență LiDAR în curs"


def test_loading_different_map_clears_manual_initial_pose():
    cells = {(x, 0) for x in range(100)} | {(0, y) for y in range(100)}
    localizer = LocalLidarLocalizer()
    localizer.configure_map(cells, 0.05, "/maps/first.pcd")
    localizer.reset({"x": 1.2, "y": 0.8, "yaw": 0.25})

    localizer.configure_map(cells, 0.05, "/maps/second.pcd")

    status = localizer.status()
    assert status["initial_pose_set"] is False
    assert status["anchor_pose"] is None


def test_input_error_replaces_pending_message_without_erasing_anchor():
    cells = {(x, 0) for x in range(100)} | {(0, y) for y in range(100)}
    localizer = LocalLidarLocalizer()
    localizer.configure_map(cells, 0.05, "/maps/test.pcd")
    anchor = {"x": 1.2, "y": 0.8, "yaw": 0.25}
    localizer.reset(anchor)

    localizer.report_input_error("nu sosesc cadre LiDAR")

    status = localizer.status()
    assert status["initial_pose_set"] is True
    assert status["anchor_pose"] == pytest.approx(anchor)
    assert status["error"] == "nu sosesc cadre LiDAR"
