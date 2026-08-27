import math

import pytest

from localization_refinement import estimate_pose_correction


def test_estimate_pose_correction_recovers_translation_and_yaw():
    reference = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.1},
        {"x": 0.2, "y": 1.2},
        {"x": -0.7, "y": 0.8},
    ]
    observed = [
        {"x": 0.5, "y": -0.3},
        {"x": 1.5, "y": -0.2},
        {"x": 0.7, "y": 0.9},
        {"x": -0.2, "y": 0.5},
    ]

    result = estimate_pose_correction(reference, observed, {"x": 0.0, "y": 0.0, "yaw": 0.0})

    assert result["ok"] is True
    # Cloud-ul observat este translatat cu (+0.5, -0.3) față de referință;
    # transformarea corectivă observat -> hartă are semnul opus.
    assert abs(result["x"] + 0.5) < 0.2
    assert abs(result["y"] - 0.3) < 0.2
    assert abs(result["yaw"]) < 0.2
    assert result["score"] < 0.35


def test_estimate_pose_correction_returns_false_when_too_few_points():
    result = estimate_pose_correction([{"x": 0.0, "y": 0.0}], [{"x": 0.1, "y": 0.1}], {"x": 0.0, "y": 0.0, "yaw": 0.0})

    assert result["ok"] is False
    assert result["reason"] == "insufficient_points"


def test_estimate_pose_correction_uses_inverse_yaw_to_align_scan_to_map():
    reference = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.1},
        {"x": 0.2, "y": 1.2},
        {"x": -0.7, "y": 0.8},
    ]
    observed_yaw = 0.20
    cosine, sine = math.cos(observed_yaw), math.sin(observed_yaw)
    observed = [
        {
            "x": cosine * point["x"] - sine * point["y"] + 0.30,
            "y": sine * point["x"] + cosine * point["y"] - 0.20,
        }
        for point in reference
    ]

    result = estimate_pose_correction(
        reference, observed, {"x": 0.0, "y": 0.0, "yaw": 0.0}
    )

    assert result["ok"] is True
    assert result["yaw"] == pytest.approx(-observed_yaw, abs=1e-6)
    assert result["x"] == pytest.approx(-0.2543, abs=0.01)
    assert result["y"] == pytest.approx(0.2556, abs=0.01)
    assert result["score"] < 1e-8
