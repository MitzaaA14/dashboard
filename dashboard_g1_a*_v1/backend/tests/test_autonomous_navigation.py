import asyncio
import base64
import math
import time
from pathlib import Path

import pytest

from autonomous_navigation import (
    AutonomousNavigator,
    NativeWaypointNavigator,
    PCDGridPlanner,
    plan_pcd_route,
)


def test_v24_astar_keeps_v23_core_with_explicit_live_route_extensions():
    """V24 păstrează API-ul v23, dar adaugă ruta densă/live cerută."""
    current = Path(__file__).parents[1] / "autonomous_navigation.py"
    reference = Path(__file__).parents[3] / "dashboard_g1_v23" / "backend" / "autonomous_navigation.py"
    current_source = current.read_text(encoding="utf-8")
    reference_source = reference.read_text(encoding="utf-8")

    for symbol in (
        "class PCDGridPlanner", "def plan_pcd_route",
        "class NativeWaypointNavigator", "class AutonomousNavigator",
    ):
        assert symbol in current_source
        assert symbol in reference_source
    assert "LIVE_ROUTE_WAYPOINT_SPACING = 0.85" in current_source
    assert "def _remaining_route_blocked" in current_source


def write_pcd(path: Path, points):
    with path.open("w") as stream:
        stream.write(
            "# .PCD v0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
            f"WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA ascii\n"
        )
        for point in points:
            stream.write("%s %s %s\n" % point)


def test_astar_routes_through_wall_opening(tmp_path):
    points = []
    # Puncte joase extind limitele hărții fără să devină obstacole.
    for x in range(-3, 4):
        for y in range(-3, 4):
            points.append((x, y, 0.0))
    # Perete la x=0, cu trecere lată în jurul y=0.
    points.extend((0.0, y / 10, 0.8) for y in range(-30, -10))
    points.extend((0.0, y / 10, 0.8) for y in range(11, 31))
    path = tmp_path / "map.pcd"
    write_pcd(path, points)
    planner = PCDGridPlanner(resolution=0.15, robot_radius=0.35)
    planner.load(str(path))
    route = planner.plan((-2.0, 0.0), (2.0, 0.0))
    assert route[0][0] < 0
    assert route[-1][0] > 0
    assert all(cell not in planner.occupied for cell in map(lambda p: planner.world_to_cell(*p), route))


def test_goal_inside_inflated_obstacle_is_rejected(tmp_path):
    path = tmp_path / "map.pcd"
    write_pcd(path, [(-2, -2, 0), (2, 2, 0), (0, 0, 0.8)])
    planner = PCDGridPlanner()
    planner.load(str(path))
    with pytest.raises(ValueError, match="Ținta"):
        planner.plan((-1.0, -1.0), (0.0, 0.0))


def test_static_costmap_keeps_raw_cells_separate_from_inflation(tmp_path):
    path = tmp_path / "costmap_layers.pcd"
    write_pcd(path, [
        (-2.0, -2.0, 0.0), (2.0, 2.0, 0.0),
        (0.0, 0.0, 0.8), (0.01, 0.01, 0.8), (0.02, 0.02, 0.8),
    ])
    planner = PCDGridPlanner(
        resolution=0.15, robot_radius=0.42, min_obstacle_points=3
    )
    planner.load(str(path))

    obstacle_cell = planner.world_to_cell(0.0, 0.0)
    assert obstacle_cell in planner.raw_static_occupied
    assert planner.raw_static_occupied <= planner.static_occupied
    assert planner.static_occupied - planner.raw_static_occupied

    planner.clear_robot_footprint(0.0, 0.0)
    assert obstacle_cell not in planner.raw_static_occupied
    assert obstacle_cell not in planner.static_occupied


def test_static_costmap_preview_returns_png_and_exact_parameters(tmp_path):
    from server import _build_static_costmap_preview

    path = tmp_path / "costmap_preview.pcd"
    write_pcd(path, [
        (-2.0, -2.0, 0.0), (2.0, 2.0, 0.0),
        (0.0, 0.0, 0.8), (0.01, 0.01, 0.8), (0.02, 0.02, 0.8),
    ])

    preview = _build_static_costmap_preview(str(path))
    png = base64.b64decode(preview["image_b64"])

    assert preview["success"] is True
    assert preview["resolution"] == pytest.approx(0.10)
    assert preview["clearance_radius"] == pytest.approx(0.25)
    assert preview["comfort_radius"] == pytest.approx(0.50)
    assert preview["planner_filter"]["min_points_per_cell"] == 3
    assert preview["counts"]["raw_obstacle"] == 1
    assert preview["counts"]["inflated_buffer"] > 0
    assert png.startswith(b"\x89PNG\r\n\x1a\n")

    tuned = _build_static_costmap_preview(
        str(path), resolution=0.05, robot_radius=0.30,
        min_obstacle_points=1, obstacle_min_z=0.05, obstacle_max_z=2.00,
    )
    assert tuned["resolution"] == pytest.approx(0.05)
    assert tuned["clearance_radius"] == pytest.approx(0.30)
    assert tuned["planner_filter"] == {
        "min_z": 0.05, "max_z": 2.00, "min_points_per_cell": 1,
        "level_to_floor": True, "floor_tolerance": 0.08,
            "comfort_radius": 0.50, "clearance_weight": 3.50,
    }
    assert tuned["width"] > preview["width"]

    with pytest.raises(ValueError, match="Z minim"):
        _build_static_costmap_preview(
            str(path), obstacle_min_z=1.0, obstacle_max_z=0.5
        )


def test_navigation_costmap_clamps_unsafe_route_radius():
    from server import _parse_navigation_costmap

    settings = _parse_navigation_costmap({
        "costmap": {"radius": 0.10, "comfort_radius": 0.20}
    })

    assert settings["robot_radius"] == pytest.approx(0.20)
    assert settings["comfort_radius"] == pytest.approx(0.20)
    assert settings["clearance_weight"] == pytest.approx(3.50)


def test_navigation_speed_is_capped_for_stable_indoor_walk():
    from server import _parse_navigation_goal

    _, _, _, speed, _ = _parse_navigation_goal({
        "x": 1.0, "y": 2.0, "yaw": 0.0, "speed": 0.90, "timeout": 0.0,
    })

    assert speed == pytest.approx(0.35)


def test_navigation_gesture_whitelist():
    from server import _parse_navigation_gesture

    assert _parse_navigation_gesture({}) == "none"
    assert _parse_navigation_gesture({"gesture": "wave"}) == "wave"
    assert _parse_navigation_gesture({"gesture": "kiss"}) == "kiss"
    assert _parse_navigation_gesture({"gesture": "handshake"}) == "handshake"
    assert _parse_navigation_gesture({"gesture": "clap"}) == "clap"
    with pytest.raises(ValueError, match="Gestul"):
        _parse_navigation_gesture({"gesture": "arbitrary_action_id"})


def test_g1_gestures_use_only_whitelisted_arm_action_ids():
    from robot_client import SportClient

    class FakeArm:
        def __init__(self):
            self.calls = []

        def ExecuteAction(self, action_id):
            self.calls.append(action_id)
            return 0

    arm = FakeArm()
    client = SportClient.__new__(SportClient)
    client._arm_action_client = arm
    client._arm_action_map = {
        "high wave": 26, "two-hand kiss": 11,
        "shake hand": 27, "clap": 17, "release arm": 99,
    }
    client.is_locomotion_ready = lambda: True

    assert client.execute_gesture("wave")["success"]
    assert client.execute_gesture("kiss")["success"]
    assert client.execute_gesture("handshake")["success"]
    assert client.execute_gesture("clap")["success"]
    assert client.release_arms()["success"]
    assert arm.calls == [26, 11, 27, 17, 99]


def test_dynamic_camera_observation_replaces_previous_position():
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.bounds = (-30, 30, -30, 30)
    planner.add_dynamic_obstacle(0.50, 0.0, radius=0.20, source="camera")
    old_cells = set(planner.dynamic_occupied)

    planner.clear_dynamic_source("camera")
    planner.add_dynamic_obstacle(1.50, 0.0, radius=0.20, source="camera")

    assert old_cells.isdisjoint(planner.dynamic_occupied)
    assert planner.dynamic_occupied


def test_tilted_floor_is_leveled_and_excluded_from_costmap(tmp_path):
    def floor_z(x, y):
        return 0.30 + 0.08*x - 0.04*y

    points=[]
    for xi in range(-20,21):
        for yi in range(-20,21):
            x,y=xi/5,yi/5
            points.append((x,y,floor_z(x,y)))
    for _ in range(4):
        points.extend([
            (0.00,0.00,floor_z(0.00,0.00)+0.75),
            (0.02,0.01,floor_z(0.02,0.01)+0.75),
        ])
    path=tmp_path/"tilted_floor.pcd"
    write_pcd(path,points)

    planner=PCDGridPlanner(resolution=0.15,robot_radius=0.42,min_obstacle_points=3)
    planner.load(str(path),obstacle_min_z=0.15,obstacle_max_z=1.60,level_to_floor=True)

    assert planner.floor_plane is not None
    assert planner.floor_plane["tilt_deg"] == pytest.approx(5.12,abs=0.35)
    assert planner.world_to_cell(0.0,0.0) in planner.raw_static_occupied
    assert planner.world_to_cell(2.0,2.0) not in planner.raw_static_occupied


def test_soft_comfort_zone_does_not_close_narrow_corridor(tmp_path):
    points=[(x/10,y/10,0.0) for x in range(-25,26) for y in range(-5,6)]
    for x in range(-25,26):
        points.extend([(x/10,-0.30,0.8)]*3)
        points.extend([(x/10,0.30,0.8)]*3)
    path=tmp_path/"soft_narrow_corridor.pcd"
    write_pcd(path,points)

    planner=PCDGridPlanner(resolution=0.10,robot_radius=0.20,
                           comfort_radius=0.35,clearance_weight=2.5,
                           min_obstacle_points=3)
    planner.load(str(path))
    route=planner.plan((-2.0,0.0),(2.0,0.0))

    assert route[0][0]<0<route[-1][0]
    assert planner.world_to_cell(0.0,0.0) in planner.clearance_cost
    assert planner.world_to_cell(0.0,0.0) not in planner.static_occupied


def test_live_lidar_guard_uses_relative_height_and_rejects_floor_points():
    from robot_client import ObstacleGuard

    guard=ObstacleGuard()
    guard.set_floor_plane({"a":0.10,"b":0.0,"c":0.30})
    pose={"x":0.0,"y":0.0,"yaw":0.0}
    floor_points=[{"x":0.50+i*0.01,"y":0.0,"z":0.30+0.10*(0.50+i*0.01)} for i in range(12)]
    guard.update_lidar_points(floor_points,pose)
    assert not guard.is_blocked(0.2,0.0)

    obstacle_points=[{"x":0.50+i*0.01,"y":0.0,"z":0.30+0.10*(0.50+i*0.01)+0.35} for i in range(5)]
    guard.update_lidar_points(obstacle_points,pose)
    assert guard.is_blocked(0.2,0.0)


def test_live_lidar_guard_preserves_obstacle_lateral_position():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    guard.set_floor_plane({"a": 0.0, "b": 0.0, "c": 0.0})
    pose = {"x": 2.0, "y": 3.0, "yaw": math.pi / 2}
    # În cadrul robotului: circa 0,55 m înainte și 0,25 m în stânga.
    # La yaw=90°, aceasta corespunde aproximativ coordonatelor map (1,75; 3,55).
    points = [
        {"x": 1.75 + i * 0.004, "y": 3.55 + i * 0.004, "z": 0.45}
        for i in range(6)
    ]

    guard.update_lidar_points(points, pose)
    forward, left = guard.front_obstacle_vector()

    assert forward == pytest.approx(0.56, abs=0.03)
    assert left == pytest.approx(0.25, abs=0.03)


def test_live_lidar_ignores_saved_wall_and_reconstructs_new_chair():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    guard.configure_navigation_map(
        {"a": 0.0, "b": 0.0, "c": 0.0},
        resolution=0.10,
        raw_static_cells={(8, 0)},
        robot_radius=0.20,
    )
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    saved_wall = [
        {"x": 0.80, "y": -0.02 + index * 0.008, "z": 0.45}
        for index in range(8)
    ]
    guard.update_lidar_points(saved_wall, pose)
    assert not guard.is_navigation_blocked()
    assert guard.front_obstacle_shape() is None

    chair = [
        {"x": 0.48 + index * 0.008, "y": 0.17 + index * 0.004, "z": 0.45}
        for index in range(8)
    ]
    guard.update_lidar_points(saved_wall + chair, pose)
    shape = guard.front_obstacle_shape()

    assert guard.is_navigation_blocked()
    assert shape is not None
    assert len(shape["points"]) >= 1
    assert all(x < 0.70 for x, _ in shape["points"])


def test_lidar_cluster_requires_confirmation_and_survives_short_occlusion():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    guard.set_floor_plane({"a": 0.0, "b": 0.0, "c": 0.0})
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    points = [
        {"x": 0.74 + index * 0.006, "y": -0.03 + index * 0.008, "z": 0.45}
        for index in range(8)
    ]

    guard.update_lidar_points(points, pose)
    assert guard.front_obstacle_shape() is None

    guard.update_lidar_points(points, pose)
    assert guard.front_obstacle_shape() is not None
    assert guard.is_navigation_blocked()

    # Un cadru gol nu șterge scaunul; evităm dispariția în unghiul mort.
    guard.update_lidar_points([], pose)
    assert guard.front_obstacle_shape() is not None

    with guard._lock:
        for track in guard._lidar_tracks.values():
            track["last_seen"] -= 3.0
    guard.update_lidar_points([], pose)
    assert guard.front_obstacle_shape() is None


def test_two_separate_side_objects_leave_center_corridor_open():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    guard.configure_navigation_map(
        {"a": 0.0, "b": 0.0, "c": 0.0},
        resolution=0.10, raw_static_cells=set(), robot_radius=0.20,
    )
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    left_object = [
        {"x": 0.60 + index * 0.006, "y": 0.48 + index * 0.004, "z": 0.45}
        for index in range(8)
    ]
    right_object = [
        {"x": 0.60 + index * 0.006, "y": -0.48 - index * 0.004, "z": 0.45}
        for index in range(8)
    ]
    points = left_object + right_object

    guard.update_lidar_points(points, pose)
    guard.update_lidar_points(points, pose)

    status = guard.sensor_status()
    assert set(status["lidar_zones"]) == {"left", "right"}
    assert status["lidar_zone_distances"]["left"] > 0.56
    assert status["lidar_zone_distances"]["right"] > 0.56
    assert not guard.is_navigation_blocked()
    assert not guard.is_lateral_clear(1)
    assert not guard.is_lateral_clear(-1)
    assert guard.front_obstacle_shape() is not None


def test_near_lidar_side_object_protects_hand_without_closing_wide_corridor():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    guard.configure_navigation_map(
        {"a": 0.0, "b": 0.0, "c": 0.0},
        resolution=0.10, raw_static_cells=set(), robot_radius=0.20,
    )
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    chair_corner = [
        {"x": 0.24 + index * 0.004, "y": 0.38 + index * 0.004, "z": 0.65}
        for index in range(8)
    ]

    guard.update_lidar_points(chair_corner, pose)
    guard.update_lidar_points(chair_corner, pose)

    status = guard.sensor_status()
    assert status["lidar_zones"] == ["left"]
    assert status["lidar_zone_distances"]["left"] < 0.56
    assert guard.is_navigation_blocked()


def test_dynamic_comfort_does_not_merge_objects_across_wide_gap():
    planner = PCDGridPlanner(
        resolution=0.05, robot_radius=0.25,
        comfort_radius=0.65, clearance_weight=6.0,
    )
    planner.add_dynamic_points(
        [(0.0, 0.0)], inflation_radius=0.30,
        observed_at=10.0, source="lidar",
    )

    assert planner.world_to_cell(0.40, 0.0) in planner.dynamic_clearance_cost
    assert planner.world_to_cell(0.50, 0.0) not in planner.dynamic_clearance_cost


def test_navigation_camera_stops_before_center_or_hand_collision():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    guard.update({"center": {"level": "danger", "dist": 0.58}})
    assert guard.is_navigation_blocked()

    guard = ObstacleGuard()
    guard.update({"left": {"level": "danger", "dist": 0.60}})
    # Un obiect doar lateral la 60 cm nu mai oprește la nesfârșit ruta
    # frontală când LiDAR-ul nu confirmă geometria pe culoar.
    assert not guard.is_navigation_blocked()
    assert guard.is_blocked(0.20, 0.0)
    forward, left = guard.front_obstacle_vector()
    assert forward == pytest.approx(0.60)
    assert left > 0.30

    guard = ObstacleGuard()
    guard.update({"right": {"level": "danger", "dist": 0.40}})
    assert guard.is_navigation_blocked()

    guard = ObstacleGuard()
    guard.update({"center": {"level": "danger", "dist": 0.40}})
    assert guard.is_navigation_blocked()

    guard = ObstacleGuard()
    guard.update({"left": {"level": "warning", "dist": 0.84}})
    # Mobilierul lateral aflat aproape la un metru nu mai generează discul fals.
    assert not guard.is_navigation_blocked()
    # Teleoperarea rămâne și ea liberă la simpla avertizare laterală îndepărtată.
    assert not guard.is_blocked(0.20, 0.0)
    forward, left = guard.front_obstacle_vector()
    assert forward == pytest.approx(0.84)
    assert left > 0.30


def test_autonomy_requires_both_realsense_and_lidar_fresh():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    guard.update({"center": {"level": "safe", "dist": 1.50}})
    status = guard.sensor_status()
    assert status["camera_fresh"]
    assert not status["lidar_fresh"]
    assert guard.has_fresh_data()
    assert not guard.navigation_sensors_ready()

    guard.update_lidar_points([], {"x": 0.0, "y": 0.0, "yaw": 0.0})
    status = guard.sensor_status()
    assert status["camera_fresh"] and status["lidar_fresh"]
    assert guard.navigation_sensors_ready()


def test_lidar_does_not_flicker_stale_between_one_and_two_seconds(monkeypatch):
    import robot_client

    now = 100.0
    monkeypatch.setattr(robot_client.time, "time", lambda: now)
    guard = robot_client.ObstacleGuard()
    guard.update({"center": {"level": "safe", "dist": 1.50}})
    guard.update_lidar_points([], {"x": 0.0, "y": 0.0, "yaw": 0.0})

    now = 101.5
    assert guard.sensor_status()["lidar_fresh"] is True
    assert guard.navigation_sensors_ready() is True

    now = 102.1
    assert guard.sensor_status()["lidar_fresh"] is False
    assert guard.navigation_sensors_ready() is False


def test_navigation_preflight_reports_camera_and_lidar_separately():
    from server import navigation_preflight

    result = asyncio.run(navigation_preflight())

    assert "camera_obstacle_sensor_fresh" in result["checks"]
    assert "lidar_obstacle_sensor_fresh" in result["checks"]
    assert "obstacle_sensors_fresh" in result["checks"]
    assert set(result["obstacle_sensors"]) >= {
        "camera_fresh", "lidar_fresh", "camera_age", "lidar_age"
    }


def test_navigation_flight_recorder_persists_event_sensor_and_slam_context(tmp_path):
    import json
    import server

    previous_path = server.navigation_flight_log_path
    previous_slam = server.slam_runtime_info
    try:
        server.navigation_flight_log_path = tmp_path / "flight.jsonl"
        server.slam_runtime_info = {
            "message_type": "pos_info",
            "current_pose": {"x": 1.0, "y": 2.0},
            "paused": False,
        }
        server._append_navigation_flight_log({
            "type": "nav_status", "state": "waiting_obstacle",
            "pose": {"x": 1.0, "y": 2.0},
        })

        record = json.loads(server.navigation_flight_log_path.read_text().strip())
        assert record["event"]["state"] == "waiting_obstacle"
        assert set(record["obstacle_sensors"]) >= {"camera_fresh", "lidar_fresh"}
        assert record["slam"]["message_type"] == "pos_info"
        assert record["slam"]["current_pose"] == {"x": 1.0, "y": 2.0}
    finally:
        server.navigation_flight_log_path = previous_path
        server.slam_runtime_info = previous_slam


def test_flight_recorder_captures_exact_1102_command_and_feedback(tmp_path, monkeypatch):
    import json
    import server

    previous_path = server.navigation_flight_log_path
    try:
        server.navigation_flight_log_path = tmp_path / "dispatch.jsonl"
        monkeypatch.setattr(
            server.slam_client, "pose_navigation",
            lambda x, y, yaw, speed: {
                "success": True, "sent": [x, y, yaw, speed],
            },
        )

        async def accepted_feedback(_api_id, _sent_at, timeout=4.0):
            return {
                "status_code": 0,
                "payload": {"succeed": True, "errorCode": 0},
            }

        monkeypatch.setattr(server, "_wait_slam_feedback", accepted_feedback)

        async def direct_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
        result = asyncio.run(server._dispatch_native_waypoint(1.2, -0.4, 0.3, 0.15))

        records = [
            json.loads(line)["event"]
            for line in server.navigation_flight_log_path.read_text().splitlines()
        ]
        assert result["success"]
        assert [record["type"] for record in records] == [
            "api_1102_dispatch", "api_1102_result"
        ]
        assert records[0]["command"] == {
            "x": 1.2, "y": -0.4, "yaw": 0.3, "speed": 0.15,
        }
        assert records[1]["state"] == "accepted"
        assert records[1]["latency_s"] >= 0.0
    finally:
        server.navigation_flight_log_path = previous_path


def test_1102_retries_once_after_late_previous_waypoint_completion(
        tmp_path, monkeypatch):
    import json
    import server

    previous_path = server.navigation_flight_log_path
    previous_completion = server.slam_last_completion
    sends = []
    try:
        server.navigation_flight_log_path = tmp_path / "late_completion.jsonl"
        server.slam_last_completion = {}

        def pose_navigation(x, y, yaw, speed):
            sends.append((x, y, yaw, speed))
            return {"success": True}

        monkeypatch.setattr(server.slam_client, "pose_navigation", pose_navigation)
        feedback_waits = 0

        async def feedback_after_retry(_api_id, sent_at, timeout=4.0):
            nonlocal feedback_waits
            feedback_waits += 1
            if feedback_waits == 1:
                # FINISHED aparține punctului anterior: robotul este încă la
                # ~1,3 m de ținta nouă, exact ca în logul real.
                server.slam_last_completion = {
                    "received_at": sent_at + 0.01,
                    "current_pose": {"x": 6.03, "y": 1.95},
                    "machine_state": "FINISHED",
                    "arrived": True,
                }
                return None
            return {
                "status_code": 0,
                "payload": {"succeed": True, "errorCode": 0},
            }

        monkeypatch.setattr(server, "_wait_slam_feedback", feedback_after_retry)

        async def direct_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
        result = asyncio.run(
            server._dispatch_native_waypoint(7.04, 2.80, 0.71, 0.158)
        )

        assert result["success"]
        assert result["attempts"] == 2
        assert result["recovered_from_late_completion"] is True
        assert len(sends) == 2

        records = [
            json.loads(line)["event"]
            for line in server.navigation_flight_log_path.read_text().splitlines()
        ]
        assert any(
            record["type"] == "api_1102_retry"
            and record["state"] == "late_previous_completion"
            for record in records
        )
    finally:
        server.navigation_flight_log_path = previous_path
        server.slam_last_completion = previous_completion


def test_1102_accepts_fresh_completion_only_near_current_target(monkeypatch):
    import server

    previous_completion = server.slam_last_completion
    try:
        server.slam_last_completion = {}
        monkeypatch.setattr(
            server.slam_client, "pose_navigation", lambda *_args: {"success": True}
        )

        async def completion_without_api_feedback(_api_id, sent_at, timeout=4.0):
            server.slam_last_completion = {
                "received_at": sent_at + 0.01,
                "current_pose": {"x": 2.01, "y": 3.02},
                "machine_state": "FINISHED",
                "arrived": True,
            }
            return None

        monkeypatch.setattr(
            server, "_wait_slam_feedback", completion_without_api_feedback
        )

        async def direct_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
        result = asyncio.run(
            server._dispatch_native_waypoint(2.0, 3.0, 0.0, 0.15)
        )

        assert result["success"]
        assert result["telemetry_confirmed"] is True
        assert result["attempts"] == 1
    finally:
        server.slam_last_completion = previous_completion


def test_intermediate_arrived_does_not_mark_final_goal_arrived():
    import server

    progress = server._slam_navigation_progress(
        {
            "arrived": True,
            "paused": False,
            "error_code": 0,
            "machine_state": "FINISHED",
            "current_pose": {"x": 6.03, "y": 1.95},
        },
        {"x": 5.48, "y": 1.98},
        {"x": 6.35, "y": 6.03},
    )

    assert progress == "navigating"


def test_unconfirmed_1201_applies_independent_zero_velocity_fallback(monkeypatch):
    import server

    calls = []

    monkeypatch.setattr(
        server.slam_client, "pause_navigation",
        lambda: calls.append("1201") or {"success": True},
    )
    monkeypatch.setattr(
        server.sport_client, "stop",
        lambda: calls.append("zero") or {"success": True, "output": "zero"},
    )

    async def no_feedback(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "_wait_slam_feedback", no_feedback)
    monkeypatch.setattr(server, "slam_runtime_info", {})

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)

    result = asyncio.run(server._pause_native_navigation())

    assert result["success"]
    assert result["locomotion_fallback"]["success"]
    assert "1201" in result["error"]
    assert calls == ["1201", "zero"]


def test_resume_requires_1202_feedback_or_fresh_unpaused_telemetry(monkeypatch):
    import server

    monkeypatch.setattr(
        server.slam_client, "resume_navigation", lambda: {"success": True}
    )

    async def no_feedback(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "_wait_slam_feedback", no_feedback)
    # O telemetrie `paused=false` proaspătă, dar anterioară comenzii, nu este
    # o confirmare validă pentru 1202 curent.
    monkeypatch.setattr(server, "slam_runtime_info", {
        "paused": False, "received_at": time.monotonic(),
    })

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)

    result = asyncio.run(server._resume_native_navigation())

    assert not result["success"]
    assert "rămâne oprit" in result["error"]


def test_realsense_detects_a_small_table_corner_in_side_zone():
    import numpy as np
    from server import detect_obstacles

    depth = np.full((480, 640), 2000, dtype=np.uint16)
    # 25x25 px în ROI: suficient de mic încât vechiul prag de 6% îl ignora.
    depth[180:205, 55:80] = 550
    zones = detect_obstacles(depth)

    assert zones["left"]["level"] == "danger"
    assert zones["left"]["dist"] == pytest.approx(0.55, abs=0.01)


def test_camera_only_obstacle_stops_without_inventing_dynamic_geometry(tmp_path):
    class CameraOnlyGuard:
        def front_obstacle_shape(self):
            return None

        def front_obstacle_vector(self, default=0.70):
            return 0.84, 0.50

    path = tmp_path / "camera_costmap.pcd"
    write_pcd(path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 31)
        for y in range(-20, 21)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.42)
    planner.load(str(path))
    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        CameraOnlyGuard(), lambda *_: asyncio.sleep(0),
        lambda: asyncio.sleep(0), lambda: asyncio.sleep(0),
        lambda _event: asyncio.sleep(0),
    )
    navigator.planner = planner

    obstacle, fallback = navigator._record_obstacle(
        {"x": 0.0, "y": 0.0, "yaw": 0.0}, time.monotonic()
    )

    assert obstacle["mode"] == "camera_stop_only"
    assert fallback is None
    assert not planner.dynamic_occupied
    assert planner.segment_is_free((0.0, 0.0), (1.50, 0.0))


def test_camera_side_warning_does_not_block_from_almost_one_meter():
    from robot_client import ObstacleGuard

    guard = ObstacleGuard()
    far_side = {
        "left": {"level": "warning", "dist": 0.85},
        "center": {"level": "safe", "dist": 2.0},
        "right": {"level": "safe", "dist": 2.0},
    }
    guard.update(far_side)
    assert not guard.is_navigation_blocked()

    close_side = dict(far_side)
    close_side["left"] = {"level": "warning", "dist": 0.40}
    guard.update(close_side)
    assert guard.is_navigation_blocked()


def test_turn_clearance_penalty_prefers_rotation_away_from_obstacles():
    planner = PCDGridPlanner(
        resolution=0.10, robot_radius=0.25,
        comfort_radius=0.40, clearance_weight=3.5,
    )
    near = (0, 0)
    far = (1, 0)
    planner.obstacle_distance = {near: 0.27, far: 0.70}

    near_cost = planner._turn_clearance_penalty(near, math.pi / 2)
    far_cost = planner._turn_clearance_penalty(far, math.pi / 2)

    assert near_cost > 1.0
    assert far_cost == 0.0


def test_safe_corner_smoothing_stays_collision_free_and_removes_redundancy(tmp_path):
    path = tmp_path / "open_corner.pcd"
    write_pcd(path, [
        (x / 5, y / 5, 0.0)
        for x in range(-10, 16)
        for y in range(-10, 16)
    ])
    planner = PCDGridPlanner(
        resolution=0.10, robot_radius=0.42,
        comfort_radius=0.65, clearance_weight=4.5,
    )
    planner.load(str(path))

    sharp = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    rounded = planner._round_safe_corners(sharp)

    assert len(rounded) <= len(sharp)
    assert rounded[0] == sharp[0]
    assert rounded[-1] == sharp[-1]
    assert all(
        planner.segment_is_free(start, end)
        for start, end in zip(rounded, rounded[1:])
    )


def test_clear_direct_corridor_has_only_endpoints_and_no_centerline_zigzag(tmp_path):
    points = [
        (x / 10, y / 10, 0.0)
        for x in range(-20, 21) for y in range(-10, 11)
    ]
    for x in range(-20, 21):
        points.extend([(x / 10, -1.0, 0.8)] * 3)
        points.extend([(x / 10, 1.0, 0.8)] * 3)
    map_path = tmp_path / "centered_corridor.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(
        resolution=0.10, robot_radius=0.10,
        comfort_radius=0.20, clearance_weight=1.0,
        min_obstacle_points=3,
    )
    planner.load(str(map_path))

    route = planner.plan((-1.8, 0.50), (1.8, 0.50))

    assert len(route) == 2
    assert all(point[1] == pytest.approx(0.50) for point in route)
    assert all(
        planner._line_cells(
            planner.world_to_cell(*a), planner.world_to_cell(*b)
        ) is not None
        for a, b in zip(route, route[1:])
    )


def test_unknown_space_is_penalized_when_observed_floor_has_a_safe_detour():
    planner = PCDGridPlanner(
        resolution=0.10, robot_radius=0.20,
        comfort_radius=0.20, unknown_space_weight=3.0,
    )
    planner.bounds = (0, 4, 0, 2)
    planner.known_free = {(x, 1) for x in range(5)}

    route = planner.plan((0.0, 0.0), (0.4, 0.0))

    assert any(y >= 0.09 for _, y in route[1:-1])


def test_side_waypoint_rotates_body_toward_travel_direction():
    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    yaw = navigator._waypoint_yaw(
        [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)],
        1,
        {"x": 0.0, "y": 0.0, "yaw": 0.0},
        0.0,
    )

    assert yaw == pytest.approx(math.pi / 2)
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    assert navigator._lateral_motion_mode(
        [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)], 1, pose
    ) == "lateral_left"
    assert navigator._lateral_motion_mode(
        [(0.0, 0.0), (0.0, -1.0), (0.0, -2.0)], 1, pose
    ) == "lateral_right"

    detour = navigator._prepare_dynamic_detour(
        [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0)], pose
    )
    assert detour == [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0)]


def test_native_control_lookahead_skips_dense_collinear_commands():
    class ClearPlanner:
        @staticmethod
        def segment_is_free(_start, _goal):
            return True

    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    navigator.planner = ClearPlanner()
    dense = [(0.0, 0.0), (0.8, 0.0), (1.6, 0.0), (2.4, 0.0)]

    assert navigator._select_control_waypoint_index(
        dense, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0}
    ) == 3


def test_future_obstacle_is_replanned_while_current_segment_keeps_moving():
    class HealthyGuard:
        @staticmethod
        def navigation_sensors_ready():
            return True

        @staticmethod
        def is_navigation_blocked():
            return False

    class DetourPlanner:
        @staticmethod
        def dynamic_costmap_points():
            return []

        @staticmethod
        def segment_is_free(_start, _goal):
            return True

        @staticmethod
        def plan(start, goal):
            return [start, (1.0, 0.6), goal]

    async def scenario():
        pauses = []

        async def pause(*_args):
            pauses.append("pause")
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
            HealthyGuard(), pause, pause, pause,
            lambda _event: asyncio.sleep(0), poll_interval=0.005,
        )
        navigator.planner = DetourPlanner()
        navigator.status["native_command_active"] = True

        result = await navigator._replan_ahead_while_moving(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], 1,
            2.0, 0.0, 0,
        )

        assert result is not None
        path, index, replans = result
        assert path == [(0.0, 0.0), (1.0, 0.6), (2.0, 0.0)]
        assert index == 2
        assert replans == 1
        assert pauses == []
        assert navigator.status["error"] == (
            "ocolire recalculată în mers; continui fără STOP"
        )

    asyncio.run(scenario())


def test_native_finished_near_start_requires_real_progress():
    # Cazul real: țintă de evitare la ~0,42 m și FINISHED la ~0,34 m după
    # numai câțiva centimetri. Nu avem voie să sărim waypoint-ul lateral.
    assert not NativeWaypointNavigator._native_completion_has_progress(
        0.42, 0.34, 0.34
    )
    # După un segment lung, firmware-ul poate termina legitim la 0,5 m.
    assert NativeWaypointNavigator._native_completion_has_progress(
        1.80, 0.50, 0.50
    )
    assert NativeWaypointNavigator._native_completion_has_progress(
        0.42, 0.25, 0.25
    )


def test_lidar_costmap_preserves_previous_chair_edge_until_ttl():
    class ShapeGuard:
        shape = None

        def front_obstacle_shape(self):
            return self.shape

    async def command(*_args):
        return {"success": True}

    guard = ShapeGuard()
    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        guard, command, command, command, lambda _event: asyncio.sleep(0),
    )
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    navigator.planner = planner
    guard.shape = {"points": [(1.0, 0.20)], "vector": (1.0, 0.20)}
    navigator._sync_lidar_costmap(observed_at=10.0)
    first_edge = planner.world_to_cell(1.0, 0.20)
    guard.shape = {"points": [(1.0, -0.20)], "vector": (1.0, -0.20)}
    navigator._sync_lidar_costmap(observed_at=10.1)
    second_edge = planner.world_to_cell(1.0, -0.20)

    assert first_edge in planner.dynamic_occupied
    assert second_edge in planner.dynamic_occupied


def test_corner_waypoint_uses_continuous_tangent_instead_of_second_rotation():
    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )

    yaw = navigator._waypoint_yaw(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        1, {"x": 0.0, "y": 0.0, "yaw": math.pi}, 0.0,
    )

    assert yaw == pytest.approx(math.pi / 4)


def test_departure_lateral_positioning_is_single_and_only_for_direct_diagonal():
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    assert NativeWaypointNavigator._departure_lateral_mode(
        [(0.0, 0.0), (1.20, 0.80)], 1, pose
    ) == "lateral_left"
    assert NativeWaypointNavigator._departure_lateral_mode(
        [(0.0, 0.0), (1.20, -0.80)], 1, pose
    ) == "lateral_right"
    assert NativeWaypointNavigator._departure_lateral_mode(
        [(0.0, 0.0), (2.00, 0.20)], 1, pose
    ) is None
    # Un traseu A* cu viraj real nu primește repoziționări laterale automate.
    assert NativeWaypointNavigator._departure_lateral_mode(
        [(0.0, 0.0), (1.20, 0.80), (2.0, 0.80)], 1, pose
    ) is None


def test_holonomic_direct_pattern_uses_forward_then_right_without_rotation():
    class ClearPlanner:
        robot_radius = 0.20
        comfort_radius = 0.65
        obstacle_distance = {}
        def segment_is_free(self, _start, _goal): return True
        def world_to_cell(self, x, y): return round(x, 2), round(y, 2)

    async def command(*_args): return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    navigator.planner = ClearPlanner()
    path, pattern = navigator._holonomic_direct_pattern(
        [(0.0, 0.0), (2.0, -0.45)],
        {"x": 0.0, "y": 0.0, "yaw": 0.0},
    )

    assert path == pytest.approx([(0.0, 0.0), (2.0, 0.0), (2.0, -0.45)])
    assert pattern["type"] == "forward_then_lateral"
    assert pattern["lateral_direction"] == "right"
    navigator.status["path_pattern"] = pattern
    assert navigator._waypoint_yaw(
        path, 1, {"x": 0.0, "y": 0.0, "yaw": 0.0}, 1.2
    ) == pytest.approx(0.0)
    assert navigator._waypoint_yaw(
        path, 2, {"x": 2.0, "y": 0.0, "yaw": 0.0}, 1.2
    ) == pytest.approx(0.0)
    assert navigator._can_smooth_handoff(
        path, 1, {"x": 1.75, "y": 0.0, "yaw": 0.0}
    )


def test_holonomic_direct_pattern_is_rejected_when_lateral_leg_is_blocked():
    class BlockedLateralPlanner:
        robot_radius = 0.20
        obstacle_distance = {}
        def segment_is_free(self, start, goal):
            return not (start == (2.0, 0.0) and goal == (2.0, 0.45))
        def world_to_cell(self, x, y): return round(x, 2), round(y, 2)

    async def command(*_args): return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    navigator.planner = BlockedLateralPlanner()
    original = [(0.0, 0.0), (2.0, 0.45)]
    path, pattern = navigator._holonomic_direct_pattern(
        original, {"x": 0.0, "y": 0.0, "yaw": 0.0}
    )

    assert path == original
    assert pattern == {"type": "direct"}


def test_departure_positioning_keeps_yaw_and_makes_useful_lateral_progress(tmp_path):
    class ClearSideGuard:
        def has_fresh_data(self): return True
        def is_lateral_clear(self, _direction): return True

    map_path = tmp_path / "departure_lateral.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 21) for y in range(-15, 16)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        commands = []

        async def command(*_args): return {"success": True}

        async def lateral(vy):
            commands.append(vy)
            pose["y"] += 0.11 if vy > 0 else -0.11
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, ClearSideGuard(),
            command, command, command, lambda _event: asyncio.sleep(0),
            lateral_velocity=lateral, stop_locomotion=command,
            poll_interval=0.005,
        )
        navigator.planner = planner
        moved = await navigator._execute_lateral_escape(
            [(0.0, 0.0), (1.20, 0.80)], 1, math.inf,
            positioning=True,
        )

        assert moved
        assert pose["yaw"] == 0.0
        assert pose["y"] >= 0.18
        assert commands and all(value > 0 for value in commands)
        assert navigator.status["state"] == "lateral_positioning"

    asyncio.run(scenario())


def test_lateral_escape_uses_direct_vy_and_slam_progress(tmp_path):
    class ClearSideGuard:
        def has_fresh_data(self): return True
        def is_lateral_clear(self, _direction): return True

    map_path = tmp_path / "direct_lateral_escape.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 21)
        for y in range(-15, 16)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        commands = []
        stops = []

        async def command(*_args): return {"success": True}

        async def lateral(vy):
            commands.append(vy)
            pose["y"] += 0.10 if vy > 0 else -0.10
            return {"success": True}

        async def stop():
            stops.append(True)
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, ClearSideGuard(),
            command, command, command, lambda _event: asyncio.sleep(0),
            lateral_velocity=lateral, stop_locomotion=stop, poll_interval=0.005,
        )
        navigator.planner = planner
        moved = await navigator._execute_lateral_escape(
            [(0.0, 0.0), (0.0, 0.40), (1.0, 0.40)], 1, math.inf,
        )

        assert moved
        assert commands and all(value > 0 for value in commands)
        # Un pas util confirmat este suficient; navigatorul oprește imediat și
        # replănuiește, fără să insiste până la vechea țintă de 0,30 m.
        assert pose["y"] >= 0.14
        assert len(commands) <= 2
        assert stops == [True]
        assert navigator.status["state"] == "lateral_evading"

    asyncio.run(scenario())


def test_recent_lidar_obstacle_on_left_selects_right_escape():
    class ClearSideGuard:
        def is_lateral_clear(self, _direction): return True

    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        ClearSideGuard(), command, command, command,
        lambda _event: asyncio.sleep(0),
        lateral_velocity=command, stop_locomotion=command,
    )
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.bounds = (-10, 10, -10, 10)
    planner.known_free = {
        (x, y) for x in range(-10, 11) for y in range(-10, 11)
    }
    navigator.planner = planner
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    navigator.recent_obstacle_info = {
        "x": 0.55, "y": 0.30, "distance": 0.63,
        "mode": "lidar_shape", "shape_points": 8,
    }
    navigator.recent_obstacle_pose = dict(pose)
    navigator.recent_obstacle_at = time.monotonic() - 4.0

    remembered = navigator._recent_obstacle_for_recovery(pose)
    recovery = navigator._stagnation_recovery_path(pose, remembered)

    assert recovery is not None
    assert recovery[1][1] < 0.0


def test_stagnation_without_lidar_retries_forward_without_lateral(tmp_path):
    class ClearGuard:
        def has_fresh_data(self): return True
        def is_navigation_blocked(self): return False
        def front_obstacle_shape(self): return None
        def is_lateral_clear(self, _direction): return True

    map_path = tmp_path / "stagnation_lateral_recovery.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-15, 21)
        for y in range(-15, 16)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.0, 0.0))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        waypoint_sends = 0
        lateral_commands = []

        async def send_waypoint(x, y, yaw, _speed):
            nonlocal waypoint_sends
            waypoint_sends += 1
            if waypoint_sends >= 2:
                pose.update({"x": x, "y": y, "yaw": yaw})
            return {"success": True}

        async def command(*_args): return {"success": True}

        async def lateral(vy):
            lateral_commands.append(vy)
            pose["y"] += 0.10 if vy > 0 else -0.10
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, ClearGuard(),
            send_waypoint, command, command, lambda _event: asyncio.sleep(0),
            lateral_velocity=lateral, stop_locomotion=command,
            poll_interval=0.005, stagnation_timeout=0.05,
        )
        result = await navigator.start(
            str(map_path), 1.0, 0.0, 0.0, timeout=2.0,
            prepared_plan={"planner": planner, "path": route, "clearance_mode": "custom"},
        )
        assert result["success"]
        await navigator.task

        assert navigator.status["state"] == "arrived", navigator.status
        assert navigator.status.get("stagnation_recoveries", 0) == 0
        assert lateral_commands == []
        assert waypoint_sends >= 2

    asyncio.run(scenario())


def test_segment_speed_slows_for_unknown_space_and_sharp_turns():
    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.bounds = (-5, 20, -5, 20)
    navigator.planner = planner
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    straight = [(0.0, 0.0), (1.0, 0.0)]
    planner.known_free = set(planner._line_cells((0, 0), (10, 0)))

    assert navigator._safe_segment_speed(straight, 1, pose, 0.40) == pytest.approx(0.40)

    planner.known_free = set()
    assert navigator._safe_segment_speed(straight, 1, pose, 0.40) == pytest.approx(0.18)

    planner.known_free = {
        (x, y) for x in range(-5, 21) for y in range(-5, 21)
    }
    sharp = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert navigator._safe_segment_speed(sharp, 1, pose, 0.40) == pytest.approx(0.20)

    navigator.status["clearance_mode"] = "narrow"
    assert navigator._safe_segment_speed(straight, 1, pose, 0.40) == pytest.approx(0.22)


def test_dynamic_lidar_shape_is_compact_and_keeps_previous_object_until_ttl(tmp_path):
    points = [(x / 5, y / 5, 0.0) for x in range(-10, 11) for y in range(-10, 11)]
    map_path = tmp_path / "dynamic_shape.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))

    planner.add_dynamic_points([(0.0, 0.0)], observed_at=10.0)
    first_cell = planner.world_to_cell(0.0, 0.0)
    far_cell = planner.world_to_cell(0.60, 0.0)
    assert first_cell in planner.dynamic_occupied
    assert planner.dynamic_sources[first_cell] == "lidar_raw"
    assert "lidar" in set(planner.dynamic_sources.values())
    assert far_cell not in planner.dynamic_occupied

    planner.add_dynamic_points([(1.0, 0.0)], observed_at=11.0)
    second_cell = planner.world_to_cell(1.0, 0.0)
    planner.expire_dynamic_obstacles(max_age=2.5, now=12.0)
    assert first_cell in planner.dynamic_occupied
    assert second_cell in planner.dynamic_occupied

    planner.expire_dynamic_obstacles(max_age=2.5, now=13.0)
    assert first_cell not in planner.dynamic_occupied
    assert second_cell in planner.dynamic_occupied


def test_lidar_shape_has_soft_clearance_outside_blocked_cells(tmp_path):
    points = [(x / 5, y / 5, 0.0) for x in range(-10, 11) for y in range(-10, 11)]
    map_path = tmp_path / "dynamic_soft_margin.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(
        resolution=0.10, robot_radius=0.20,
        comfort_radius=0.65, clearance_weight=6.0,
    )
    planner.load(str(map_path))

    planner.add_dynamic_points(
        [(0.0, 0.0)], inflation_radius=0.25, observed_at=10.0
    )
    soft_cell = planner.world_to_cell(0.40, 0.0)

    assert soft_cell not in planner.dynamic_occupied
    assert planner.dynamic_clearance_cost[soft_cell] > 0.0
    assert planner._navigation_penalty(soft_cell) > 0.0
    assert planner.clear_dynamic_source("lidar") > 0
    assert planner._navigation_penalty(soft_cell) == 0.0


def test_execution_simplifier_keeps_only_required_turns(tmp_path):
    points = [(x / 10, y / 10, 0.0) for x in range(-10, 61) for y in range(-10, 31)]
    map_path = tmp_path / "long_open_route.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))

    route = planner._simplify_polyline_for_execution([
        (0.0, 0.0), (1.0, 0.02), (2.0, 0.0), (4.20, 0.0),
    ])

    assert route == [(0.0, 0.0), (4.20, 0.0)]


def test_smooth_handoff_is_early_only_when_next_segment_is_free(tmp_path):
    points = [(x / 5, y / 5, 0.0) for x in range(-10, 16) for y in range(-10, 11)]
    map_path = tmp_path / "smooth_handoff.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))

    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.65, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    navigator.planner = planner
    gentle_path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.10)]
    tight_path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    pose = {"x": 0.65, "y": 0.0, "yaw": 0.0}

    assert navigator._handoff_distance(gentle_path, 1, 0.30) >= 0.38
    assert navigator._handoff_distance(tight_path, 1, 0.30) == pytest.approx(0.22)
    assert navigator._can_smooth_handoff(gentle_path, 1, pose)

    planner.add_dynamic_points([(1.45, 0.05)], observed_at=10.0)
    assert not navigator._can_smooth_handoff(gentle_path, 1, pose)


def test_native_short_waypoint_is_merged_before_dispatch():
    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.10)
    planner.bounds = (-2, 20, -10, 10)
    planner.known_free = {
        (x, y) for x in range(-2, 21) for y in range(-10, 11)
    }
    navigator.planner = planner
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    path = [(0.0, 0.0), (0.39, 0.02), (0.85, 0.20), (1.50, 0.20)]

    assert navigator._advance_past_native_tolerance(path, 1, pose) == 2


def test_short_departure_is_extended_in_same_safe_direction():
    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        object(), command, command, command, lambda _event: asyncio.sleep(0),
    )
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.10)
    planner.bounds = (-2, 20, -10, 10)
    planner.known_free = {
        (x, y) for x in range(-2, 21) for y in range(-10, 11)
    }
    navigator.planner = planner
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    path = [(0.0, 0.0), (0.39, 0.02), (0.85, 0.20), (1.50, 0.20)]

    stabilized = navigator._stabilize_departure_path(path, pose)

    assert math.hypot(*stabilized[1]) == pytest.approx(0.70)
    assert stabilized[1][1] / stabilized[1][0] == pytest.approx(
        path[1][1] / path[1][0]
    )
    assert planner.segment_is_free(stabilized[1], stabilized[2])


def test_shortcut_does_not_trade_clearance_peak_for_shorter_distance():
    planner = PCDGridPlanner(
        resolution=1.0, robot_radius=0.10,
        comfort_radius=0.50, clearance_weight=3.5,
    )
    planner.bounds = (0, 4, 0, 1)
    planner.known_free = {(x, y) for x in range(5) for y in range(2)}
    # Linia directă are un singur vârf de risc. Costul ei total este încă mai
    # mic decât ocolul, dar nu trebuie să taie acel colț.
    planner.clearance_cost = {(2, 0): 0.20}

    assert not planner.smooth_handoff_is_safe(
        (0.0, 0.0), (2.0, 1.0), (4.0, 0.0)
    )


def test_direct_first_ignores_soft_cost_when_hard_segment_is_clear():
    planner = PCDGridPlanner(
        resolution=1.0, robot_radius=0.10,
        comfort_radius=0.50, clearance_weight=3.5,
    )
    planner.bounds = (0, 6, -2, 2)
    planner.known_free = {
        (x, y) for x in range(7) for y in range(-2, 3)
    }
    planner.clearance_cost = {(x, 0): 2.0 for x in range(1, 6)}

    route = planner.plan((0.0, 0.0), (6.0, 0.0))

    assert route[0] == (0.0, 0.0)
    assert route[-1] == (6.0, 0.0)
    assert len(route) == 2
    assert all(point[1] == pytest.approx(0.0) for point in route)


def test_dynamic_obstacle_replans_around_blocked_straight_line(tmp_path):
    points = [(x / 2, y / 2, 0.0) for x in range(-6, 7) for y in range(-6, 7)]
    path = tmp_path / "open_map.pcd"
    write_pcd(path, points)
    planner = PCDGridPlanner(resolution=0.15, robot_radius=0.48)
    planner.load(str(path))

    direct = planner.plan((-2.0, 0.0), (2.0, 0.0))
    planner.add_dynamic_obstacle(0.0, 0.0, radius=0.58)
    detour = planner.plan((-2.0, 0.0), (2.0, 0.0))

    # În spațiu liber traseul este A→B. Apariția obstacolului invalidează
    # segmentul complet și A* introduce doar colțurile ocolirii necesare.
    assert len(direct) == 2
    assert all(y == pytest.approx(0.0) for _, y in direct)
    assert len(detour) > 2
    assert any(abs(y) >= 0.55 for _, y in detour[1:-1])
    assert all(planner.world_to_cell(*point) not in planner.occupied for point in detour)


def test_dynamic_costmap_clears_without_touching_static_map(tmp_path):
    points = [(x / 2, y / 2, 0.0) for x in range(-6, 7) for y in range(-6, 7)]
    points.append((0.0, 0.0, 0.8))
    path = tmp_path / "static_and_dynamic.pcd"
    write_pcd(path, points)
    planner = PCDGridPlanner(resolution=0.15, robot_radius=0.35)
    planner.load(str(path))
    static_before = set(planner.static_occupied)

    planner.add_dynamic_obstacle(1.5, 0.0, radius=0.30, observed_at=10.0)

    assert planner.dynamic_occupied
    assert planner.static_occupied == static_before
    assert not (set(planner.dynamic_occupied) & planner.static_occupied)
    assert planner.expire_dynamic_obstacles(max_age=8.0, now=19.0) > 0
    assert not planner.dynamic_occupied
    assert planner.static_occupied == static_before


def test_temporary_obstacle_is_cleared_and_original_route_resumes(tmp_path):
    class TemporaryGuard:
        def __init__(self):
            self.block_checks = 0

        def has_fresh_data(self):
            return True

        def is_blocked(self, _vx, _vy):
            self.block_checks += 1
            return self.block_checks <= 2

        def front_obstacle_distance(self, default=0.70):
            return 0.55

    points = [(x / 2, y / 2, 0.0) for x in range(-6, 7) for y in range(-6, 7)]
    map_path = tmp_path / "temporary_obstacle.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.15, robot_radius=0.35)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.0, 0.0))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        events = []
        commands = []

        async def send_waypoint(x, y, _yaw, _speed):
            commands.append("1102")
            pose.update({"x": x, "y": y})
            return {"success": True}

        async def pause():
            commands.append("1201")
            return {"success": True}

        async def resume():
            commands.append("1202")
            return {"success": True}

        async def event_callback(event):
            events.append(event)

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, TemporaryGuard(),
            send_waypoint, pause, resume, event_callback,
            obstacle_wait_before_replan=0.20,
            obstacle_clear_stable=0.02,
            poll_interval=0.01,
        )
        result = await navigator.start(
            str(map_path), 1.0, 0.0, 0.0, timeout=2.0,
            prepared_plan={"planner": planner, "path": route, "clearance_mode": "normal"},
        )
        assert result["success"]
        try:
            await asyncio.wait_for(navigator.task, timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail(
                f"navigator blocat: status={navigator.status}, "
                f"events={[event.get('state') for event in events]}, commands={commands}"
            )

        assert navigator.status["state"] == "arrived"
        # Două cadre izolate nu mai provoacă STOP/replanificare.
        assert "replanning" not in [event["state"] for event in events]
        assert not any(event["state"] == "awaiting_replan_confirmation" for event in events)
        # După pauza de siguranță reluăm 1202 înainte să trimitem noul 1102;
        # controllerul Unitree ignoră uneori țintele publicate cât este paused.
        # Dreapta este o singură comandă 1102 stabilă; LiDAR rasterizează
        # segmentul complet, fără waypoint-uri artificiale.
        assert commands.count("1102") == 1
        assert commands[-1] == "1201"
        assert not planner.dynamic_occupied

    asyncio.run(scenario())


def test_narrow_clearance_fallback_opens_realistic_corridor(tmp_path):
    points = [(x / 10, y / 10, 0.0) for x in range(-25, 26) for y in range(-12, 13)]
    # Coridor de 0,90 m: garda mare nu încape, profilul controlat de 0,20 m încape.
    points.extend((x / 10, -0.45, 0.8) for x in range(-25, 26) for _ in range(3))
    points.extend((x / 10, 0.45, 0.8) for x in range(-25, 26) for _ in range(3))
    path = tmp_path / "narrow_corridor.pcd"
    write_pcd(path, points)

    planner, route, mode = plan_pcd_route(
        str(path), (-2.0, 0.0), (2.0, 0.0), robot_radius=0.38
    )

    assert mode == "narrow"
    assert planner.robot_radius == pytest.approx(0.20)
    assert route[0][0] < 0 < route[-1][0]


def test_small_requested_radius_uses_safe_profile_when_space_exists(tmp_path):
    points = [
        (x / 10, y / 10, 0.0)
        for x in range(-25, 26) for y in range(-8, 9)
    ]
    points.extend(
        (x / 10, side, 0.8)
        for x in range(-25, 26)
        for side in (-0.60, 0.60)
        for _ in range(3)
    )
    map_path = tmp_path / "safe_profile_corridor.pcd"
    write_pcd(map_path, points)

    planner, route, mode = plan_pcd_route(
        str(map_path), (-2.0, 0.0), (2.0, 0.0), robot_radius=0.10
    )

    assert mode == "safe"
    assert planner.robot_radius == pytest.approx(0.25)
    assert max(abs(y) for _, y in route) <= 0.15


def test_safe_profile_snaps_near_wall_goal_instead_of_falling_back(tmp_path):
    points = [
        (x / 10, y / 10, 0.0)
        for x in range(-20, 21) for y in range(-10, 11)
    ]
    points.extend(
        (x / 10, 1.0, 0.8)
        for x in range(-20, 21) for _ in range(3)
    )
    map_path = tmp_path / "safe_goal_snap.pcd"
    write_pcd(map_path, points)

    planner, route, mode = plan_pcd_route(
        str(map_path), (-1.5, 0.0), (1.5, 0.72), robot_radius=0.10
    )

    assert mode == "safe"
    assert planner.robot_radius == pytest.approx(0.25)
    assert math.hypot(route[-1][0] - 1.5, route[-1][1] - 0.72) <= 0.20


def test_production_planner_removes_saved_robot_ghost_at_start(tmp_path):
    points = [(x / 5, y / 5, 0.0) for x in range(-15, 16) for y in range(-15, 16)]
    # Corpul robotului rămas în PCD la poziția de pornire.
    points.extend((0.10, 0.02, z / 10) for z in range(2, 14) for _ in range(4))
    path = tmp_path / "map_with_robot_ghost.pcd"
    write_pcd(path, points)

    planner, route, _ = plan_pcd_route(str(path), (0.0, 0.0), (2.0, 0.0))

    assert planner.world_to_cell(0.0, 0.0) not in planner.occupied
    assert route[0] == pytest.approx((0.0, 0.0))


def test_stagnation_retries_waypoint_without_virtual_obstacle(tmp_path):
    class ClearGuard:
        def has_fresh_data(self): return True
        def is_navigation_blocked(self): return False
        def front_obstacle_shape(self): return None

    points = [(x / 5, y / 5, 0.0) for x in range(-10, 11) for y in range(-10, 11)]
    map_path = tmp_path / "retry_waypoint.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.0, 0.0))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        sends = []
        resumes = []

        async def send_waypoint(x, y, _yaw, _speed):
            sends.append((x, y))
            # Fiecare segment răspunde numai la retry-ul său.
            if len(sends) % 2 == 0:
                pose.update({"x": x, "y": y})
            return {"success": True}

        async def pause():
            return {"success": True}

        async def resume():
            resumes.append("1202")
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, ClearGuard(),
            send_waypoint, pause, resume, lambda _event: asyncio.sleep(0),
            poll_interval=0.01,
            navigation_paused=lambda: True,
            stagnation_timeout=0.05,
            max_waypoint_retries=1,
        )
        result = await navigator.start(
            str(map_path), 1.0, 0.0, 0.0, timeout=2.0,
            prepared_plan={"planner": planner, "path": route, "clearance_mode": "custom"},
        )
        assert result["success"]
        await navigator.task

        assert navigator.status["state"] == "arrived", navigator.status
        # 1 comandă de control + retry-ul ei; nu există punct intermediar
        # artificial care să provoace încă un ciclu de oprire și rotație.
        assert len(sends) == 2
        assert resumes == ["1202"]
        assert not any(source == "stagnation" for source in planner.dynamic_sources.values())

    asyncio.run(scenario())


def test_persistent_stagnation_stops_without_replanning_fake_obstacle(tmp_path):
    class ClearGuard:
        def has_fresh_data(self): return True
        def is_navigation_blocked(self): return False
        def front_obstacle_shape(self): return None

    points = [(x / 5, y / 5, 0.0) for x in range(-10, 11) for y in range(-10, 11)]
    map_path = tmp_path / "persistent_stall.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.0, 0.0))

    async def scenario():
        sends = []
        events = []

        async def send_waypoint(*_args):
            sends.append("1102")
            return {"success": True}

        async def command(*_args):
            return {"success": True}

        async def event_callback(event):
            events.append(dict(event))

        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
            ClearGuard(), send_waypoint, command, command, event_callback,
            poll_interval=0.01,
            stagnation_timeout=0.05,
            max_waypoint_retries=1,
        )
        result = await navigator.start(
            str(map_path), 1.0, 0.0, 0.0, timeout=2.0,
            prepared_plan={"planner": planner, "path": route, "clearance_mode": "custom"},
        )
        assert result["success"]
        await navigator.task

        assert navigator.status["state"] == "failed"
        assert len(sends) == 2
        assert "nu inventez o rută" in navigator.status["error"]
        assert not planner.dynamic_occupied
        assert not any(event.get("state") == "awaiting_replan_confirmation" for event in events)

    asyncio.run(scenario())


def test_navigator_refuses_stale_localization(tmp_path):
    class Sport:
        def is_sdk_available(self): return True
    navigator = AutonomousNavigator(
        Sport(), lambda: {"x": 0, "y": 0, "yaw": 0}, lambda: False,
        object(), lambda event: asyncio.sleep(0),
    )
    result = asyncio.run(navigator.start(str(tmp_path / "missing.pcd"), 1, 1, 0))
    assert not result["success"]
    assert "Localizarea" in result["error"]


def test_replan_is_automatic_when_lidar_detects_persistent_obstacle(tmp_path):
    class FlickerGuard:
        def __init__(self):
            self.blocked = True

        def has_fresh_data(self):
            return True

        def is_navigation_blocked(self):
            return self.blocked

        def front_obstacle_shape(self):
            if not self.blocked:
                return None
            return {
                "points": [(0.55, 0.0), (0.55, 0.10)],
                "distance": 0.55,
                "vector": (0.55, 0.0),
            }

    points = [(x / 5, y / 5, 0.0) for x in range(-10, 11) for y in range(-10, 11)]
    map_path = tmp_path / "confirmation_lock.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.50, 0.0))

    async def scenario():
        guard = FlickerGuard()
        commands = []

        async def command(*_args):
            return {"success": True}

        async def pause():
            commands.append("1201")
            return {"success": True}

        async def resume():
            commands.append("1202")
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
            guard, command, pause, resume, lambda _event: asyncio.sleep(0),
            obstacle_wait_before_replan=0.0,
            obstacle_clear_stable=0.02,
            dynamic_obstacle_ttl=0.05,
            poll_interval=0.01,
        )
        navigator.planner = planner
        wait_task = asyncio.create_task(navigator._wait_for_clear_or_replan(
            route, 1, 1.50, 0.0, 0, time.monotonic() + 3.0,
        ))
        new_path, new_index, replans, _ = await asyncio.wait_for(wait_task, timeout=1.0)

        assert navigator.status["state"] == "replanning"
        assert replans == 1
        assert new_index == 1
        assert new_path != route
        assert navigator.replan_confirmation_id is None
        assert commands == ["1201"]

    asyncio.run(scenario())


def test_blocked_replan_uses_one_verified_lateral_step_then_replans(tmp_path):
    class FrontObstacleGuard:
        def __init__(self):
            self.blocked = True

        def has_fresh_data(self): return True
        def is_navigation_blocked(self): return self.blocked
        def is_lateral_clear(self, _direction): return True

        def front_obstacle_shape(self):
            if not self.blocked:
                return None
            # Obstacol puțin în stânga: partea preferată trebuie să fie dreapta.
            return {
                "points": [(0.55, 0.12), (0.58, 0.16)],
                "distance": 0.56,
                "vector": (0.55, 0.14),
            }

    map_path = tmp_path / "dynamic_lateral_unlock.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 31) for y in range(-20, 21)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (2.0, 0.0))

    async def scenario():
        guard = FrontObstacleGuard()
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        lateral_commands = []
        pauses = []

        async def send_waypoint(*_args): return {"success": True}
        async def pause():
            pauses.append("1201")
            return {"success": True}
        async def resume(): return {"success": True}
        async def stop(): return {"success": True}

        async def lateral(vy):
            lateral_commands.append(vy)
            pose["y"] += 0.15 if vy > 0.0 else -0.15
            guard.blocked = False
            return {"success": True}

        original_plan = planner.plan

        def blocked_until_lateral(start, goal):
            if guard.blocked:
                raise ValueError("ieșirea frontală este blocată")
            return original_plan(start, goal)

        planner.plan = blocked_until_lateral
        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, guard,
            send_waypoint, pause, resume, lambda _event: asyncio.sleep(0),
            obstacle_wait_before_replan=0.0,
            obstacle_clear_stable=0.0,
            poll_interval=0.01,
            lateral_velocity=lateral,
            stop_locomotion=stop,
        )
        navigator.planner = planner
        new_path, new_index, replans, _ = await asyncio.wait_for(
            navigator._wait_for_clear_or_replan(
                route, 1, 2.0, 0.0, 0, time.monotonic() + 3.0,
            ),
            timeout=2.0,
        )

        assert pauses == ["1201"]
        assert lateral_commands and all(value < 0.0 for value in lateral_commands)
        assert pose["y"] < 0.0
        assert navigator.status["dynamic_lateral_unlocks"] == 1
        assert navigator.status["state"] == "replanning"
        assert new_index == 1
        assert new_path[-1] == (2.0, 0.0)
        assert replans == 0

    asyncio.run(scenario())


def test_legacy_dynamic_replan_confirmation_is_disabled():
    async def command(*args):
        return {"success": True}

    async def scenario():
        navigator = NativeWaypointNavigator(
            lambda: {"x": 0, "y": 0, "yaw": 0}, lambda: True,
            object(), command, command, command, lambda event: asyncio.sleep(0),
        )
        result = await navigator.confirm_replan("orice-id")
        assert not result["success"]
        assert "automat" in result["error"]

    asyncio.run(scenario())


def test_route_guard_allows_safe_lateral_detour_while_front_sensor_is_blocked(tmp_path):
    class FrontObstacleGuard:
        def has_fresh_data(self): return True
        def is_navigation_blocked(self): return True
        def front_obstacle_shape(self):
            return {
                "points": [(0.68, -0.03), (0.70, 0.0), (0.72, 0.03)],
                "distance": 0.70,
                "vector": (0.70, 0.0),
            }

    map_path = tmp_path / "route_aware_guard.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 21)
        for y in range(-15, 16)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))

    async def command(*_args):
        return {"success": True}

    navigator = NativeWaypointNavigator(
        lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
        FrontObstacleGuard(), command, command, command,
        lambda _event: asyncio.sleep(0),
    )
    navigator.planner = planner
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    straight_blocked, sensor_blocked, *_ = navigator._refresh_route_obstacles(
        [(0.0, 0.0), (1.5, 0.0)], 1, pose,
    )
    lateral_blocked, sensor_still_blocked, *_ = navigator._refresh_route_obstacles(
        [(0.0, 0.0), (0.0, 0.8), (1.5, 0.8)], 1, pose,
    )

    assert sensor_blocked and sensor_still_blocked
    assert straight_blocked
    assert not lateral_blocked


def test_dynamic_replanning_has_no_six_replan_limit(tmp_path):
    class PersistentGuard:
        def has_fresh_data(self): return True
        def is_navigation_blocked(self): return True
        def front_obstacle_shape(self):
            return {
                "points": [(0.65, -0.05), (0.68, 0.0), (0.65, 0.05)],
                "distance": 0.68,
                "vector": (0.68, 0.0),
            }

    map_path = tmp_path / "unlimited_replans.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 31)
        for y in range(-20, 21)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (2.0, 0.0))

    async def command(*_args):
        return {"success": True}

    async def scenario():
        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
            PersistentGuard(), command, command, command,
            lambda _event: asyncio.sleep(0),
            obstacle_wait_before_replan=0.0,
            poll_interval=0.005,
        )
        navigator.planner = planner
        current_path = route
        current_index = 1
        replans = 0
        for _ in range(8):
            current_path, current_index, replans, _ = await asyncio.wait_for(
                navigator._wait_for_clear_or_replan(
                    current_path, current_index, 2.0, 0.0, replans, math.inf,
                ),
                timeout=0.5,
            )
        assert replans == 8
        assert navigator.status["state"] == "replanning"
        assert navigator.replan_confirmation_id is None

    asyncio.run(scenario())


def test_zero_navigation_timeout_means_unlimited(tmp_path):
    class ClearGuard:
        def has_fresh_data(self): return True
        def is_navigation_blocked(self): return False
        def front_obstacle_shape(self): return None

    map_path = tmp_path / "unlimited_navigation.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 21)
        for y in range(-10, 11)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.0, 0.0))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        async def send_waypoint(x, y, yaw, _speed):
            await asyncio.sleep(0.02)
            pose.update({"x": x, "y": y, "yaw": yaw})
            return {"success": True}

        async def command(*_args):
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, ClearGuard(),
            send_waypoint, command, command, lambda _event: asyncio.sleep(0),
            poll_interval=0.005,
        )
        result = await navigator.start(
            str(map_path), 1.0, 0.0, 0.0, speed=0.3, timeout=0.0,
            prepared_plan={"planner": planner, "path": route, "clearance_mode": "custom"},
        )
        assert result["success"]
        await asyncio.wait_for(navigator.task, timeout=1.0)
        assert navigator.status["state"] == "arrived"

    asyncio.run(scenario())


def test_end_to_end_a_to_b_stops_replans_automatically_and_arrives(tmp_path):
    class SideCornerGuard:
        def __init__(self):
            self.blocked = False

        def has_fresh_data(self):
            return True

        def is_navigation_blocked(self):
            return self.blocked

        def front_obstacle_shape(self):
            if not self.blocked:
                return None
            return {
                # Colț de masă nou, lateral față de pelvis, dar în anvelopa mâinii.
                "points": [(0.70, 0.30), (0.72, 0.34), (0.74, 0.38)],
                "distance": 0.76,
                "vector": (0.72, 0.34),
            }

    map_path = tmp_path / "end_to_end_open_room.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-15, 31)
        for y in range(-20, 21)
    ])
    planner = PCDGridPlanner(
        resolution=0.10, robot_radius=0.42,
        min_obstacle_points=3, comfort_radius=0.65,
        clearance_weight=4.5,
    )
    planner.load(str(map_path))
    planner.clear_robot_footprint(0.0, 0.0)
    initial_route = planner.plan((0.0, 0.0), (2.0, 0.0))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        guard = SideCornerGuard()
        sent_waypoints = []
        events = []

        async def send_waypoint(x, y, yaw, speed):
            sent_waypoints.append((x, y, yaw, speed))
            if len(sent_waypoints) == 1:
                # Obstacolul apare după pornirea primului segment.
                guard.blocked = True
                # Simulează exact robotul real: 1102 expiră după STOP-ul 1201.
                await asyncio.sleep(0.35)
                return {"success": False, "error": "Robotul nu a răspuns la waypoint-ul 1102 în 4s"}
            else:
                pose.update({"x": x, "y": y, "yaw": yaw})
            return {"success": True}

        async def command():
            return {"success": True}

        async def event_callback(event):
            events.append(dict(event))

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, guard,
            send_waypoint, command, command, event_callback,
            obstacle_wait_before_replan=0.0,
            obstacle_clear_stable=0.02,
            dynamic_obstacle_ttl=2.5,
            poll_interval=0.005,
            stagnation_timeout=1.0,
        )
        result = await navigator.start(
            str(map_path), 2.0, 0.0, 0.0,
            speed=0.30, timeout=4.0,
            prepared_plan={
                "planner": planner,
                "path": initial_route,
                "clearance_mode": "normal",
            },
        )
        assert result["success"]

        await asyncio.wait_for(navigator.task, timeout=3.0)
        replanned_events = [
            event for event in events
            if event.get("state") == "replanning" and event.get("replans", 0) >= 1
        ]
        assert replanned_events
        replanned_path = list(replanned_events[-1]["path"])
        assert any(abs(y) >= 0.10 for _, y in replanned_path[1:-1]), replanned_path
        assert navigator.status["state"] == "arrived"
        assert math.hypot(pose["x"] - 2.0, pose["y"]) <= 0.22
        assert len(sent_waypoints) >= 2
        states = [event.get("state") for event in events]
        assert "replanning" in states
        assert "awaiting_replan_confirmation" not in states
        assert states[-1] == "arrived"

    asyncio.run(scenario())


def test_waypoint_feedback_wait_has_live_emergency_pause(tmp_path):
    class Guard:
        def __init__(self):
            self.blocked = False

        def has_fresh_data(self):
            return True

        def is_navigation_blocked(self):
            return self.blocked

        def front_obstacle_shape(self):
            return None

        def front_obstacle_vector(self, default=0.70):
            return 0.70, 0.0

    map_path = tmp_path / "dispatch_guard.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 21)
        for y in range(-10, 11)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.42)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.50, 0.0))

    async def scenario():
        guard = Guard()
        paused_at = []
        feedback_returned_at = []
        events = []

        async def delayed_waypoint(*_args):
            guard.blocked = True
            await asyncio.sleep(0.40)
            feedback_returned_at.append(time.monotonic())
            return {"success": True}

        async def pause():
            paused_at.append(time.monotonic())
            return {"success": True}

        async def event_callback(event):
            events.append((time.monotonic(), dict(event)))

        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
            guard, delayed_waypoint, pause, pause,
            event_callback, poll_interval=0.01,
        )
        navigator.planner = planner

        result, hazard = await navigator._send_waypoint_guarded(
            1.50, 0.0, 0.0, 0.20, route, 1
        )

        assert result["success"]
        assert hazard == "obstacle"
        assert paused_at
        assert feedback_returned_at
        assert paused_at[0] < feedback_returned_at[0]
        assert events
        assert events[0][1]["state"] == "waiting_obstacle"
        assert events[0][0] < feedback_returned_at[0]

    asyncio.run(scenario())


def test_waypoint_feedback_ignores_one_short_sensor_gap(tmp_path):
    class Guard:
        def __init__(self):
            self.started = time.monotonic()

        def navigation_sensors_ready(self):
            elapsed = time.monotonic() - self.started
            return not (0.02 <= elapsed <= 0.10)

        def is_navigation_blocked(self): return False
        def front_obstacle_shape(self): return None

    map_path = tmp_path / "dispatch_sensor_glitch.pcd"
    write_pcd(map_path, [
        (x / 10, y / 10, 0.0)
        for x in range(-10, 21)
        for y in range(-10, 11)
    ])
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.42)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.50, 0.0))

    async def scenario():
        guard = Guard()
        pauses = []

        async def delayed_waypoint(*_args):
            await asyncio.sleep(0.14)
            return {"success": True}

        async def pause():
            pauses.append(True)
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}, lambda: True,
            guard, delayed_waypoint, pause, pause,
            lambda _event: asyncio.sleep(0), poll_interval=0.01,
        )
        navigator.planner = planner

        result, hazard = await navigator._send_waypoint_guarded(
            1.50, 0.0, 0.0, 0.20, route, 1
        )

        assert result["success"]
        assert hazard is None
        assert pauses == []

    asyncio.run(scenario())


def test_active_1102_is_resumed_without_resending_after_sensor_gap(tmp_path):
    class GapGuard:
        def __init__(self):
            self.fresh = True

        def navigation_sensors_ready(self):
            return self.fresh

        @staticmethod
        def is_navigation_blocked():
            return False

        @staticmethod
        def front_obstacle_shape():
            return None

    points = [
        (x / 5, y / 5, 0.0)
        for x in range(-10, 11) for y in range(-10, 11)
    ]
    map_path = tmp_path / "sensor_resume.pcd"
    write_pcd(map_path, points)
    planner = PCDGridPlanner(resolution=0.10, robot_radius=0.20)
    planner.load(str(map_path))
    route = planner.plan((0.0, 0.0), (1.0, 0.0))

    async def scenario():
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        guard = GapGuard()
        sends = []
        resumes = []

        async def create_gap(x, y):
            await asyncio.sleep(0.02)
            guard.fresh = False
            await asyncio.sleep(0.35)
            guard.fresh = True
            await asyncio.sleep(0.03)
            pose.update({"x": x, "y": y})

        async def send_waypoint(x, y, _yaw, _speed):
            sends.append((x, y))
            asyncio.create_task(create_gap(x, y))
            return {"success": True}

        async def pause():
            return {"success": True}

        async def resume():
            resumes.append("1202")
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: dict(pose), lambda: True, guard,
            send_waypoint, pause, resume, lambda _event: asyncio.sleep(0),
            poll_interval=0.01, sensor_loss_timeout=1.0,
        )
        result = await navigator.start(
            str(map_path), 1.0, 0.0, 0.0, timeout=2.0,
            prepared_plan={
                "planner": planner, "path": route,
                "clearance_mode": "normal",
            },
        )
        assert result["success"]
        await navigator.task

        assert navigator.status["state"] == "arrived"
        assert sends == [(route[-1][0], route[-1][1])]
        assert resumes == ["1202"]

    asyncio.run(scenario())


def test_native_localization_gap_requires_three_stable_samples_before_resume():
    async def scenario():
        checks = iter([False, False, True, True, True])
        pauses = []
        events = []

        def localization_ok():
            return next(checks, True)

        async def pause():
            pauses.append("1201")
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0},
            localization_ok,
            object(),
            lambda *_args: asyncio.sleep(0),
            pause,
            lambda: asyncio.sleep(0),
            lambda event: events.append(event) or asyncio.sleep(0),
            poll_interval=0.01,
        )

        await navigator._wait_for_localization_recovery(
            time.monotonic() + 1.0, timeout=0.5,
        )

        assert pauses == ["1201"]
        assert navigator.status["state"] == "navigating"
        assert navigator.status["localization_recoveries"] == 1
        assert [event["state"] for event in events] == [
            "waiting_localization", "navigating",
        ]

    asyncio.run(scenario())


def test_persistent_native_localization_loss_stays_stopped():
    async def scenario():
        pauses = []

        async def pause():
            pauses.append("1201")
            return {"success": True}

        navigator = NativeWaypointNavigator(
            lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0},
            lambda: False,
            object(),
            lambda *_args: asyncio.sleep(0),
            pause,
            lambda: asyncio.sleep(0),
            lambda _event: asyncio.sleep(0),
            poll_interval=0.01,
        )

        with pytest.raises(RuntimeError, match="nu a revenit stabil"):
            await navigator._wait_for_localization_recovery(
                time.monotonic() + 1.0, timeout=0.04,
            )
        assert pauses == ["1201"]
        assert navigator.status["state"] == "waiting_localization"

    asyncio.run(scenario())
