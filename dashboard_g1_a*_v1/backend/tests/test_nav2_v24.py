import math
import sys
from pathlib import Path

import pytest


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from nav2_observer import Nav2ObserverPublisher, compose_planar, map_to_odom


def test_map_to_odom_reconstructs_localized_robot_pose():
    global_base = {"x": 4.2, "y": -1.3, "yaw": 1.1}
    odom_base = {"x": 0.8, "y": 0.2, "yaw": -0.25}

    reconstructed = compose_planar(map_to_odom(global_base, odom_base), odom_base)

    assert reconstructed["x"] == pytest.approx(global_base["x"])
    assert reconstructed["y"] == pytest.approx(global_base["y"])
    assert reconstructed["yaw"] == pytest.approx(global_base["yaw"])


def test_initialpose_keeps_a_fixed_map_to_odom_anchor_while_local_odom_moves():
    observer = Nav2ObserverPublisher.__new__(Nav2ObserverPublisher)
    observer.global_pose = None
    observer.global_source = ""
    observer.global_received_at = 0.0
    observer.local_pose = {"x": 1.0, "y": 0.0, "yaw": 0.0}
    observer.anchored_map_to_odom = None
    observer._publish_dynamic_transforms = lambda: None

    observer.update_global_pose(
        {"x": 5.0, "y": 2.0, "yaw": 0.0},
        "/initialpose + anchored_pelvis",
    )
    fixed_anchor = dict(observer.anchored_map_to_odom)
    observer.local_pose = {"x": 1.8, "y": 0.0, "yaw": 0.0}
    progressed = compose_planar(fixed_anchor, observer.local_pose)

    assert observer.anchored_map_to_odom == fixed_anchor
    assert progressed["x"] == pytest.approx(5.8)
    assert progressed["y"] == pytest.approx(2.0)


def test_raw_livox_points_are_aligned_to_anchored_map_and_pcd_floor():
    from server import _transform_livox_points_to_map

    roll = 3.14
    pitch = 0.04014257279586953
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)

    def raw_for_base(base_x, base_y, base_z):
        vx = base_x - 0.0002835
        vy = base_y - 0.00003
        vz = base_z - 0.46018
        return {
            "x": cp * vx - sp * vz,
            "y": sp * sr * vx + cr * vy + cp * sr * vz,
            "z": sp * cr * vx - sr * vy + cp * cr * vz,
        }

    raw = [raw_for_base(0.4 + index * 0.08, 0.0, -0.75) for index in range(20)]
    raw.append(raw_for_base(1.0, 0.0, -0.25))
    transformed = _transform_livox_points_to_map(
        raw,
        {"x": 2.0, "y": 3.0, "yaw": math.pi / 2},
        {"a": 0.0, "b": 0.0, "c": -1.2},
    )
    obstacle = transformed[-1]

    assert obstacle["x"] == pytest.approx(2.0, abs=1e-5)
    assert obstacle["y"] == pytest.approx(4.0, abs=1e-5)
    assert obstacle["z"] == pytest.approx(-0.7, abs=1e-5)


def test_single_point_lidar_packets_cannot_refresh_safety_or_localization():
    from server import _pointcloud_has_geometry

    class Cloud:
        point_step = 22
        height = 1

        def __init__(self, width):
            self.width = width
            self.data = bytes(width * self.point_step)

    assert _pointcloud_has_geometry(Cloud(1)) is False
    assert _pointcloud_has_geometry(Cloud(20064)) is True


def test_nav2_uses_the_lidar_topic_that_is_live_without_1804():
    source = (
        BACKEND / "nav2" / "nav2_observer_params.yaml"
    ).read_text(encoding="utf-8")

    assert "topic: /utlidar/cloud_livox_mid360" in source
    assert "topic: /unitree/slam_localization/points" not in source
    assert "min_obstacle_height: -0.60" in source
    assert "allow_unknown: true" in source
    assert "robot_radius: 0.20" in source
    assert "inflation_radius: 0.21" in source


def test_nav2_launcher_stops_the_real_child_process_groups():
    source = (
        BACKEND / "nav2" / "start_nav2_observer.sh"
    ).read_text(encoding="utf-8")

    assert "setsid ros2 run nav2_planner planner_server" in source
    assert "setsid ros2 run nav2_lifecycle_manager lifecycle_manager" in source
    assert 'stop_process_group "${PLANNER_PID:-}"' in source
    assert 'stop_process_group "${LIFECYCLE_PID:-}"' in source
    assert 'kill -TERM -- "-${leader}"' in source
    assert 'kill -KILL -- "-${leader}"' in source


def test_nav2_waits_for_manual_tf_without_bond_killing_planner():
    source = (
        BACKEND / "nav2" / "nav2_observer_params.yaml"
    ).read_text(encoding="utf-8")

    assert "bond_timeout: 0.0" in source


def test_camera_receiver_retries_port_during_dashboard_restart():
    source = (BACKEND / "server.py").read_text(encoding="utf-8")

    assert "while server_sock is None:" in source
    assert "portul 5005 este temporar indisponibil" in source


def test_nav2_dense_straight_path_becomes_one_execution_segment():
    observer = Nav2ObserverPublisher.__new__(Nav2ObserverPublisher)
    observer.nav2_costmap = {
        "width": 31, "height": 21, "resolution": 0.10,
        "origin_x": -0.05, "origin_y": -1.05,
        "data": [0] * (31 * 21),
    }
    dense = [
        {"x": index / 10.0, "y": 0.0, "yaw": 0.0}
        for index in range(21)
    ]

    assert observer.simplify_execution_path(dense) == [
        (0.0, 0.0), (2.0, 0.0)
    ]


def test_nav2_simplifier_keeps_detour_around_lethal_cells():
    observer = Nav2ObserverPublisher.__new__(Nav2ObserverPublisher)
    width, height = 31, 21
    data = [0] * (width * height)
    for column in range(9, 13):
        for row in range(9, 12):
            data[row * width + column] = 100
    observer.nav2_costmap = {
        "width": width, "height": height, "resolution": 0.10,
        "origin_x": -0.05, "origin_y": -1.05, "data": data,
    }
    detour = [
        {"x": 0.0, "y": 0.0}, {"x": 0.6, "y": 0.5},
        {"x": 1.4, "y": 0.5}, {"x": 2.0, "y": 0.0},
    ]

    reduced = observer.simplify_execution_path(detour)

    assert len(reduced) >= 3
    assert all(
        observer._nav2_segment_is_safe(start, end)
        for start, end in zip(reduced, reduced[1:])
    )


def test_nav2_simplifier_preserves_valid_path_through_high_inflation_cost():
    observer = Nav2ObserverPublisher.__new__(Nav2ObserverPublisher)
    width, height = 31, 21
    data = [0] * (width * height)
    for column in range(9, 13):
        data[10 * width + column] = 90
    observer.nav2_costmap = {
        "width": width, "height": height, "resolution": 0.10,
        "origin_x": -0.05, "origin_y": -1.05, "data": data,
    }
    dense = [
        {"x": index / 10.0, "y": 0.0, "yaw": 0.0}
        for index in range(21)
    ]

    reduced = observer.simplify_execution_path(dense, maximum_cost=80)

    assert reduced[0] == (0.0, 0.0)
    assert reduced[-1] == (2.0, 0.0)
    assert len(reduced) >= 3
    assert all(
        observer._nav2_segment_is_safe(start, end, 98)
        for start, end in zip(reduced, reduced[1:])
    )


def test_nav2_simplifier_repairs_corner_cut_with_safe_costmap_detour():
    observer = Nav2ObserverPublisher.__new__(Nav2ObserverPublisher)
    width = height = 7
    data = [0] * (width * height)
    # Diagonala publicată trece exact printre aceste două celule letale.
    for column, row in ((2, 3), (1, 2)):
        data[row * width + column] = 100
    observer.nav2_costmap = {
        "width": width, "height": height, "resolution": 0.10,
        "origin_x": -0.05, "origin_y": -0.05, "data": data,
    }
    corner_cut = [
        {"x": 0.1, "y": 0.3},
        {"x": 0.2, "y": 0.2},
        {"x": 0.5, "y": 0.3},
    ]

    repaired = observer.simplify_execution_path(corner_cut)

    assert repaired[0] == pytest.approx((0.1, 0.3))
    assert repaired[-1] == pytest.approx((0.5, 0.3))
    assert len(repaired) >= 3
    assert all(
        observer._nav2_segment_is_safe(start, end, 98)
        for start, end in zip(repaired, repaired[1:])
    )


def test_v24_driver_selection_preserves_v22_default():
    from server import _parse_navigation_driver

    assert _parse_navigation_driver({}) == "legacy"
    assert _parse_navigation_driver({"driver": "v22"}) == "legacy"
    assert _parse_navigation_driver({"driver": "nav2"}) == "nav2"
    with pytest.raises(ValueError):
        _parse_navigation_driver({"driver": "unknown"})


def test_executor_selection_is_explicit_and_keeps_native_default():
    from server import _parse_navigation_executor

    assert _parse_navigation_executor({}) == "native_1102"
    assert _parse_navigation_executor({"executor": "1102"}) == "native_1102"
    assert _parse_navigation_executor({"executor": "local_safe"}) == "local_velocity"
    with pytest.raises(ValueError):
        _parse_navigation_executor({"executor": "automatic_fallback"})


def test_all_fsm_changes_require_exact_ok(monkeypatch):
    import asyncio
    import server

    calls = []
    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    # Izolează testul de executorul global al mediului ROS/OpenCV; verificăm
    # aici exclusiv bariera de confirmare, nu thread-pool-ul Python.
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
    method_names = (
        "wake_up_sequence", "damp", "zero_torque",
        "stand_up", "start_locomotion", "set_fsm_id",
    )
    for method_name in method_names:
        monkeypatch.setattr(
            server.sport_client,
            method_name,
            lambda *args, _name=method_name: calls.append((_name, args)) or {"success": True},
        )

    handlers = (
        lambda body: server.wake_up_robot(body),
        lambda body: server.damp_robot(body),
        lambda body: server.zero_torque_robot(body),
        lambda body: server.stand_up_robot(body),
        lambda body: server.start_robot(body),
        lambda body: server.set_robot_fsm(801, body),
    )
    invalid_confirmations = ({}, {"confirmed": True}, {"confirmation": "OK"}, {"confirmation": " ok "})

    for handler in handlers:
        for body in invalid_confirmations:
            refused = asyncio.run(handler(body))
            assert refused["success"] is False
            assert refused["confirmation_required"] is True
    assert calls == []

    for handler in handlers:
        accepted = asyncio.run(handler({"confirmation": "ok"}))
        assert accepted["success"] is True
    assert [name for name, _ in calls] == list(method_names)
    assert calls[-1] == ("set_fsm_id", (801,))


def test_frontend_consumes_exact_ok_for_each_fsm_command():
    source = (BACKEND.parent / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "fsm-confirmation-input" in source
    assert "input.value === \"ok\"" in source
    assert "input.value = \"\"" in source
    assert source.count("fsm-state-button") == 6
    for endpoint in (
        "/api/robot/wake_up", "/api/robot/damp", "/api/robot/zero_torque",
        "/api/robot/stand_up", "/api/robot/start",
    ):
        assert endpoint in source


def test_frontend_can_clear_goal_and_hide_map_without_deleting_pcd():
    source = (BACKEND.parent / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="nav-clear-goal-btn"' in source
    assert "async function clearNavigationGoal()" in source
    assert "document.getElementById('nav-goal-x').value = '';" in source
    assert "document.getElementById('nav-goal-y').value = '';" in source
    assert "S.goalMarker.visible = false" in source
    assert "'/api/nav/preview/cancel'" in source
    assert 'id="unload-map-btn"' in source
    assert "function clearMapVisualization(suppressUpdates = true)" in source
    assert "mapDisplaySuppressed" in source
    assert "sessionStorage.setItem(MAP_DISPLAY_STORAGE_KEY, '1')" in source
    clear_display = source.split("async function clearDisplayedMap()", 1)[1].split(
        "async function startRelocationRobot()", 1
    )[0]
    assert "/api/slam/unload_robot" not in clear_display


def test_frontend_defaults_to_deadline_astar_and_native_executor():
    source = (BACKEND.parent / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="nav-executor"' in source
    assert 'value="native_1102" selected' in source
    assert 'value="legacy" selected' in source
    assert 'value="local_velocity"' not in source
    assert 'value="nav2"' not in source
    assert "executor:data.executor" in source
    assert "sensors.lidar_source" in source
    assert "|| 'native_1102'" in source
    assert "local_only: false" in source
    assert "force_native: true" in source
    assert "data.executor_override.reason" in source
    assert "goal_adjustment_m" in source
    assert "mișcare blocată până la RUN" in source
    assert "response.status === 401" in source
    assert "sessionStorage.removeItem(TOKEN_STORAGE_KEY)" in source
    assert "radius:0.25" in source
    assert "minPoints:3" in source
    assert "backend HTTP ${response.status}" in source


def test_costmap_endpoint_returns_structured_result(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "map.pcd"
    pcd.write_text("pcd", encoding="utf-8")

    monkeypatch.setattr(server, "loaded_map_path", str(pcd))
    monkeypatch.setattr(server, "_resolve_map_path", lambda _path: str(pcd))
    monkeypatch.setattr(server, "_get_nav2_observer", lambda: None)
    monkeypatch.setattr(server.obstacle_guard, "set_floor_plane", lambda _plane: None)
    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(
        server, "_build_static_costmap_preview",
        lambda *_args: {
            "_planner": object(), "success": True,
            "floor_plane": {"a": 0.0, "b": 0.0, "c": 0.0},
            "clearance_radius": 0.20,
        },
    )

    result = asyncio.run(server.get_navigation_costmap(
        resolution=0.08, min_points=3, min_z=0.15, max_z=1.85,
        radius=0.20, level_floor=True, floor_tolerance=0.08,
        comfort_radius=0.65, clearance_weight=6.0, driver="nav2",
    ))

    assert result["success"] is True
    assert result["clearance_radius"] == pytest.approx(0.20)
    assert "_planner" not in result


def test_nav2_runtime_uses_selected_hard_radius_not_astar_comfort(monkeypatch):
    import asyncio
    import server

    runtime = {}

    class PathClient:
        @staticmethod
        def server_is_ready():
            return True

    class Observer:
        path_client = PathClient()
        map_published_at = 1.0
        nav2_costmap_received_at = 1.0

        def set_runtime_parameters(self, values):
            runtime.update(values)
            return values

        def publish_map(self, _planner):
            return {"counts": {}}

        def compute_path(self, x, y, _yaw, _timeout):
            return {
                "path": [{"x": 0.0, "y": 0.0}, {"x": x, "y": y}],
                "poses": 2,
            }

        def simplify_execution_path(self, path, _maximum_cost):
            return [(point["x"], point["y"]) for point in path]

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "_get_nav2_observer", lambda: Observer())
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)

    result, path = asyncio.run(server._request_nav2_plan(
        object(), 2.0, 1.0, 0.0,
        {
            "robot_radius": 0.20,
            "resolution": 0.08,
            "comfort_radius": 0.65,
            "obstacle_min_z": 0.15,
            "obstacle_max_z": 1.85,
        },
    ))

    assert result["poses"] == 2
    assert path[-1] == (2.0, 1.0)
    assert runtime["inflation_layer.inflation_radius"] == pytest.approx(0.21)
    assert runtime["voxel_layer.lidar.obstacle_min_range"] == pytest.approx(0.30)


def test_dashboard_start_persists_token_and_reserves_mid360_for_native_slam():
    source = (BACKEND.parent / "start_dashboard.sh").read_text(encoding="utf-8")

    assert 'TOKEN_FILE="$SCRIPT_DIR/.dashboard_token_astar_v1"' in source
    assert 'umask 077' in source
    assert 'head -n 1 "$TOKEN_FILE"' in source
    assert "[l]ivox_ros_driver2_node" in source
    assert "--stop-native-sensors" in source
    assert "restore_mid360_native.cpp" in source
    assert "192.168.123.161" in source
    assert "--restart-slam" in source
    assert "ros2 run livox_ros_driver2" not in source


def test_astar_v1_launcher_uses_safe_cwd_and_its_own_token():
    source = (BACKEND.parent / "start_dashboard.sh").read_text(encoding="utf-8")

    assert ".dashboard_token_astar_v1" in source
    assert "/tmp/g1_dashboard_astar_v1.lock" in source
    assert 'export PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert "cd /home/unitree" in source

    nav2_start = (BACKEND / "nav2" / "start_nav2_observer.sh").read_text(
        encoding="utf-8"
    )
    assert 'dashboard_g1_v24_nav2_observer.pids' in nav2_start
    assert 'stop_recorded_group' in nav2_start
    assert '"${command}" == *"${PARAMS_FILE}"*' in nav2_start


def test_stale_native_preview_never_falls_back_to_local_executor(monkeypatch):
    import server

    monkeypatch.setattr(server, "_native_localization_fresh", lambda: False)
    monkeypatch.setattr(server, "_localization_fresh", lambda: True)
    monkeypatch.setattr(
        server.local_lidar_localizer, "status", lambda *args, **kwargs: {"ready": True}
    )

    selected, override = server._resolve_navigation_executor_for_runtime("native_1102")

    assert selected == "native_1102"
    assert override is None


def test_native_preview_is_not_rerouted_without_fresh_local_icp(monkeypatch):
    import server

    monkeypatch.setattr(server, "_native_localization_fresh", lambda: False)
    monkeypatch.setattr(server, "_localization_fresh", lambda: False)

    selected, override = server._resolve_navigation_executor_for_runtime("native_1102")

    assert selected == "native_1102"
    assert override is None


def test_failed_1804_keeps_explicit_local_anchor_available(monkeypatch):
    import asyncio
    import server

    modes = []

    class FakeNode:
        def publish_initial_pose(self, x, y, yaw):
            return True

        def set_slam_mode(self, mode):
            modes.append(mode)

    async def native_failure(*_args, **_kwargs):
        return {
            "success": False,
            "error": "Load pcd failed.",
            "attempts": [{"address": "/native/missing.pcd", "detail": "Load pcd failed."}],
        }

    previous_node = server.node_instance
    previous_mode = server.map_state["slam_mode"]
    try:
        monkeypatch.setattr(server, "node_instance", FakeNode())
        monkeypatch.setattr(server, "_initialize_native_pose_1804", native_failure)

        result = asyncio.run(server.relocalize({
            "map_name": "/map/existing.pcd", "x": 1.0, "y": 2.0, "yaw": 0.3,
        }))

        assert result["success"] is False
        assert result["display_pose_published"] is True
        assert result["local_controller_pose_ready"] is False
        assert result["local_controller_anchor_ready"] is True
        assert "API 1102 rămâne indisponibil" in result["local_controller_hint"]
        assert server.map_state["slam_mode"] == "localization"
        assert modes == ["localization"]
    finally:
        server.node_instance = previous_node
        server.map_state["slam_mode"] = previous_mode


def test_failed_1804_arms_and_verifies_local_icp_on_same_pose(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "existing.pcd"
    pcd.write_text("pcd", encoding="utf-8")
    resets = []

    class FakeNode:
        def publish_initial_pose(self, *_args): return True
        def set_slam_mode(self, _mode): pass

    class FakeLocalizer:
        def __init__(self): self.ready = False
        def reset(self, pose):
            resets.append(dict(pose))
            self.ready = True
        def status(self, *args, **kwargs):
            return {
                "map_configured": True, "map_path": str(pcd),
                "ready": self.ready, "inliers": 210,
                "inlier_ratio": 0.72, "score": 0.04, "error": "",
            }

    async def native_failure(*_args, **_kwargs):
        return {"success": False, "error": "Load pcd failed.", "attempts": []}

    async def no_sleep(_seconds):
        return None

    previous_mode = server.map_state["slam_mode"]
    try:
        monkeypatch.setattr(server, "node_instance", FakeNode())
        monkeypatch.setattr(server, "local_lidar_localizer", FakeLocalizer())
        monkeypatch.setattr(server, "_initialize_native_pose_1804", native_failure)
        monkeypatch.setattr(server.asyncio, "sleep", no_sleep)

        result = asyncio.run(server.relocalize({
            "map_name": str(pcd), "x": 1.0, "y": 2.0, "yaw": 0.3,
        }))

        assert result["success"] is False
        assert result["local_controller_fallback_attempted"] is True
        assert result["local_controller_pose_ready"] is True
        assert result["recommended_executor"] == "local_velocity"
        assert resets == [{"x": 1.0, "y": 2.0, "yaw": 0.3}]
        assert any(step["step"] == "local_lidar_icp_fallback" and step["ok"]
                   for step in result["steps"])
    finally:
        server.map_state["slam_mode"] = previous_mode


def test_loading_map_publishes_same_indexed_grid_to_nav2(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "loaded.pcd"
    pcd.write_text("pcd", encoding="utf-8")
    planner = object()
    published = []

    class Observer:
        def publish_map(self, candidate):
            published.append(candidate)
            return {"frame_id": "map", "counts": {"free": 42}}

    async def no_broadcast(_event):
        return None

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    previous_path = server.loaded_map_path
    previous_mode = server.map_state["slam_mode"]
    try:
        monkeypatch.setattr(server, "_any_navigation_active", lambda: False)
        monkeypatch.setattr(
            server.slam_client, "read_pcd_points", lambda _path: [{"x": 0, "y": 0, "z": 0}]
        )
        monkeypatch.setattr(
            server, "_configure_local_lidar_map",
            lambda _path: {"map_points": 42, "_planner": planner},
        )
        monkeypatch.setattr(server, "_get_nav2_observer", lambda: Observer())
        monkeypatch.setattr(server, "broadcast", no_broadcast)
        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)

        result = asyncio.run(server.load_robot_map({"map_name": str(pcd)}))

        assert result["success"] is True
        assert published == [planner]
        assert result["local_localization"]["nav2_map"]["counts"]["free"] == 42
        assert result["local_localization"]["initial_pose_preserved"] is False
        assert server.map_state["slam_mode"] == "idle"
        assert "_planner" not in result["local_localization"]
    finally:
        server.loaded_map_path = previous_path
        server.map_state["slam_mode"] = previous_mode


def test_loading_same_map_cannot_cancel_concurrent_localization_anchor(
        monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "anchored.pcd"
    pcd.write_text("pcd", encoding="utf-8")
    modes = []

    class AnchoredLocalizer:
        def status(self):
            return {
                "map_path": str(pcd),
                "map_configured": True,
                "initial_pose_set": True,
            }

    class Node:
        def set_slam_mode(self, mode):
            modes.append(mode)

    async def no_broadcast(_event):
        return None

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    previous_path = server.loaded_map_path
    previous_node = server.node_instance
    previous_mode = server.map_state["slam_mode"]
    try:
        monkeypatch.setattr(server, "_any_navigation_active", lambda: False)
        monkeypatch.setattr(
            server.slam_client, "read_pcd_points",
            lambda _path: [{"x": 0, "y": 0, "z": 0}],
        )
        monkeypatch.setattr(
            server, "_configure_local_lidar_map",
            lambda _path: {"map_points": 42, "_planner": object()},
        )
        monkeypatch.setattr(server, "local_lidar_localizer", AnchoredLocalizer())
        monkeypatch.setattr(server, "_get_nav2_observer", lambda: None)
        monkeypatch.setattr(server, "node_instance", Node())
        monkeypatch.setattr(server, "broadcast", no_broadcast)
        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)

        result = asyncio.run(server.load_robot_map({"map_name": str(pcd)}))

        assert result["success"] is True
        assert result["local_localization"]["initial_pose_preserved"] is True
        assert server.map_state["slam_mode"] == "localization"
        assert modes[-1] == "localization"
    finally:
        server.loaded_map_path = previous_path
        server.node_instance = previous_node
        server.map_state["slam_mode"] = previous_mode


def test_local_relocalization_skips_1804_and_requires_icp_convergence(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "existing.pcd"
    pcd.write_text("pcd", encoding="utf-8")
    native_calls = []
    previous_mode = server.map_state["slam_mode"]
    statuses = [
        {"map_configured": True, "map_path": str(pcd), "ready": False, "error": "starting"},
        {"map_configured": True, "map_path": str(pcd), "ready": True,
         "inliers": 120, "inlier_ratio": 0.55, "score": 0.04, "error": "",
         "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
    ]

    class FakeLocalizer:
        def status(self, *args, **kwargs):
            return dict(statuses.pop(0) if len(statuses) > 1 else statuses[0])

        def reset(self, pose):
            self.pose = dict(pose)

    class FakeNode:
        def publish_initial_pose(self, *_args): return True
        def set_slam_mode(self, _mode): pass

    monkeypatch.setattr(server, "local_lidar_localizer", FakeLocalizer())
    monkeypatch.setattr(server, "node_instance", FakeNode())
    monkeypatch.setattr(
        server, "_initialize_native_pose_1804",
        lambda *_args, **_kwargs: native_calls.append(True),
    )

    result = asyncio.run(server.relocalize({
        "map_name": str(pcd), "x": 1.0, "y": 2.0, "yaw": 0.3,
        "local_only": True,
    }))

    assert result["success"] is True
    assert result["native_localization_skipped"] is True
    assert result["local_controller_pose_ready"] is True
    assert result["refinement"] == {
        "ok": True, "x": 0.0, "y": 0.0, "yaw": 0.0,
        "score": 0.04, "inliers": 120, "inlier_ratio": 0.55,
    }
    assert native_calls == []
    server.map_state["slam_mode"] = previous_mode


def test_local_relocalization_reports_missing_lidar_frames(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "existing.pcd"
    pcd.write_text("pcd", encoding="utf-8")

    class FakeLocalizer:
        error = "convergență LiDAR în curs"

        def status(self, *args, **kwargs):
            return {
                "map_configured": True, "map_path": str(pcd),
                "ready": False, "inliers": 0, "inlier_ratio": 0.0,
                "score": None, "scan_points": 0, "error": self.error,
            }

        def reset(self, _pose):
            self.error = "convergență LiDAR în curs"

        def report_input_error(self, reason):
            self.error = str(reason)

    class FakeNode:
        last_raw_lidar_frame_time = 0.0
        last_raw_lidar_frame_points = 0
        last_raw_lidar_obstacle_points = 0

        def publish_initial_pose(self, *_args): return True
        def set_slam_mode(self, _mode): pass

    async def no_sleep(_seconds):
        return None

    previous_mode = server.map_state["slam_mode"]
    try:
        monkeypatch.setattr(server, "local_lidar_localizer", FakeLocalizer())
        monkeypatch.setattr(server, "node_instance", FakeNode())
        monkeypatch.setattr(server.asyncio, "sleep", no_sleep)

        result = asyncio.run(server.relocalize({
            "map_name": str(pcd), "x": 1.0, "y": 2.0, "yaw": 0.3,
            "local_only": True,
        }))

        assert result["success"] is False
        assert "nu sosesc cadre pe /utlidar/cloud_livox_mid360" in result["error"]
        assert result["local_localization"]["scan_points"] == 0
    finally:
        server.map_state["slam_mode"] = previous_mode


def test_cached_native_relocalization_skips_known_impossible_1804(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "indexed.pcd"
    pcd.write_text("pcd", encoding="utf-8")
    native_calls = []

    class FakeLocalizer:
        ready = False

        def status(self, *args, **kwargs):
            return {
                "map_configured": True, "map_path": str(pcd),
                "ready": self.ready, "inliers": 180 if self.ready else 0,
                "inlier_ratio": 0.75 if self.ready else 0.0,
                "score": 0.03 if self.ready else None, "error": "",
            }

        def reset(self, _pose):
            self.ready = True

    class FakeNode:
        def publish_initial_pose(self, *_args): return True
        def set_slam_mode(self, _mode): pass

    async def native_init(*_args, **_kwargs):
        native_calls.append(True)
        return {"success": False}

    monkeypatch.setattr(server, "slam_runtime_info", {"controller": "not init"})
    monkeypatch.setattr(server, "local_lidar_localizer", FakeLocalizer())
    monkeypatch.setattr(server, "node_instance", FakeNode())
    monkeypatch.setattr(server, "_initialize_native_pose_1804", native_init)

    result = asyncio.run(server.relocalize({
        "map_name": str(pcd), "x": 1.0, "y": 2.0, "yaw": 0.3,
        # Forma trimisă de frontendurile vechi: nu cunoșteau `force_native`.
        "local_only": False,
    }))

    assert result["success"] is True
    assert result["native_localization_skipped"] is True
    assert result["recommended_executor"] == "local_velocity"
    assert result["executor_override"]["requested"] == "native_1102"
    assert native_calls == []


@pytest.mark.parametrize("driver", ["legacy", "nav2"])
def test_navigation_preview_keeps_requested_goal_for_both_planners(
        monkeypatch, tmp_path, driver):
    import asyncio
    import server

    pcd = tmp_path / "navigation.pcd"
    pcd.write_text("pcd", encoding="utf-8")

    class Planner:
        floor_plane = {"a": 0.0, "b": 0.0, "c": 0.0}
        resolution = 0.08
        raw_static_occupied = set()
        robot_radius = 0.25
        obstacle_min_z = 0.15
        obstacle_max_z = 1.85
        min_obstacle_points = 3
        comfort_radius = 0.65
        clearance_weight = 6.0

        def dynamic_costmap_points(self):
            return []

    class PathTools:
        planner = None

        def _stabilize_departure_path(self, path, _pose):
            return list(path)

    class Localizer:
        def status(self, *args, **kwargs):
            return {"ready": True}

    async def nav2_plan(_planner, x, y, _yaw, _settings):
        return ({"planner_id": "GridBased", "poses": 2,
                 "planning_time_s": 0.01}, [(1.0, 1.0), (x, y)])

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    previous_path = server.loaded_map_path
    previous_mode = server.map_state["slam_mode"]
    previous_pose = dict(server.map_state["pose"])
    previous_previews = server.pending_nav_previews
    try:
        server.loaded_map_path = str(pcd)
        server.map_state["slam_mode"] = "localization"
        server.map_state["pose"] = {"x": 1.0, "y": 1.0, "yaw": 0.0}
        server.pending_nav_previews = {}
        path_tools = PathTools()
        monkeypatch.setattr(server, "_localization_fresh", lambda: True)
        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
        monkeypatch.setattr(server, "local_lidar_localizer", Localizer())
        monkeypatch.setattr(
            server, "plan_pcd_route",
            lambda *_args, **_kwargs: (
                Planner(), [(1.0, 1.0), (2.1, 2.0)], "normal"
            ),
        )
        monkeypatch.setattr(
            server.obstacle_guard, "configure_navigation_map",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(server, "_get_autonomous_navigator", lambda: path_tools)
        monkeypatch.setattr(server, "_get_native_waypoint_navigator", lambda: path_tools)
        monkeypatch.setattr(server, "_request_nav2_plan", nav2_plan)

        result = asyncio.run(server._preview_navigation({
            "x": 2.0, "y": 2.0, "yaw": 0.4,
            "speed": 0.2, "timeout": 0,
            "driver": driver, "executor": "local_velocity",
        }))

        assert result["success"] is True
        assert result["requested_goal"] == {"x": 2.0, "y": 2.0, "yaw": 0.4}
        assert result["goal"] == {"x": 2.1, "y": 2.0, "yaw": 0.4}
        assert result["goal_adjustment_m"] == pytest.approx(0.1)
        assert result["driver"] == driver
        preview = next(iter(server.pending_nav_previews.values()))
        assert preview["requested_goal"] == result["requested_goal"]
    finally:
        server.loaded_map_path = previous_path
        server.map_state["slam_mode"] = previous_mode
        server.map_state["pose"] = previous_pose
        server.pending_nav_previews = previous_previews


def test_unload_map_only_clears_runtime_selection(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "kept_map.pcd"
    pcd.write_text("# .PCD v0.7\nDATA ascii\n", encoding="utf-8")
    previous_path = server.loaded_map_path
    previous_previews = server.pending_nav_previews
    previous_mode = server.map_state["slam_mode"]

    class IdleNavigator:
        task = None

    events = []

    async def capture(event):
        events.append(event)

    try:
        monkeypatch.setattr(server, "_get_native_waypoint_navigator", lambda: IdleNavigator())
        monkeypatch.setattr(server, "broadcast", capture)
        server.loaded_map_path = str(pcd)
        server.pending_nav_previews = {"preview": {}}
        server.map_state["slam_mode"] = "localization"

        result = asyncio.run(server.unload_robot_map())

        assert result["success"] is True
        assert result["file_deleted"] is False
        assert result["saved_path"] == str(pcd)
        assert pcd.exists()
        assert server.loaded_map_path is None
        assert server.pending_nav_previews == {}
        assert events[-1]["type"] == "loaded_map_cleared"
    finally:
        server.loaded_map_path = previous_path
        server.pending_nav_previews = previous_previews
        server.map_state["slam_mode"] = previous_mode


def test_1804_retries_selected_pcd_after_stale_native_address(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "harta_retry.pcd"
    pcd.write_text("# .PCD v0.7\nDATA ascii\n", encoding="utf-8")
    stale = "/home/unitree/.slam_save_harta_retry_stale.pcd"
    calls = []
    feedbacks = iter([
        {
            "status_code": 0,
            "payload": {"succeed": False, "errorCode": 1, "info": "Load pcd failed."},
        },
        {
            "status_code": 0,
            "payload": {"succeed": True, "errorCode": 0, "info": "Load pcd success."},
        },
    ])

    monkeypatch.setitem(server.NATIVE_SLAM_MAP_PATHS, pcd.name, stale)
    monkeypatch.setattr(
        server.slam_client, "set_initial_pose",
        lambda x, y, yaw, address: calls.append(address) or {"success": True},
    )

    async def next_feedback(*_args, **_kwargs):
        return next(feedbacks)

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    remembered = []
    monkeypatch.setattr(server, "_wait_slam_feedback", next_feedback)
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(
        server, "_remember_native_slam_map",
        lambda local, native: remembered.append((local, native)),
    )

    result = asyncio.run(server._initialize_native_pose_1804(str(pcd), 1.0, 2.0, 0.3))

    assert result["success"] is True
    assert result["address"] == str(pcd)
    assert calls == [stale, str(pcd)]
    assert [attempt["accepted"] for attempt in result["attempts"]] == [False, True]
    assert remembered == [(str(pcd), str(pcd))]


def test_1804_retries_recent_yaw_from_same_native_map(monkeypatch, tmp_path):
    import asyncio
    import math
    import time
    import server

    pcd = tmp_path / "harta_same_map.pcd"
    pcd.write_text("# .PCD v0.7\nDATA ascii\n", encoding="utf-8")
    native = "/home/unitree/.slam_save_harta_same_map_123.pcd"
    calls = []
    feedbacks = iter([
        {"status_code": 0, "payload": {
            "succeed": False, "errorCode": 1,
            "info": "The current location matching degree is low.",
        }},
        {"status_code": 0, "payload": {
            "succeed": True, "errorCode": 0,
            "info": "Successfully started localization.",
        }},
    ])

    monkeypatch.setitem(server.NATIVE_SLAM_MAP_PATHS, pcd.name, native)
    monkeypatch.setattr(server, "slam_pose_info", {
        "received_at": time.monotonic(),
        "current_pose": {"x": 4.13, "y": -2.94, "yaw": 0.50},
        "map_address": native,
        "pcd_name": pcd.name,
    })
    monkeypatch.setattr(
        server.slam_client, "set_initial_pose",
        lambda x, y, yaw, address:
            calls.append((x, y, yaw, address)) or {"success": True},
    )

    async def next_feedback(*_args, **_kwargs):
        return next(feedbacks)

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "_wait_slam_feedback", next_feedback)
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(server, "_remember_native_slam_map", lambda *_args: None)

    result = asyncio.run(server._initialize_native_pose_1804(
        str(pcd), 4.53, -3.01, math.radians(104.0)
    ))

    assert result["success"] is True
    assert result["pose_source"] == "operator_xy_last_native_yaw"
    assert calls[0][2] == pytest.approx(math.radians(104.0))
    assert calls[1][:3] == pytest.approx((4.53, -3.01, 0.50))
    assert result["pose"] == pytest.approx({
        "x": 4.53, "y": -3.01, "yaw": 0.50,
    })


def test_1804_recovers_missing_lidar_imu_once_then_retries(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "harta_sensor_recovery.pcd"
    pcd.write_text("# .PCD v0.7\nDATA ascii\n", encoding="utf-8")
    calls = []
    feedbacks = iter([
        {
            "status_code": 0,
            "payload": {
                "succeed": False,
                "errorCode": 1,
                "info": "Lack of lidar or imu data.",
            },
        },
        {
            "status_code": 0,
            "payload": {
                "succeed": True,
                "errorCode": 0,
                "info": "Successfully started localization.",
            },
        },
    ])
    recoveries = []

    monkeypatch.delitem(server.NATIVE_SLAM_MAP_PATHS, pcd.name, raising=False)
    monkeypatch.setattr(
        server.slam_client, "set_initial_pose",
        lambda *_args: calls.append(_args[-1]) or {"success": True},
    )

    async def next_feedback(*_args, **_kwargs):
        return next(feedbacks)

    async def recovered():
        recoveries.append(True)
        return {"success": True}

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "_wait_slam_feedback", next_feedback)
    monkeypatch.setattr(server, "_recover_native_lidar_imu", recovered)
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(server, "_remember_native_slam_map", lambda *_args: None)

    result = asyncio.run(
        server._initialize_native_pose_1804(str(pcd), 1.0, 2.0, 0.3)
    )

    assert result["success"] is True
    assert calls == [str(pcd), str(pcd)]
    assert recoveries == [True]
    assert result["sensor_recovery"] == {"success": True}
    assert [attempt["accepted"] for attempt in result["attempts"]] == [False, True]


def test_1804_accepts_fresh_same_map_pos_info_when_response_is_missing(
        monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "harta_pos_info.pcd"
    pcd.write_text("# .PCD v0.7\nDATA ascii\n", encoding="utf-8")
    native = "/home/unitree/.slam_save_harta_pos_info_123.pcd"
    recoveries = []

    monkeypatch.setitem(server.NATIVE_SLAM_MAP_PATHS, pcd.name, native)
    monkeypatch.setattr(
        server.slam_client, "set_initial_pose", lambda *_args: {"success": True},
    )
    monkeypatch.setattr(server, "slam_runtime_info", {"error_code": 0})

    async def no_response(*_args, **_kwargs):
        monkeypatch.setattr(server, "slam_pose_info", {
            "received_at": server.time.monotonic(),
            "current_pose": {"x": 1.08, "y": 2.04, "yaw": 0.32},
            "map_address": native,
            "pcd_name": pcd.name,
        })
        return None

    async def must_not_recover():
        recoveries.append(True)
        return {"success": True}

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "_wait_slam_feedback", no_response)
    monkeypatch.setattr(server, "_recover_native_lidar_imu", must_not_recover)
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(server, "_remember_native_slam_map", lambda *_args: None)

    result = asyncio.run(
        server._initialize_native_pose_1804(str(pcd), 1.0, 2.0, 0.30)
    )

    assert result["success"] is True
    assert result["attempts"][0]["feedback"] is False
    assert result["attempts"][0]["pose_confirmation"]["source"] == (
        "/slam_info pos_info"
    )
    assert recoveries == []


def test_1804_missing_response_triggers_one_native_sensor_recovery(
        monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "harta_no_response.pcd"
    pcd.write_text("# .PCD v0.7\nDATA ascii\n", encoding="utf-8")
    calls = []
    feedbacks = iter([
        None,
        {
            "status_code": 0,
            "payload": {"succeed": True, "errorCode": 0},
        },
    ])
    recoveries = []

    monkeypatch.delitem(server.NATIVE_SLAM_MAP_PATHS, pcd.name, raising=False)
    monkeypatch.setattr(server, "slam_pose_info", {
        "received_at": 0.0, "current_pose": {}, "map_address": "",
    })
    monkeypatch.setattr(server, "slam_runtime_info", {"error_code": 0})
    monkeypatch.setattr(
        server.slam_client, "set_initial_pose",
        lambda *_args: calls.append(_args[-1]) or {"success": True},
    )

    async def next_feedback(*_args, **_kwargs):
        return next(feedbacks)

    async def recovered():
        recoveries.append(True)
        return {"success": True}

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "_wait_slam_feedback", next_feedback)
    monkeypatch.setattr(server, "_recover_native_lidar_imu", recovered)
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(server, "_remember_native_slam_map", lambda *_args: None)

    result = asyncio.run(
        server._initialize_native_pose_1804(str(pcd), 1.0, 2.0, 0.3)
    )

    assert result["success"] is True
    assert calls == [str(pcd), str(pcd)]
    assert recoveries == [True]
    assert [attempt["accepted"] for attempt in result["attempts"]] == [False, True]


def test_1804_explains_missing_native_copy_without_deleting_local_pcd(monkeypatch, tmp_path):
    import asyncio
    import server

    pcd = tmp_path / "local_only.pcd"
    pcd.write_text("# .PCD v0.7\nDATA ascii\n", encoding="utf-8")
    native = "/home/unitree/.slam_save_local_only_remote.pcd"
    monkeypatch.setitem(server.NATIVE_SLAM_MAP_PATHS, pcd.name, native)
    monkeypatch.setattr(
        server.slam_client, "set_initial_pose", lambda *_args: {"success": True},
    )

    async def rejected(*_args, **_kwargs):
        return {
            "status_code": 0,
            "payload": {"succeed": False, "errorCode": 1, "info": "Load pcd failed."},
        }

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(server, "_wait_slam_feedback", rejected)
    monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)

    result = asyncio.run(server._initialize_native_pose_1804(str(pcd), 0.0, 0.0, 0.0))

    assert result["success"] is False
    assert result["native_copy_missing"] is True
    assert result["local_pcd_preserved"] is True
    assert "PCD-ul local rămâne valid" in result["error"]
    assert "nu impune recartografierea" in result["error"]
    assert pcd.exists()


def test_1802_keeps_native_address_and_writes_separate_local_copy(monkeypatch, tmp_path):
    import asyncio
    import server

    remembered = []
    previous_mode = server.map_state["slam_mode"]

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def accepted(*_args, **_kwargs):
        return {
            "status_code": 0,
            "payload": {"succeed": True, "errorCode": 0, "info": "Save pcd successfully."},
        }

    async def native_file_not_visible(*_args, **_kwargs):
        return False

    try:
        server.map_state["slam_mode"] = "mapping"
        monkeypatch.setattr(server, "_map_dir", lambda: str(tmp_path))
        monkeypatch.setattr(server.slam_client, "save_map", lambda address: {"success": True})
        monkeypatch.setattr(server, "_wait_slam_feedback", accepted)
        monkeypatch.setattr(server, "_wait_for_map_file", native_file_not_visible)
        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
        monkeypatch.setattr(
            server, "_save_accumulated_pcd",
            lambda path: Path(path).write_text("pcd", encoding="utf-8") or 17,
        )
        monkeypatch.setattr(
            server, "_remember_native_slam_map",
            lambda local, native: remembered.append((local, native)),
        )

        result = asyncio.run(server.save_robot_map({"map_name": "separate"}))

        assert result["success"] is True
        assert result["native_save_acknowledged"] is True
        assert result["native_path"].startswith("/home/unitree/.slam_save_separate_")
        assert result["native_path"].endswith(".pcd")
        assert result["local_copy_source"] == "dashboard_accumulator"
        assert remembered == [(str(tmp_path / "separate.pcd"), result["native_path"])]
        assert (tmp_path / "separate.pcd").exists()
    finally:
        server.map_state["slam_mode"] = previous_mode


def test_first_keyboard_command_stops_autonomy_before_move(monkeypatch):
    import asyncio
    import json
    import server

    order = []

    class ActiveTask:
        def done(self):
            return False

    class ActiveNavigator:
        task = ActiveTask()

        async def stop(self, reason):
            order.append(("stop_autonomy", reason))
            return {"success": True}

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send_text(self, message):
            self.messages.append(json.loads(message))

    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    previous_owner = server._teleop_owner
    previous_time = server._teleop_last_cmd_time
    try:
        monkeypatch.setattr(server, "DASHBOARD_TELEOP_ENABLED", True)
        monkeypatch.setattr(server, "_get_native_waypoint_navigator", lambda: ActiveNavigator())
        monkeypatch.setattr(server.obstacle_guard, "is_blocked", lambda *_args: False)
        monkeypatch.setattr(
            server.sport_client, "move_to",
            lambda vx, vy, vyaw: order.append(("move", vx, vy, vyaw)) or {"success": True},
        )
        monkeypatch.setattr(server.asyncio, "to_thread", direct_to_thread)
        server._teleop_owner = None
        server._teleop_last_cmd_time = 0.0
        ws = FakeWebSocket()

        asyncio.run(server.handle_teleop_command(ws, 0.20, 0.0, 0.0))

        assert order == [
            ("stop_autonomy", "control preluat de tastatură"),
            ("move", 0.20, 0.0, 0.0),
        ]
        assert ws.messages == [{
            "type": "teleop_takeover",
            "message": "Autonomia a fost oprită; controlul aparține tastaturii.",
        }]
    finally:
        server._teleop_owner = previous_owner
        server._teleop_last_cmd_time = previous_time


def test_frontend_waits_for_first_1102_acceptance_before_saying_started():
    source = (BACKEND.parent / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "Aștept acceptarea primei comenzi Unitree 1102" in source
    assert "Navigare pornită pe" not in source


def test_native_1102_readiness_never_uses_ros_odom_as_substitute(monkeypatch):
    import server

    class RelocationOdom:
        active_odom_source = "relocation"
        last_primary_odom_time = 10**12

    monkeypatch.setattr(server, "node_instance", RelocationOdom())
    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": 0.0,
        "current_pose": {},
        "controller": "not init",
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")

    assert server._native_localization_fresh() is False


def test_native_localization_survives_transient_not_init_ctrl_info(monkeypatch):
    """Un ctrl_info oscilant nu anulează un pos_info nativ recent."""
    import server

    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": server.time.monotonic(),
        "current_pose": {"x": 1.12, "y": 0.36, "q_w": 1.0, "q_z": 0.0},
        "controller": "not init",
        "message_type": "ctrl_info",
        "error_code": 0,
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")

    assert server._native_localization_fresh() is True


def test_native_localization_still_rejects_stale_pos_info(monkeypatch):
    import server

    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": server.time.monotonic() - 2.1,
        "current_pose": {"x": 1.12, "y": 0.36, "q_w": 1.0, "q_z": 0.0},
        "controller": "pregătit",
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")

    assert server._native_localization_fresh() is False


def test_native_following_bridges_short_pos_info_gap_with_relocation(monkeypatch):
    import server

    now_mono = server.time.monotonic()
    now_wall = server.time.time()

    class RelocationNode:
        active_odom_source = "relocation"
        last_primary_odom_time = now_wall

    monkeypatch.setattr(server, "node_instance", RelocationNode())
    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": now_mono - 3.0,
        "received_at": now_mono,
        "current_pose": {"x": 1.0, "y": 2.0, "q_w": 1.0, "q_z": 0.0},
        "machine_state": "FOLLOWING",
        "controller": "not init",
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")
    monkeypatch.setitem(server.map_state, "pose_updated_at", now_wall)
    monkeypatch.setitem(server.map_state, "pose", {"x": 1.4, "y": 2.2, "yaw": 0.3})

    assert server._native_localization_fresh() is True
    assert server._native_pose() == {"x": 1.4, "y": 2.2, "yaw": 0.3}


def test_native_pose_prefers_fresh_pos_info_over_relocation_yaw(monkeypatch):
    import server

    now_mono = server.time.monotonic()
    now_wall = server.time.time()

    class RelocationNode:
        active_odom_source = "relocation"
        last_primary_odom_time = now_wall

    monkeypatch.setattr(server, "node_instance", RelocationNode())
    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": now_mono,
        "received_at": now_mono,
        "current_pose": {"x": 1.0, "y": 2.0, "yaw": -0.7},
        "machine_state": "FOLLOWING",
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")
    monkeypatch.setitem(server.map_state, "pose_updated_at", now_wall)
    monkeypatch.setitem(
        server.map_state, "pose", {"x": 1.4, "y": 2.2, "yaw": 0.9}
    )

    assert server._native_pose() == {"x": 1.0, "y": 2.0, "yaw": -0.7}


def test_native_relocation_bridge_survives_pos_info_gap_after_failed_route(
        monkeypatch, tmp_path):
    import server

    local_map = tmp_path / "bridge_map.pcd"
    local_map.write_text("VERSION .7\n", encoding="utf-8")
    now_mono = server.time.monotonic()
    now_wall = server.time.time()

    class RelocationNode:
        active_odom_source = "relocation"
        last_primary_odom_time = now_wall

    monkeypatch.setattr(server, "node_instance", RelocationNode())
    monkeypatch.setattr(server, "loaded_map_path", str(local_map))
    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": now_mono - 20.0,
        "received_at": now_mono,
        "current_pose": {"x": 6.61, "y": 2.24, "yaw": -2.84},
        "map_address": "/home/unitree/.slam_save_bridge_map_123.pcd",
        "machine_state": "FOLLOWING",
        "controller": "not init",
        "error_code": 0,
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")
    monkeypatch.setitem(server.map_state, "pose_updated_at", now_wall)
    monkeypatch.setitem(
        server.map_state, "pose", {"x": 6.62, "y": 2.23, "yaw": -2.83}
    )

    assert server._native_localization_fresh() is True
    assert server._native_pose() == {"x": 6.62, "y": 2.23, "yaw": -2.83}


def test_active_1102_task_tolerates_firmware_pos_info_pause_without_odom(monkeypatch):
    import server

    now_mono = server.time.monotonic()

    class RunningTask:
        @staticmethod
        def done():
            return False

    class Navigator:
        task = RunningTask()

    monkeypatch.setattr(server, "node_instance", None)
    monkeypatch.setattr(server, "native_waypoint_navigator", Navigator())
    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": now_mono - 5.0,
        "received_at": now_mono,
        "current_pose": {"x": 1.21, "y": 0.54, "q_w": 1.0, "q_z": 0.0},
        "machine_state": "FINISHED",
        "controller": "not init",
        "error_code": 0,
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")

    assert server._native_localization_fresh() is True
    assert server._native_pose()["x"] == pytest.approx(1.21)


def test_active_1102_uses_fresh_anchored_pelvis_during_long_pos_gap(
        monkeypatch, tmp_path):
    import server

    local_map = tmp_path / "pelvis_bridge.pcd"
    local_map.write_text("VERSION .7\n", encoding="utf-8")
    now_mono = server.time.monotonic()
    now_wall = server.time.time()

    class RunningTask:
        @staticmethod
        def done():
            return False

    class Navigator:
        task = RunningTask()

    class AnchoredNode:
        has_pelvis_offset = True
        active_odom_source = None
        last_primary_odom_time = 0.0

    monkeypatch.setattr(server, "node_instance", AnchoredNode())
    monkeypatch.setattr(server, "native_waypoint_navigator", Navigator())
    monkeypatch.setattr(server, "loaded_map_path", str(local_map))
    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": now_mono - 25.0,
        "received_at": now_mono,
        "current_pose": {"x": 4.2, "y": 1.1, "yaw": 0.2},
        "map_address": "/home/unitree/.slam_save_pelvis_bridge_123.pcd",
        "machine_state": "ADJUSTMENT",
        "error_code": 0,
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")
    monkeypatch.setitem(server.map_state, "pose_source", "anchored_pelvis")
    monkeypatch.setitem(server.map_state, "pose_updated_at", now_wall)
    monkeypatch.setitem(
        server.map_state, "pose", {"x": 4.85, "y": 1.32, "yaw": 0.28}
    )

    assert server._native_localization_fresh() is True
    assert server._native_pose() == {"x": 4.85, "y": 1.32, "yaw": 0.28}


def test_anchored_pelvis_cannot_bootstrap_native_1102(monkeypatch, tmp_path):
    import server

    local_map = tmp_path / "no_bootstrap.pcd"
    local_map.write_text("VERSION .7\n", encoding="utf-8")

    class AnchoredNode:
        has_pelvis_offset = True
        active_odom_source = None
        last_primary_odom_time = 0.0

    monkeypatch.setattr(server, "node_instance", AnchoredNode())
    monkeypatch.setattr(server, "native_waypoint_navigator", None)
    monkeypatch.setattr(server, "loaded_map_path", str(local_map))
    monkeypatch.setattr(server, "slam_runtime_info", {
        "pose_received_at": 0.0,
        "received_at": server.time.monotonic(),
        "current_pose": {},
        "map_address": "",
        "machine_state": "READY",
        "error_code": 0,
    })
    monkeypatch.setitem(server.map_state, "slam_mode", "localization")
    monkeypatch.setitem(server.map_state, "pose_source", "anchored_pelvis")
    monkeypatch.setitem(server.map_state, "pose_updated_at", server.time.time())
    monkeypatch.setitem(
        server.map_state, "pose", {"x": 1.0, "y": 2.0, "yaw": 0.0}
    )

    assert server._native_localization_fresh() is False


def test_native_finished_is_correlated_with_current_dense_waypoint(monkeypatch):
    import server

    now = server.time.monotonic()
    monkeypatch.setattr(server, "slam_last_completion", {
        "received_at": now,
        "machine_state": "FINISHED",
        "arrived": True,
        "current_pose": {"x": 0.25, "y": 0.21},
    })

    assert server._native_waypoint_completed(0.66, 0.27, now - 1.0) is True
    assert server._native_waypoint_completed(0.66, 0.27, now + 1.0) is False
    assert server._native_waypoint_completed(1.20, 0.27, now - 1.0) is False


def test_frontend_map_clear_and_token_are_scoped_to_v24():
    source = (BACKEND.parent / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "g1_dashboard_v24_token:${window.location.host}${window.location.pathname}" in source
    assert "g1_dashboard_v24_map_hidden:${window.location.host}${window.location.pathname}" in source
    assert "mesh.geometry = new THREE.BufferGeometry()" in source
    assert "!mapDisplaySuppressed && mapViewFilters.loadedVisible" in source


def test_map_load_replaces_view_atomically_without_clear_first():
    source = (BACKEND / "server.py").read_text(encoding="utf-8")
    load_section = source[source.index("async def load_robot_map"):source.index(
        "async def unload_robot_map"
    )]

    assert '"replace": True' in load_section
    assert 'broadcast({"type": "loaded_map_cleared"})' not in load_section


def test_v24_defaults_to_current_harta_03aug():
    import server

    assert server.GOOD_MAP_PATH == "/home/unitree/g1_ws/map/harta_03aug.pcd"
