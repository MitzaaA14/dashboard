#!/usr/bin/env python3
"""
server.py - Versiune Monolit Optimizată pentru Eliminarea Lag-ului și Oglindirii
"""

import asyncio
import json
import os
import secrets
import subprocess
import threading
import time
import sys
import math
import struct
import base64
import socket
import select
import re
import shutil
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.append("/opt/ros/humble/lib/python3.10/site-packages")

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String as RosString
    from unitree_api.msg import Response as UnitreeResponse
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from robot_client import slam_client, sport_client, obstacle_guard
from localization_refinement import estimate_pose_correction
from autonomous_navigation import (
    AutonomousNavigator,
    NativeWaypointNavigator,
    PCDGridPlanner,
    plan_pcd_route,
)
from nav2_observer import Nav2ObserverPublisher
from local_lidar_localization import LocalLidarLocalizer

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
POINTS_PER_SCAN = 7000
MAX_VOXELS_BEFORE_DOWNSAMPLE = 250000
# Nav2 calculeaza o raza inscrisa de aproximativ 0.206 m pentru amprenta G1.
# InflationLayer trebuie sa fie cel putin la fel de mare ca aceasta raza.
NAV2_MIN_INFLATION_RADIUS = 0.21

# Hărțile create de serviciul SLAM trăiesc pe hostul intern al robotului,
# nu în filesystemul Orin. PCD-ul local este folosit doar pentru viewer.
NATIVE_SLAM_MAP_PATHS = {
    "harta_test1.pcd": "/home/unitree/.slam_save_harta_test1_1784807423643.pcd",
    "harta_noua_buna.pcd": "/home/unitree/.slam_save_harta_noua_buna_1785152512189.pcd",
    "lala.pcd": "/home/unitree/.slam_save_lala_1785154169722.pcd",
}
LEGACY_NATIVE_SLAM_MAP_REGISTRY = Path(__file__).with_name("native_slam_maps.json")
ROBOT_NATIVE_SLAM_MAP_REGISTRY = Path("/home/unitree/g1_ws/map/.native_slam_maps.json")
NATIVE_SLAM_MAP_REGISTRY = Path(os.environ.get(
    "G1_NATIVE_SLAM_MAP_REGISTRY",
    str(ROBOT_NATIVE_SLAM_MAP_REGISTRY)
    if ROBOT_NATIVE_SLAM_MAP_REGISTRY.parent.is_dir()
    else str(LEGACY_NATIVE_SLAM_MAP_REGISTRY),
))


def _write_native_slam_map_registry() -> None:
    """Scrie atomic registrul în locația persistentă a hărților."""
    temporary = NATIVE_SLAM_MAP_REGISTRY.with_name(NATIVE_SLAM_MAP_REGISTRY.name + ".tmp")
    temporary.write_text(
        json.dumps(NATIVE_SLAM_MAP_PATHS, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, NATIVE_SLAM_MAP_REGISTRY)


def _load_native_slam_map_registry() -> None:
    """Încarcă registrul stabil și migrează automat registrul vechi din surse."""
    registries = {LEGACY_NATIVE_SLAM_MAP_REGISTRY, NATIVE_SLAM_MAP_REGISTRY}
    loaded = False
    for registry in registries:
        try:
            stored = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(stored, dict):
            NATIVE_SLAM_MAP_PATHS.update({
                os.path.basename(str(name)): str(native_path)
                for name, native_path in stored.items()
                if name and native_path
            })
            loaded = True
    if (loaded
            and NATIVE_SLAM_MAP_REGISTRY != LEGACY_NATIVE_SLAM_MAP_REGISTRY
            and not NATIVE_SLAM_MAP_REGISTRY.exists()):
        try:
            _write_native_slam_map_registry()
        except OSError:
            pass


def _remember_native_slam_map(local_path: str, native_path: str) -> None:
    """Persistă calea nativă separat de sursele dashboard-ului."""
    key = os.path.basename(str(local_path))
    NATIVE_SLAM_MAP_PATHS[key] = str(native_path)
    _write_native_slam_map_registry()


_load_native_slam_map_registry()


def _native_slam_map_path(local_or_remote_path: str) -> str:
    return _native_slam_map_candidates(local_or_remote_path)[0]


def _native_slam_map_candidates(local_or_remote_path: str) -> List[str]:
    """Adrese posibile pentru 1804, în ordinea în care merită încercate.

    Registrul poate conține o cale din filesystemul procesului SLAM, în timp
    ce viewerul folosește copia locală din g1_ws/map. Nu presupunem că una
    dintre ele este accesibilă: acceptarea este decisă numai de feedback-ul
    real al API-ului 1804.
    """
    basename = os.path.basename(str(local_or_remote_path))
    candidates = []

    def add(path) -> None:
        value = str(path or "").strip()
        if value and value not in candidates:
            candidates.append(value)

    # `pos_info` publică adresa reală din filesystemul serviciului SLAM.
    # Aceasta nu este neapărat vizibilă în filesystemul dashboardului.
    runtime = globals().get("slam_runtime_info", {})
    address = str(runtime.get("map_address") or "")
    match = re.match(r"^/home/unitree/\.slam_save_(.+)_\d+\.pcd$", address)
    if match and basename == f"{match.group(1)}.pcd":
        NATIVE_SLAM_MAP_PATHS[basename] = address
        add(address)
    add(NATIVE_SLAM_MAP_PATHS.get(basename))
    add(local_or_remote_path)
    if not candidates:
        candidates.append(str(local_or_remote_path))
    return candidates

class Real3DMapAccumulator:
    def __init__(self):
        self.voxels = {}  
        self.current_cell_size = None  

    def add_points(self, points, cell_size, min_hits):
        if self.current_cell_size is None:
            self.current_cell_size = cell_size
        effective_cell_size = self.current_cell_size
        for p in points:
            gx = round(p["x"] / effective_cell_size)
            gy = round(p["y"] / effective_cell_size)
            gz = round(p["z"] / effective_cell_size)
            key = (gx, gy, gz)

            if key not in self.voxels:
                self.voxels[key] = {"x": p["x"], "y": p["y"], "z": p["z"], "hits": 1}
            else:
                self.voxels[key]["hits"] += 1

        while len(self.voxels) > MAX_VOXELS_BEFORE_DOWNSAMPLE:
            self._downsample()

    def _downsample(self):
        base_cell = self.current_cell_size or 0.035
        new_cell = base_cell * 2.0
        merged = {}
        for v in self.voxels.values():
            gx = round(v["x"] / new_cell)
            gy = round(v["y"] / new_cell)
            gz = round(v["z"] / new_cell)
            key = (gx, gy, gz)
            if key not in merged:
                merged[key] = {"x": v["x"], "y": v["y"], "z": v["z"], "hits": v["hits"]}
            else:
                merged[key]["hits"] += v["hits"]
        self.voxels = merged
        self.current_cell_size = new_cell

    def get_filtered_points(self, min_hits):
        return [v for v in self.voxels.values() if v["hits"] >= min_hits]

    def reset(self):
        self.voxels.clear()
        self.current_cell_size = None

accumulator_3d = Real3DMapAccumulator()

OBSTACLE_DANGER_M  = 0.60
OBSTACLE_WARNING_M = 1.00
# Excludem partea de jos a imaginii, unde podeaua era interpretată ca obstacol.
OBSTACLE_ROI_ROWS  = (0.25, 0.68)
# Un colț de masă/poliță poate ocupa puțini pixeli, dar poate lovi mâna G1.
# Pragurile procentuale mici rămân protejate de numărul minim de pixeli validați.
OBSTACLE_DANGER_FRACTION = 0.01
OBSTACLE_WARNING_FRACTION = 0.02
OBSTACLE_MIN_PIXELS = 40

def detect_obstacles(depth_img):
    h, w = depth_img.shape
    r0, r1 = int(h * OBSTACLE_ROI_ROWS[0]), int(h * OBSTACLE_ROI_ROWS[1])
    roi = depth_img[r0:r1, :]

    zones = {}
    thirds = w // 3
    for name, (c0, c1) in [("left", (0, thirds)), ("center", (thirds, 2*thirds)), ("right", (2*thirds, w))]:
        col = roi[:, c0:c1].astype(np.float32) / 1000.0
        valid = col[(col > 0.15) & (col < 5.0)]
        if valid.size == 0:
            zones[name] = {"dist": None, "level": "safe"}
            continue
        danger = valid[valid < OBSTACLE_DANGER_M]
        warning = valid[valid < OBSTACLE_WARNING_M]
        danger_ratio = danger.size / valid.size
        warning_ratio = warning.size / valid.size
        if danger.size >= OBSTACLE_MIN_PIXELS and danger_ratio >= OBSTACLE_DANGER_FRACTION:
            level = "danger"
            d = float(np.percentile(danger, 20))
        elif warning.size >= OBSTACLE_MIN_PIXELS and warning_ratio >= OBSTACLE_WARNING_FRACTION:
            level = "warning"
            d = float(np.percentile(warning, 20))
        else:
            level = "safe"
            d = float(np.percentile(valid, 20))
        zones[name] = {"dist": round(d, 2), "level": level}
    return zones

app = FastAPI(title="G1 3D Monolith Engine")
CONTROL_TOKEN = os.environ.get("G1_DASHBOARD_TOKEN", "")
DASHBOARD_TELEOP_ENABLED = os.environ.get("G1_DASHBOARD_TELEOP_ENABLED", "1") == "1"

def _valid_control_token(candidate: Optional[str]) -> bool:
    return bool(
        CONTROL_TOKEN
        and candidate
        and secrets.compare_digest(candidate, CONTROL_TOKEN)
    )

def _safe_map_name(value) -> Optional[str]:
    name = str(value or "").strip()
    if name.lower().endswith(".pcd"):
        name = name[:-4]
    if not name or len(name) > 64:
        return None
    if not all(ch.isalnum() or ch in "_-" for ch in name):
        return None
    return name

@app.middleware("http")
async def require_control_token(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        if not CONTROL_TOKEN:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "G1_DASHBOARD_TOKEN nu este configurat; controlul este blocat implicit",
                },
            )
        if not _valid_control_token(request.headers.get("X-G1-Token")):
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "Token dashboard invalid"},
            )
    return await call_next(request)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

active_ws: List[WebSocket] = []
points_lock = threading.Lock()
loop = None
node_instance = None
GOOD_MAP_PATH = "/home/unitree/g1_ws/map/harta_03aug.pcd"
TEMP_MAP_PATH = "/home/unitree/g1_ws/map/temp_map.pcd"
loaded_map_path = GOOD_MAP_PATH if os.path.isfile(GOOD_MAP_PATH) else TEMP_MAP_PATH
autonomous_navigator = None
native_waypoint_navigator = None
active_navigation_driver: Optional[str] = None
active_navigation_executor: Optional[str] = None
pending_nav_previews = {}
navigation_goal_queue = []
queue_transition_scheduled = False
navigation_post_action_task: Optional[asyncio.Task] = None
active_navigation_post_action: Optional[str] = None
NAVIGATION_GESTURES = {
    "none": {"label": "Fără gest", "duration": 0.0},
    "wave": {"label": "Face cu mâna", "duration": 4.0},
    "kiss": {"label": "Dă pupic", "duration": 4.0},
    "handshake": {"label": "Dă noroc", "duration": 4.0},
    "clap": {"label": "Bate din palme", "duration": 4.0},
}
NAVIGATION_LOG_DIR = Path(__file__).parent.parent / "logs"
navigation_flight_log_path: Optional[Path] = None
native_nav_status = {"state": "idle", "path": [], "goal": None, "error": "", "driver": "unitree_slam_1102"}
slam_runtime_info = {}
slam_pose_info = {"received_at": 0.0, "current_pose": {}, "map_address": "", "pcd_name": ""}
slam_api_feedback = {}
# FINISHED/arrived trebuie păstrat separat: următorul pos_info poate
# suprascrie imediat ctrl_info între două segmente A*.
slam_last_completion = {}
NATIVE_POS_INFO_MAX_AGE = 2.0
# În FOLLOWING/ROTATION firmware-ul Mid360 poate întrerupe temporar pos_info,
# deși relocation odom și ctrl_info continuă. Folosim această punte numai după
# ce aceeași sesiune a avut deja o poziție nativă validă.
# În jurnalele reale pos_info a lipsit 10–11 s în FOLLOWING, în timp ce
# ctrl_info, LiDAR-ul și controllerul nativ au rămas active. Puntea este doar
# pentru o cursă 1102 deja activă; la pornire rămân valabile verificările hărții.
NATIVE_ACTIVE_ODOM_BRIDGE_MAX_AGE = 45.0
NATIVE_ACTIVE_UNVERIFIED_GRACE_MAX_AGE = 15.0
# Odometria pelvisului este continuă chiar când firmware-ul 1102 întrerupe
# temporar pos_info. Nu o folosim pentru inițializare și nici nelimitat: este
# doar o punte recentă, în cadrul unei curse native deja validate pe aceeași
# hartă.
NATIVE_ANCHORED_ODOM_MAX_AGE = 0.60

map_state = {
    "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    "pose_updated_at": 0.0,
    "pose_source": None,
    "scan_paused": False,
    "slam_active": False,
    "slam_mode": "idle",
    "last_refinement": None,
    "local_localization": {},
}
local_lidar_localizer = LocalLidarLocalizer()

map_filter = {
    # Doar parametri de stocare; filtrarea vizuală este controlată din UI.
    "grid_cell": 0.025,
    "min_hits": 1,
}

MAP_BROADCAST_INTERVAL = 0.2  
_map_dirty = False
_map_dirty_lock = threading.Lock()

def _mark_map_dirty():
    global _map_dirty
    with _map_dirty_lock:
        _map_dirty = True

# -----------------------------------------------------------------------
# Teleop din dashboard (WASD) — watchdog server-side de siguranță:
# dacă nu mai vine nicio comandă în TELEOP_WATCHDOG_TIMEOUT secunde
# (tab închis, conexiune pierdută etc.), robotul se oprește automat.
# -----------------------------------------------------------------------
TELEOP_WATCHDOG_TIMEOUT = 0.6  # sec, > TELEOP_SEND_INTERVAL din frontend
_teleop_last_cmd_time = 0.0
_teleop_owner: Optional[WebSocket] = None
_teleop_command_lock = asyncio.Lock()
_teleop_block_notice_time = 0.0
TELEOP_MAX_LINEAR = 1.0
TELEOP_MAX_YAW = 1.5

async def teleop_watchdog_loop():
    global _teleop_last_cmd_time, _teleop_owner
    while True:
        await asyncio.sleep(0.15)
        async with _teleop_command_lock:
            last = _teleop_last_cmd_time
            if last and (time.time() - last) > TELEOP_WATCHDOG_TIMEOUT:
                await asyncio.to_thread(sport_client.stop)
                _teleop_last_cmd_time = 0.0
                _teleop_owner = None

async def handle_teleop_command(ws: WebSocket, vx: float, vy: float, vyaw: float) -> None:
    global _teleop_last_cmd_time, _teleop_owner, _teleop_block_notice_time
    if not DASHBOARD_TELEOP_ENABLED:
        await ws.send_text(json.dumps({
            "type": "teleop_error",
            "error": "Teleop dashboard este dezactivat cât timp bridge-ul /cmd_vel este activ",
        }))
        return


    values_are_safe = (
        all(math.isfinite(v) for v in (vx, vy, vyaw))
        and abs(vx) <= TELEOP_MAX_LINEAR
        and abs(vy) <= TELEOP_MAX_LINEAR
        and abs(vyaw) <= TELEOP_MAX_YAW
    )
    is_stop = all(abs(v) < 1e-4 for v in (vx, vy, vyaw))

    async with _teleop_command_lock:
        if not values_are_safe:
            if _teleop_owner is ws:
                await asyncio.to_thread(sport_client.stop)
                _teleop_owner = None
                _teleop_last_cmd_time = 0.0
            await ws.send_text(json.dumps({
                "type": "teleop_error",
                "error": "Comandă respinsă: valori nefinite sau peste limitele de siguranță",
            }))
            return

        if is_stop:
            if _teleop_owner is None or _teleop_owner is ws:
                await asyncio.to_thread(sport_client.stop)
                _teleop_owner = None
                _teleop_last_cmd_time = 0.0
            return

        if _teleop_owner is not None and _teleop_owner is not ws:
            await ws.send_text(json.dumps({
                "type": "teleop_error",
                "error": "Teleop este deja controlat de alt client",
            }))
            return

        # Barieră de ownership: prima comandă manuală nenulă oprește întâi
        # executorul autonom (1201 + STOP direct), apoi permite Move-ul teleop.
        # Astfel 1102 și tastatura nu pot controla simultan robotul.
        active_navigator = None
        # Trecem prin accessorul nativ și când instanța globală nu a fost încă
        # materializată. Asta păstrează o singură barieră de ownership pentru
        # toate adaptoarele și face imposibilă prima comandă teleop înainte de
        # confirmarea STOP-ului autonomiei.
        native_candidate = native_waypoint_navigator or _get_native_waypoint_navigator()
        if _navigator_is_active(native_candidate):
            active_navigator = native_candidate
        elif autonomous_navigator is not None and _navigator_is_active(autonomous_navigator):
            active_navigator = autonomous_navigator
        if active_navigator is not None:
            takeover = await active_navigator.stop("control preluat de tastatură")
            await asyncio.sleep(0)
            if not takeover.get("success"):
                await ws.send_text(json.dumps({
                    "type": "teleop_error",
                    "error": "Preluarea manuală a fost refuzată: autonomia nu a confirmat STOP",
                }))
                return
            await ws.send_text(json.dumps({
                "type": "teleop_takeover",
                "message": "Autonomia a fost oprită; controlul aparține tastaturii.",
            }))

        _teleop_owner = ws
        if obstacle_guard.is_blocked(vx, vy):
            await asyncio.to_thread(sport_client.stop)
            now = time.time()
            if now - _teleop_block_notice_time >= 0.75:
                await ws.send_text(json.dumps({
                    "type": "teleop_blocked",
                    "error": "Comanda a fost oprită: RealSense sau LiDAR vede un obstacol în direcția cerută",
                }))
                _teleop_block_notice_time = now
        else:
            result = await asyncio.to_thread(sport_client.move_to, vx, vy, vyaw)
            if not result.get("success"):
                await asyncio.to_thread(sport_client.stop)
                _teleop_owner = None
                _teleop_last_cmd_time = 0.0
                await ws.send_text(json.dumps({
                    "type": "teleop_error",
                    "error": result.get("error", "Comanda de mișcare a eșuat"),
                }))
                return
        _teleop_last_cmd_time = time.time()

async def release_teleop_owner(ws: WebSocket) -> None:
    global _teleop_last_cmd_time, _teleop_owner
    async with _teleop_command_lock:
        if _teleop_owner is ws:
            await asyncio.to_thread(sport_client.stop)
            _teleop_owner = None
            _teleop_last_cmd_time = 0.0

def _set_slam_mode(mode: str) -> None:
    map_state["slam_mode"] = mode
    map_state["slam_active"] = mode in {"mapping", "localization"}
    if node_instance and hasattr(node_instance, "set_slam_mode"):
        node_instance.set_slam_mode(mode)

async def map_broadcast_loop():
    global _map_dirty
    while True:
        await asyncio.sleep(MAP_BROADCAST_INTERVAL)
        with _map_dirty_lock:
            dirty = _map_dirty
            _map_dirty = False
        if not dirty:
            continue
        with points_lock:
            all_pts = accumulator_3d.get_filtered_points(map_filter["min_hits"])
        await broadcast({"type": "map_points", "points": all_pts})


def _livox_points_to_base(points: List[dict]) -> tuple:
    """Întoarce punctele în base_link și cota locală estimată a podelei."""
    if not points:
        return [], -0.75
    roll = 3.14
    pitch = 0.04014257279586953
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    base_points = []
    ground_candidates = []
    for point in points:
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        z = float(point.get("z", 0.0))
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        base_x = 0.0002835 + cp * x + sp * sr * y + sp * cr * z
        base_y = 0.00003 + cr * y - sr * z
        base_z = 0.46018 - sp * x + cp * sr * y + cp * cr * z
        horizontal = math.hypot(base_x, base_y)
        base_points.append((base_x, base_y, base_z))
        if 0.30 <= horizontal <= 3.0 and -1.8 <= base_z <= 0.30:
            ground_candidates.append(base_z)
    if ground_candidates:
        ground_candidates.sort()
        index = min(
            len(ground_candidates) - 1,
            max(0, int(len(ground_candidates) * 0.12)),
        )
        ground_base_z = max(-1.45, min(-0.35, ground_candidates[index]))
    else:
        ground_base_z = -0.75
    return base_points, ground_base_z


def _pointcloud_has_geometry(message, minimum_points: int = 100) -> bool:
    """Acceptă numai cadre LiDAR suficient de mari pentru siguranță și ICP.

    Mid360 alternează pe acest robot norii compleți cu mesaje de un singur
    punct. Mesajele mici nu trebuie să împrospăteze watchdog-ul LiDAR.
    """
    try:
        point_step = int(message.point_step)
        declared = int(message.width) * max(1, int(message.height))
        payload_points = len(message.data) // point_step if point_step >= 12 else 0
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return False
    return declared >= int(minimum_points) and payload_points >= int(minimum_points)


def _livox_obstacles_in_base(points: List[dict]) -> List[dict]:
    base_points, ground_base_z = _livox_points_to_base(points)
    return [
        {"x": x, "y": y, "z": z - ground_base_z}
        for x, y, z in base_points
        if 0.12 <= z - ground_base_z <= 1.85
    ]


def _transform_livox_points_to_map(points: List[dict], pose: dict,
                                    floor_plane: Optional[dict]) -> List[dict]:
    """Transformă cloudul brut livox_frame în map pentru protecția locală.

    Transformarea rigidă este aceeași cu TF-ul publicat de adaptorul Nav2.
    Cota verticală este aliniată la planul podelei din PCD folosind podeaua
    observată în cadrul curent; astfel nu presupunem că pelvisul are z=0 în
    harta Unitree.
    """
    base_points, ground_base_z = _livox_points_to_base(points)
    if not base_points:
        return []

    yaw = float(pose.get("yaw", 0.0))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    pose_x, pose_y = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
    transformed = []
    for base_x, base_y, base_z in base_points:
        world_x = pose_x + cosine * base_x - sine * base_y
        world_y = pose_y + sine * base_x + cosine * base_y
        floor_z = 0.0
        if floor_plane:
            floor_z = (
                float(floor_plane.get("a", 0.0)) * world_x
                + float(floor_plane.get("b", 0.0)) * world_y
                + float(floor_plane.get("c", 0.0))
            )
        transformed.append({
            "x": world_x,
            "y": world_y,
            "z": floor_z + (base_z - ground_base_z),
        })
    return transformed

if ROS2_AVAILABLE:
    class DualLidarSLAMSubscriber(Node):
        def __init__(self):
            super().__init__('g1_dense_monolith_subscriber')
            self.last_primary_odom_time = 0.0
            self.last_fusion_odom_time = 0.0
            self.last_primary_cloud_time = 0.0
            self.active_odom_source = None
            self.active_cloud_source = None

            # O singură familie de topicuri primare este acceptată conform modului SLAM.
            self.odom_pelvis_sub = self.create_subscription(Odometry, '/state_estimator/odom_pelvis', self.odom_pelvis_callback, 10)
            self.odom_slam_sub = self.create_subscription(Odometry, '/unitree/slam_mapping/odom', lambda msg: self.primary_odom_callback(msg, 'mapping', 'mapping'), 10)
            self.odom_fusion_sub = self.create_subscription(Odometry, '/state_estimator/fusion_odom', self.fusion_odom_callback, 10)
            self.odom_loc_sub = self.create_subscription(Odometry, '/unitree/slam_localization/odom', lambda msg: self.primary_odom_callback(msg, 'localization', 'slam_localization'), 10)
            self.odom_loc2_sub = self.create_subscription(Odometry, '/localization/odom', lambda msg: self.primary_odom_callback(msg, 'localization', 'localization'), 10)
            self.odom_reloc_sub = self.create_subscription(Odometry, '/unitree/slam_relocation/odom', lambda msg: self.primary_odom_callback(msg, 'localization', 'relocation'), 10)

            self.slam_sub = self.create_subscription(PointCloud2, '/unitree/slam_mapping/points', lambda msg: self.primary_cloud_callback(msg, 'mapping', 'mapping'), 10)
            self.loc_cloud_sub = self.create_subscription(PointCloud2, '/unitree/slam_localization/points', lambda msg: self.primary_cloud_callback(msg, 'localization', 'slam_localization'), 10)
            self.reloc_cloud_sub = self.create_subscription(PointCloud2, '/unitree/slam_relocation/points', lambda msg: self.primary_cloud_callback(msg, 'localization', 'relocation'), 10)
            # Când 1804 nu pornește relocalizarea, cloudurile SLAM rămân fără
            # cadre, însă LiDAR-ul Mid360 continuă să publice în livox_frame.
            # BEST_EFFORT este compatibil atât cu publisherul brut RELIABLE,
            # cât și cu profilul uzual de senzori ROS 2.
            self.raw_lidar_sub = self.create_subscription(
                PointCloud2, '/utlidar/cloud_livox_mid360',
                self.raw_lidar_callback, qos_profile_sensor_data,
            )
            self.slam_info_sub = self.create_subscription(RosString, '/slam_info', self.slam_info_callback, 10)
            self.slam_response_sub = self.create_subscription(UnitreeResponse, '/api/slam_operate/response', self.slam_response_callback, 10)
            self.last_raw_slam_msg = None
            self.suppress_pelvis_until = 0.0
            self.last_live_scan_points = []
            self.last_live_scan_time = 0.0
            self.last_local_lidar_attempt = 0.0
            self.last_raw_lidar_frame_time = 0.0
            self.last_raw_lidar_frame_points = 0
            self.last_raw_lidar_obstacle_points = 0
            self.last_raw_lidar_ground_base_z = -0.75
            self.nav2_observer = None

            # Profilul de deadline este A* + 1102 și nu inițializează deloc
            # adaptorul Nav2. Acesta poate fi reactivat explicit pentru teste,
            # fără să mai poată bloca pornirea traseului stabil.
            if os.environ.get("G1_ENABLE_NAV2_OBSERVER") == "1":
                try:
                    self.nav2_observer = Nav2ObserverPublisher(self)
                    self.get_logger().info(
                        "Nav2 planner adapter pregătit (fără publisher cmd_vel)"
                    )
                except Exception as exc:
                    self.get_logger().warning(f"Nav2 adapter indisponibil: {exc}")
            else:
                self.get_logger().info(
                    "Profil A* stabil activ; adaptorul Nav2 este dezactivat"
                )
            
            # Offsets pentru a preveni snap-back la pelvis odom
            self.pelvis_offset_x = 0.0
            self.pelvis_offset_y = 0.0
            self.pelvis_offset_yaw = 0.0
            self.has_pelvis_offset = False
            self.last_raw_pelvis_x = 0.0
            self.last_raw_pelvis_y = 0.0
            self.last_raw_pelvis_yaw = 0.0

            # Publisher pentru poziția inițială de localizare (/initialpose standard Nav2)
            try:
                from geometry_msgs.msg import PoseWithCovarianceStamped
                self._PoseWithCovStamped = PoseWithCovarianceStamped
                self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
            except Exception as e:
                self.get_logger().warning(f'Nu pot crea publisher /initialpose: {e}')
                self.init_pose_pub = None
                self._PoseWithCovStamped = None

        def slam_response_callback(self, msg):
            """Păstrează răspunsul real al serviciului SLAM, indexat după API ID."""
            try:
                api_id = int(msg.header.identity.api_id)
                raw_data = msg.data or ""
                payload = json.loads(raw_data) if raw_data else {}
                slam_api_feedback[api_id] = {
                    "received_at": time.monotonic(),
                    "request_id": int(msg.header.identity.id),
                    "api_id": api_id,
                    "status_code": int(msg.header.status.code),
                    "payload": payload,
                    "raw_data": raw_data,
                }
            except Exception as exc:
                self.get_logger().warning(f'Răspuns SLAM invalid: {exc}')

        def slam_info_callback(self, msg):
            """Transformă telemetria planificatorului Unitree în status pentru UI."""
            global slam_runtime_info, slam_pose_info, slam_last_completion
            global native_nav_status, loop
            try:
                root = json.loads(msg.data)
                data = root.get("data", {})
                machine = data.get("stateMachine", {})
                received_at = time.monotonic()
                message_type = str(root.get("type", ""))
                # Doar pos_info este sursa poziției de localizare. ctrl_info
                # raportează frecvent (0,0,0) când controllerul este `not init`.
                if message_type == "pos_info" and _pose_xy(data.get("currentPose")):
                    slam_pose_info = {
                        "received_at": received_at,
                        "current_pose": dict(data.get("currentPose") or {}),
                        "map_address": str(data.get("address") or ""),
                        "pcd_name": str(data.get("pcdName") or ""),
                    }
                    pos_x = float(slam_pose_info["current_pose"]["x"])
                    pos_y = float(slam_pose_info["current_pose"]["y"])
                    pos_yaw = _pose_yaw(slam_pose_info["current_pose"])
                    self._process_odom_coords(pos_x, pos_y, pos_yaw, "/slam_info pos_info")
                    if self.nav2_observer:
                        self.nav2_observer.update_global_pose(
                            {
                                "x": pos_x,
                                "y": pos_y,
                                "yaw": pos_yaw,
                            },
                            "/slam_info pos_info",
                        )
                slam_runtime_info = {
                    "received_at": received_at,
                    "pose_received_at": slam_pose_info.get("received_at", 0.0),
                    "message_type": message_type,
                    "error_code": root.get("errorCode", 0),
                    "info": root.get("info", ""),
                    "controller": machine.get("ctrName", ""),
                    "planner_open": bool(machine.get("isOpenPlan", False)),
                    "paused": bool(machine.get("isPause", False)),
                    "machine_state": machine.get("state", ""),
                    "arrived": bool(data.get("is_arrived", False)),
                    "current_pose": dict(slam_pose_info.get("current_pose") or {}),
                    "map_address": slam_pose_info.get("map_address", ""),
                    "pcd_name": slam_pose_info.get("pcd_name", ""),
                    "raw": root,
                }
                machine_state = str(slam_runtime_info["machine_state"]).strip().lower()
                if (
                    slam_runtime_info["arrived"]
                    or any(token in machine_state for token in (
                        "finished", "arrived", "reached"
                    ))
                ):
                    slam_last_completion = {
                        "received_at": received_at,
                        "current_pose": dict(slam_runtime_info["current_pose"]),
                        "machine_state": slam_runtime_info["machine_state"],
                        "arrived": slam_runtime_info["arrived"],
                        "info": slam_runtime_info["info"],
                        "raw": root,
                    }
                if native_nav_status.get("state") in {"starting", "navigating"}:
                    current = slam_runtime_info["current_pose"]
                    if "x" in current and "y" in current:
                        native_nav_status["pose"] = {
                            "x": current["x"],
                            "y": current["y"],
                            "yaw": _pose_yaw(current),
                        }
                    progress = _slam_navigation_progress(
                        slam_runtime_info,
                        native_nav_status.get("start_pose"),
                        native_nav_status.get("goal"),
                    )
                    if slam_runtime_info["error_code"]:
                        native_nav_status.update({
                            "state": "failed",
                            "error": root.get("info") or f"SLAM errorCode={slam_runtime_info['error_code']}",
                        })
                    elif progress == "arrived":
                        native_nav_status.update({"state": "arrived", "error": ""})
                    elif progress == "navigating":
                        native_nav_status.update({"state": "navigating", "error": ""})
                    if loop:
                        asyncio.run_coroutine_threadsafe(
                            broadcast({"type": "nav_status", **native_nav_status}), loop
                        )
            except Exception:
                return

        def publish_initial_pose(self, x: float, y: float, yaw: float) -> bool:
            """Publică pozitia initiala pe /initialpose pentru a anchora localizarea."""
            if not self.init_pose_pub or not self._PoseWithCovStamped:
                return False
            try:
                # Suspendăm odometria pelvis pentru 5 secunde
                self.suppress_pelvis_until = time.time() + 5.0
                
                # Transformare SE(2) pelvis -> map, nu simplă translație aditivă.
                self.pelvis_offset_yaw = float(yaw) - self.last_raw_pelvis_yaw
                c, s = math.cos(self.pelvis_offset_yaw), math.sin(self.pelvis_offset_yaw)
                self.pelvis_offset_x = float(x) - (c * self.last_raw_pelvis_x - s * self.last_raw_pelvis_y)
                self.pelvis_offset_y = float(y) - (s * self.last_raw_pelvis_x + c * self.last_raw_pelvis_y)
                self.has_pelvis_offset = True
                
                msg = self._PoseWithCovStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.pose.pose.position.x = float(x)
                msg.pose.pose.position.y = float(y)
                msg.pose.pose.position.z = 0.0
                msg.pose.pose.orientation.x = 0.0
                msg.pose.pose.orientation.y = 0.0
                msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
                msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
                # Covarianță moderată — robot știe approximativ unde e
                cov = [0.0] * 36
                cov[0]  = 0.25   # x
                cov[7]  = 0.25   # y
                cov[35] = 0.0685 # yaw
                msg.pose.covariance = cov
                self.init_pose_pub.publish(msg)
                # Ancora selectată este imediat poziția de afișare/controller.
                # Fluxul pelvis o va actualiza continuu după fereastra de
                # stabilizare, folosind aceeași transformare SE(2).
                map_state["pose"] = {
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "yaw": float(yaw),
                }
                map_state["pose_updated_at"] = time.time()
                map_state["pose_source"] = "anchored_pelvis"
                if self.nav2_observer:
                    self.nav2_observer.update_global_pose(
                        {"x": float(x), "y": float(y), "yaw": float(yaw)},
                        "/initialpose + anchored_pelvis",
                    )
                self.get_logger().info(f'[InitialPose] Publicat x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}°')
                return True
            except Exception as e:
                self.get_logger().error(f'Eroare publish_initial_pose: {e}')
                return False

        def set_slam_mode(self, mode: str):
            self.active_odom_source = None
            self.active_cloud_source = None
            self.last_primary_odom_time = 0.0
            self.last_fusion_odom_time = 0.0
            self.last_primary_cloud_time = 0.0

        def primary_odom_callback(self, msg, mode: str, source: str):
            if map_state.get("slam_mode") != mode:
                return
            if source == "relocation" and self.nav2_observer:
                self.nav2_observer.update_global_odometry(msg)
            now = time.time()
            if self.active_odom_source not in (None, source) and now - self.last_primary_odom_time <= 1.0:
                return
            self.active_odom_source = source
            self.last_primary_odom_time = now
            self._process_odom(msg, source)

        def fusion_odom_callback(self, msg):
            # În localizare, fusion_odom este în cadrul odometric brut. Dacă avem
            # ancora pelvis->map de la /initialpose, lăsăm pelvis callback să
            # publice poziția transformată; altfel fusion ar bloca exact fallback-ul.
            if map_state.get("slam_mode") == "localization" and self.has_pelvis_offset:
                return
            now = time.time()
            if now - self.last_primary_odom_time <= 1.0:
                return
            self.last_fusion_odom_time = now
            self._process_odom(msg, "fusion_odom")

        def primary_cloud_callback(self, msg, mode: str, source: str):
            if map_state.get("slam_mode") != mode:
                return
            now = time.time()
            if self.active_cloud_source not in (None, source) and now - self.last_primary_cloud_time <= 1.0:
                return
            self.active_cloud_source = source
            self.last_primary_cloud_time = now
            self.slam_callback(msg, source)

        def raw_lidar_callback(self, msg):
            """Fallback sigur LiDAR când serviciul 1804 nu emite cloud global."""
            if time.time() - self.last_primary_cloud_time <= 0.50:
                return
            if not _pointcloud_has_geometry(msg):
                return
            points = self._extract_points_from_cloud(msg)
            if len(points) < 100:
                return
            self.last_raw_lidar_frame_time = time.monotonic()
            self.last_raw_lidar_frame_points = len(points)
            pose = dict(map_state["pose"])
            base_points, ground_base_z = _livox_points_to_base(points)
            self.last_raw_lidar_ground_base_z = ground_base_z
            base_obstacles = [
                {"x": x, "y": y, "z": z - ground_base_z}
                for x, y, z in base_points
                if 0.12 <= z - ground_base_z <= 1.85
            ]
            self.last_raw_lidar_obstacle_points = len(base_obstacles)
            floor_plane = obstacle_guard.floor_plane()
            transformed = _transform_livox_points_to_map(
                points, pose, floor_plane
            )
            if transformed:
                obstacle_guard.update_lidar_points(
                    transformed,
                    pose,
                    source="/utlidar/cloud_livox_mid360 -> map",
                )
            now = time.monotonic()
            if (map_state.get("slam_mode") != "localization"
                    or not self.has_pelvis_offset
                    or len(base_obstacles) < 35
                    or now - self.last_local_lidar_attempt < 0.25):
                return
            self.last_local_lidar_attempt = now
            result = local_lidar_localizer.match(base_obstacles, pose)
            map_state["local_localization"] = local_lidar_localizer.status()
            if result.get("ok"):
                self.apply_local_lidar_pose(result["pose"])
            if loop:
                asyncio.run_coroutine_threadsafe(
                    broadcast({
                        "type": "localization_status",
                        **map_state["local_localization"],
                    }),
                    loop,
                )

        def apply_local_lidar_pose(self, pose: dict) -> None:
            """Aplică ICP ca o corecție SE(2) peste odometria pelvisului."""
            yaw = float(pose["yaw"])
            self.pelvis_offset_yaw = yaw - self.last_raw_pelvis_yaw
            cosine, sine = math.cos(self.pelvis_offset_yaw), math.sin(self.pelvis_offset_yaw)
            self.pelvis_offset_x = float(pose["x"]) - (
                cosine * self.last_raw_pelvis_x - sine * self.last_raw_pelvis_y
            )
            self.pelvis_offset_y = float(pose["y"]) - (
                sine * self.last_raw_pelvis_x + cosine * self.last_raw_pelvis_y
            )
            self.has_pelvis_offset = True
            self._process_odom_coords(
                float(pose["x"]), float(pose["y"]), yaw,
                "local_lidar_icp",
            )
            if self.nav2_observer:
                self.nav2_observer.update_localized_pose(pose)

        def odom_pelvis_callback(self, msg):
            # Salvăm poziţia brută a pelvisului în caz de click relocalizare
            q = msg.pose.pose.orientation
            raw_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.last_raw_pelvis_x = msg.pose.pose.position.x
            self.last_raw_pelvis_y = msg.pose.pose.position.y
            self.last_raw_pelvis_yaw = raw_yaw

            if time.time() < self.suppress_pelvis_until:
                return
            if self.has_pelvis_offset:
                c, s = math.cos(self.pelvis_offset_yaw), math.sin(self.pelvis_offset_yaw)
                x = c * self.last_raw_pelvis_x - s * self.last_raw_pelvis_y + self.pelvis_offset_x
                y = s * self.last_raw_pelvis_x + c * self.last_raw_pelvis_y + self.pelvis_offset_y
                yaw = self.last_raw_pelvis_yaw + self.pelvis_offset_yaw
                # Normalizăm yaw la [-pi, pi]
                yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
                local_status = local_lidar_localizer.status()
                source = (
                    "local_lidar_icp"
                    if local_status.get("ready") else "anchored_pelvis"
                )
                self._process_odom_coords(x, y, yaw, source)
            elif time.time() - max(self.last_primary_odom_time, self.last_fusion_odom_time) > 1.0:
                self._process_odom(msg, "pelvis_unanchored")

        def _process_odom(self, msg, source: str = "odom"):
            q = msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self._process_odom_coords(msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, source)

        def _process_odom_coords(self, x: float, y: float, yaw: float, source: str = "odom"):
            global loop
            map_state["pose"] = {"x": round(x, 3), "y": round(y, 3), "yaw": yaw}
            map_state["pose_updated_at"] = time.time()
            map_state["pose_source"] = source
            if self.nav2_observer:
                self.nav2_observer.update_global_pose(map_state["pose"], source)
            if loop: asyncio.run_coroutine_threadsafe(broadcast({
                "type": "pose", **map_state["pose"], "source": source,
            }), loop)

        def _extract_points_from_cloud(self, msg):
            raw_data = msg.data
            if not raw_data:
                return []
            total_pts = len(raw_data) // msg.point_step
            points = []
            step = max(1, total_pts // POINTS_PER_SCAN)
            try:
                for i in range(0, total_pts, step):
                    offset = i * msg.point_step
                    x_g, y_g, z_g = struct.unpack_from('<fff', raw_data, offset)
                    if math.isfinite(x_g) and math.isfinite(y_g) and math.isfinite(z_g):
                        points.append({"x": round(x_g, 2), "y": round(y_g, 2), "z": round(z_g, 2)})
            except Exception:
                return []
            return points

        def slam_callback(self, msg, source: str = "unknown"):
            global loop
            self.last_raw_slam_msg = msg
            points = self._extract_points_from_cloud(msg)
            if points:
                self.last_live_scan_points = points[:400]
                self.last_live_scan_time = time.time()
                lidar_topic = {
                    "mapping": "/unitree/slam_mapping/points",
                    "slam_localization": "/unitree/slam_localization/points",
                    "relocation": "/unitree/slam_relocation/points",
                }.get(source, str(source))
                obstacle_guard.update_lidar_points(
                    points, map_state["pose"], source=lidar_topic
                )
                if (not map_state.get("scan_paused")
                        and map_state.get("slam_mode") in {"mapping", "localization"}):
                    with points_lock:
                        accumulator_3d.add_points(points, map_filter["grid_cell"], map_filter["min_hits"])
                    _mark_map_dirty()

def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def camera_receiver_thread():
    global loop
    server_sock = None
    while server_sock is None:
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind(('0.0.0.0', 5005))
            candidate.listen(1)
            server_sock = candidate
        except OSError as exc:
            candidate.close()
            # La restart, vechiul receiver poate elibera portul câteva sute de
            # milisecunde după 3003. Threadul de siguranță nu trebuie să moară;
            # reîncearcă până când poate primi din nou RealSense.
            print(f"[camera] portul 5005 este temporar indisponibil: {exc}; reîncerc în 1s")
            time.sleep(1.0)

    while True:
        try:
            conn, addr = server_sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            while True:
                size_data = _recv_exact(conn, 4)
                if size_data is None: break
                payload_size = struct.unpack(">I", size_data)[0]
                payload = _recv_exact(conn, payload_size)
                if payload is None: break

                # DRENARE AGRESIVĂ BUFFER (Elimină Delay-ul/Lag-ul cumulativ)
                while True:
                    ready, _, _ = select.select([conn], [], [], 0)
                    if not ready:
                        break
                    next_size_data = _recv_exact(conn, 4)
                    if next_size_data is None:
                        break
                    next_payload_size = struct.unpack(">I", next_size_data)[0]
                    next_payload = _recv_exact(conn, next_payload_size)
                    if next_payload is None:
                        break
                    payload = next_payload  

                if len(payload) < 8: continue
                color_len, depth_len = struct.unpack(">II", payload[:8])
                color_bytes = payload[8:8+color_len]
                depth_bytes = payload[8+color_len:8+color_len+depth_len]

                depth_img = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                color_img = cv2.imdecode(np.frombuffer(color_bytes, np.uint8), cv2.IMREAD_COLOR)

                if color_img is None or depth_img is None:
                    print(f"[camera] decode eșuat: color_len={color_len} depth_len={depth_len} payload_total={len(payload)}")
                    continue
                zones = detect_obstacles(depth_img)
                obstacle_guard.update(zones)
                print(f"[obstacle] {zones}")
                if loop:
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"type": "obstacle_status", "zones": zones}), loop
                    )
                h, w = depth_img.shape
                cam_points = []
                fx, fy = 460.0, 460.0
                cx, cy = 320.0, 240.0

                step = 16 # Am mărit ușor pasul pentru a scădea dramatic utilizarea CPU și lag-ul
                for v in range(0, h, step):
                    for u in range(0, w, step):
                        z_m = depth_img[v, u] / 1000.0
                        if 0.2 < z_m < 3.0:
                            # REPARARE OGLINDIRE 3D PUNCTE CAMERĂ: 
                            # Oglindim u pe orizontală (w - u) pentru a anula efectul de oglindă din lentilă
                            u_mirrored = w - u
                            x_m = (u_mirrored - cx) * z_m / fx
                            y_m = (v - cy) * z_m / fy
                            b, g, r = color_img[v, u]

                            cam_points.append({
                                "x": float(x_m), "y": float(-y_m), "z": float(z_m),
                                "r": int(r), "g": int(g), "b": int(b)
                            })

                # REPARARE OGLINDIRE LIVE 2D FEED:
                # Oglindim imaginea pe orizontală înainte de trimitere folosind OpenCV
                color_img_flipped = cv2.flip(color_img, 1)
                _, color_bytes_flipped = cv2.imencode(".jpg", color_img_flipped, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                encoded_video = base64.b64encode(color_bytes_flipped.tobytes()).decode('utf-8')

                if loop and cam_points:
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"type": "camera_3d_data", "points": cam_points, "video_b64": encoded_video}), loop
                    )
        except Exception as e:
            time.sleep(1)

async def broadcast(data: dict):
    msg = json.dumps(data)
    for ws in active_ws[:]:
        try: await ws.send_text(msg)
        except Exception:
            if ws in active_ws: active_ws.remove(ws)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    if not _valid_control_token(ws.query_params.get("token")):
        await ws.close(code=1008, reason="Token dashboard invalid")
        return
    active_ws.append(ws)
    try:
        with points_lock: all_pts = accumulator_3d.get_filtered_points(map_filter["min_hits"])
        await ws.send_text(json.dumps({
            "type": "init", "pose": map_state["pose"], "slam_points": all_pts,
            "pose_source": map_state.get("pose_source"),
            "map_filter": map_filter, "scan_paused": map_state["scan_paused"],
            "loaded_map_path": _resolve_map_path(loaded_map_path),
        }))
        current_map = _resolve_map_path(loaded_map_path)
        if current_map:
            points = await asyncio.to_thread(slam_client.read_pcd_points, current_map)
            if points:
                step = max(1, math.ceil(len(points) / 100000))
                await ws.send_text(json.dumps({
                    "type": "loaded_map_points",
                    "points": points[::step],
                    "path": current_map,
                }))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if msg.get("type") == "teleop":
                try:
                    vx = float(msg.get("vx", 0.0))
                    vy = float(msg.get("vy", 0.0))
                    vyaw = float(msg.get("vyaw", 0.0))
                except (TypeError, ValueError):
                    await ws.send_text(json.dumps({
                        "type": "teleop_error", "error": "Comandă teleop invalidă"
                    }))
                    continue
                await handle_teleop_command(ws, vx, vy, vyaw)
    except WebSocketDisconnect: pass
    finally:
        if ws in active_ws: active_ws.remove(ws)
        await release_teleop_owner(ws)

@app.post("/api/slam/clear")
async def clear_map():
    with points_lock: accumulator_3d.reset()
    await broadcast({"type": "map_reset"})
    return {"success": True}


@app.post("/api/slam/stop_scan")
async def stop_and_clear_realtime_scan():
    """Oprește acumularea înainte de reset, astfel încât harta să rămână goală."""
    map_state["scan_paused"] = True
    with points_lock:
        accumulator_3d.reset()
    await broadcast({"type": "map_reset"})
    await broadcast({"type": "scan_state", "active": False})
    return {"success": True, "active": False, "cleared": True}


@app.post("/api/slam/scan")
async def set_realtime_scan(body: dict = Body({})):
    """Pauză/continuare pentru acumularea și randarea cloud-ului live.

    Detecția de obstacole și odometria continuă să ruleze pentru siguranță;
    doar harta live nu mai este modificată cât scanarea este oprită.
    """
    active = bool(body.get("active", True))
    map_state["scan_paused"] = not active
    event = {"type": "scan_state", "active": active}
    await broadcast(event)
    return {"success": True, "active": active}

@app.post("/api/slam/save")
async def save_map_pcd(body: dict = Body({})):
    name = _safe_map_name(body.get("map_name", "harta_dense_monolith"))
    if not name:
        return {"success": False, "error": "Nume hartă invalid; folosește maximum 64 caractere alfanumerice, _ sau -"}
    filepath = os.path.join("/home/unitree/g1_ws/map" if os.path.exists("/home/unitree/g1_ws/map") else "/home/unitree", f"{name}.pcd")
    try:
        with points_lock: pts = accumulator_3d.get_filtered_points(map_filter["min_hits"])
        with open(filepath, 'w') as f:
            f.write(f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\nPOINTS {len(pts)}\nDATA ascii\n")
            for p in pts: f.write(f"{p['x']:.4f} {p['y']:.4f} {p['z']:.4f}\n")
        return {"success": True, "output": f"Hartă salvată în {filepath}"}
    except Exception as e: return {"success": False, "error": str(e)}

@app.post("/api/slam/start_mapping")
async def start_mapping():
    """Pornește cartografierea SLAM pe robot (API ID 1801)."""
    result = await asyncio.to_thread(slam_client.start_mapping)
    if result.get("success"):
        with points_lock: accumulator_3d.reset()
        _set_slam_mode("mapping")
        await broadcast({"type": "map_reset"})
    return result

def _map_dir() -> str:
    return "/home/unitree/g1_ws/map" if os.path.isdir("/home/unitree/g1_ws/map") else "/home/unitree"


async def _wait_for_map_file(path: str, timeout: float = 12.0) -> bool:
    """Robotul scrie harta asincron. Așteptăm fișierul, nu presupunem succes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        await asyncio.sleep(0.4)
    return False


def _save_accumulated_pcd(path: str) -> int:
    """Persistă atomic norul SLAM primit de dashboard; întoarce numărul de puncte."""
    with points_lock:
        pts = [
            (float(p["x"]), float(p["y"]), float(p["z"]))
            for p in accumulator_3d.get_filtered_points(map_filter["min_hits"])
        ]
    if not pts:
        return 0

    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(
                "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\n"
                "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {len(pts)}\nHEIGHT 1\nPOINTS {len(pts)}\nDATA ascii\n"
            )
            for x, y, z in pts:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    return len(pts)


@app.post("/api/slam/save_robot")
async def save_robot_map(body: dict = Body({})):
    """Oprește cartografierea și salvează harta pe robot (API ID 1802).

    1802 = End Mapping: încheie sesiunea ȘI scrie fișierul. Se poate apela o
    singură dată per sesiune de mapping, deci verificăm întâi că mapping-ul
    chiar rulează — altfel comanda pleacă în gol și pare că "nu merge salvarea".
    """
    name = _safe_map_name(body.get("map_name", "my_map"))
    if not name:
        return {"success": False, "error": "Nume hartă invalid; maximum 64 caractere alfanumerice, _ sau -"}

    if map_state["slam_mode"] != "mapping":
        return {
            "success": False,
            "error": f"Cartografierea nu rulează (mod curent: {map_state['slam_mode']}). "
                     f"Apasă întâi Start Cartografiere (1801); 1802 încheie sesiunea și nu are ce salva.",
        }

    map_dir = _map_dir()
    if not os.path.isdir(map_dir) or not os.access(map_dir, os.W_OK):
        return {
            "success": False,
            "error": f"Directorul pentru hărți nu există sau nu este scriibil: {map_dir}",
        }

    # API-ul Unitree 1802 primește o cale din filesystemul PROCESULUI SLAM.
    # Acesta poate rula pe alt host/container decât dashboardul. Prin urmare,
    # lipsa lui `staging` din filesystemul local nu infirmă salvarea nativă,
    # iar mutarea lui ar putea chiar șterge copia necesară ulterior lui 1804.
    # Păstrăm separat adresa nativă și copia locală pentru viewer/planner.
    request_id = f"{int(time.time() * 1000)}"
    staging = f"/home/unitree/.slam_save_{name}_{request_id}.pcd"
    expected = os.path.join(map_dir, f"{name}.pcd")

    sent_at = time.monotonic()
    result = await asyncio.to_thread(slam_client.save_map, staging)
    if not result.get("success"):
        return {
            "success": False,
            "error": f"Comanda 1802 a eșuat: {result.get('error') or result.get('output') or 'fără detalii'}",
        }

    feedback = await _wait_slam_feedback(1802, sent_at, timeout=8.0)
    if feedback is None:
        return {
            "success": False,
            "error": "Comanda 1802 a fost publicată, dar robotul nu a confirmat salvarea în 8s.",
            "publisher_result": result,
        }
    payload = feedback.get("payload") or {}
    rejected = (
        feedback.get("status_code", 0) != 0
        or payload.get("succeed") is False
        or int(payload.get("errorCode", 0) or 0) != 0
    )
    if rejected:
        code = payload.get("errorCode", feedback.get("status_code"))
        return {
            "success": False,
            "error": payload.get("info") or f"API 1802 a respins salvarea (cod {code})",
            "api_feedback": feedback,
        }

    # Feedback-ul pozitiv 1802 este singura confirmare pe care dashboardul o
    # poate primi pentru copia din spațiul nativ. Memorăm exact acea adresă;
    # disponibilitatea ei reală va fi validată ulterior numai de feedback 1804.
    try:
        _remember_native_slam_map(expected, staging)
    except OSError as exc:
        return {
            "success": False,
            "error": f"1802 a confirmat salvarea, dar registrul căii native nu a putut fi scris: {exc}",
            "native_path": staging,
            "api_feedback": feedback,
        }

    local_staging_visible = await _wait_for_map_file(staging, timeout=5.0)
    if not local_staging_visible:
        # Caz normal când SLAM rulează în alt namespace. Dashboardul deține
        # norul folosit în viewer și îl salvează independent în g1_ws/map.
        try:
            point_count = await asyncio.to_thread(_save_accumulated_pcd, expected)
        except Exception as exc:
            return {
                "success": False,
                "error": f"1802 nu a creat fișierul, iar salvarea locală a eșuat: {exc}",
            }
        if point_count == 0:
            return {
                "success": False,
                "error": "1802 a confirmat copia nativă, dar dashboardul nu a primit încă puncte SLAM "
                         "pentru copia locală. Așteaptă să apară harta în viewer înainte de salvare.",
                "native_path": staging,
                "native_save_acknowledged": True,
            }
        _set_slam_mode("idle")
        size_mb = os.path.getsize(expected) / (1024 * 1024)
        return {
            "success": True,
            "output": (
                f"Hartă salvată local: {expected} ({size_mb:.1f} MB, {point_count} puncte). "
                f"Copia nativă 1802: {staging}; va fi verificată la 1804."
            ),
            "local_copy_source": "dashboard_accumulator",
            "native_path": staging,
            "native_save_acknowledged": True,
            "native_copy_available": "unverified_until_1804",
            "api_feedback": feedback,
        }

    try:
        # Copiem, nu mutăm: dacă dashboardul și SLAM-ul văd același filesystem,
        # `staging` trebuie să rămână disponibil exact la adresa dată lui 1802.
        await asyncio.to_thread(shutil.copy2, staging, expected)
    except OSError as exc:
        return {
            "success": False,
            "error": f"Harta nativă a fost salvată, dar copia locală nu a putut fi creată în {expected}: {exc}",
            "native_path": staging,
            "native_save_acknowledged": True,
        }
    _set_slam_mode("idle")
    size_mb = os.path.getsize(expected) / (1024 * 1024)
    return {
        "success": True,
        "output": (
            f"Hartă salvată local: {expected} ({size_mb:.1f} MB). "
            f"Copia nativă 1802 păstrată la {staging}; va fi verificată la 1804."
        ),
        "local_copy_source": "native_staging_copy",
        "native_path": staging,
        "native_save_acknowledged": True,
        "native_copy_available": "unverified_until_1804",
        "api_feedback": feedback,
    }

@app.post("/api/slam/load_robot")
async def load_robot_map(body: dict = Body({})):
    """Selectează și afișează PCD-ul; poziția reală va porni apoi 1804.

    Nu inițializăm robotul implicit la (0, 0): API 1804 trebuie trimis o
    singură dată, cu poziția indicată de operator pe hartă.
    """
    global loaded_map_path, pending_nav_previews
    name = body.get("map_name")
    if not name:
        return {"success": False, "error": "Numele hărții este obligatoriu"}

    if _any_navigation_active():
        return {
            "success": False,
            "error": "Nu schimb harta cât autonomia rulează. Apasă mai întâi STOP navigare.",
        }

    # Citim PCD-ul de pe disc și îl trimitem la frontend
    filepath = name if os.path.exists(name) else None
    if filepath is None:
        base_dir = "/home/unitree/g1_ws/map"
        candidate = os.path.join(base_dir, os.path.basename(name))
        if os.path.exists(candidate):
            filepath = candidate
    
    if filepath and os.path.exists(filepath):
        pts = await asyncio.to_thread(slam_client.read_pcd_points, filepath)
        if not pts:
            return {"success": False, "error": f"Fișierul PCD nu conține puncte valide: {filepath}"}
        try:
            local_map = await asyncio.to_thread(
                _configure_local_lidar_map, filepath
            )
        except Exception as exc:
            return {
                "success": False,
                "error": f"Harta se poate afișa, dar localizatorul LiDAR local nu o poate indexa: {exc}",
            }
        nav2_planner = local_map.pop("_planner", None)

        # Abia după validare înlocuim harta veche din viewer. PCD-ul de pe disc
        # nu este modificat sau șters de acest flux.
        pending_nav_previews = {}
        with points_lock:
            accumulator_3d.reset()
        await broadcast({"type": "map_reset"})
        loaded_map_path = filepath
        # Păstrăm suficientă densitate pentru vizualizarea de tip Unitree.
        # Filtrul de densitate din frontend poate reduce ulterior fără a reciti PCD-ul.
        step = max(1, len(pts) // 100000)
        pts_sub = pts[::step]
        # Înlocuire atomică în viewer: nu trimitem `loaded_map_cleared` înaintea
        # payloadului mare. Dacă WebSocketul întârzie/repornește, harta veche
        # rămâne vizibilă până când geometria nouă a ajuns complet.
        await broadcast({
            "type": "loaded_map_points", "points": pts_sub,
            "path": filepath, "replace": True,
        })
        observer = _get_nav2_observer()
        if observer is not None and nav2_planner is not None:
            try:
                local_map["nav2_map"] = await asyncio.to_thread(
                    observer.publish_map, nav2_planner
                )
            except Exception as exc:
                local_map["nav2_warning"] = str(exc)
        # configure_map rulează într-un worker, iar operatorul poate termina
        # marcarea poziției înainte ca indexarea să revină aici. În acel caz
        # LocalLidarLocalizer păstrează ancora pentru aceeași hartă; nu anulăm
        # apoi fereastra ICP trecând orbește în idle.
        local_status = local_lidar_localizer.status()
        configured_path = local_status.get("map_path")
        localization_armed = bool(
            local_status.get("initial_pose_set")
            and configured_path
            and os.path.realpath(str(configured_path))
                == os.path.realpath(filepath)
        )
        _set_slam_mode("localization" if localization_armed else "idle")
        local_map["initial_pose_preserved"] = localization_armed
        return {
            "success": True,
            "output": (
                "Harta este afișată și indexată pentru localizare LiDAR locală. "
                "Marchează poziția și orientarea reală a robotului."
            ),
            "path": filepath,
            "local_localization": local_map,
        }

    return {"success": False, "error": f"Fișierul PCD nu există: {name}"}


@app.post("/api/slam/unload_robot")
async def unload_robot_map():
    """Ascunde harta salvată fără să șteargă fișierul PCD de pe disc."""
    global loaded_map_path, pending_nav_previews
    if _any_navigation_active():
        return {
            "success": False,
            "error": "Nu ascund harta cât autonomia rulează. Apasă mai întâi STOP navigare.",
        }

    previous_path = _resolve_map_path(loaded_map_path)
    pending_nav_previews = {}
    loaded_map_path = None
    local_lidar_localizer.clear()
    _set_slam_mode("idle")
    await broadcast({"type": "loaded_map_cleared", "previous_path": previous_path})
    return {
        "success": True,
        "cleared_from_view": True,
        "file_deleted": False,
        "saved_path": previous_path,
    }

def _localization_fresh(max_age: float = 2.0) -> bool:
    """True numai pentru localizare măsurată, nu pentru simpla ancoră manuală."""
    if map_state.get("slam_mode") != "localization":
        return False
    if not node_instance:
        return False
    src = getattr(node_instance, "active_odom_source", None)
    last = getattr(node_instance, "last_primary_odom_time", 0.0)
    if src and (time.time() - last) < max_age:
        return True
    local_status = local_lidar_localizer.status(max_age=max_age)
    pose_fresh = time.time() - float(map_state.get("pose_updated_at", 0.0)) < max_age
    return bool(
        local_status.get("ready")
        and map_state.get("pose_source") == "local_lidar_icp"
        and pose_fresh
    )


def _relocation_odom_fresh() -> bool:
    """Odometrie relocation recentă, ancorată în frame-ul hărții native."""
    if not node_instance or map_state.get("slam_mode") != "localization":
        return False
    odom_age = time.time() - float(
        getattr(node_instance, "last_primary_odom_time", 0.0) or 0.0
    )
    return bool(
        getattr(node_instance, "active_odom_source", None) == "relocation"
        and odom_age < 1.0
        and time.time() - float(map_state.get("pose_updated_at", 0.0)) < 1.0
    )


def _anchored_pelvis_odom_fresh() -> bool:
    """Poziție pelvis recentă, transformată în harta sesiunii curente."""
    if not node_instance or map_state.get("slam_mode") != "localization":
        return False
    return bool(
        getattr(node_instance, "has_pelvis_offset", False)
        and map_state.get("pose_source") in {
            "anchored_pelvis", "local_lidar_icp",
        }
        and _pose_xy(map_state.get("pose"))
        and time.time() - float(map_state.get("pose_updated_at", 0.0) or 0.0)
            < NATIVE_ANCHORED_ODOM_MAX_AGE
    )


def _native_runtime_same_map() -> bool:
    runtime_address = str(slam_runtime_info.get("map_address") or "").strip()
    active_map = _resolve_map_path(loaded_map_path)
    return bool(
        runtime_address and active_map
        and runtime_address in _native_slam_map_candidates(active_map)
    )


def _native_active_pose_grace_fresh() -> bool:
    """Tolerează pos_info lipsă folosind numai odometrie map recentă."""
    if map_state.get("slam_mode") != "localization":
        return False
    native_age = time.monotonic() - float(
        slam_runtime_info.get("pose_received_at", 0.0) or 0.0
    )
    telemetry_age = time.monotonic() - float(
        slam_runtime_info.get("received_at", 0.0) or 0.0
    )
    machine_state = str(
        slam_runtime_info.get("machine_state") or ""
    ).strip().lower()
    navigator_task = getattr(native_waypoint_navigator, "task", None)
    navigator_active = bool(navigator_task and not navigator_task.done())
    return bool(
        _pose_xy(slam_runtime_info.get("current_pose"))
        and NATIVE_POS_INFO_MAX_AGE <= native_age < NATIVE_ACTIVE_ODOM_BRIDGE_MAX_AGE
        and telemetry_age < 1.0
        and int(slam_runtime_info.get("error_code", 0) or 0) == 0
        and machine_state in {
            "ready", "following", "rotation", "adjustment", "finished",
        }
        and (
            (
                native_age < NATIVE_ACTIVE_UNVERIFIED_GRACE_MAX_AGE
                and (navigator_active or _relocation_odom_fresh())
            )
            or (
                navigator_active
                and _native_runtime_same_map()
                and (_relocation_odom_fresh() or _anchored_pelvis_odom_fresh())
            )
        )
    )


def _native_localization_fresh(max_age: float = NATIVE_POS_INFO_MAX_AGE) -> bool:
    """Dovadă de pose coerentă pentru o comandă API 1102.

    Preferăm întotdeauna ``pos_info``. Firmware-ul îl întrerupe însă tocmai
    după unele FINISHED/FAILED, deși relocation continuă pe aceeași hartă și
    ``ctrl_info`` rămâne sănătos. În acel caz permitem o punte controlată;
    acceptarea efectivă a noii rute rămâne decisă de răspunsul API 1102.
    """
    if map_state.get("slam_mode") != "localization":
        return False
    info_age = time.monotonic() - float(slam_runtime_info.get("pose_received_at", 0.0))
    # `ctrl_info` and `pos_info` are independent streams.  On the Unitree
    # firmware `ctrl_info` briefly reports ``not init`` between otherwise
    # valid `pos_info` packets (including while the same map and pose keep
    # updating).  Treating that controller label as loss of localization made
    # an active A* run fail on a single transient packet.  The authoritative
    # localization health signal is a recent native `pos_info`; readiness of
    # API 1102 is verified separately by the response to the command itself.
    native_fresh = bool(
        info_age < max_age
        and _pose_xy(slam_runtime_info.get("current_pose"))
    )
    telemetry_age = time.monotonic() - float(
        slam_runtime_info.get("received_at", 0.0) or 0.0
    )
    machine_state = str(
        slam_runtime_info.get("machine_state") or ""
    ).strip().lower()
    same_map = _native_runtime_same_map()
    relocation_bridge = bool(
        _pose_xy(slam_runtime_info.get("current_pose"))
        and _relocation_odom_fresh()
        and telemetry_age < 2.0
        and int(slam_runtime_info.get("error_code", 0) or 0) == 0
        and machine_state in {
            "ready", "following", "rotation", "adjustment", "finished",
        }
        and same_map
    )
    return (
        native_fresh
        or relocation_bridge
        or _native_active_pose_grace_fresh()
    )


def _resolve_map_path(map_name: Optional[str]) -> Optional[str]:
    if not map_name:
        return None
    if os.path.exists(map_name):
        return map_name
    candidates = [
        map_name,
        os.path.join("/home/unitree/g1_ws/map", os.path.basename(map_name)),
        os.path.join("/home/unitree", os.path.basename(map_name)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _configure_local_lidar_map(map_path: str) -> dict:
    """Extrage o singură dată geometria statică folosită de ICP local."""
    planner = PCDGridPlanner(
        resolution=0.10,
        robot_radius=0.25,
        min_obstacle_points=2,
        comfort_radius=0.65,
        clearance_weight=6.0,
    )
    planner.load(
        map_path,
        obstacle_min_z=0.12,
        obstacle_max_z=1.85,
        level_to_floor=True,
        floor_tolerance=0.08,
    )
    configured = local_lidar_localizer.configure_map(
        planner.raw_static_occupied, planner.resolution, map_path
    )
    configured["floor_plane"] = planner.floor_plane
    # Obiect intern, eliminat înainte de răspunsul JSON. Refolosirea aceluiași
    # planner publică imediat aceeași geometrie către Nav2, fără o a doua
    # rasterizare și fără ca operatorul să fie obligat să deschidă Costmap.
    configured["_planner"] = planner
    return configured


def _load_map_points(map_name: Optional[str]) -> List[dict]:
    path = _resolve_map_path(map_name)
    if not path:
        return []
    pts = slam_client.read_pcd_points(path)
    if not pts:
        return []
    step = max(1, len(pts) // 220)
    return pts[::step]


def _refine_pose_with_live_scan(initial_pose: dict, map_name: Optional[str]) -> Optional[dict]:
    live_points = []
    if node_instance and hasattr(node_instance, "last_live_scan_points"):
        live_points = getattr(node_instance, "last_live_scan_points", [])
    if not live_points:
        return None
    map_points = _load_map_points(map_name)
    if len(map_points) < 3 or len(live_points) < 3:
        return None
    return estimate_pose_correction(map_points, live_points, initial_pose)


def _slam_feedback_success(feedback: Optional[dict]) -> bool:
    payload = (feedback or {}).get("payload") or {}
    return bool(
        feedback
        and feedback.get("status_code", 0) == 0
        and payload.get("succeed", True) is not False
        and int(payload.get("errorCode", 0) or 0) == 0
    )


async def _recover_native_lidar_imu() -> dict:
    """Reia exclusiv Mid360 pentru serviciul Unitree după un eșec 1804."""
    external = await asyncio.to_thread(
        subprocess.run,
        ["pgrep", "-f", "[l]ivox_ros_driver2_node"],
        capture_output=True, text=True, timeout=3,
    )
    external_pids = [
        int(value) for value in external.stdout.split() if value.isdigit()
    ]
    if external_pids:
        return {
            "success": False,
            "error": (
                "un livox_ros_driver2_node extern deține Mid360 "
                f"(PID {external_pids}); oprește-l și repetă repoziționarea"
            ),
        }

    helper = Path(__file__).with_name("ensure_native_slam_services.py")
    restore_bin = os.environ.get("G1_MID360_RESTORE_BIN", "")
    if not helper.is_file() or not restore_bin or not os.path.isfile(restore_bin):
        return {
            "success": False,
            "error": "utilitarele de recuperare Mid360 nu sunt disponibile",
        }

    env = os.environ.copy()
    sdk_python = "/home/unitree/unitree_sdk2_python"
    env["PYTHONPATH"] = sdk_python + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    helper_base = [
        sys.executable, str(helper), "--interface", "enP8p1s0",
        "--timeout", "20",
    ]

    async def run(command, *, timeout: float, command_env=None):
        try:
            completed = await asyncio.to_thread(
                subprocess.run, command, env=command_env, capture_output=True,
                text=True, timeout=timeout,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {
            "success": completed.returncode == 0,
            "output": completed.stdout.strip(),
            "error": completed.stderr.strip(),
            "returncode": completed.returncode,
        }

    stopped = await run(
        helper_base + ["--stop-native-sensors"], timeout=75, command_env=env
    )
    if not stopped.get("success"):
        return {"success": False, "stop": stopped,
                "error": stopped.get("error") or "serviciile native nu s-au oprit"}

    restore_env = env.copy()
    sdk_lib = "/home/unitree/Livox-SDK2/build/sdk_core"
    restore_env["LD_LIBRARY_PATH"] = sdk_lib + (
        os.pathsep + restore_env["LD_LIBRARY_PATH"]
        if restore_env.get("LD_LIBRARY_PATH") else ""
    )
    restored = await run(
        [restore_bin, "/home/unitree/g1_ws/assets/mid360.robot.json",
         "192.168.123.161"],
        timeout=45, command_env=restore_env,
    )
    if not restored.get("success"):
        return {"success": False, "stop": stopped, "restore": restored,
                "error": restored.get("error") or "destinațiile Mid360 nu s-au restaurat"}

    restarted = await run(
        helper_base + ["--restart-slam"], timeout=75, command_env=env
    )
    return {
        "success": bool(restarted.get("success")),
        "stop": stopped,
        "restore": restored,
        "restart": restarted,
        "error": restarted.get("error"),
    }


def _native_pose_confirms_1804(
        addresses: List[str], x: float, y: float, yaw: float,
        sent_at: float,
) -> Optional[dict]:
    """Confirmă 1804 prin pos_info când firmware-ul omite response-ul API.

    Publisherul singur nu este dovadă. Acceptăm numai o poziție nativă nouă,
    pe una dintre adresele aceleiași hărți și suficient de aproape de ancora
    indicată de operator; astfel nu pornim 1102 într-o cameră greșită.
    """
    received_at = float(slam_pose_info.get("received_at", 0.0) or 0.0)
    pose = dict(slam_pose_info.get("current_pose") or {})
    address = str(slam_pose_info.get("map_address") or "").strip()
    position = _pose_xy(pose)
    if (
        received_at < sent_at
        or address not in addresses
        or not position
        or int(slam_runtime_info.get("error_code", 0) or 0) != 0
    ):
        return None
    position_error = math.hypot(position[0] - float(x), position[1] - float(y))
    yaw_error = abs(
        (_pose_yaw(pose) - float(yaw) + math.pi) % (2.0 * math.pi) - math.pi
    )
    if position_error > 1.25 or yaw_error > math.radians(55.0):
        return None
    return {
        "source": "/slam_info pos_info",
        "received_at": received_at,
        "address": address,
        "pose": pose,
        "position_error_m": round(position_error, 3),
        "yaw_error_deg": round(math.degrees(yaw_error), 1),
    }


async def _initialize_native_pose_1804(
    map_name: str, x: float, y: float, yaw: float
) -> dict:
    """Încearcă adresele plauzibile și crede numai feedback-ul API 1804."""
    attempts = []
    sensor_recovery = None
    addresses = _native_slam_map_candidates(map_name)
    pose_requests = [
        (address, "operator", float(x), float(y), float(yaw))
        for address in addresses
    ]

    # Firmware-ul poate rămâne `not init`, dar ultimul pos_info conține
    # ultima poză validă și adresa exactă a hărții. O folosim numai
    # pentru aceeași hartă, într-o fereastră scurtă și dacă poziția
    # indicată de operator este aproape. Astfel repară o săgeată cu yaw
    # greșit fără să caute arbitrar o potrivire în altă cameră.
    last_pose = dict(slam_pose_info.get("current_pose") or {})
    last_address = str(slam_pose_info.get("map_address") or "")
    last_received = float(slam_pose_info.get("received_at", 0.0) or 0.0)
    last_age = time.monotonic() - last_received if last_received else math.inf

    # După restart, memoria pos_info este goală. Recuperăm numai ultima
    # poziție măsurată dintr-un jurnal foarte recent al aceleiași hărți.
    if (last_address not in addresses or last_age > 600.0 or not _pose_xy(last_pose)):
        try:
            recent_logs = sorted(
                NAVIGATION_LOG_DIR.glob("nav_*.jsonl"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:5]
            recovered = None
            for log_path in recent_logs:
                lines = log_path.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines[-120:]):
                    entry = json.loads(line)
                    if time.time() - float(entry.get("recorded_at", 0.0)) > 600.0:
                        continue
                    slam = entry.get("slam") or {}
                    candidate_address = str(slam.get("map_address") or "")
                    candidate_pose = dict(slam.get("current_pose") or {})
                    if candidate_address in addresses and _pose_xy(candidate_pose):
                        recovered = (candidate_address, candidate_pose)
                        break
                if recovered:
                    break
            if recovered:
                last_address, last_pose = recovered
                last_received = time.monotonic()
                last_age = 0.0
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    last_xy = _pose_xy(last_pose)
    if (
        last_address in addresses
        and last_xy
        and last_age <= 600.0
        and math.hypot(last_xy[0] - x, last_xy[1] - y) <= 0.75
    ):
        last_yaw = _pose_yaw(last_pose)
        yaw_error = abs((float(yaw) - last_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        fallback_requests = []
        if yaw_error >= math.radians(8.0):
            fallback_requests.append(
                (last_address, "operator_xy_last_native_yaw",
                 float(x), float(y), last_yaw)
            )
        if math.hypot(last_xy[0] - x, last_xy[1] - y) >= 0.08:
            fallback_requests.append(
                (last_address, "last_native_pose",
                 float(last_xy[0]), float(last_xy[1]), last_yaw)
            )
        insertion = next(
            (index + 1 for index, request in enumerate(pose_requests)
             if request[0] == last_address),
            len(pose_requests),
        )
        pose_requests[insertion:insertion] = fallback_requests

    for address, pose_source, pose_x, pose_y, pose_yaw in pose_requests:
        # Exact o reîncercare este permisă după recuperarea senzorilor.
        # Nu repetăm la nesfârșit o comandă 1804 respinsă.
        for _attempt_index in range(2):
            sent_at = time.monotonic()
            publisher_result = await asyncio.to_thread(
                slam_client.set_initial_pose,
                pose_x, pose_y, pose_yaw, address,
            )
            feedback = (
                await _wait_slam_feedback(1804, sent_at, timeout=8.0)
                if publisher_result.get("success")
                else None
            )
            payload = (feedback or {}).get("payload") or {}
            pose_confirmation = _native_pose_confirms_1804(
                addresses, pose_x, pose_y, pose_yaw, sent_at,
            )
            accepted = bool(publisher_result.get("success")) and bool(
                _slam_feedback_success(feedback) or pose_confirmation
            )
            detail = (
                payload.get("info")
                or publisher_result.get("error")
                or (
                    "1804 confirmat prin pos_info nativ"
                    if pose_confirmation else ""
                )
                or ("fără feedback 1804 în 8s" if feedback is None else "1804 respins")
            )
            attempt = {
                "address": address,
                "pose_source": pose_source,
                "pose": {"x": pose_x, "y": pose_y, "yaw": pose_yaw},
                "published": bool(publisher_result.get("success")),
                "feedback": bool(feedback),
                "pose_confirmation": pose_confirmation,
                "accepted": accepted,
                "detail": detail,
            }
            attempts.append(attempt)
            if accepted:
                registry_warning = ""
                try:
                    _remember_native_slam_map(map_name, address)
                except OSError as exc:
                    registry_warning = str(exc)
                return {
                    "success": True,
                    "address": address,
                    "pose_source": pose_source,
                    "pose": {"x": pose_x, "y": pose_y, "yaw": pose_yaw},
                    "attempts": attempts,
                    "publisher_result": publisher_result,
                    "api_feedback": feedback,
                    "sensor_recovery": sensor_recovery,
                    "registry_warning": registry_warning,
                }

            missing_sensors = bool(
                feedback is None
                or "lack of lidar or imu data" in str(detail).lower()
            )
            if missing_sensors and sensor_recovery is None:
                sensor_recovery = await _recover_native_lidar_imu()
                attempt["sensor_recovery"] = sensor_recovery
                if sensor_recovery.get("success"):
                    continue
                # Expunem cauza recuperării, nu doar mesajul generic din 1804.
                attempt["detail"] = (
                    f"{detail}; recuperare LiDAR/IMU eșuată: "
                    f"{sensor_recovery.get('error') or 'motiv necunoscut'}"
                )
            break

    last_detail = attempts[-1]["detail"] if attempts else "nicio adresă PCD disponibilă"
    load_failed = bool(attempts) and all(
        "load pcd failed" in str(attempt.get("detail") or "").lower()
        for attempt in attempts
    )
    error = last_detail
    if load_failed:
        error = (
            "serviciul SLAM nu găsește/nu poate deschide copia nativă a acestei hărți. "
            "PCD-ul local rămâne valid pentru afișare și planificare, dar API 1804 nu îl încarcă "
            "automat. Aceasta nu dovedește că geometria hărții este defectă și nu impune "
            "recartografierea. Pentru 1102, copia trebuie restaurată printr-un mecanism Unitree "
            "acceptat în filesystemul controlerului SLAM; până atunci folosește numai executorul "
            "local explicit, cu poziția ancorată"
        )
    return {
        "success": False,
        "address": None,
        "attempts": attempts,
        "error": error,
        "sensor_recovery": sensor_recovery,
        "native_copy_missing": load_failed,
        "local_pcd_preserved": bool(_resolve_map_path(map_name)),
    }


@app.post("/api/slam/relocalize")
async def relocalize(body: dict = Body({})):
    """Secvența completă de repoziționare, cu verificare la fiecare pas.

    API 1804 încarcă harta și inițializează poziția în aceeași comandă.
    Un al doilea 1804 trimis în timpul inițializării poate fi ignorat de robot.
    """
    map_name = body.get("map_name") or loaded_map_path
    if not map_name:
        return {"success": False, "error": "Nicio hartă specificată sau încărcată"}

    x = float(body.get("x", 0.0))
    y = float(body.get("y", 0.0))
    yaw = float(body.get("yaw", 0.0))
    local_only = bool(body.get("local_only", False))
    force_native = bool(body.get("force_native", False))
    automatic_local_override = False
    if not local_only and not force_native:
        # Compatibilitate sigură cu paginile rămase în cache: versiunile vechi
        # trimiteau `local_only=false` chiar dacă selectorul restaurat de browser
        # rămânea pe 1102. Dacă telemetria Unitree spune fără echivoc `not init`
        # și harta aleasă este deja indexată local, nu mai repetăm un 1804 care
        # nu poate reuși. Schimbarea rămâne explicită în răspuns.
        native_controller = str(
            slam_runtime_info.get("controller") or ""
        ).strip().lower()
        local_status = local_lidar_localizer.status()
        resolved_map = _resolve_map_path(map_name)
        configured_path = local_status.get("map_path")
        automatic_local_override = bool(
            native_controller == "not init"
            and resolved_map
            and local_status.get("map_configured")
            and configured_path
            and os.path.realpath(str(configured_path))
                == os.path.realpath(resolved_map)
        )
        local_only = automatic_local_override
    steps = []

    if local_only:
        local_status = local_lidar_localizer.status()
        resolved_map = _resolve_map_path(map_name)
        if (not resolved_map or not local_status.get("map_configured")
                or os.path.realpath(str(local_status.get("map_path") or ""))
                != os.path.realpath(resolved_map)):
            return {
                "success": False,
                "error": "Localizatorul LiDAR nu este indexat pentru harta selectată. Reîncarcă harta.",
                "local_localization": local_status,
            }
        # Oprim întâi vechea fereastră de localizare, apoi instalăm ancora.
        # Astfel niciun cadru concurent nu este asociat cu poziția precedentă.
        _set_slam_mode("idle")
        local_lidar_localizer.reset({"x": x, "y": y, "yaw": yaw})
        map_state["local_localization"] = local_lidar_localizer.status()
        ros2_ok = bool(
            node_instance
            and hasattr(node_instance, "publish_initial_pose")
            and node_instance.publish_initial_pose(x, y, yaw)
        )
        if not ros2_ok:
            _set_slam_mode("idle")
            return {
                "success": False,
                "error": "Nu am putut publica ancora locală /initialpose.",
                "local_localization": local_lidar_localizer.status(),
            }
        _set_slam_mode("localization")
        lidar_wait_started = time.monotonic()
        for _ in range(32):
            await asyncio.sleep(0.125)
            local_status = local_lidar_localizer.status()
            if local_status.get("ready"):
                break
        converged = bool(local_status.get("ready"))
        if not converged:
            last_frame_at = float(
                getattr(node_instance, "last_raw_lidar_frame_time", 0.0) or 0.0
            )
            raw_points = int(
                getattr(node_instance, "last_raw_lidar_frame_points", 0) or 0
            )
            obstacle_points = int(
                getattr(node_instance, "last_raw_lidar_obstacle_points", 0) or 0
            )
            if last_frame_at < lidar_wait_started:
                local_lidar_localizer.report_input_error(
                    "nu sosesc cadre pe /utlidar/cloud_livox_mid360; "
                    "driverul Mid360 nu publică"
                )
            elif obstacle_points < 35:
                local_lidar_localizer.report_input_error(
                    f"cadrul LiDAR are {raw_points} puncte, dar numai "
                    f"{obstacle_points} puncte-obstacol utile pentru ICP"
                )
            local_status = local_lidar_localizer.status()
        failure_reason = str(local_status.get("error") or "").strip()
        refined_pose = dict(local_status.get("pose") or {})
        refinement = (
            {
                "ok": True,
                "x": float(refined_pose["x"]),
                "y": float(refined_pose["y"]),
                "yaw": float(refined_pose["yaw"]),
                "score": float(local_status.get("score") or 0.0),
                "inliers": int(local_status.get("inliers") or 0),
                "inlier_ratio": float(local_status.get("inlier_ratio") or 0.0),
            }
            if converged and all(key in refined_pose for key in ("x", "y", "yaw"))
            else None
        )
        return {
            "success": converged,
            "steps": [{
                "step": "local_lidar_icp",
                "ok": converged,
                "detail": (
                    f"inliers={local_status.get('inliers', 0)} "
                    f"ratio={float(local_status.get('inlier_ratio') or 0.0):.2f} "
                    f"score={local_status.get('score')}"
                    if converged else local_status.get("error")
                ),
            }],
            "output": (
                "Localizare LiDAR locală convergentă pe harta PCD; Nav2 poate folosi TF-ul map→odom."
                if converged else ""
            ),
            "error": (
                "Localizarea LiDAR locală nu a convergat în 4s: "
                f"{failure_reason or 'potrivire ICP insuficientă'}. "
                "Verifică poziția/orientarea marcată și geometria vizibilă."
                if not converged else ""
            ),
            "native_localization_skipped": True,
            "display_pose_published": True,
            "local_controller_pose_ready": converged,
            "refinement": refinement,
            "local_localization": local_status,
            "recommended_executor": "local_velocity",
            "executor_override": ({
                "requested": "native_1102",
                "selected": "local_velocity",
                "reason": (
                    "controllerul SLAM Unitree este «not init»; 1804 a fost omis, "
                    "iar ancora a fost verificată prin ICP LiDAR local"
                ),
            } if automatic_local_override else None),
        }

    # O poză pos_info veche nu poate confirma noua inițializare.
    global slam_pose_info
    slam_pose_info = {
        "received_at": 0.0, "current_pose": {}, "map_address": "", "pcd_name": ""
    }
    native_result = await _initialize_native_pose_1804(map_name, x, y, yaw)
    accepted_pose = dict(native_result.get("pose") or {})
    if native_result.get("success") and all(
        key in accepted_pose for key in ("x", "y", "yaw")
    ):
        x = float(accepted_pose["x"])
        y = float(accepted_pose["y"])
        yaw = float(accepted_pose["yaw"])
    ros2_ok = False
    if node_instance and hasattr(node_instance, "publish_initial_pose"):
        ros2_ok = node_instance.publish_initial_pose(x, y, yaw)
    pose_ok = bool(native_result.get("success"))
    attempt_text = "; ".join(
        f"{attempt['address']} -> {attempt['detail']}"
        for attempt in native_result.get("attempts", [])
    )
    detail = attempt_text or native_result.get("error") or ""
    steps.append({"step": "initialize_pose_1804", "ok": pose_ok,
                  "detail": f"/initialpose={ros2_ok}; {detail}".strip()})
    if not pose_ok:
        # 1804 și /initialpose sunt două rezultate distincte. Dacă ancora ROS
        # a fost publicată, pornim imediat și ICP-ul local cu exact aceeași
        # poziție aleasă de operator. Eșecul 1102 rămâne explicit, însă nu mai
        # obligăm operatorul să deseneze încă o dată aceeași ancoră.
        local_attempted = False
        local_ready = False
        local_status = local_lidar_localizer.status()
        if ros2_ok:
            resolved_map = _resolve_map_path(map_name)
            configured_path = local_status.get("map_path")
            local_attempted = bool(
                resolved_map
                and local_status.get("map_configured")
                and configured_path
                and os.path.realpath(str(configured_path))
                    == os.path.realpath(resolved_map)
            )
            refined = _refine_pose_with_live_scan({"x": x, "y": y, "yaw": yaw}, map_name)
            pose_to_use = (
                {"x": float(refined["x"]), "y": float(refined["y"]), "yaw": float(refined["yaw"])}
                if (refined and refined.get("ok")) else {"x": x, "y": y, "yaw": yaw}
            )
            if node_instance and hasattr(node_instance, "apply_local_lidar_pose"):
                node_instance.apply_local_lidar_pose(pose_to_use)
            if local_attempted:
                local_lidar_localizer.reset(pose_to_use)
            _set_slam_mode("localization")
            if local_attempted:
                for _ in range(32):
                    await asyncio.sleep(0.125)
                    local_status = local_lidar_localizer.status()
                    if local_status.get("ready"):
                        break
                local_ready = bool(local_status.get("ready"))
                map_state["local_localization"] = dict(local_status)
                steps.append({
                    "step": "local_lidar_icp_fallback",
                    "ok": local_ready,
                    "detail": (
                        f"inliers={local_status.get('inliers', 0)} "
                        f"ratio={float(local_status.get('inlier_ratio') or 0.0):.2f} "
                        f"score={local_status.get('score')}"
                        if local_ready else local_status.get("error")
                    ),
                })
        reason = native_result.get("error") or "1804 respins"
        if local_ready:
            local_hint = (
                "1102 rămâne indisponibil, dar localizarea LiDAR–PCD a convergat. "
                "Interfața trece vizibil la «Controller local Nav2»; Nav2 poate "
                "planifica și executorul local poate urmări ruta după preflight."
            )
        elif local_attempted:
            local_hint = (
                "Ancora locală a fost instalată, dar ICP nu a convergat încă. "
                "Interfața trece la controllerul local; ajustează poziția/orientarea "
                "pe hartă și seteaz-o din nou."
            )
        else:
            local_hint = (
                "Ancora de afișare a fost publicată, dar harta nu este încă indexată "
                "pentru ICP local. Reîncarcă harta și folosește Controller local Nav2; "
                "API 1102 rămâne indisponibil."
            ) if ros2_ok else None
        return {
            "success": False,
            "steps": steps,
            "error": f"Inițializarea localizării native a eșuat: {reason}",
            "native_localization": native_result,
            "display_pose_published": ros2_ok,
            "local_controller_pose_ready": local_ready,
            "local_controller_anchor_ready": ros2_ok,
            "local_controller_fallback_attempted": local_attempted,
            "local_localization": local_status,
            "recommended_executor": "local_velocity" if ros2_ok else None,
            "local_controller_hint": local_hint,
        }

    _set_slam_mode("localization")

    # Confirmăm convergența: fără odometrie de localizare, repoziționarea nu a prins.
    refinement_result = None
    for _ in range(40):
        await asyncio.sleep(0.5)
        if _native_localization_fresh():
            break
    converged = _native_localization_fresh()
    if converged:
        refinement_result = _refine_pose_with_live_scan({"x": x, "y": y, "yaw": yaw}, map_name)
        if refinement_result and refinement_result.get("ok"):
            map_state["pose"] = {"x": round(refinement_result["x"], 3), "y": round(refinement_result["y"], 3), "yaw": refinement_result["yaw"]}
            map_state["last_refinement"] = {
                "x": map_state["pose"]["x"],
                "y": map_state["pose"]["y"],
                "yaw": map_state["pose"]["yaw"],
                "score": refinement_result.get("score"),
            }
            await broadcast({"type": "pose", **map_state["pose"]})
            await broadcast({"type": "pose_refined", "pose": map_state["pose"], "score": refinement_result.get("score")})
    steps.append({"step": "verify_localization", "ok": converged,
                  "detail": getattr(node_instance, "active_odom_source", None) or "niciun topic de localizare activ"})
    if refinement_result:
        steps.append({"step": "refine_pose", "ok": bool(refinement_result.get("ok")),
                      "detail": f"score={refinement_result.get('score', float('inf')):.3f}"})

    return {
        "success": converged,
        "steps": steps,
        "output": f"Repoziționat la ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f}°)" if converged else "",
        "error": "" if converged else "Localizarea nu a publicat odometrie în 10s după setarea pozei",
        "refinement": refinement_result,
        "native_localization": native_result,
        "display_pose_published": ros2_ok,
    }


@app.post("/api/slam/start_relocation")
async def start_relocation():
    """Repornește localizarea pe harta deja încărcată."""
    result = await asyncio.to_thread(slam_client.start_relocation)
    if result.get("success"): _set_slam_mode("localization")
    return result

@app.post("/api/slam/set_initial_pose")
async def set_initial_pose_endpoint(body: dict = Body({})):
    """Alias compatibil care păstrează verificarea reală a feedback-ului 1804."""
    return await relocalize(body)

@app.get("/api/slam/list_maps")
async def list_robot_maps():
    """Obține lista fișierelor de hartă de pe robot."""
    maps = await asyncio.to_thread(slam_client.list_maps)
    import os
    # Eliminăm duplicatele păstrând ordinea
    unique_maps = []
    seen = set()
    for m in maps:
        if m not in seen:
            seen.add(m)
            unique_maps.append(m)
    short_names = [{"path": m, "name": os.path.basename(m)} for m in unique_maps]
    return {"success": True, "maps": short_names}

@app.get("/api/robot/status")
async def robot_status():
    """Diagnostic rapid: sdk_available=False inseamna ca SDK-ul (G1LocoClient)
    nu s-a instantiat la pornirea serverului - verifica log-ul consolei
    (unde ruleaza uvicorn) pentru eroarea exacta de import/Init()."""
    return await asyncio.to_thread(sport_client.get_status)

def _fsm_confirmation_error(body: dict) -> Optional[dict]:
    """Blochează orice schimbare FSM fără confirmarea textuală exactă `ok`."""
    if not isinstance(body, dict) or body.get("confirmation") != "ok":
        return {
            "success": False,
            "confirmation_required": True,
            "error": "Comanda nu a fost trimisă. Scrie exact ok pentru a confirma această schimbare de stare.",
        }
    return None


@app.post("/api/robot/wake_up")
async def wake_up_robot(body: dict = Body({})):
    """Rulează secvența Damp -> Ready -> Run numai cu confirmare textuală."""
    if error := _fsm_confirmation_error(body):
        return error
    return await asyncio.to_thread(sport_client.wake_up_sequence)


@app.post("/api/robot/damp")
async def damp_robot(body: dict = Body({})):
    if error := _fsm_confirmation_error(body):
        return error
    return await asyncio.to_thread(sport_client.damp)


@app.post("/api/robot/zero_torque")
async def zero_torque_robot(body: dict = Body({})):
    if error := _fsm_confirmation_error(body):
        return error
    return await asyncio.to_thread(sport_client.zero_torque)


@app.post("/api/robot/stand_up")
async def stand_up_robot(body: dict = Body({})):
    if error := _fsm_confirmation_error(body):
        return error
    return await asyncio.to_thread(sport_client.stand_up)


@app.post("/api/robot/start")
async def start_robot(body: dict = Body({})):
    if error := _fsm_confirmation_error(body):
        return error
    return await asyncio.to_thread(sport_client.start_locomotion)


@app.post("/api/robot/set_fsm/{fsm_id}")
async def set_robot_fsm(fsm_id: int, body: dict = Body({})):
    if error := _fsm_confirmation_error(body):
        return error
    return await asyncio.to_thread(sport_client.set_fsm_id, fsm_id)

@app.post("/api/robot/navigate")
async def navigate_robot(x: float, y: float, yaw: float = 0.0):
    """Compatibilitate: folosește navigatorul autonom real, nu un topic fără subscriber."""
    return await start_navigation({"x": x, "y": y, "yaw": yaw})


def _get_autonomous_navigator() -> AutonomousNavigator:
    global autonomous_navigator
    if autonomous_navigator is None:
        autonomous_navigator = AutonomousNavigator(
            sport_client=sport_client,
            pose_provider=lambda: {
                **dict(map_state["pose"]),
                "source": map_state.get("pose_source"),
            },
            localization_ok=_localization_fresh,
            obstacle_guard=obstacle_guard,
            event_callback=_navigation_event,
        )
    return autonomous_navigator


def _native_pose() -> dict:
    """Poziția curentă în același frame map în care sunt țintele și PCD-ul."""
    live_pose = dict(slam_runtime_info.get("current_pose") or {})
    native_age = time.monotonic() - float(
        slam_runtime_info.get("pose_received_at", 0.0) or 0.0
    )
    # 1102 planifică în pose-ul său nativ. Cât pos_info este proaspăt, acesta
    # trebuie să fie și pose-ul executorului; alegerea relocation înaintea lui
    # amesteca două yaw-uri și producea rotații aparent aleatoare. Odometria
    # relocation este folosită numai ca punte în pauza cunoscută de pos_info.
    if _pose_xy(live_pose) and native_age < NATIVE_POS_INFO_MAX_AGE:
        pose = live_pose
    elif _relocation_odom_fresh() or (
        _native_active_pose_grace_fresh() and _anchored_pelvis_odom_fresh()
    ):
        pose = dict(map_state["pose"])
    else:
        pose = live_pose if _pose_xy(live_pose) else dict(map_state["pose"])
    return {
        "x": float(pose.get("x", 0.0)),
        "y": float(pose.get("y", 0.0)),
        "yaw": _pose_yaw(pose),
    }


def _native_waypoint_completed(
        target_x: float, target_y: float, dispatched_at: float,
) -> bool:
    """Corelează FINISHED cu waypoint-ul curent, nu cu o comandă veche."""
    completion = dict(slam_last_completion)
    if float(completion.get("received_at", 0.0) or 0.0) < float(dispatched_at):
        return False
    machine_state = str(completion.get("machine_state") or "").strip().lower()
    if not (
        completion.get("arrived")
        or any(token in machine_state for token in ("finished", "arrived", "reached"))
    ):
        return False
    position = _pose_xy(completion.get("current_pose"))
    if not position:
        return False
    # Firmware-ul 1102 declară frecvent FINISHED la 0,4–0,5 m de țintele
    # intermediare. Sub 0,60 m acceptăm confirmarea explicită; timestampul
    # împiedică folosirea FINISHED-ului rămas de la waypoint-ul precedent.
    return math.hypot(position[0] - target_x, position[1] - target_y) <= 0.60


async def _dispatch_native_waypoint(x: float, y: float, yaw: float, speed: float) -> dict:
    """Trimite segmentul A* exact ca v23, inclusiv sincronizarea FINISHED."""
    started_at = time.monotonic()
    command = {"x": x, "y": y, "yaw": yaw, "speed": speed}
    _append_navigation_flight_log({
        "type": "api_1102_dispatch", "state": "sending", "command": command,
    })

    def finish(result: dict) -> dict:
        _append_navigation_flight_log({
            "type": "api_1102_result",
            "state": "accepted" if result.get("success") else "rejected",
            "command": command,
            "latency_s": round(max(0.0, time.monotonic() - started_at), 4),
            "result": result,
        })
        return result

    publisher_results = []
    for attempt in range(2):
        sent_at = time.monotonic()
        publisher_result = await asyncio.to_thread(
            slam_client.pose_navigation, x, y, yaw, speed
        )
        publisher_results.append(publisher_result)
        if not publisher_result.get("success"):
            return finish({
                **publisher_result, "slam_info": dict(slam_runtime_info),
                "attempts": attempt + 1,
            })
        feedback = await _wait_slam_feedback(1102, sent_at)
        if feedback is not None:
            payload = feedback.get("payload") or {}
            rejected = (
                feedback.get("status_code", 0) != 0
                or payload.get("succeed") is False
                or int(payload.get("errorCode", 0) or 0) != 0
            )
            if rejected:
                code = payload.get("errorCode", feedback.get("status_code"))
                return finish({
                    "success": False,
                    "error": payload.get("info") or
                             f"Waypoint-ul 1102 a fost respins (cod {code})",
                    "api_feedback": feedback, "attempts": attempt + 1,
                })
            return finish({
                "success": True, "api_feedback": feedback,
                "publisher_result": publisher_result,
                "publisher_results": publisher_results,
                "attempts": attempt + 1,
                "recovered_from_late_completion": attempt > 0,
            })

        completion = dict(slam_last_completion)
        completion_is_new = float(completion.get("received_at", 0.0)) >= sent_at
        completion_pose = _pose_xy(completion.get("current_pose"))
        target_distance = (
            math.hypot(completion_pose[0] - x, completion_pose[1] - y)
            if completion_pose else math.inf
        )
        if completion_is_new and target_distance <= NAV_GOAL_TOLERANCE_M:
            return finish({
                "success": True, "publisher_result": publisher_result,
                "publisher_results": publisher_results,
                "slam_completion": completion, "telemetry_confirmed": True,
                "attempts": attempt + 1,
            })
        if attempt == 0 and completion_is_new and completion_pose:
            _append_navigation_flight_log({
                "type": "api_1102_retry", "state": "late_previous_completion",
                "command": command,
                "target_distance_m": round(target_distance, 3),
                "slam_completion": completion,
            })
            continue
        return finish({
            "success": False,
            "error": "Robotul nu a răspuns la waypoint-ul 1102 în 4s"
                     + (" nici după retrimiterea sincronizată" if attempt else ""),
            "publisher_result": publisher_result,
            "publisher_results": publisher_results,
            "slam_info": dict(slam_runtime_info),
            "slam_completion": completion if completion_is_new else None,
            "attempts": attempt + 1,
        })
    return finish({
        "success": False,
        "error": "Waypoint-ul 1102 nu a putut fi sincronizat",
        "publisher_results": publisher_results,
    })


async def _pause_native_navigation() -> dict:
    sent_at = time.monotonic()
    publisher_result = await asyncio.to_thread(slam_client.pause_navigation)
    if not publisher_result.get("success"):
        fallback = await asyncio.to_thread(sport_client.stop)
        result = {
            **publisher_result,
            "success": bool(fallback.get("success")),
            "error": publisher_result.get("error") or
                     "Publicarea 1201 a eșuat; am aplicat STOP locomotion",
            "locomotion_fallback": fallback,
        }
    else:
        # Barieră redundantă lansată imediat, nu după timeout-ul feedback-ului.
        # 1201 oprește planificatorul, iar viteza zero oprește locomotion-ul.
        fallback = await asyncio.to_thread(sport_client.stop)
        feedback = await _wait_slam_feedback(1201, sent_at, timeout=1.25)
        payload = (feedback or {}).get("payload") or {}
        feedback_ok = bool(
            feedback
            and feedback.get("status_code", 0) == 0
            and payload.get("succeed", True) is not False
            and int(payload.get("errorCode", 0) or 0) == 0
        )
        telemetry_ok = bool(
            slam_runtime_info.get("paused") is True
            and float(slam_runtime_info.get("received_at", 0.0)) >= sent_at
        )
        if feedback_ok or telemetry_ok:
            result = {
                "success": True, "publisher_result": publisher_result,
                "api_feedback": feedback, "telemetry_confirmed": telemetry_ok,
                "locomotion_fallback": fallback,
            }
        else:
            result = {
                "success": bool(fallback.get("success")),
                "error": "API 1201 nu a fost confirmat; am aplicat STOP locomotion"
                         if fallback.get("success") else
                         "Nici API 1201, nici STOP locomotion nu au fost confirmate",
                "publisher_result": publisher_result,
                "api_feedback": feedback,
                "locomotion_fallback": fallback,
            }
    _append_navigation_flight_log({
        "type": "api_1201_pause",
        "command": {"api_id": 1201, "parameter": {}},
        "result": result,
    })
    return result


async def _resume_native_navigation() -> dict:
    sent_at = time.monotonic()
    publisher_result = await asyncio.to_thread(slam_client.resume_navigation)
    if not publisher_result.get("success"):
        result = publisher_result
    else:
        feedback = await _wait_slam_feedback(1202, sent_at, timeout=1.25)
        payload = (feedback or {}).get("payload") or {}
        feedback_ok = bool(
            feedback
            and feedback.get("status_code", 0) == 0
            and payload.get("succeed", True) is not False
            and int(payload.get("errorCode", 0) or 0) == 0
        )
        telemetry_ok = bool(
            slam_runtime_info.get("paused") is False
            and float(slam_runtime_info.get("received_at", 0.0)) >= sent_at
        )
        result = {
            "success": bool(feedback_ok or telemetry_ok),
            "error": "API 1202 nu a fost confirmat; robotul rămâne oprit"
                     if not (feedback_ok or telemetry_ok) else "",
            "publisher_result": publisher_result,
            "api_feedback": feedback,
            "telemetry_confirmed": telemetry_ok,
        }
    _append_navigation_flight_log({"type": "api_1202_resume", "result": result})
    return result


async def _command_lateral_velocity(vy: float) -> dict:
    """Pas lateral scurt prin controllerul locomotion G1, în afara lui 1102."""
    # Comanda rămâne activă suficient cât monitorizarea SLAM să primească o
    # poziție nouă. Navigatorul o reînnoiește înainte să expire și trimite STOP
    # imediat ce pasul util a fost confirmat.
    result = await asyncio.to_thread(
        sport_client.move_to, 0.0, float(vy), 0.0, 0.85
    )
    _append_navigation_flight_log({
        "type": "lateral_velocity", "vy": float(vy), "result": result,
    })
    return result


async def _stop_direct_locomotion() -> dict:
    return await asyncio.to_thread(sport_client.stop)


def _public_navigation_queue() -> list:
    return [
        {key: value for key, value in item.items() if key != "created_at"}
        for item in navigation_goal_queue
    ]


async def _activate_next_queued_goal() -> None:
    global navigation_goal_queue, queue_transition_scheduled
    try:
        await asyncio.sleep(0.20)
        if not navigation_goal_queue:
            return
        item = navigation_goal_queue[0]
        result = await _preview_navigation(dict(item), allow_active=True)
        if result.get("success"):
            # Adăugarea în coadă este confirmarea explicită a operatorului.
            # Ruta se recalculează din poziția reală după taskul anterior.
            await broadcast({"type": "nav_scheduled_route", **result})
            start_result = await start_navigation({
                "preview_id": result["preview_id"]
            })
            if start_result.get("success"):
                navigation_goal_queue.pop(0)
                await broadcast({
                    "type": "nav_task_started", "task": {
                        key: value for key, value in item.items()
                        if key != "created_at"
                    },
                    "result": start_result,
                })
                await broadcast({
                    "type": "nav_queue", "queue": _public_navigation_queue()
                })
            else:
                await broadcast({
                    "type": "nav_queue", "queue": _public_navigation_queue(),
                    "error": f"Taskul programat nu a pornit: {start_result.get('error', 'eroare')}",
                })
        else:
            await broadcast({
                "type": "nav_queue", "queue": _public_navigation_queue(),
                "error": f"Următorul punct nu poate fi planificat: {result.get('error', 'eroare')}",
            })
    finally:
        queue_transition_scheduled = False


def _append_navigation_flight_log(event: dict) -> None:
    """Persistă deciziile de navigare împreună cu datele care le-au provocat."""
    if navigation_flight_log_path is None:
        return
    logged_event = dict(event)
    dynamic_points = logged_event.get("dynamic_costmap")
    if isinstance(dynamic_points, list) and len(dynamic_points) > 80:
        # Costmapul complet rămâne în UI, dar scrierea sincronă a sutelor de
        # celule la fiecare stare întârzia bucla de decizie.
        logged_event["dynamic_costmap_count"] = len(dynamic_points)
        logged_event["dynamic_costmap"] = dynamic_points[:80]
    sensor_status = obstacle_guard.sensor_status()
    now_mono = time.monotonic()
    now_wall = time.time()
    native_pose_received_at = float(
        slam_runtime_info.get("pose_received_at", 0.0) or 0.0
    )
    native_telemetry_received_at = float(
        slam_runtime_info.get("received_at", 0.0) or 0.0
    )
    map_pose_updated_at = float(map_state.get("pose_updated_at", 0.0) or 0.0)
    slam_snapshot = {
        key: slam_runtime_info.get(key)
        for key in (
            "message_type", "error_code", "info", "controller", "planner_open",
            "paused", "machine_state", "arrived", "current_pose", "map_address",
        )
    }
    payload = {
        "recorded_at": time.time(),
        "monotonic_at": time.monotonic(),
        "event": logged_event,
        "obstacle_sensors": sensor_status,
        # Câmpuri explicite, stabile, pentru analiza fiecărui STOP fără a
        # depinde de forma internă a snapshot-ului de senzori.
        "camera_zones": sensor_status.get("camera_zones", {}),
        "camera_distances": sensor_status.get("camera_distances", {}),
        "lidar_zones": sensor_status.get("lidar_zones", []),
        "lidar_center_distance": sensor_status.get("lidar_center_distance"),
        "slam": slam_snapshot,
        "pose_source": map_state.get("pose_source"),
        "localization_diagnostic": {
            "native_pose_age_s": (
                round(now_mono - native_pose_received_at, 3)
                if native_pose_received_at else None
            ),
            "native_telemetry_age_s": (
                round(now_mono - native_telemetry_received_at, 3)
                if native_telemetry_received_at else None
            ),
            "map_pose_age_s": (
                round(now_wall - map_pose_updated_at, 3)
                if map_pose_updated_at else None
            ),
            "same_map": _native_runtime_same_map(),
            "relocation_bridge": _relocation_odom_fresh(),
            "anchored_pelvis_bridge": _anchored_pelvis_odom_fresh(),
            "accepted": _native_localization_fresh(),
        },
        "fsm": getattr(sport_client, "_last_fsm_id", None),
    }
    try:
        with navigation_flight_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    except OSError:
        # Jurnalizarea nu trebuie să întrerupă bariera de siguranță sau STOP-ul.
        return


def _start_navigation_flight_log(preview: dict, pose: dict) -> Optional[str]:
    global navigation_flight_log_path
    try:
        NAVIGATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        navigation_flight_log_path = NAVIGATION_LOG_DIR / (
            f"nav_{stamp}_{secrets.token_hex(3)}.jsonl"
        )
        _append_navigation_flight_log({
            "type": "run_start",
            "state": "confirmed",
            "map_path": preview.get("map_path"),
            "start_pose": pose,
            "goal": preview.get("goal"),
            "path": preview.get("path"),
            "speed": preview.get("speed"),
            "timeout": preview.get("timeout"),
            "driver": preview.get("driver", "legacy"),
            "executor": preview.get("executor", "native_1102"),
            "nav2_plan": preview.get("nav2_plan"),
            "costmap_settings": preview.get("costmap_settings"),
        })
        return str(navigation_flight_log_path)
    except OSError:
        navigation_flight_log_path = None
        return None


async def _navigation_event(event: dict) -> None:
    global queue_transition_scheduled, navigation_post_action_task
    global active_navigation_post_action, active_navigation_driver
    global active_navigation_executor
    _append_navigation_flight_log(event)
    await broadcast(event)
    state = event.get("state")
    if state == "arrived":
        gesture = active_navigation_post_action or "none"
        active_navigation_post_action = None
        active_navigation_driver = None
        active_navigation_executor = None
        if navigation_post_action_task is None or navigation_post_action_task.done():
            navigation_post_action_task = asyncio.create_task(
                _finish_navigation_task(gesture)
            )
    elif state in {"failed", "cancelled", "paused"}:
        active_navigation_post_action = None
        active_navigation_driver = None
        active_navigation_executor = None


async def _finish_navigation_task(gesture: str) -> None:
    """Execută gestul numai după sosire, apoi avansează coada FIFO."""
    global queue_transition_scheduled
    try:
        if gesture != "none":
            definition = NAVIGATION_GESTURES[gesture]
            await broadcast({
                "type": "nav_task_action", "state": "running",
                "gesture": gesture, "label": definition["label"],
            })
            _append_navigation_flight_log({
                "type": "gesture_start", "gesture": gesture,
                "label": definition["label"],
            })
            result = await asyncio.to_thread(sport_client.execute_gesture, gesture)
            if result.get("success"):
                await asyncio.sleep(float(definition["duration"]))
                release = await asyncio.to_thread(sport_client.release_arms)
                state = "done" if release.get("success") else "failed"
                error = release.get("error", "")
            else:
                release = None
                state = "failed"
                error = result.get("error", "Gestul a eșuat")
            event = {
                "type": "nav_task_action", "state": state,
                "gesture": gesture, "label": definition["label"],
                "result": result, "release": release, "error": error,
            }
            _append_navigation_flight_log({**event, "type": "gesture_result"})
            await broadcast(event)
    except asyncio.CancelledError:
        await asyncio.to_thread(sport_client.release_arms)
        await broadcast({
            "type": "nav_task_action", "state": "cancelled",
            "gesture": gesture, "error": "Gest oprit de operator",
        })
        raise
    finally:
        if navigation_goal_queue and not queue_transition_scheduled:
            queue_transition_scheduled = True
            asyncio.create_task(_activate_next_queued_goal())


def _get_native_waypoint_navigator() -> NativeWaypointNavigator:
    global native_waypoint_navigator
    if native_waypoint_navigator is None:
        native_waypoint_navigator = NativeWaypointNavigator(
            pose_provider=_native_pose,
            localization_ok=_native_localization_fresh,
            obstacle_guard=obstacle_guard,
            send_waypoint=_dispatch_native_waypoint,
            pause_navigation=_pause_native_navigation,
            resume_navigation=_resume_native_navigation,
            event_callback=_navigation_event,
            navigation_paused=lambda: bool(slam_runtime_info.get("paused", False)),
            # Disponibile permanent pentru unica repoziționare laterală de la
            # plecare. Recuperarea laterală repetitivă rămâne opt-in separat.
            lateral_velocity=_command_lateral_velocity,
            stop_locomotion=_stop_direct_locomotion,
            enable_stagnation_lateral_recovery=(
                os.environ.get("G1_ENABLE_LATERAL_RECOVERY") == "1"
            ),
            waypoint_completed=_native_waypoint_completed,
        )
    return native_waypoint_navigator


def _get_nav2_observer() -> Optional[Nav2ObserverPublisher]:
    node = node_instance
    return getattr(node, "nav2_observer", None) if node is not None else None


def _parse_navigation_driver(body: dict) -> str:
    """Normalizează modul ales fără să schimbe implicit comportamentul v22."""
    value = str(body.get("driver") or "legacy").strip().lower()
    aliases = {
        "legacy": "legacy", "v22": "legacy", "astar": "legacy",
        "astar_waypoints": "legacy", "native": "legacy",
        "nav2": "nav2", "gridbased": "nav2", "nav2_gridbased": "nav2",
    }
    if value not in aliases:
        raise ValueError("Modul de navigare trebuie să fie legacy sau nav2")
    return aliases[value]


def _parse_navigation_executor(body: dict) -> str:
    """Executorul este separat de planner și nu face fallback implicit."""
    value = str(body.get("executor") or "native_1102").strip().lower()
    aliases = {
        "native": "native_1102", "native_1102": "native_1102", "1102": "native_1102",
        "local": "local_velocity", "local_safe": "local_velocity",
        "local_velocity": "local_velocity",
    }
    if value not in aliases:
        raise ValueError("Executorul trebuie să fie native_1102 sau local_velocity")
    return aliases[value]


def _resolve_navigation_executor_for_runtime(requested: str) -> tuple:
    """Păstrează executorul cerut; profilul A* nu schimbă ascuns controlerul.

    În profilul de deadline interfața cere exclusiv 1102. Dacă localizarea
    nativă lipsește, preflight-ul/preview-ul raportează problema și robotul nu
    pornește; nu mai comutăm automat pe controllerul experimental local.
    """
    return requested, None


def _navigator_is_active(navigator) -> bool:
    task = getattr(navigator, "task", None)
    return bool(task and not task.done())


def _any_navigation_active() -> bool:
    return bool(
        (native_waypoint_navigator is not None and _navigator_is_active(native_waypoint_navigator))
        or (autonomous_navigator is not None and _navigator_is_active(autonomous_navigator))
    )


async def _request_nav2_plan(
        planner: PCDGridPlanner, x: float, y: float, yaw: float,
        costmap_settings: dict) -> tuple:
    """Publică harta filtrată și cere numai planul global de la Nav2."""
    observer = _get_nav2_observer()
    if observer is None:
        raise RuntimeError(
            "Adaptorul Nav2 nu este disponibil. Verifică rclpy/nav2_msgs și repornește dashboardul."
        )
    # VoxelLayer vede z în frame-ul base_link (pelvis), în timp ce valorile
    # configurate în UI sunt înălțimi față de podea. Convertim folosind podeaua
    # observată în ultimul cadru complet Mid360.
    ground_base_z = float(
        getattr(node_instance, "last_raw_lidar_ground_base_z", -0.75)
    )
    await asyncio.to_thread(
        observer.set_runtime_parameters,
        {
            "robot_radius": float(costmap_settings["robot_radius"]),
            "resolution": float(costmap_settings["resolution"]),
            "inflation_layer.inflation_radius": float(
                max(costmap_settings["robot_radius"], NAV2_MIN_INFLATION_RADIUS)
            ),
            "voxel_layer.enabled": True,
            "voxel_layer.lidar.obstacle_min_range": 0.30,
            "voxel_layer.lidar.min_obstacle_height": ground_base_z + float(
                costmap_settings["obstacle_min_z"]
            ),
            "voxel_layer.lidar.max_obstacle_height": ground_base_z + float(
                costmap_settings["obstacle_max_z"]
            ),
            # PCD-ul are podeaua observată fragmentată în insule. A* tratează
            # deja necunoscutul ca traversabil cu penalizare; Navfn trebuie să
            # poată lega aceleași insule, păstrând obstacolele/inflația letale.
            "GridBased.allow_unknown": True,
        },
    )
    await asyncio.to_thread(observer.publish_map, planner)
    published_at = observer.map_published_at
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if (observer.path_client.server_is_ready()
                and observer.nav2_costmap_received_at >= published_at):
            break
        await asyncio.sleep(0.10)
    if not observer.path_client.server_is_ready():
        raise RuntimeError(
            "planner_server Nav2 nu rulează. Pornește v24 cu bash start_dashboard.sh."
        )
    if observer.nav2_costmap_received_at < published_at:
        raise RuntimeError(
            "Nav2 nu a reconstruit costmapul după publicarea hărții; verifică planner_server."
        )
    nav2_plan = await asyncio.to_thread(observer.compute_path, x, y, yaw, 8.0)
    execution_path = await asyncio.to_thread(
        observer.simplify_execution_path, nav2_plan["path"], 80
    )
    if len(execution_path) < 2:
        raise RuntimeError("Nav2 a întors un traseu fără segmente executabile")
    return nav2_plan, execution_path


async def _local_navigation_replan(
        planner: PCDGridPlanner, pose: dict, x: float, y: float, yaw: float,
        driver: str, costmap_settings: dict) -> dict:
    """Replanifică executorul local cu plannerul ales inițial de operator."""
    if driver == "nav2":
        nav2_plan, path = await _request_nav2_plan(
            planner, x, y, yaw, costmap_settings
        )
        return {
            "path": path,
            "driver": "nav2",
            "nav2_plan": nav2_plan,
            "path_pattern": {
                "type": "nav2_gridbased_replan",
                "planner_id": nav2_plan.get("planner_id", "GridBased"),
                "raw_poses": nav2_plan.get("poses", 0),
                "execution_segments": max(0, len(path) - 1),
                "planning_time_s": nav2_plan.get("planning_time_s", 0.0),
            },
        }
    path = await asyncio.to_thread(
        planner.plan, (pose["x"], pose["y"]), (x, y)
    )
    return {
        "path": path, "driver": "legacy",
        "path_pattern": {
            "type": "astar_local_replan",
            "execution_segments": max(0, len(path) - 1),
        },
    }


def _build_static_costmap_preview(
        map_path: str, resolution: float = 0.10, robot_radius: float = 0.25,
        min_obstacle_points: int = 3, obstacle_min_z: float = 0.15,
        obstacle_max_z: float = 1.85, level_to_floor: bool = True,
        floor_tolerance: float = 0.08, comfort_radius: float = 0.50,
        clearance_weight: float = 3.50, planner_mode: str = "legacy") -> dict:
    """Rasterizează costmapul A*: obstacol brut, inflație configurabilă și liber."""
    if obstacle_min_z >= obstacle_max_z:
        raise ValueError("Z minim trebuie să fie mai mic decât Z maxim")
    planner = PCDGridPlanner(
        resolution=resolution,
        robot_radius=robot_radius,
        min_obstacle_points=min_obstacle_points,
        comfort_radius=comfort_radius,
        clearance_weight=clearance_weight,
    )
    planner.load(
        map_path,
        obstacle_min_z=obstacle_min_z,
        obstacle_max_z=obstacle_max_z,
        level_to_floor=level_to_floor,
        floor_tolerance=floor_tolerance,
    )
    if not planner.bounds:
        raise ValueError("Harta nu are limite valide")
    min_x, max_x, min_y, max_y = planner.bounds
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    pixels = width * height
    if width <= 0 or height <= 0:
        raise ValueError("Dimensiunile costmapului sunt invalide")
    if width > 4096 or height > 4096 or pixels > 16_000_000:
        raise ValueError(
            f"Costmap prea mare pentru vizualizare ({width}×{height} celule); "
            "PCD-ul conține probabil puncte izolate foarte îndepărtate"
        )

    # OpenCV folosește BGR. Fiecare pixel reprezintă o celulă la rezoluția aleasă.
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = (38, 38, 38)       # gri: spațiu necunoscut, puternic penalizat

    def paint(cells, bgr):
        for cell_x, cell_y in cells:
            if min_x <= cell_x <= max_x and min_y <= cell_y <= max_y:
                image[max_y - cell_y, cell_x - min_x] = bgr

    inflated_only = planner.static_occupied - planner.raw_static_occupied
    planner_mode = "nav2" if str(planner_mode).lower() == "nav2" else "legacy"
    comfort_only = (
        set(planner.clearance_cost) - planner.static_occupied
        if planner_mode == "legacy" else set()
    )
    paint(planner.known_free,(43,53,18))    # verde: podea observată
    paint(comfort_only,(32,180,210))       # galben: traversabil, dar penalizat
    paint(inflated_only, (11, 158, 245))   # portocaliu: raza blocată aleasă
    paint(planner.raw_static_occupied, (68, 68, 239))  # roșu: puncte-obstacol PCD
    encoded_ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not encoded_ok:
        raise ValueError("Imaginea costmap nu a putut fi comprimată")
    occupied_inside = sum(
        1 for x, y in planner.static_occupied
        if min_x <= x <= max_x and min_y <= y <= max_y
    )
    known_inside = sum(
        1 for x, y in planner.known_free
        if min_x <= x <= max_x and min_y <= y <= max_y
        and (x, y) not in planner.static_occupied
    )
    # A* stabil v23 nu separă podeaua observată de completările ulterioare;
    # pentru preview raportăm `known_free` ca podea, fără să schimbăm plannerul.
    observed_floor = getattr(planner, "observed_floor", planner.known_free)
    inferred_free = getattr(planner, "inferred_free", set())
    observed_floor_inside = sum(
        1 for x, y in observed_floor
        if min_x <= x <= max_x and min_y <= y <= max_y
        and (x, y) not in planner.static_occupied
    )
    inferred_free_inside = sum(
        1 for x, y in inferred_free
        if min_x <= x <= max_x and min_y <= y <= max_y
        and (x, y) not in planner.static_occupied
    )
    return {
        "_planner": planner,
        "success": True,
        "map_path": map_path,
        "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "width": width,
        "height": height,
        "resolution": planner.resolution,
        "clearance_radius": planner.robot_radius,
        "nav2_inflation_radius": max(
            planner.robot_radius, NAV2_MIN_INFLATION_RADIUS
        ),
        "comfort_radius": planner.comfort_radius,
        "clearance_weight": planner.clearance_weight,
        "planner_mode": planner_mode,
        "bounds_filter": getattr(planner, "bounds_filter", {
            "applied": False,
            "active_bounds": planner.bounds,
            "discarded_points": 0,
        }),
        "bounds": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
        "world_bounds": {
            "min_x": round(min_x * planner.resolution, 3),
            "max_x": round(max_x * planner.resolution, 3),
            "min_y": round(min_y * planner.resolution, 3),
            "max_y": round(max_y * planner.resolution, 3),
        },
        "counts": {
            "raw_obstacle": len(planner.raw_static_occupied),
            "inflated_buffer": len(inflated_only),
            "comfort_penalty": len(comfort_only),
            "occupied_total": occupied_inside,
            "known_free": known_inside,
            "observed_floor": observed_floor_inside,
            "inferred_free": inferred_free_inside,
            "unknown_penalized": max(0, pixels - occupied_inside - known_inside),
            "considered_free": max(0, pixels - occupied_inside),
        },
        "planner_filter": {
            "min_z": obstacle_min_z,
            "max_z": obstacle_max_z,
            "min_points_per_cell": min_obstacle_points,
            "level_to_floor": level_to_floor,
            "floor_tolerance": floor_tolerance,
            "comfort_radius": planner.comfort_radius,
            "clearance_weight": planner.clearance_weight,
        },
        "floor_plane": planner.floor_plane,
        "note": (
            "Nav2 permite griul necunoscut; numai roșul și portocaliul sunt blocate."
            if planner_mode == "nav2" else
            "A* penalizează puternic griul necunoscut; galbenul este traversabil, dar costisitor."
        ),
    }


@app.get("/api/nav/costmap")
async def get_navigation_costmap(
        resolution: float = Query(0.10, ge=0.05, le=0.30),
        min_points: int = Query(4, ge=1, le=20),
        min_z: float = Query(0.15, ge=-5.0, le=5.0),
        max_z: float = Query(1.85, ge=-5.0, le=5.0),
        radius: float = Query(0.20, ge=0.20, le=0.80),
        level_floor: bool = Query(True),
        floor_tolerance: float = Query(0.08, ge=0.03, le=0.25),
        comfort_radius: float = Query(0.50, ge=0.10, le=1.00),
        clearance_weight: float = Query(3.50, ge=3.5, le=10.0),
        driver: str = Query("legacy")):
    map_path = _resolve_map_path(loaded_map_path)
    if not map_path:
        return {"success": False, "error": "Nu există o hartă PCD încărcată"}
    if min_z >= max_z:
        return {"success": False, "error": "Z minim trebuie să fie mai mic decât Z maxim"}
    if comfort_radius < radius:
        return {"success":False,"error":"Raza de confort nu poate fi mai mică decât raza blocată"}
    planner_mode = "nav2" if str(driver).strip().lower() == "nav2" else "legacy"
    try:
        result=await asyncio.to_thread(
            _build_static_costmap_preview,
            map_path,
            resolution,
            radius,
            min_points,
            min_z,
            max_z,
            level_floor,
            floor_tolerance,
            comfort_radius,
            clearance_weight,
            planner_mode,
        )
        planner = result.pop("_planner", None)
        obstacle_guard.set_floor_plane(result.get("floor_plane"))
        observer = _get_nav2_observer()
        if observer is not None and planner is not None:
            try:
                result["nav2_map"] = await asyncio.to_thread(
                    observer.publish_map, planner
                )
            except Exception as exc:
                result["nav2_warning"] = str(exc)
        return result
    except Exception as exc:
        return {"success": False, "error": f"Costmapul 2D nu a putut fi generat: {exc}"}


async def _wait_slam_feedback(api_id: int, sent_at: float, timeout: float = 4.0) -> Optional[dict]:
    """Așteaptă răspunsul robotului, nu doar confirmarea publisherului ROS."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feedback = slam_api_feedback.get(api_id)
        if feedback and feedback.get("received_at", 0.0) >= sent_at:
            return feedback
        await asyncio.sleep(0.05)
    return None


NAV_START_DISPLACEMENT_M = 0.04
NAV_GOAL_TOLERANCE_M = 0.20


def _pose_xy(pose: Optional[dict]) -> Optional[tuple]:
    if not isinstance(pose, dict):
        return None
    try:
        x = float(pose["x"])
        y = float(pose["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _pose_yaw(pose: dict) -> float:
    if "yaw" in pose:
        return float(pose.get("yaw", 0.0) or 0.0)
    qw = float(pose.get("q_w", 1.0) or 1.0)
    qx = float(pose.get("q_x", 0.0) or 0.0)
    qy = float(pose.get("q_y", 0.0) or 0.0)
    qz = float(pose.get("q_z", 0.0) or 0.0)
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _slam_navigation_progress(info: dict, start_pose: Optional[dict], goal: Optional[dict]) -> Optional[str]:
    """Interpretează atât stateMachine, cât și fluxul pos_info al firmware-ului G1."""
    if not info or info.get("error_code") or info.get("paused"):
        return None
    current_xy = _pose_xy(info.get("current_pose"))
    goal_xy = _pose_xy(goal)
    if current_xy and goal_xy and math.hypot(current_xy[0] - goal_xy[0], current_xy[1] - goal_xy[1]) <= NAV_GOAL_TOLERANCE_M:
        return "arrived"
    if info.get("arrived") and not goal_xy:
        return "arrived"

    start_xy = _pose_xy(start_pose)
    if current_xy and start_xy and math.hypot(current_xy[0] - start_xy[0], current_xy[1] - start_xy[1]) >= NAV_START_DISPLACEMENT_M:
        return "navigating"

    state = str(info.get("machine_state", "")).strip().lower()
    controller = str(info.get("controller", "")).strip().lower()
    active_tokens = ("navigat", "planning", "planner", "running", "moving", "tracking")
    if (
        info.get("planner_open")
        or any(token in state for token in active_tokens)
        or any(token in controller for token in active_tokens)
    ):
        return "navigating"
    return None


def _slam_info_confirms_navigation(info: dict, start_pose: Optional[dict] = None, goal: Optional[dict] = None) -> bool:
    return _slam_navigation_progress(info, start_pose, goal) is not None


async def _wait_navigation_confirmation(
    sent_at: float,
    start_pose: dict,
    goal: dict,
    timeout: float = 20.0,
) -> Optional[dict]:
    """Așteaptă stare explicită sau deplasare reală raportată prin pos_info."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = dict(slam_runtime_info)
        if info.get("received_at", 0.0) >= sent_at:
            if info.get("error_code") or _slam_info_confirms_navigation(info, start_pose, goal):
                return info
        await asyncio.sleep(0.05)
    return None


def _parse_navigation_goal(body: dict) -> tuple:
    try:
        x = float(body["x"])
        y = float(body["y"])
        yaw = float(body.get("yaw", 0.0))
        speed = float(body.get("speed", 0.3))
        timeout = float(body.get("timeout", 0.0))
    except (KeyError, TypeError, ValueError):
        raise ValueError("x, y, yaw, speed și timeout trebuie să fie numere valide")
    if not all(math.isfinite(value) for value in (x, y, yaw, speed, timeout)):
        raise ValueError("Coordonatele nu pot fi NaN sau infinite")
    if not 0.1 <= speed <= 1.0:
        raise ValueError("Viteza trebuie să fie între 0.1 și 1.0 m/s")
    # Profil indoor fluent: 1102 poate păstra 0,30-0,35 m/s pe segmentele
    # drepte; navigatorul reduce separat viteza în necunoscut, la colțuri și
    # în culoare înguste. Nu transmitem niciodată valoarea brută de 1 m/s.
    speed = min(speed, 0.35)
    if timeout != 0.0 and not 5.0 <= timeout <= 86400.0:
        raise ValueError("Timeout-ul trebuie să fie 0 (nelimitat) sau între 5 și 86400 secunde")
    return x, y, yaw, speed, timeout


def _parse_navigation_gesture(body: dict) -> str:
    gesture = str(body.get("gesture") or "none").strip().lower()
    if gesture not in NAVIGATION_GESTURES:
        raise ValueError(
            "Gestul trebuie să fie: none, wave, kiss, handshake sau clap"
        )
    return gesture


def _parse_navigation_costmap(body: dict) -> dict:
    raw=body.get("costmap") or {}
    try:
        settings={
            "resolution":float(raw.get("resolution",0.10)),
            "min_obstacle_points":int(raw.get("min_points",3)),
            "obstacle_min_z":float(raw.get("min_z",0.15)),
            "obstacle_max_z":float(raw.get("max_z",1.85)),
            "robot_radius":float(raw.get("radius",0.25)),
            "level_to_floor":bool(raw.get("level_floor",True)),
            "floor_tolerance":float(raw.get("floor_tolerance",0.08)),
            "comfort_radius":float(raw.get("comfort_radius",0.50)),
            "clearance_weight":float(raw.get("clearance_weight",3.50)),
        }
    except (TypeError,ValueError):
        raise ValueError("Parametrii costmap trebuie să fie numere valide")
    # Raza hard este numai dilatarea obstacolelor în planul 2D. Protecția live
    # a corpului și mâinilor rămâne în gardul redundant RealSense + LiDAR.
    settings["robot_radius"] = max(0.20, settings["robot_radius"])
    settings["comfort_radius"] = max(
        settings["robot_radius"], settings["comfort_radius"]
    )
    settings["clearance_weight"] = max(3.50, settings["clearance_weight"])
    numeric=[value for key,value in settings.items() if key!="level_to_floor"]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Parametrii costmap nu pot fi NaN sau infiniți")
    if not 0.05<=settings["resolution"]<=0.30:
        raise ValueError("Rezoluția costmap trebuie să fie între 0,05 și 0,30 m")
    if not 1<=settings["min_obstacle_points"]<=20:
        raise ValueError("Minimum puncte/celulă trebuie să fie între 1 și 20")
    if settings["obstacle_min_z"]>=settings["obstacle_max_z"]:
        raise ValueError("Înălțimea minimă trebuie să fie mai mică decât cea maximă")
    if not 0.20<=settings["robot_radius"]<=0.80:
        raise ValueError("Raza hard costmap trebuie să fie între 0,20 și 0,80 m")
    if not 0.03<=settings["floor_tolerance"]<=0.25:
        raise ValueError("Banda podelei trebuie să fie între 0,03 și 0,25 m")
    if not settings["robot_radius"]<=settings["comfort_radius"]<=1.00:
        raise ValueError("Raza de confort trebuie să fie între raza blocată și 1,00 m")
    if not 0.0<=settings["clearance_weight"]<=10.0:
        raise ValueError("Greutatea de confort trebuie să fie între 0 și 10")
    return settings


async def _preview_navigation(body: dict, allow_active: bool = False):
    """Calculează ruta selectată fără mișcare și cere confirmarea operatorului."""
    global pending_nav_previews
    if not allow_active and _any_navigation_active():
        return {"success": False, "error": "Există deja o navigare activă. Oprește-o înainte de o rută nouă."}
    try:
        x, y, yaw, speed, timeout = _parse_navigation_goal(body)
        costmap_settings = _parse_navigation_costmap(body)
        gesture = _parse_navigation_gesture(body)
        driver = _parse_navigation_driver(body)
        requested_executor = _parse_navigation_executor(body)
        executor, executor_override = _resolve_navigation_executor_for_runtime(
            requested_executor
        )
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    map_path = _resolve_map_path(loaded_map_path)
    if not map_path:
        return {"success": False, "error": "Nu există o hartă PCD încărcată pentru navigare"}
    if map_state.get("slam_mode") != "localization":
        return {"success": False, "error": "Robotul trebuie să fie în modul localization"}
    if executor == "native_1102":
        if not _native_localization_fresh():
            return {
                "success": False,
                "error": (
                    "SLAM Unitree nu furnizează poziția nativă pentru 1102. "
                    "Alege explicit «Controller local sigur» dacă vrei să folosești ancora locală."
                ),
            }
        pose = _native_pose()
    else:
        if not _localization_fresh():
            return {
                "success": False,
                "error": "Poziția locală nu este recentă. Setează din nou poziția și orientarea pe hartă.",
            }
        pose = {**dict(map_state["pose"]), "source": map_state.get("pose_source")}
    requested_goal = {"x": x, "y": y, "yaw": yaw}
    try:
        planner, path, clearance_mode = await asyncio.to_thread(
            plan_pcd_route,map_path,(pose["x"],pose["y"]),(x,y),None,
            **costmap_settings,allow_narrow_fallback=True
        )
        # A* poate deplasa controlat un click aflat în inflația unui obstacol
        # spre cea mai apropiată celulă hard-validă. De aici înainte Nav2,
        # preview-ul și executorul trebuie să folosească exact același capăt.
        resolved_x, resolved_y = path[-1]
        goal_adjustment_m = math.hypot(
            resolved_x - requested_goal["x"],
            resolved_y - requested_goal["y"],
        )
        x, y = float(resolved_x), float(resolved_y)
        obstacle_guard.configure_navigation_map(
            planner.floor_plane,
            planner.resolution,
            planner.raw_static_occupied,
            planner.robot_radius,
            planner.obstacle_min_z,
            planner.obstacle_max_z,
        )
        if executor == "native_1102":
            _get_native_waypoint_navigator().planner = planner
        else:
            _get_autonomous_navigator().planner = planner
        nav2_plan = None
        if driver == "nav2":
            nav2_plan, path = await _request_nav2_plan(
                planner, x, y, yaw, costmap_settings
            )
            path_pattern = {
                "type": "nav2_gridbased",
                "planner_id": nav2_plan.get("planner_id", "GridBased"),
                "raw_poses": nav2_plan.get("poses", 0),
                "execution_segments": max(0, len(path) - 1),
                "planning_time_s": nav2_plan.get("planning_time_s", 0.0),
            }
            clearance_mode = "nav2"
        else:
            # Preview-ul v22 include exact waypoint-ul de plecare folosit la execuție.
            path_tools = _get_native_waypoint_navigator()
            path = path_tools._stabilize_departure_path(path, pose)
            if executor == "native_1102":
                path, path_pattern = path_tools._holonomic_direct_pattern(path, pose)
            else:
                path_pattern = {
                    "type": "astar_local_tracking",
                    "execution_segments": max(0, len(path) - 1),
                }
    except Exception as exc:
        planner_name = "Nav2" if locals().get("driver") == "nav2" else "A* v22"
        return {
            "success": False,
            "error": f"Planificarea {planner_name} a eșuat: {exc}",
            "diagnostics": {
                "driver": locals().get("driver"),
                "executor": locals().get("executor"),
                "start_pose": locals().get("pose"),
                "requested_goal": requested_goal,
                "resolved_goal": ({"x": locals().get("x"), "y": locals().get("y")}
                                  if "x" in locals() and "y" in locals() else None),
                "goal_adjustment_m": locals().get("goal_adjustment_m"),
            },
        }

    preview_id = secrets.token_urlsafe(18)
    path_length = sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])
    )
    pending_nav_previews = {
        preview_id: {
            "created_at": time.monotonic(), "map_path": map_path,
            "start_pose": pose, "goal": {"x": x, "y": y, "yaw": yaw},
            "requested_goal": requested_goal,
            "goal_adjustment_m": goal_adjustment_m,
            "speed": speed, "timeout": timeout, "path": path, "planner": planner,
            "clearance_mode": clearance_mode,
            "costmap_settings": costmap_settings,
            "gesture": gesture,
            "path_pattern": path_pattern,
            "driver": driver,
            "executor": executor,
            "requested_executor": requested_executor,
            "executor_override": executor_override,
            "nav2_plan": nav2_plan,
        }
    }
    return {
        "success": True, "state": "awaiting_confirmation",
        "preview_id": preview_id, "expires_in": 90,
        "path": path, "waypoints": len(path), "length_m": round(path_length, 2),
        "clearance_mode": clearance_mode,
        "costmap_settings": {
            "resolution":planner.resolution,"min_points":planner.min_obstacle_points,
            "min_z":costmap_settings["obstacle_min_z"],"max_z":costmap_settings["obstacle_max_z"],
            "radius":planner.robot_radius,"level_floor":costmap_settings["level_to_floor"],
            "floor_tolerance":costmap_settings["floor_tolerance"],
            "comfort_radius":planner.comfort_radius,"clearance_weight":planner.clearance_weight,
        },
        "floor_plane": planner.floor_plane,
        "dynamic_costmap": planner.dynamic_costmap_points(),
        "start_pose": pose, "goal": {"x": x, "y": y, "yaw": yaw},
        "requested_goal": requested_goal,
        "goal_adjustment_m": round(goal_adjustment_m, 3),
        "gesture": gesture,
        "path_pattern": path_pattern,
        "driver": driver,
        "executor": executor,
        "requested_executor": requested_executor,
        "executor_override": executor_override,
        "planner": "Nav2 GridBased" if driver == "nav2" else "A* v22",
    }


@app.post("/api/nav/preview")
async def preview_navigation(body: dict = Body({})):
    return await _preview_navigation(body, allow_active=False)


@app.post("/api/nav/goal")
async def start_navigation(body: dict = Body({})):
    """Execută numai ruta A* afișată și confirmată de operator."""
    global pending_nav_previews, active_navigation_post_action
    global active_navigation_driver, active_navigation_executor
    preview_id = str(body.get("preview_id") or "")
    preview = pending_nav_previews.pop(preview_id, None)
    if not preview:
        return {
            "success": False,
            "error": "Ruta nu a fost confirmată sau a expirat. Previzualizează ruta din nou.",
        }
    if time.monotonic() - preview["created_at"] > 90.0:
        return {"success": False, "error": "Previzualizarea a expirat după 90s. Recalculează ruta."}
    map_path = _resolve_map_path(loaded_map_path)
    if not map_path or os.path.realpath(map_path) != os.path.realpath(preview["map_path"]):
        return {"success": False, "error": "Harta s-a schimbat după previzualizare. Recalculează ruta."}
    executor = preview.get("executor", "native_1102")
    if executor == "native_1102":
        if not _native_localization_fresh():
            return {"success": False, "error": "Localizarea nativă 1102 s-a pierdut după previzualizare."}
        pose = _native_pose()
        if autonomous_navigator is not None and _navigator_is_active(autonomous_navigator):
            return {"success": False, "error": "Controllerul local este încă activ; apasă STOP înainte de 1102."}
    else:
        if not _localization_fresh():
            return {"success": False, "error": "Poziția locală s-a pierdut după previzualizare."}
        pose = {**dict(map_state["pose"]), "source": map_state.get("pose_source")}
        if native_waypoint_navigator is not None and _navigator_is_active(native_waypoint_navigator):
            return {"success": False, "error": "Executorul 1102 este încă activ; apasă STOP înainte de control local."}
    start_pose = preview["start_pose"]
    moved = math.hypot(pose["x"] - start_pose["x"], pose["y"] - start_pose["y"])
    if moved > 0.20:
        return {
            "success": False,
            "error": f"Robotul s-a deplasat {moved:.2f}m după previzualizare. Recalculează ruta.",
        }

    goal = preview["goal"]
    driver = preview.get("driver", "legacy")
    active_navigation_post_action = preview.get("gesture", "none")
    active_navigation_driver = driver
    active_navigation_executor = executor
    flight_log = _start_navigation_flight_log(preview, pose)
    navigator = (
        _get_native_waypoint_navigator()
        if executor == "native_1102" else _get_autonomous_navigator()
    )
    if executor == "native_1102":
        result = await navigator.start(
            map_path, goal["x"], goal["y"], goal["yaw"],
            speed=preview["speed"], timeout=preview["timeout"],
            prepared_plan={
                "path": preview["path"], "planner": preview["planner"],
                "clearance_mode": preview.get("clearance_mode", "normal"),
                "path_pattern": preview.get("path_pattern") or {"type": "astar"},
                "preserve_path": driver == "nav2",
                "driver": driver,
                "costmap_settings": preview.get("costmap_settings") or {},
            },
            motion_profile="stable",
        )
    else:
        # Semnătura executorului local este cea originală din v23.
        result = await navigator.start(
            map_path, goal["x"], goal["y"], goal["yaw"],
            timeout=preview["timeout"] or 120.0,
        )
    if result.get("success"):
        result.update({
            "api_id": 1102 if executor == "native_1102" else None,
            "executor": executor,
            "driver": (
                "nav2_gridbased_1102" if driver == "nav2" else "v22_astar_1102"
            ) if executor == "native_1102" else (
                "nav2_gridbased_local_velocity" if driver == "nav2"
                else "v22_astar_local_velocity"
            ),
            "planner": "Nav2 GridBased" if driver == "nav2" else "A* v22",
            "map_path": map_path,
            "flight_log": flight_log,
        })
    else:
        active_navigation_post_action = None
        active_navigation_driver = None
        active_navigation_executor = None
        _append_navigation_flight_log({
            "type": "run_rejected", "state": "failed", "error": result.get("error", "")
        })
    return result


@app.post("/api/nav/preview/cancel")
async def cancel_navigation_preview():
    global pending_nav_previews, queue_transition_scheduled
    pending_nav_previews = {}
    if navigation_goal_queue and not queue_transition_scheduled:
        queue_transition_scheduled = True
        asyncio.create_task(_activate_next_queued_goal())
    return {"success": True}


@app.post("/api/nav/replan/confirm")
async def confirm_dynamic_replan(body: dict = Body({})):
    confirmation_id = str(body.get("confirmation_id") or "")
    return await _get_native_waypoint_navigator().confirm_replan(confirmation_id)


@app.post("/api/nav/replan/reject")
async def reject_dynamic_replan(body: dict = Body({})):
    confirmation_id = str(body.get("confirmation_id") or "")
    return await _get_native_waypoint_navigator().reject_replan(confirmation_id)


@app.post("/api/nav/queue")
async def add_navigation_queue(body: dict = Body({})):
    """Adaugă FIFO sau oprește taskul curent și pregătește o rută înlocuitoare."""
    global navigation_goal_queue, pending_nav_previews
    try:
        x, y, yaw, speed, timeout = _parse_navigation_goal(body)
        gesture = _parse_navigation_gesture(body)
        driver = _parse_navigation_driver(body)
        executor = _parse_navigation_executor(body)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    action = str(body.get("action") or "append")
    if action not in {"append", "replace"}:
        return {"success": False, "error": "action trebuie să fie append sau replace"}
    item = {
        "id": secrets.token_urlsafe(10), "x": x, "y": y, "yaw": yaw,
        "speed": speed, "timeout": timeout, "created_at": time.time(),
        "costmap": body.get("costmap") or {},
        "gesture": gesture,
        "driver": driver,
        "executor": executor,
    }
    active = _any_navigation_active()

    if action == "replace":
        navigation_goal_queue = []
        pending_nav_previews = {}
        if active:
            if native_waypoint_navigator is not None and _navigator_is_active(native_waypoint_navigator):
                await native_waypoint_navigator.stop("task înlocuit de operator")
            if autonomous_navigator is not None and _navigator_is_active(autonomous_navigator):
                await autonomous_navigator.stop("task înlocuit de operator")
            await asyncio.sleep(0)
        result = await _preview_navigation(item, allow_active=True)
        if result.get("success"):
            await broadcast({"type": "nav_preview", "source": "replace", **result})
        await broadcast({"type": "nav_queue", "queue": []})
        return {**result, "action": "replace", "queue": []}

    if not active and not pending_nav_previews:
        result = await _preview_navigation(item, allow_active=True)
        if result.get("success"):
            await broadcast({"type": "nav_preview", "source": "queue_immediate", **result})
        return {**result, "action": "preview_now", "queue": _public_navigation_queue()}

    navigation_goal_queue.append(item)
    public_queue = _public_navigation_queue()
    await broadcast({"type": "nav_queue", "queue": public_queue})
    return {"success": True, "action": "queued", "queue": public_queue, "item": public_queue[-1]}


@app.get("/api/nav/queue")
async def get_navigation_queue():
    return {"success": True, "queue": _public_navigation_queue()}


@app.delete("/api/nav/queue/{item_id}")
async def remove_navigation_queue_item(item_id: str):
    global navigation_goal_queue
    original = len(navigation_goal_queue)
    navigation_goal_queue = [item for item in navigation_goal_queue if item.get("id") != item_id]
    queue = _public_navigation_queue()
    await broadcast({"type": "nav_queue", "queue": queue})
    return {"success": len(navigation_goal_queue) != original, "queue": queue}


@app.post("/api/nav/stop")
async def stop_navigation():
    """Pune pe pauză planificatorul SLAM și trimite și viteza zero ca barieră."""
    global navigation_goal_queue, pending_nav_previews
    global navigation_post_action_task, active_navigation_post_action
    global active_navigation_driver, active_navigation_executor
    navigation_goal_queue = []
    pending_nav_previews = {}
    active_navigation_post_action = None
    active_navigation_driver = None
    active_navigation_executor = None
    gesture_was_running = bool(
        navigation_post_action_task and not navigation_post_action_task.done()
    )
    if gesture_was_running:
        navigation_post_action_task.cancel()
        try:
            await navigation_post_action_task
        except asyncio.CancelledError:
            pass
    navigation_post_action_task = None
    native_result = (
        await native_waypoint_navigator.stop()
        if native_waypoint_navigator is not None and _navigator_is_active(native_waypoint_navigator)
        else {"success": True, "output": "executor 1102 inactiv"}
    )
    local_result = (
        await autonomous_navigator.stop()
        if autonomous_navigator is not None and _navigator_is_active(autonomous_navigator)
        else {"success": True, "output": "executor local inactiv"}
    )
    stop_result = await asyncio.to_thread(sport_client.stop)
    await broadcast({"type": "nav_queue", "queue": []})
    return {
        "success": bool(
            native_result.get("success") and local_result.get("success")
            and stop_result.get("success")
        ),
        "slam": native_result,
        "local_executor": local_result,
        "locomotion": stop_result,
        "gesture_cancelled": gesture_was_running,
    }


@app.get("/api/nav/status")
async def navigation_status():
    navigator = (
        _get_autonomous_navigator()
        if active_navigation_executor == "local_velocity"
        else _get_native_waypoint_navigator()
    )
    return {
        "success": True, **navigator.status,
        "selected_driver": active_navigation_driver,
        "selected_executor": active_navigation_executor,
        "flight_log": str(navigation_flight_log_path) if navigation_flight_log_path else None,
    }


@app.get("/api/nav2/status")
async def nav2_status():
    observer = _get_nav2_observer()
    if observer is None:
        return {"success": False, "available": False, "error": "Adaptorul Nav2 nu este disponibil"}
    return {"success": True, **observer.status()}


@app.get("/api/nav/flight_log")
async def navigation_flight_log(limit: int = Query(200, ge=1, le=1000)):
    """Întoarce ultimele înregistrări ale cursei curente pentru audit fizic."""
    if navigation_flight_log_path is None or not navigation_flight_log_path.exists():
        return {"success": False, "error": "Nu există încă un jurnal de navigare"}
    try:
        lines = navigation_flight_log_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines[-limit:] if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return {"success": False, "error": f"Jurnalul nu poate fi citit: {exc}"}
    return {
        "success": True,
        "path": str(navigation_flight_log_path),
        "records": records,
    }


@app.get("/api/nav/flight_log/download")
async def download_navigation_flight_log():
    """Descarcă jurnalul curent; calea este generată exclusiv de server."""
    path = navigation_flight_log_path
    if (path is None or not path.is_file()
            or path.resolve().parent != NAVIGATION_LOG_DIR.resolve()):
        return JSONResponse(status_code=404, content={
            "success": False, "error": "Nu există încă un jurnal de navigare",
        })
    return FileResponse(path, filename=path.name, media_type="application/x-ndjson")


@app.get("/api/nav/preflight")
async def navigation_preflight(
        wait_s: float = 0.0,
        driver: str = "legacy",
        executor: str = "native_1102"):
    """Verifică preflight-ul; opțional așteaptă reconectarea scurtă a senzorilor."""
    wait_s = max(0.0, min(3.0, float(wait_s)))
    try:
        selected_driver = _parse_navigation_driver({"driver": driver})
        selected_executor = _parse_navigation_executor({"executor": executor})
    except ValueError as exc:
        return {"success": False, "checks": {}, "error": str(exc)}
    started = time.monotonic()
    sensor_keys = {
        "camera_obstacle_sensor_fresh",
        "lidar_obstacle_sensor_fresh",
        "obstacle_sensors_fresh",
        "nav2_adapter_available",
        "nav2_planner_ready",
        "nav2_tf_ready",
        "nav2_map_published",
    }
    while True:
        obstacle_sensor_status = obstacle_guard.sensor_status()
        # Derivăm starea combinată din același snapshot. Două citiri separate
        # puteau cădea de o parte și de alta a pragului de prospețime.
        sensors_ready = bool(
            obstacle_sensor_status["camera_fresh"]
            and obstacle_sensor_status["lidar_fresh"]
        )
        native_pose_ready = _native_localization_fresh()
        native_controller = str(slam_runtime_info.get("controller") or "").strip().lower()
        local_fsm_id = (
            sport_client.get_current_fsm_id()
            if selected_executor == "local_velocity" else None
        )
        checks = {
            "map_loaded": bool(_resolve_map_path(loaded_map_path)),
            "localization_mode": map_state.get("slam_mode") == "localization",
            "camera_obstacle_sensor_fresh": obstacle_sensor_status["camera_fresh"],
            "lidar_obstacle_sensor_fresh": obstacle_sensor_status["lidar_fresh"],
            "obstacle_sensors_fresh": sensors_ready,
        }
        if selected_executor == "native_1102":
            checks.update({
                "localization_fresh": _localization_fresh(),
                "unitree_pose_ready": native_pose_ready,
                # `not init` from ctrl_info is transient on this firmware and
                # must not invalidate a simultaneous, fresh native pos_info.
                # Dispatch still waits for the real API 1102 acknowledgement.
                "slam_native_navigation": native_pose_ready,
                "slam_telemetry_fresh": (
                    time.monotonic() - float(slam_runtime_info.get("received_at", 0.0)) < 2.0
                ),
            })
        else:
            local_lidar_status = local_lidar_localizer.status()
            checks.update({
                "localization_fresh": _localization_fresh(),
                "local_lidar_localization_ready": bool(
                    local_lidar_status.get("ready")
                ),
                "local_velocity_sdk_available": sport_client.is_sdk_available(),
                "fsm_locomotion_ready": local_fsm_id in {500, 501, 502, 801, 802},
            })
        nav2 = _get_nav2_observer()
        nav2_details = nav2.status() if nav2 is not None else {
            "available": False, "planner_ready": False,
            "tf_ready": False, "map_published": False,
        }
        if selected_driver == "nav2":
            checks.update({
                "nav2_adapter_available": bool(nav2_details.get("available")),
                "nav2_planner_ready": bool(nav2_details.get("planner_ready")),
                "nav2_tf_ready": bool(nav2_details.get("tf_ready")),
                "nav2_map_published": bool(nav2_details.get("map_published")),
            })
        success = all(checks.values())
        non_sensor_failure = any(
            not ok for name, ok in checks.items() if name not in sensor_keys
        )
        elapsed = time.monotonic() - started
        if success or non_sensor_failure or elapsed >= wait_s:
            return {
                "success": success,
                "checks": checks,
                "pose": dict(map_state["pose"]),
                "map_path": _resolve_map_path(loaded_map_path),
                "slam": dict(slam_runtime_info),
                "localization_sources": {
                    "display": {
                        "ready": _localization_fresh(),
                        "source": map_state.get("pose_source"),
                    },
                    "unitree_1102": {
                        "ready": native_pose_ready,
                        "source": "/slam_info pos_info" if native_pose_ready else None,
                        "controller": slam_runtime_info.get("controller"),
                        "current_pose": dict(slam_runtime_info.get("current_pose") or {}),
                    },
                    "nav2_tf": {
                        "ready": bool(nav2_details.get("tf_ready")),
                        "source": nav2_details.get("tf_source"),
                    },
                    "local_lidar": local_lidar_localizer.status(),
                },
                "obstacle_sensors": obstacle_sensor_status,
                "driver": selected_driver,
                "executor": selected_executor,
                "robot": {
                    "fsm_id": local_fsm_id,
                    "locomotion_ready": (
                        local_fsm_id in {500, 501, 502, 801, 802}
                        if selected_executor == "local_velocity" else None
                    ),
                },
                "nav2": nav2_details,
                "waited_s": round(elapsed, 3),
            }
        await asyncio.sleep(0.10)

@app.get("/api/robot/fsm_id")
async def get_robot_fsm_id():
    """Diagnostic: citește starea FSM curentă a robotului (util ca să
    vezi dacă tranzițiile Damp/stand_up/Start chiar au avut efect)."""
    return await asyncio.to_thread(sport_client.get_fsm_id)

@app.get("/")
async def get_dashboard():
    # Frontendul conține logica de siguranță și tokenul de storage pe versiune;
    # nu permitem browserului să păstreze o copie veche după restart.
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )

@app.on_event("startup")
async def startup_event():
    global loop, node_instance
    loop = asyncio.get_event_loop()
    if ROS2_AVAILABLE:
        threading.Thread(target=ros2_thread_entry, daemon=True).start()
    threading.Thread(target=camera_receiver_thread, daemon=True).start()
    asyncio.create_task(map_broadcast_loop())
    asyncio.create_task(teleop_watchdog_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Ultima barieră de siguranță: niciun restart/CTRL-C nu lasă G1 în mers."""
    global autonomous_navigator, native_waypoint_navigator
    global navigation_post_action_task
    if navigation_post_action_task and not navigation_post_action_task.done():
        navigation_post_action_task.cancel()
        try:
            await navigation_post_action_task
        except asyncio.CancelledError:
            pass
        navigation_post_action_task = None
    if native_waypoint_navigator is not None:
        await native_waypoint_navigator.stop("serverul se oprește")
    if autonomous_navigator is not None:
        await autonomous_navigator.stop("serverul se oprește")
    else:
        await asyncio.to_thread(sport_client.stop)

def ros2_thread_entry():
    global node_instance
    if not ROS2_AVAILABLE: return
    rclpy.init()
    node_instance = DualLidarSLAMSubscriber()
    rclpy.spin(node_instance)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3003)
