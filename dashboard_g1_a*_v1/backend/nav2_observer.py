"""Adaptoare ROS 2 pentru observarea sigură a intrărilor Nav2.

Modulul nu pornește navigarea și nu publică ``cmd_vel``. El expune doar:

* harta statică filtrată ca ``/map`` (OccupancyGrid, transient-local);
* odometria locală Unitree normalizată ca ``/nav2_observer/odom``;
* lanțul TF ``map -> odom -> base_link -> torso_link -> livox_frame``.

Funcțiile geometrice și rasterizarea rămân independente de ROS pentru a putea
fi verificate pe laptop.
"""

import math
import heapq
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def compose_planar(first: Dict[str, float], second: Dict[str, float]) -> Dict[str, float]:
    """Întoarce transformarea SE(2) ``first * second``."""
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
    """Calculează ``T_map_odom = T_map_base * inverse(T_odom_base)``."""
    return compose_planar(global_base, inverse_planar(odom_base))


def build_occupancy_grid(
    planner,
    clear_xy: Optional[Tuple[float, float]] = None,
    clear_radius: float = 0.32,
) -> Dict[str, object]:
    """Convertește straturile PCD într-un OccupancyGrid fără inflație.

    Podeaua observată devine liberă (0), obstacolele brute devin letale (100),
    iar restul rămâne necunoscut (-1). InflationLayer din Nav2 este singurul
    responsabil de inflația folosită la planificarea Nav2.
    """
    if not planner.bounds:
        raise ValueError("Plannerul nu are limite pentru OccupancyGrid")
    min_x, max_x, min_y, max_y = planner.bounds
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width <= 0 or height <= 0:
        raise ValueError("Dimensiuni OccupancyGrid invalide")
    if width * height > 16_000_000:
        raise ValueError("OccupancyGrid prea mare")

    data: List[int] = [-1] * (width * height)

    def index(cell: Tuple[int, int]) -> Optional[int]:
        x, y = cell
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return None
        return (y - min_y) * width + (x - min_x)

    for cell in planner.known_free:
        offset = index(cell)
        if offset is not None:
            data[offset] = 0
    for cell in planner.raw_static_occupied:
        offset = index(cell)
        if offset is not None:
            data[offset] = 100

    resolution = float(planner.resolution)
    cleared_robot_cells = 0
    if clear_xy is not None:
        center_x = round(float(clear_xy[0]) / resolution)
        center_y = round(float(clear_xy[1]) / resolution)
        radius_cells = max(1, math.ceil(float(clear_radius) / resolution))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if math.hypot(dx * resolution, dy * resolution) > clear_radius:
                    continue
                offset = index((center_x + dx, center_y + dy))
                if offset is not None:
                    if data[offset] != 0:
                        cleared_robot_cells += 1
                    data[offset] = 0
    return {
        "frame_id": "map",
        "width": width,
        "height": height,
        "resolution": resolution,
        # world_to_cell folosește round(), deci centrul celulei min este
        # min*resolution, iar colțul OccupancyGrid este cu jumătate de celulă jos.
        "origin_x": (min_x - 0.5) * resolution,
        "origin_y": (min_y - 0.5) * resolution,
        "data": data,
        "counts": {
            "free": sum(value == 0 for value in data),
            "occupied": sum(value == 100 for value in data),
            "unknown": sum(value == -1 for value in data),
            "robot_cleared": cleared_robot_cells,
        },
        "robot_clear_radius": float(clear_radius) if clear_xy is not None else 0.0,
    }


try:
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import OccupancyGrid, Odometry
    from nav2_msgs.action import ComputePathToPose
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import GetParameters, SetParameters
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    ROS2_OBSERVER_AVAILABLE = True
except ImportError:
    ROS2_OBSERVER_AVAILABLE = False


class Nav2ObserverPublisher:
    """Publică numai intrările Nav2; nu poate comanda locomotion."""

    def __init__(self, node):
        if not ROS2_OBSERVER_AVAILABLE:
            raise RuntimeError("Mesajele ROS 2/tf2_ros nu sunt disponibile")
        self.node = node
        self.global_pose: Optional[Dict[str, float]] = None
        self.global_source = ""
        self.local_pose: Optional[Dict[str, float]] = None
        self.local_source = ""
        # Pentru /initialpose + odometrie pelvis, map->odom este o ancoră
        # constantă. Recalcularea ei la fiecare cadru ar ține base_link blocat
        # în poziția inițială în loc să lase odometria să progreseze.
        self.anchored_map_to_odom: Optional[Dict[str, float]] = None
        self.global_received_at = 0.0
        self.local_received_at = 0.0
        self.primary_local_received_at = 0.0
        self.map_spec: Optional[Dict[str, object]] = None
        self.map_grid: Optional[Dict[str, object]] = None
        self.map_published_at = 0.0
        self.nav2_costmap: Optional[Dict[str, object]] = None
        self.nav2_costmap_received_at = 0.0
        self.error = ""

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_publisher = node.create_publisher(OccupancyGrid, "/map", map_qos)
        self.odom_publisher = node.create_publisher(
            Odometry, "/nav2_observer/odom", 10
        )
        self.tf_broadcaster = TransformBroadcaster(node)
        self.path_client = ActionClient(node, ComputePathToPose, "/compute_path_to_pose")
        self.static_broadcaster = StaticTransformBroadcaster(node)
        local_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.local_subscription = node.create_subscription(
            Odometry,
            "/unitree_slam/high_rate_odometry",
            self.update_primary_local_odometry,
            local_qos,
        )
        self.local_fallback_subscription = node.create_subscription(
            Odometry,
            "/state_estimator/odom_pelvis",
            self.update_fallback_local_odometry,
            local_qos,
        )
        self.nav2_costmap_subscription = node.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            self.update_nav2_costmap,
            map_qos,
        )
        self.costmap_get_parameters = node.create_client(
            GetParameters, "/global_costmap/global_costmap/get_parameters"
        )
        self.costmap_set_parameters = node.create_client(
            SetParameters, "/global_costmap/global_costmap/set_parameters"
        )
        self.planner_get_parameters = node.create_client(
            GetParameters, "/planner_server/get_parameters"
        )
        self.planner_set_parameters = node.create_client(
            SetParameters, "/planner_server/set_parameters"
        )
        self._publish_static_transforms()

    @staticmethod
    def _pose_from_odometry(message) -> Dict[str, float]:
        return {
            "x": float(message.pose.pose.position.x),
            "y": float(message.pose.pose.position.y),
            "yaw": quaternion_to_yaw(message.pose.pose.orientation),
        }

    def _uses_fixed_correction(self) -> bool:
        return bool(
            self.global_source.startswith("/initialpose")
            or self.global_source.startswith("/local_lidar/icp")
        )

    @staticmethod
    def _set_yaw(rotation, yaw: float) -> None:
        rotation.x = 0.0
        rotation.y = 0.0
        rotation.z = math.sin(yaw / 2.0)
        rotation.w = math.cos(yaw / 2.0)

    def _transform(self, parent: str, child: str, pose: Dict[str, float], stamp):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x = float(pose["x"])
        message.transform.translation.y = float(pose["y"])
        message.transform.translation.z = float(pose.get("z", 0.0))
        self._set_yaw(message.transform.rotation, float(pose["yaw"]))
        return message

    def _publish_static_transforms(self) -> None:
        stamp = self.node.get_clock().now().to_msg()

        torso = TransformStamped()
        torso.header.stamp = stamp
        torso.header.frame_id = "base_link"
        torso.child_frame_id = "torso_link"
        torso.transform.translation.x = -0.0039635
        torso.transform.translation.y = 0.0
        torso.transform.translation.z = 0.054
        torso.transform.rotation.w = 1.0

        lidar = TransformStamped()
        lidar.header.stamp = stamp
        lidar.header.frame_id = "torso_link"
        lidar.child_frame_id = "livox_frame"
        lidar.transform.translation.x = 0.0002835
        lidar.transform.translation.y = 0.00003
        lidar.transform.translation.z = 0.40618
        x, y, z, w = quaternion_from_rpy(
            3.14, 0.04014257279586953, 0.0
        )
        lidar.transform.rotation.x = x
        lidar.transform.rotation.y = y
        lidar.transform.rotation.z = z
        lidar.transform.rotation.w = w
        self.static_broadcaster.sendTransform([torso, lidar])

    def update_global_odometry(self, message) -> None:
        """Primește localizarea Unitree ``map -> base_link``."""
        self.update_global_pose(
            self._pose_from_odometry(message),
            "/unitree/slam_relocation/odom",
        )

    def update_global_pose(self, pose: Dict[str, float], source: str) -> None:
        """Acceptă și poziția funcțională din ``/slam_info pos_info``."""
        self.global_pose = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": wrap_angle(float(pose["yaw"])),
        }
        self.global_source = str(source)
        self.global_received_at = time.monotonic()
        if self._uses_fixed_correction():
            self.anchored_map_to_odom = (
                map_to_odom(self.global_pose, self.local_pose)
                if self.local_pose else None
            )
        else:
            self.anchored_map_to_odom = None
        self._publish_dynamic_transforms()

    def update_localized_pose(self, pose: Dict[str, float]) -> None:
        """Actualizează corecția map->odom din ICP, fără a îngheța odometria."""
        self.update_global_pose(pose, "/local_lidar/icp")

    def update_primary_local_odometry(self, message) -> None:
        self.primary_local_received_at = time.monotonic()
        self.update_local_odometry(message, "/unitree_slam/high_rate_odometry")

    def update_fallback_local_odometry(self, message) -> None:
        if time.monotonic() - self.primary_local_received_at < 1.0:
            return
        self.update_local_odometry(message, "/state_estimator/odom_pelvis")

    def update_local_odometry(self, message, source: str) -> None:
        """Primește odometria fluidă ``odom -> pelvis``; pelvis=base_link."""
        self.local_pose = self._pose_from_odometry(message)
        self.local_source = str(source)
        self.local_received_at = time.monotonic()

        normalized = Odometry()
        normalized.header.stamp = self.node.get_clock().now().to_msg()
        normalized.header.frame_id = "odom"
        normalized.child_frame_id = "base_link"
        normalized.pose = message.pose
        normalized.twist = message.twist
        self.odom_publisher.publish(normalized)
        self._publish_dynamic_transforms()

    def _publish_dynamic_transforms(self) -> None:
        if not self.global_pose or not self.local_pose:
            return
        stamp = self.node.get_clock().now().to_msg()
        if self._uses_fixed_correction():
            if self.anchored_map_to_odom is None:
                self.anchored_map_to_odom = map_to_odom(
                    self.global_pose, self.local_pose
                )
            correction = self.anchored_map_to_odom
        else:
            correction = map_to_odom(self.global_pose, self.local_pose)
        map_odom = self._transform("map", "odom", correction, stamp)
        odom_base = self._transform("odom", "base_link", self.local_pose, stamp)
        self.tf_broadcaster.sendTransform([map_odom, odom_base])
        self.error = ""

    def update_nav2_costmap(self, message) -> None:
        """Păstrează costmapul combinat folosit efectiv de planner_server."""
        width = int(message.info.width)
        height = int(message.info.height)
        expected = width * height
        if width <= 0 or height <= 0 or expected != len(message.data):
            self.error = "Nav2 a publicat un costmap cu dimensiuni invalide"
            return
        self.nav2_costmap = {
            "frame_id": str(message.header.frame_id or "map"),
            "width": width,
            "height": height,
            "resolution": float(message.info.resolution),
            "origin_x": float(message.info.origin.position.x),
            "origin_y": float(message.info.origin.position.y),
            "data": [int(value) for value in message.data],
        }
        self.nav2_costmap_received_at = time.monotonic()

    @staticmethod
    def _parameter_value(value):
        if value.type == ParameterType.PARAMETER_BOOL:
            return bool(value.bool_value)
        if value.type == ParameterType.PARAMETER_INTEGER:
            return int(value.integer_value)
        if value.type == ParameterType.PARAMETER_DOUBLE:
            return float(value.double_value)
        if value.type == ParameterType.PARAMETER_STRING:
            return str(value.string_value)
        return None

    @staticmethod
    def _parameter_message(name: str, value):
        parameter_value = ParameterValue()
        if isinstance(value, bool):
            parameter_value.type = ParameterType.PARAMETER_BOOL
            parameter_value.bool_value = value
        elif isinstance(value, int):
            parameter_value.type = ParameterType.PARAMETER_INTEGER
            parameter_value.integer_value = value
        else:
            parameter_value.type = ParameterType.PARAMETER_DOUBLE
            parameter_value.double_value = float(value)
        return Parameter(name=name, value=parameter_value)

    @staticmethod
    def _wait_service(client, request, timeout: float = 3.0):
        if not client.wait_for_service(timeout_sec=min(1.5, timeout)):
            raise RuntimeError("Serviciul de parametri Nav2 nu este disponibil")
        future = client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(timeout):
            raise TimeoutError("Nav2 nu a răspuns la serviciul de parametri")
        result = future.result()
        if result is None:
            raise RuntimeError("Serviciul de parametri Nav2 a răspuns fără rezultat")
        return result

    def _get_parameters(self, client, names: List[str]) -> Dict[str, object]:
        request = GetParameters.Request()
        request.names = list(names)
        response = self._wait_service(client, request)
        return {
            name: self._parameter_value(value)
            for name, value in zip(names, response.values)
        }

    def _set_parameters(self, client, values: Dict[str, object]) -> None:
        if not values:
            return
        request = SetParameters.Request()
        request.parameters = [
            self._parameter_message(name, value) for name, value in values.items()
        ]
        response = self._wait_service(client, request)
        failures = [
            result.reason or name
            for name, result in zip(values, response.results)
            if not result.successful
        ]
        if failures:
            raise RuntimeError("Nav2 a refuzat parametrii: " + "; ".join(failures))

    def runtime_parameters(self) -> Dict[str, object]:
        costmap_names = [
            "robot_radius",
            "resolution",
            "inflation_layer.inflation_radius",
            "inflation_layer.cost_scaling_factor",
            "voxel_layer.enabled",
            "voxel_layer.lidar.min_obstacle_height",
            "voxel_layer.lidar.max_obstacle_height",
            "voxel_layer.lidar.obstacle_min_range",
            "voxel_layer.lidar.obstacle_max_range",
            "voxel_layer.lidar.raytrace_max_range",
        ]
        planner_names = ["GridBased.allow_unknown", "GridBased.tolerance"]
        values = self._get_parameters(self.costmap_get_parameters, costmap_names)
        values.update(self._get_parameters(self.planner_get_parameters, planner_names))
        return values

    def set_runtime_parameters(self, values: Dict[str, object]) -> Dict[str, object]:
        costmap_names = {
            "robot_radius",
            "resolution",
            "inflation_layer.inflation_radius",
            "inflation_layer.cost_scaling_factor",
            "voxel_layer.enabled",
            "voxel_layer.lidar.min_obstacle_height",
            "voxel_layer.lidar.max_obstacle_height",
            "voxel_layer.lidar.obstacle_min_range",
            "voxel_layer.lidar.obstacle_max_range",
            "voxel_layer.lidar.raytrace_max_range",
        }
        planner_names = {"GridBased.allow_unknown", "GridBased.tolerance"}
        self._set_parameters(
            self.costmap_set_parameters,
            {name: value for name, value in values.items() if name in costmap_names},
        )
        self._set_parameters(
            self.planner_set_parameters,
            {name: value for name, value in values.items() if name in planner_names},
        )
        return self.runtime_parameters()

    def nav2_costmap_snapshot(self) -> Dict[str, object]:
        if not self.nav2_costmap:
            raise RuntimeError(
                "Nav2 nu a publicat încă /global_costmap/costmap; verifică planner_server și /map"
            )
        snapshot = dict(self.nav2_costmap)
        data = list(snapshot["data"])
        snapshot["data"] = data
        snapshot["age"] = time.monotonic() - self.nav2_costmap_received_at
        snapshot["counts"] = {
            "free": sum(value == 0 for value in data),
            "inflated": sum(0 < value < 99 for value in data),
            "lethal": sum(value >= 99 for value in data),
            "unknown": sum(value < 0 for value in data),
        }
        return snapshot

    def _map_value(self, x: float, y: float) -> Optional[int]:
        if not self.map_grid:
            return None
        resolution = float(self.map_grid["resolution"])
        column = round((float(x) - float(self.map_grid["origin_x"])) / resolution - 0.5)
        row = round((float(y) - float(self.map_grid["origin_y"])) / resolution - 0.5)
        width = int(self.map_grid["width"])
        height = int(self.map_grid["height"])
        if not (0 <= column < width and 0 <= row < height):
            return None
        return int(self.map_grid["data"][row * width + column])

    def _nav2_costmap_value(self, x: float, y: float) -> Optional[int]:
        if not self.nav2_costmap:
            return None
        resolution = float(self.nav2_costmap["resolution"])
        column = math.floor((float(x) - float(self.nav2_costmap["origin_x"])) / resolution)
        row = math.floor((float(y) - float(self.nav2_costmap["origin_y"])) / resolution)
        width = int(self.nav2_costmap["width"])
        height = int(self.nav2_costmap["height"])
        if not (0 <= column < width and 0 <= row < height):
            return None
        return int(self.nav2_costmap["data"][row * width + column])

    def _nav2_segment_is_safe(
            self, start: Tuple[float, float], end: Tuple[float, float],
            maximum_cost: int = 80,
    ) -> bool:
        """Verifică o scurtătură în costmapul combinat folosit chiar de Nav2."""
        if not self.nav2_costmap:
            return False
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        step = max(0.02, float(self.nav2_costmap["resolution"]) * 0.45)
        samples = max(1, math.ceil(distance / step))
        for index in range(samples + 1):
            ratio = index / samples
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            value = self._nav2_costmap_value(x, y)
            if value is None or value < 0 or value > maximum_cost:
                return False
        return True

    def _nav2_world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if not self.nav2_costmap:
            return None
        resolution = float(self.nav2_costmap["resolution"])
        column = math.floor(
            (float(x) - float(self.nav2_costmap["origin_x"])) / resolution
        )
        row = math.floor(
            (float(y) - float(self.nav2_costmap["origin_y"])) / resolution
        )
        if not (0 <= column < int(self.nav2_costmap["width"])
                and 0 <= row < int(self.nav2_costmap["height"])):
            return None
        return column, row

    def _nav2_cell_center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        resolution = float(self.nav2_costmap["resolution"])
        return (
            float(self.nav2_costmap["origin_x"]) + (cell[0] + 0.5) * resolution,
            float(self.nav2_costmap["origin_y"]) + (cell[1] + 0.5) * resolution,
        )

    def _nav2_cell_cost(self, cell: Tuple[int, int]) -> Optional[int]:
        if not self.nav2_costmap:
            return None
        column, row = cell
        width = int(self.nav2_costmap["width"])
        height = int(self.nav2_costmap["height"])
        if not (0 <= column < width and 0 <= row < height):
            return None
        return int(self.nav2_costmap["data"][row * width + column])

    def _repair_path_on_nav2_costmap(
            self, start: Tuple[float, float], goal: Tuple[float, float],
            maximum_cost: int = 98,
    ) -> Optional[List[Tuple[float, float]]]:
        """Repară un plan ROS care taie colțul unei celule letale.

        Navfn poate publica o diagonală chiar pe frontiera dintre celule. G1
        execută segmente continue, deci refacem numai acea geometrie pe același
        costmap combinat Static/Voxel/Inflation. Necunoscutul și costul letal
        rămân blocate, iar diagonalele nu pot tăia colțuri.
        """
        start_cell = self._nav2_world_to_cell(*start)
        goal_cell = self._nav2_world_to_cell(*goal)
        if start_cell is None or goal_cell is None:
            return None

        def traversable(cell: Tuple[int, int]) -> bool:
            value = self._nav2_cell_cost(cell)
            return value is not None and 0 <= value <= maximum_cost

        if not traversable(start_cell) or not traversable(goal_cell):
            return None

        neighbors = (
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        )
        frontier = [(0.0, start_cell)]
        came_from = {start_cell: None}
        distance = {start_cell: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            for dx, dy, step in neighbors:
                candidate = current[0] + dx, current[1] + dy
                if not traversable(candidate):
                    continue
                if dx and dy and (
                        not traversable((current[0] + dx, current[1]))
                        or not traversable((current[0], current[1] + dy))):
                    continue
                cell_cost = float(self._nav2_cell_cost(candidate) or 0)
                # Preferăm mijlocul spațiului liber, dar 81..98 rămâne
                # traversabil când Nav2 nu are altă ieșire sigură.
                risk = 1.0 + 4.0 * (cell_cost / max(1.0, maximum_cost)) ** 2
                candidate_distance = distance[current] + step * risk
                if (candidate not in distance
                        or candidate_distance < distance[candidate] - 1e-9):
                    distance[candidate] = candidate_distance
                    heuristic = math.hypot(
                        goal_cell[0] - candidate[0], goal_cell[1] - candidate[1]
                    )
                    heapq.heappush(
                        frontier, (candidate_distance + heuristic, candidate)
                    )
                    came_from[candidate] = current
        if goal_cell not in came_from:
            return None

        cells = []
        current = goal_cell
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        points = [self._nav2_cell_center(cell) for cell in cells]
        # Păstrăm coordonatele exacte furnizate de Nav2 numai dacă legătura
        # scurtă până la centrul celulei este sigură.
        if self._nav2_segment_is_safe(start, points[0], maximum_cost):
            points[0] = start
        if self._nav2_segment_is_safe(points[-1], goal, maximum_cost):
            points[-1] = goal
        return points

    def simplify_execution_path(
            self, path: List[Dict[str, float]], maximum_cost: int = 80,
            traversable_maximum_cost: int = 98,
    ) -> List[Tuple[float, float]]:
        """Reduce planul dens Nav2 la segmente lungi executabile de G1.

        Navfn publică de regulă câte o poziție pe celulă. Trimiterea lor directă
        către locomotion ar produce opriri și rotiri dese. String-pulling-ul
        păstrează numai colțurile necesare și validează fiecare scurtătură în
        `/global_costmap/costmap`, inclusiv Static/Voxel/InflationLayer.
        """
        points = [
            (float(point["x"]), float(point["y"])) for point in path
        ]
        if len(points) <= 2:
            return points

        # `maximum_cost` este intenționat conservator numai pentru scurtăturile
        # adăugate de noi. Nav2 poate întoarce în mod valid pași prin inflație
        # (cost 81..98); aceștia nu sunt obstacole letale și trebuie păstrați,
        # nu respinși ca și cum plannerul ar fi traversat un perete.
        if traversable_maximum_cost < maximum_cost:
            traversable_maximum_cost = maximum_cost

        # Unele planuri Navfn trec dintr-o celulă în alta exact pe colțul unei
        # celule letale. Interpolarea liniară dintre centre poate tăia acea
        # celulă, chiar dacă polilinia ROS pare corectă în RViz. Introducem un
        # cot cartezian numai când ambele jumătăți sunt validate de costmap.
        # Astfel executorul G1 nu primește o diagonală periculoasă la colț.
        repaired = [points[0]]
        unsafe_segment = None
        for segment_index, point in enumerate(points[1:], start=1):
            previous = repaired[-1]
            if not self._nav2_segment_is_safe(
                    previous, point, traversable_maximum_cost):
                elbows = (
                    (point[0], previous[1]),
                    (previous[0], point[1]),
                )
                safe_elbows = [
                    elbow for elbow in elbows
                    if self._nav2_segment_is_safe(
                        previous, elbow, traversable_maximum_cost
                    )
                    and self._nav2_segment_is_safe(
                        elbow, point, traversable_maximum_cost
                    )
                ]
                if not safe_elbows:
                    unsafe_segment = (segment_index, previous, point)
                    break
                elbow = min(
                    safe_elbows,
                    key=lambda candidate: (
                        math.hypot(candidate[0] - previous[0], candidate[1] - previous[1])
                        + math.hypot(point[0] - candidate[0], point[1] - candidate[1])
                    ),
                )
                if elbow != previous and elbow != point:
                    repaired.append(elbow)
            repaired.append(point)
        if unsafe_segment is None:
            points = repaired
        else:
            repaired_grid = self._repair_path_on_nav2_costmap(
                points[0], points[-1], traversable_maximum_cost
            )
            if not repaired_grid:
                index, previous, point = unsafe_segment
                previous_cost = self._nav2_costmap_value(*previous)
                point_cost = self._nav2_costmap_value(*point)
                raise RuntimeError(
                    "Planul Nav2 nu poate fi reparat în costmapul curent: "
                    f"segment {index}, cost capete {previous_cost}/{point_cost}. "
                    "A apărut un obstacol letal sau o regiune necunoscută după planificare"
                )
            points = repaired_grid

        reduced = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                if self._nav2_segment_is_safe(
                    points[anchor], points[candidate], maximum_cost
                ):
                    break
                candidate -= 1
            reduced.append(points[candidate])
            anchor = candidate
        return reduced

    def compute_path(self, x: float, y: float, yaw: float, timeout: float = 8.0) -> Dict[str, object]:
        """Cere un traseu de la planner_server fără să comande mișcarea."""
        if not self.global_pose or not self.local_pose:
            raise RuntimeError("TF map -> odom -> base_link nu este pregătit")
        start_value = self._map_value(self.global_pose["x"], self.global_pose["y"])
        goal_value = self._map_value(x, y)
        if start_value is None:
            raise RuntimeError("Poziția robotului este în afara hărții Nav2")
        if start_value != 0:
            raise RuntimeError("Celula de start nu este liberă; redeschide Costmap pentru republicare")
        if goal_value is None:
            raise RuntimeError("Destinația este în afara hărții Nav2")
        if goal_value < 0:
            raise RuntimeError("Destinația este în spațiu necunoscut; alege o celulă verde")
        if goal_value >= 100:
            raise RuntimeError("Destinația este pe un obstacol; alege o celulă liberă")
        nav2_start_value = self._nav2_costmap_value(
            self.global_pose["x"], self.global_pose["y"]
        )
        nav2_goal_value = self._nav2_costmap_value(x, y)
        if nav2_start_value is not None and nav2_start_value >= 99:
            raise RuntimeError(
                f"Poziția robotului este blocată în costmapul final Nav2 (cost {nav2_start_value})"
            )
        if nav2_goal_value is not None and nav2_goal_value >= 99:
            raise RuntimeError(
                f"Destinația este blocată în costmapul final Nav2 (cost {nav2_goal_value})"
            )
        if nav2_goal_value is not None and nav2_goal_value < 0:
            raise RuntimeError("Destinația este necunoscută în costmapul final Nav2")
        if not self.path_client.wait_for_server(timeout_sec=min(2.0, timeout)):
            raise RuntimeError("Acțiunea Nav2 /compute_path_to_pose nu este disponibilă")

        goal = ComputePathToPose.Goal()
        goal.goal = PoseStamped()
        goal.goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.goal.header.frame_id = "map"
        goal.goal.pose.position.x = float(x)
        goal.goal.pose.position.y = float(y)
        goal.goal.pose.orientation.z = math.sin(float(yaw) / 2.0)
        goal.goal.pose.orientation.w = math.cos(float(yaw) / 2.0)
        goal.planner_id = "GridBased"
        goal.use_start = False

        sent = self.path_client.send_goal_async(goal)
        sent_event = threading.Event()
        sent.add_done_callback(lambda _future: sent_event.set())
        if not sent_event.wait(timeout):
            raise TimeoutError("Nav2 nu a acceptat cererea de traseu în timp util")
        goal_handle = sent.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 a refuzat cererea de traseu")

        result_future = goal_handle.get_result_async()
        result_event = threading.Event()
        result_future.add_done_callback(lambda _future: result_event.set())
        if not result_event.wait(timeout):
            goal_handle.cancel_goal_async()
            raise TimeoutError("Nav2 nu a calculat traseul în timp util")
        wrapped = result_future.result()
        path_message = wrapped.result.path
        path = [
            {
                "x": float(pose.pose.position.x),
                "y": float(pose.pose.position.y),
                "yaw": quaternion_to_yaw(pose.pose.orientation),
            }
            for pose in path_message.poses
        ]
        if not path:
            start_cost = self._nav2_costmap_value(
                self.global_pose["x"], self.global_pose["y"]
            )
            goal_cost = self._nav2_costmap_value(x, y)
            raise RuntimeError(
                "Nav2 nu a găsit traseu nici cu spațiul necunoscut permis: "
                "obstacolele sau inflația separă regiunile "
                f"(cost start {start_cost}, cost destinație {goal_cost}). "
                "Verifică roșul și inflația portocalie din tabul Costmap."
            )
        length = sum(
            math.hypot(second["x"] - first["x"], second["y"] - first["y"])
            for first, second in zip(path, path[1:])
        )
        return {
            "path": path,
            "length_m": length,
            "poses": len(path),
            "planner_id": "GridBased",
            "planning_time_s": float(wrapped.result.planning_time.sec)
                + float(wrapped.result.planning_time.nanosec) / 1e9,
        }

    def publish_map(self, planner) -> Dict[str, object]:
        clear_xy = None
        if self.global_pose:
            clear_xy = (self.global_pose["x"], self.global_pose["y"])
        spec = build_occupancy_grid(planner, clear_xy=clear_xy)
        message = OccupancyGrid()
        stamp = self.node.get_clock().now().to_msg()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        message.info.map_load_time = stamp
        message.info.resolution = float(spec["resolution"])
        message.info.width = int(spec["width"])
        message.info.height = int(spec["height"])
        message.info.origin.position.x = float(spec["origin_x"])
        message.info.origin.position.y = float(spec["origin_y"])
        message.info.origin.orientation.w = 1.0
        message.data = list(spec["data"])
        self.map_publisher.publish(message)
        self.map_grid = spec
        self.map_spec = {key: value for key, value in spec.items() if key != "data"}
        self.map_published_at = time.monotonic()
        return dict(self.map_spec)

    def status(self) -> Dict[str, object]:
        now = time.monotonic()
        global_age = now - self.global_received_at if self.global_received_at else None
        local_age = now - self.local_received_at if self.local_received_at else None
        fixed_correction = bool(
            self._uses_fixed_correction()
            and self.anchored_map_to_odom is not None
        )
        initial_anchor = self.global_source.startswith("/initialpose")
        global_fresh = bool(
            (fixed_correction and initial_anchor)
            or (global_age is not None and global_age < 2.0)
        )
        local_fresh = local_age is not None and local_age < 1.0
        current_pose = (
            compose_planar(self.anchored_map_to_odom, self.local_pose)
            if fixed_correction and self.local_pose else self.global_pose
        )
        return {
            "available": True,
            "observer_only": True,
            "cmd_vel_enabled": False,
            "planner_ready": bool(self.path_client.server_is_ready()),
            "tf_ready": bool(global_fresh and local_fresh),
            "map_published": self.map_spec is not None,
            "global_source": self.global_source or "aștept localizarea Unitree",
            "local_source": self.local_source or "aștept odometria locală",
            "odom_topic": "/nav2_observer/odom",
            "global_age": global_age,
            "local_age": local_age,
            "pose": dict(current_pose) if current_pose else None,
            "tf_source": (
                f"{self.global_source or 'fără global'} + "
                f"{self.local_source or 'fără odom'}"
            ),
            "anchored_tf": fixed_correction,
            "map": dict(self.map_spec) if self.map_spec else None,
            "waist_assumption": "waist_yaw_joint blocat la 0 rad",
            "error": self.error,
        }
