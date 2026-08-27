"""Navigație A→B pe harta PCD, fără dependență de un stack Nav2 extern."""

import asyncio
import heapq
import math
import random
import time
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple


GridCell = Tuple[int, int]
NORMAL_CLEARANCE_RADIUS = 0.25
# Culoarul controlat păstrează cel puțin 20 cm între axa traseului și punctele
# ocupate. Protecția live pentru corp și brațe rămâne activă separat.
NARROW_CLEARANCE_RADIUS = 0.20
SAFE_CLEARANCE_RADIUS = 0.25
GOAL_SNAP_TOLERANCE = 0.20
LATERAL_RECOVERY_COOLDOWN = 8.0
DYNAMIC_OBSTACLE_RADIUS = 0.30
DYNAMIC_OBSTACLE_TTL = 2.5
DYNAMIC_SENSOR_PADDING = 0.05
LIVE_OBSTACLE_MIN_CLEARANCE = 0.28
# Mobilierul live are margine hard separată mai jos. Confortul dinamic nu
# trebuie să moștenească automat raza statică de 0,65 m, altfel două scaune cu
# culoar real între ele produc două câmpuri moi suprapuse și par un singur zid.
DYNAMIC_OBSTACLE_COMFORT_RADIUS = 0.45
CENTERLINE_CLEARANCE_RADIUS = 0.75
CENTERLINE_CLEARANCE_WEIGHT = 1.50
TURN_CLEARANCE_RADIUS = 0.55
TURN_CLEARANCE_WEIGHT = 7.0
OBSTACLE_WAIT_BEFORE_REPLAN = 0.05
# Trackurile LiDAR cer deja confirmare temporală. După ce geometria a rămas
# liberă continuu aproape o jumătate de secundă putem relua fără secunda de
# ezitare care făcea mersul sacadat.
OBSTACLE_CLEAR_STABLE = 0.30
OBSTACLE_SENSOR_LOSS_TIMEOUT = 5.0
# Un singur cadru întârziat nu justifică ciclul costisitor 1201/1202. În
# această fereastră nu lansăm comenzi noi, dar nici nu declarăm senzorul pierdut.
SENSOR_GLITCH_GRACE = 0.25
ROUTE_OBSTACLE_CONFIRMATION = 0.15
REPLAN_ROUTE_STABLE = 0.12
# Dacă A* nu găsește o ieșire deși LiDAR-ul confirmă un obstacol apropiat,
# permitem o singură degajare laterală verificată. Fereastra scurtă evită
# rotațiile/replanificările repetate, fără să transforme orice STOP într-un pas.
DYNAMIC_LATERAL_UNLOCK_DELAY = 0.55
NATIVE_WAYPOINT_MIN_DISTANCE = 0.65
# 1102 primește o polilinie controlată, nu o singură țintă aflată la câțiva
# metri. Valoarea rămâne peste toleranța nativă de waypoint ca să nu fie sărite
# punctele, dar este suficient de mică pentru curbe și replănuire live precisă.
LIVE_ROUTE_WAYPOINT_SPACING = 0.85
# Ruta densă rămâne adevărul geometric pentru viewer și verificarea LiDAR,
# însă 1102 nu trebuie reprogramat la fiecare 85 cm. Un punct de control este
# ales cât mai departe pe aceeași porțiune sigură, fără să taie colțurile A*.
NATIVE_CONTROL_LOOKAHEAD = 3.20
STARTUP_SPEED_LIMIT = 0.20
STARTUP_PROGRESS_DISTANCE = 0.10
STARTUP_WAYPOINT_DISTANCE = 0.70
RECOVERY_OBSTACLE_MEMORY = 12.0
RECOVERY_OBSTACLE_MAX_ROBOT_TRAVEL = 0.55


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def navigation_sensors_ready(obstacle_guard) -> bool:
    """Folosește redundanța strictă dacă implementarea reală o oferă.

    Fake-urile/testele mai vechi rămân compatibile prin `has_fresh_data`.
    """
    if hasattr(obstacle_guard, "navigation_sensors_ready"):
        return bool(obstacle_guard.navigation_sensors_ready())
    return bool(obstacle_guard.has_fresh_data())


class PCDGridPlanner:
    def __init__(self, resolution: float = 0.15, robot_radius: float = 0.45,
                 min_obstacle_points: int = 1, comfort_radius: Optional[float] = None,
                 clearance_weight: float = 2.5, unknown_space_weight: float = 3.0):
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.comfort_radius = max(robot_radius, float(comfort_radius if comfort_radius is not None else robot_radius))
        self.clearance_weight = max(0.0, float(clearance_weight))
        self.unknown_space_weight = max(0.0, float(unknown_space_weight))
        self.min_obstacle_points = max(1, int(min_obstacle_points))
        self.raw_static_occupied: set[GridCell] = set()
        self.static_occupied: set[GridCell] = set()
        self.dynamic_occupied: Dict[GridCell, float] = {}
        self.dynamic_sources: Dict[GridCell, str] = {}
        # Marjă moale pentru obstacolele temporare: nu blochează culoarele,
        # dar A* preferă ocolirea cât mai largă permisă de spațiul disponibil.
        self.dynamic_clearance_cost: Dict[GridCell, float] = {}
        self.dynamic_clearance_seen: Dict[GridCell, float] = {}
        self.dynamic_clearance_sources: Dict[GridCell, str] = {}
        self.bounds: Optional[Tuple[int, int, int, int]] = None
        self.floor_plane: Optional[dict] = None
        self.obstacle_min_z = 0.15
        self.obstacle_max_z = 1.60
        self.clearance_cost: Dict[GridCell, float] = {}
        self.centerline_cost: Dict[GridCell, float] = {}
        self.obstacle_distance: Dict[GridCell, float] = {}
        self.known_free: set[GridCell] = set()
        self.goal_snap_tolerance = 0.0

    @property
    def occupied(self) -> set[GridCell]:
        """Compatibilitate pentru diagnosticare: costmapul static + cel dinamic."""
        return self.static_occupied | set(self.dynamic_occupied)

    def add_dynamic_obstacle(self, x: float, y: float, radius: float = 0.55,
                             observed_at: Optional[float] = None,
                             source: str = "sensor") -> int:
        """Adaugă/reîmprospătează numai celulele care nu sunt deja în PCD."""
        timestamp = time.monotonic() if observed_at is None else float(observed_at)
        center = self.world_to_cell(x, y)
        changed = 0
        for ox, oy in self._inflation_offsets(radius):
            cell = (center[0] + ox, center[1] + oy)
            if cell in self.static_occupied:
                continue
            if cell not in self.dynamic_occupied:
                changed += 1
            self.dynamic_occupied[cell] = timestamp
            self.dynamic_sources[cell] = source
        return changed

    def add_dynamic_points(self, points, inflation_radius: Optional[float] = None,
                           observed_at: Optional[float] = None,
                           source: str = "lidar") -> int:
        """Inserează forma 2D observată, nu un disc artificial în jurul unui centroid."""
        timestamp = time.monotonic() if observed_at is None else float(observed_at)
        radius = max(0.0, float(
            self.robot_radius + DYNAMIC_SENSOR_PADDING
            if inflation_radius is None else inflation_radius
        ))
        centers = {
            self.world_to_cell(float(point[0]), float(point[1]))
            for point in points
            if len(point) >= 2 and math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
        }
        changed = 0
        offsets = tuple(self._inflation_offsets(radius))
        for center_x, center_y in centers:
            # Un punct care coincide cu harta salvată este perete/mobilier static,
            # nu un obstacol dinamic nou.
            if (center_x, center_y) in self.raw_static_occupied:
                continue
            for offset_x, offset_y in offsets:
                cell = (center_x + offset_x, center_y + offset_y)
                if cell in self.static_occupied:
                    continue
                if cell not in self.dynamic_occupied:
                    changed += 1
                self.dynamic_occupied[cell] = timestamp
                self.dynamic_sources[cell] = source
        comfort_radius = max(
            radius,
            min(self.comfort_radius, DYNAMIC_OBSTACLE_COMFORT_RADIUS),
        )
        comfort_span = max(self.resolution, comfort_radius - radius)
        comfort_offsets = tuple(self._inflation_offsets(comfort_radius))
        for center_x, center_y in centers:
            if (center_x, center_y) in self.raw_static_occupied:
                continue
            for offset_x, offset_y in comfort_offsets:
                distance = math.hypot(
                    offset_x * self.resolution, offset_y * self.resolution
                )
                if distance <= radius or distance > comfort_radius:
                    continue
                cell = (center_x + offset_x, center_y + offset_y)
                if cell in self.static_occupied or cell in self.dynamic_occupied:
                    continue
                ratio = max(0.0, min(
                    1.0, (comfort_radius - distance) / comfort_span
                ))
                cost = self.clearance_weight * (ratio ** 1.25)
                self.dynamic_clearance_cost[cell] = max(
                    cost, self.dynamic_clearance_cost.get(cell, 0.0)
                )
                self.dynamic_clearance_seen[cell] = timestamp
                self.dynamic_clearance_sources[cell] = source + "_comfort"
        if source == "lidar":
            # Forma măsurată rămâne distinctă vizual de marginea de siguranță.
            for center in centers:
                if center in self.dynamic_occupied and center not in self.static_occupied:
                    self.dynamic_sources[center] = "lidar_raw"
        return changed

    def clear_dynamic_source(self, source: str) -> int:
        cells = [
            cell for cell, value in self.dynamic_sources.items()
            if value == source or value.startswith(source + "_")
        ]
        for cell in cells:
            self.dynamic_occupied.pop(cell, None)
            self.dynamic_sources.pop(cell, None)
        comfort_cells = [
            cell for cell, value in self.dynamic_clearance_sources.items()
            if value == source or value.startswith(source + "_")
        ]
        for cell in comfort_cells:
            self.dynamic_clearance_cost.pop(cell, None)
            self.dynamic_clearance_seen.pop(cell, None)
            self.dynamic_clearance_sources.pop(cell, None)
        return len(cells) + len(comfort_cells)

    def clear_dynamic_obstacle(self, x: float, y: float, radius: float = 0.80) -> int:
        """Șterge o observație live confirmată liberă, fără a atinge harta statică."""
        center = self.world_to_cell(x, y)
        removed = 0
        for ox, oy in self._inflation_offsets(radius):
            cell = (center[0] + ox, center[1] + oy)
            if self.dynamic_occupied.pop(cell, None) is not None:
                self.dynamic_sources.pop(cell,None)
                removed += 1
        return removed

    def expire_dynamic_obstacles(self, max_age: float = 8.0,
                                 now: Optional[float] = None) -> int:
        """Elimină obstacolele live care nu au mai fost observate."""
        timestamp = time.monotonic() if now is None else float(now)
        expired = [
            cell for cell, last_seen in self.dynamic_occupied.items()
            if timestamp - last_seen > max_age
        ]
        for cell in expired:
            self.dynamic_occupied.pop(cell, None)
            self.dynamic_sources.pop(cell,None)
        expired_comfort = [
            cell for cell, last_seen in self.dynamic_clearance_seen.items()
            if timestamp - last_seen > max_age
        ]
        for cell in expired_comfort:
            self.dynamic_clearance_cost.pop(cell, None)
            self.dynamic_clearance_seen.pop(cell, None)
            self.dynamic_clearance_sources.pop(cell, None)
        return len(expired) + len(expired_comfort)

    def clear_dynamic_costmap(self) -> int:
        removed = len(self.dynamic_occupied) + len(self.dynamic_clearance_cost)
        self.dynamic_occupied.clear()
        self.dynamic_sources.clear()
        self.dynamic_clearance_cost.clear()
        self.dynamic_clearance_seen.clear()
        self.dynamic_clearance_sources.clear()
        return removed

    def dynamic_costmap_points(self) -> List[dict]:
        """Celulele temporare trimise UI-ului, distinct de PCD."""
        now = time.monotonic()
        blocked = [
            {
                "x": round(cell[0] * self.resolution, 3),
                "y": round(cell[1] * self.resolution, 3),
                "age": round(max(0.0, now - last_seen), 2),
                "source": self.dynamic_sources.get(cell, "sensor"),
            }
            for cell, last_seen in self.dynamic_occupied.items()
        ]
        comfort = [
            {
                "x": round(cell[0] * self.resolution, 3),
                "y": round(cell[1] * self.resolution, 3),
                "age": round(max(0.0, now - last_seen), 2),
                "source": self.dynamic_clearance_sources.get(cell, "sensor_comfort"),
            }
            for cell, last_seen in self.dynamic_clearance_seen.items()
            if cell not in self.dynamic_occupied
        ]
        return blocked + comfort

    def clear_robot_footprint(self, x: float, y: float) -> None:
        """Poziția curentă a robotului nu poate fi un obstacol static din PCD."""
        center = self.world_to_cell(x, y)
        for ox, oy in self._inflation_offsets(self.robot_radius):
            cell = (center[0] + ox, center[1] + oy)
            self.raw_static_occupied.discard(cell)
            self.static_occupied.discard(cell)
            self.dynamic_occupied.pop(cell, None)
            self.dynamic_sources.pop(cell,None)
            self.known_free.add(cell)

    def _inflation_offsets(self, radius: float):
        """Celule aflate realmente în rază, fără rotunjirea 0,48 m la 0,60 m."""
        padding = max(1, math.ceil(radius / self.resolution))
        tolerance = self.resolution * 0.5
        for ox in range(-padding, padding + 1):
            for oy in range(-padding, padding + 1):
                if math.hypot(ox * self.resolution, oy * self.resolution) <= radius + tolerance:
                    yield ox, oy

    def _build_obstacle_distance_field(self, sources: set[GridCell],
                                       max_radius: float) -> Dict[GridCell, float]:
        """Distanță 2D continuă până la obstacol, pentru axa culoarului."""
        if not sources or not self.bounds:
            return {}
        min_x, max_x, min_y, max_y = self.bounds
        distances: Dict[GridCell, float] = {cell: 0.0 for cell in sources}
        frontier = [(0.0, cell) for cell in sources]
        heapq.heapify(frontier)
        neighbors = (
            (1, 0, self.resolution), (-1, 0, self.resolution),
            (0, 1, self.resolution), (0, -1, self.resolution),
            (1, 1, self.resolution * math.sqrt(2)),
            (1, -1, self.resolution * math.sqrt(2)),
            (-1, 1, self.resolution * math.sqrt(2)),
            (-1, -1, self.resolution * math.sqrt(2)),
        )
        while frontier:
            distance, cell = heapq.heappop(frontier)
            if distance > distances.get(cell, math.inf) + 1e-9:
                continue
            for dx, dy, step in neighbors:
                nxt = (cell[0] + dx, cell[1] + dy)
                next_distance = distance + step
                if (next_distance > max_radius
                        or not min_x <= nxt[0] <= max_x
                        or not min_y <= nxt[1] <= max_y
                        or next_distance >= distances.get(nxt, math.inf)):
                    continue
                distances[nxt] = next_distance
                heapq.heappush(frontier, (next_distance, nxt))
        return distances

    @staticmethod
    def _iter_pcd_points(path: str):
        data = False
        with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                if not data:
                    if line.lstrip().upper().startswith("DATA"):
                        if "ascii" not in line.lower():
                            raise ValueError("Navigatorul suportă momentan doar PCD ASCII")
                        data = True
                    continue
                fields = line.split()
                if len(fields) < 3:
                    continue
                try:
                    yield tuple(map(float, fields[:3]))
                except ValueError:
                    continue

    @staticmethod
    def _least_squares_plane(points):
        sx=sy=sz=sxx=syy=sxy=sxz=syz=0.0
        for x, y, z in points:
            sx+=x; sy+=y; sz+=z; sxx+=x*x; syy+=y*y
            sxy+=x*y; sxz+=x*z; syz+=y*z
        matrix=[[sxx,sxy,sx,sxz],[sxy,syy,sy,syz],[sx,sy,float(len(points)),sz]]
        for column in range(3):
            pivot=max(range(column,3), key=lambda row: abs(matrix[row][column]))
            if abs(matrix[pivot][column]) < 1e-9:
                return None
            matrix[column],matrix[pivot]=matrix[pivot],matrix[column]
            divisor=matrix[column][column]
            matrix[column]=[value/divisor for value in matrix[column]]
            for row in range(3):
                if row==column:
                    continue
                factor=matrix[row][column]
                matrix[row]=[value-factor*base for value,base in zip(matrix[row],matrix[column])]
        return matrix[0][3],matrix[1][3],matrix[2][3]

    @classmethod
    def _estimate_floor_plane(cls, sample, tolerance: float = 0.08):
        if len(sample) < 3:
            return None
        if len(sample) < 200:
            # Compatibilitate pentru hărți/teste foarte rare: percentila joasă
            # este mai sigură decât planul dominant, care poate fi un birou.
            floor_z=sorted(point[2] for point in sample)[max(0,int((len(sample)-1)*0.15))]
            return {"a":0.0,"b":0.0,"c":floor_z,"tilt_deg":0.0,
                    "inliers":sum(abs(p[2]-floor_z)<=tolerance for p in sample),
                    "sample_size":len(sample),"rms":0.0}
        rng=random.Random(len(sample)*2654435761)
        threshold=max(0.045,min(0.10,float(tolerance)))
        best=None
        best_inliers=[]
        for _ in range(140):
            p1,p2,p3=rng.sample(sample,3)
            determinant=(p1[0]-p3[0])*(p2[1]-p3[1])-(p2[0]-p3[0])*(p1[1]-p3[1])
            if abs(determinant)<0.04:
                continue
            a=((p1[2]-p3[2])*(p2[1]-p3[1])-(p2[2]-p3[2])*(p1[1]-p3[1]))/determinant
            b=((p1[0]-p3[0])*(p2[2]-p3[2])-(p2[0]-p3[0])*(p1[2]-p3[2]))/determinant
            if math.hypot(a,b)>0.60:
                continue
            c=p1[2]-a*p1[0]-b*p1[1]
            inliers=[p for p in sample if abs(p[2]-(a*p[0]+b*p[1]+c))<=threshold]
            if len(inliers)>len(best_inliers):
                best=(a,b,c); best_inliers=inliers
        if best is None or len(best_inliers)<max(100,len(sample)*0.06):
            return None
        a,b,c=cls._least_squares_plane(best_inliers) or best
        inliers=[p for p in sample if abs(p[2]-(a*p[0]+b*p[1]+c))<=threshold]
        rms=math.sqrt(sum((p[2]-(a*p[0]+b*p[1]+c))**2 for p in inliers)/max(1,len(inliers)))
        return {"a":a,"b":b,"c":c,"tilt_deg":math.degrees(math.atan(math.hypot(a,b))),
                "inliers":len(inliers),"sample_size":len(sample),"rms":rms}

    def load(self, path: str, obstacle_min_z: float = 0.15,
             obstacle_max_z: float = 1.60, level_to_floor: bool = True,
             floor_tolerance: float = 0.08,
             clear_xy: Optional[Tuple[float, float]] = None,
             clear_radius: float = 0.38) -> None:
        self.obstacle_min_z = float(obstacle_min_z)
        self.obstacle_max_z = float(obstacle_max_z)
        raw_hits: Dict[GridCell, int] = {}
        floor_cells: set[GridCell] = set()
        sample=[]
        point_count=0
        min_cell_x=min_cell_y=math.inf
        max_cell_x=max_cell_y=-math.inf
        rng=random.Random(0xC05A4A9)
        for x,y,z in self._iter_pcd_points(path):
            point_count+=1
            cell=self.world_to_cell(x,y)
            min_cell_x=min(min_cell_x,cell[0]); max_cell_x=max(max_cell_x,cell[0])
            min_cell_y=min(min_cell_y,cell[1]); max_cell_y=max(max_cell_y,cell[1])
            if len(sample)<5000:
                sample.append((x,y,z))
            else:
                index=rng.randrange(point_count)
                if index<5000:
                    sample[index]=(x,y,z)
        if not point_count:
            raise ValueError("Harta PCD nu conține puncte")

        self.floor_plane=self._estimate_floor_plane(sample,floor_tolerance) if level_to_floor else None
        if level_to_floor and self.floor_plane is None:
            raise ValueError("Planul podelei nu a putut fi detectat sigur")
        for x,y,z in self._iter_pcd_points(path):
            relative_z=z
            if self.floor_plane:
                relative_z=z-(self.floor_plane["a"]*x+self.floor_plane["b"]*y+self.floor_plane["c"])
                if abs(relative_z)<=floor_tolerance:
                    # Podeaua detectată nu este niciodată obstacol, chiar dacă
                    # operatorul coboară manual limita minimă a filtrului.
                    floor_cells.add(self.world_to_cell(x, y))
                    continue
            if obstacle_min_z <= relative_z <= obstacle_max_z:
                if clear_xy and math.hypot(x-clear_xy[0],y-clear_xy[1])<=clear_radius:
                    continue
                cell=self.world_to_cell(x,y)
                raw_hits[cell]=raw_hits.get(cell,0)+1

        padding = max(1, math.ceil(self.robot_radius / self.resolution))
        inflated: set[GridCell] = set()
        offsets = tuple(self._inflation_offsets(self.robot_radius))
        raw = {cell for cell, hits in raw_hits.items() if hits >= self.min_obstacle_points}
        for cx, cy in raw:
            for ox, oy in offsets:
                inflated.add((cx + ox, cy + oy))
        self.raw_static_occupied = raw
        self.static_occupied = inflated
        # Podeaua măsurată este spațiu cunoscut. Extinderea mică umple golurile
        # dintre razele LiDAR fără a transforma întreaga hartă în liber.
        self.known_free = set()
        known_offsets = tuple(self._inflation_offsets(max(0.18, self.resolution * 1.5)))
        for cell_x, cell_y in floor_cells:
            for offset_x, offset_y in known_offsets:
                cell = (cell_x + offset_x, cell_y + offset_y)
                if cell not in inflated:
                    self.known_free.add(cell)
        self.dynamic_occupied = {}
        self.dynamic_sources = {}
        self.dynamic_clearance_cost = {}
        self.dynamic_clearance_seen = {}
        self.dynamic_clearance_sources = {}
        margin = max(2, padding)
        self.bounds = (int(min_cell_x)-margin,int(max_cell_x)+margin,
                       int(min_cell_y)-margin,int(max_cell_y)+margin)
        self.obstacle_distance = self._build_obstacle_distance_field(
            raw, max(self.comfort_radius, CENTERLINE_CLEARANCE_RADIUS)
        )
        self.clearance_cost = {}
        self.centerline_cost = {}
        comfort_span = max(1e-6, self.comfort_radius - self.robot_radius)
        for cell, distance in self.obstacle_distance.items():
            if cell in inflated:
                continue
            if self.comfort_radius > self.robot_radius and self.clearance_weight > 0:
                ratio = max(0.0, min(
                    1.0, (self.comfort_radius - distance) / comfort_span
                ))
                if ratio > 0.0:
                    self.clearance_cost[cell] = (
                        self.clearance_weight * (ratio ** 1.25)
                    )
            center_ratio = max(
                0.0,
                (CENTERLINE_CLEARANCE_RADIUS - distance)
                / CENTERLINE_CLEARANCE_RADIUS,
            )
            if center_ratio > 0.0:
                # Termen moderat, continuu: separă zona plată și alege mijlocul
                # culoarului, dar rămâne traversabil în spații înguste.
                self.centerline_cost[cell] = (
                    CENTERLINE_CLEARANCE_WEIGHT * center_ratio * center_ratio
                )

    def world_to_cell(self, x: float, y: float) -> GridCell:
        return (round(x / self.resolution), round(y / self.resolution))

    def cell_to_world(self, cell: GridCell) -> Tuple[float, float]:
        return (cell[0] * self.resolution, cell[1] * self.resolution)

    def _valid(self, cell: GridCell) -> bool:
        if not self.bounds:
            return False
        min_x, max_x, min_y, max_y = self.bounds
        return (min_x <= cell[0] <= max_x and min_y <= cell[1] <= max_y
                and cell not in self.static_occupied
                and cell not in self.dynamic_occupied)

    def _line_is_free(self, start: GridCell, end: GridCell) -> bool:
        """Supercover aproximativ: nu scurtăm traseul prin pereți/colțuri."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return self._valid(start)
        previous = start
        for index in range(1, steps + 1):
            t = index / steps
            cell = (round(start[0] + dx * t), round(start[1] + dy * t))
            if not self._valid(cell):
                return False
            if cell[0] != previous[0] and cell[1] != previous[1]:
                if not self._valid((cell[0], previous[1])) or not self._valid((previous[0], cell[1])):
                    return False
            previous = cell
        return True

    def segment_is_free(self, start_xy: Tuple[float, float],
                        end_xy: Tuple[float, float]) -> bool:
        """Verificare publică folosită înainte de trecerea anticipată la următorul waypoint."""
        return self._line_is_free(
            self.world_to_cell(*start_xy), self.world_to_cell(*end_xy)
        )

    def smooth_handoff_is_safe(self, start_xy: Tuple[float, float],
                               waypoint_xy: Tuple[float, float],
                               following_xy: Tuple[float, float]) -> bool:
        """Permite scurtătura numai dacă nu taie colțul sau zona de confort."""
        start = self.world_to_cell(*start_xy)
        waypoint = self.world_to_cell(*waypoint_xy)
        following = self.world_to_cell(*following_xy)
        direct = self._line_cells(start, following)
        first = self._line_cells(start, waypoint)
        second = self._line_cells(waypoint, following)
        if direct is None or first is None or second is None:
            return False
        via = first + second[1:]
        return self._shortcut_is_safer_or_equal(direct, via)

    def _line_cells(self, start: GridCell, end: GridCell) -> Optional[List[GridCell]]:
        dx,dy=end[0]-start[0],end[1]-start[1]
        steps=max(abs(dx),abs(dy))
        if steps==0:
            return [start] if self._valid(start) else None
        cells=[start]
        previous=start
        for index in range(1,steps+1):
            t=index/steps
            cell=(round(start[0]+dx*t),round(start[1]+dy*t))
            if not self._valid(cell):
                return None
            if cell[0]!=previous[0] and cell[1]!=previous[1]:
                if not self._valid((cell[0],previous[1])) or not self._valid((previous[0],cell[1])):
                    return None
            if cell!=cells[-1]:
                cells.append(cell)
            previous=cell
        return cells

    def _cells_cost(self, cells: List[GridCell]) -> float:
        total=0.0
        for previous,cell in zip(cells,cells[1:]):
            step=math.hypot(cell[0]-previous[0],cell[1]-previous[1])
            unknown_penalty = 0.0 if cell in self.known_free else self.unknown_space_weight
            total+=step*(1.0+self._navigation_penalty(cell)+unknown_penalty)
        return total

    def _navigation_penalty(self, cell: GridCell) -> float:
        return (
            self.clearance_cost.get(cell, 0.0)
            + self.centerline_cost.get(cell, 0.0)
            + self.dynamic_clearance_cost.get(cell, 0.0)
        )

    def _turn_clearance_penalty(self, cell: GridCell, turn: float) -> float:
        """Cost separat pentru locul unde corpul își schimbă orientarea.

        Un segment poate încăpea pe lângă un colț, dar rotația umerilor și a
        brațelor cere mai mult spațiu. Penalizarea este continuă: nu închide
        culoarele reale, însă mută virajul spre centrul zonei libere.
        """
        if turn <= 1e-6:
            return 0.0
        desired = max(
            self.robot_radius + 0.10,
            min(TURN_CLEARANCE_RADIUS, self.comfort_radius),
        )
        clearance = self.obstacle_distance.get(cell, math.inf)
        if clearance >= desired:
            return 0.0
        ratio = max(0.0, min(1.0, (desired - clearance) / max(desired, 1e-6)))
        return (
            turn / (math.pi / 4.0)
        ) * TURN_CLEARANCE_WEIGHT * ratio * ratio

    def _cells_clearance_risk(self, cells: List[GridCell]) -> Tuple[float, float]:
        """Întoarce riscul maxim și mediu; un colț periculos nu se pierde în medie."""
        if not cells:
            return 0.0, 0.0
        penalties = [self._navigation_penalty(cell) for cell in cells]
        return max(penalties), sum(penalties) / len(penalties)

    def _shortcut_is_safer_or_equal(self, direct: List[GridCell],
                                     via: List[GridCell]) -> bool:
        """O scurtătură trebuie să păstreze și distanța, nu doar costul total."""
        direct_peak, direct_mean = self._cells_clearance_risk(direct)
        via_peak, via_mean = self._cells_clearance_risk(via)
        return (
            direct_peak <= via_peak + 0.01
            and direct_mean <= via_mean + 0.01
            and self._cells_cost(direct) <= self._cells_cost(via) * 1.005
        )

    def _polyline_cost(self, points: List[Tuple[float, float]]) -> Optional[float]:
        total = 0.0
        for start_xy, end_xy in zip(points, points[1:]):
            cells = self._line_cells(
                self.world_to_cell(*start_xy), self.world_to_cell(*end_xy)
            )
            if cells is None:
                return None
            total += self._cells_cost(cells)
        return total

    def _polyline_cells(self, points: List[Tuple[float, float]]) -> Optional[List[GridCell]]:
        cells: List[GridCell] = []
        for start_xy, end_xy in zip(points, points[1:]):
            segment = self._line_cells(
                self.world_to_cell(*start_xy), self.world_to_cell(*end_xy)
            )
            if segment is None:
                return None
            cells.extend(segment if not cells else segment[1:])
        return cells

    def _simplify_polyline_preserving_risk(
            self, points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Elimină punctele fillet redundante fără a părăsi axa sigură."""
        if len(points) <= 2:
            return points
        reduced = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                direct = self._line_cells(
                    self.world_to_cell(*points[anchor]),
                    self.world_to_cell(*points[candidate]),
                )
                via = self._polyline_cells(points[anchor:candidate + 1])
                if (direct is not None and via is not None
                        and self._shortcut_is_safer_or_equal(direct, via)):
                    break
                candidate -= 1
            reduced.append(points[candidate])
            anchor = candidate
        return reduced

    def _shortcut_preserves_execution_clearance(
            self, direct: List[GridCell], via: List[GridCell],
    ) -> bool:
        """Permite o comandă 1102 lungă când geometria hard rămâne sigură.

        Costul moale al costmapului variază la fiecare celulă și producea
        micro-viraje fără valoare practică. Acceptăm o diferență mai mică decât
        jumătate din rezoluția hărții, dar nu trecem niciodată printr-o celulă
        blocată și nu degradăm marginea dinamică LiDAR.
        """
        if not direct or not via:
            return False
        if self._shortcut_is_safer_or_equal(direct, via):
            return True
        clearance_loss = min(0.05, self.resolution * 0.50)
        direct_clearance = min(
            self.obstacle_distance.get(cell, math.inf) for cell in direct
        )
        via_clearance = min(
            self.obstacle_distance.get(cell, math.inf) for cell in via
        )
        direct_dynamic_peak = max(
            (self.dynamic_clearance_cost.get(cell, 0.0) for cell in direct),
            default=0.0,
        )
        via_dynamic_peak = max(
            (self.dynamic_clearance_cost.get(cell, 0.0) for cell in via),
            default=0.0,
        )
        direct_unknown = sum(cell not in self.known_free for cell in direct) / len(direct)
        via_unknown = sum(cell not in self.known_free for cell in via) / len(via)
        return (
            direct_clearance + clearance_loss >= via_clearance
            and direct_dynamic_peak <= via_dynamic_peak + 0.01
            and direct_unknown <= via_unknown + 0.01
            and self._cells_cost(direct) <= self._cells_cost(via) * 1.12
        )

    def _simplify_polyline_for_execution(
            self, points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Păstrează doar schimbările de direcție necesare pentru siguranță."""
        if len(points) <= 2:
            return points
        reduced = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                direct = self._line_cells(
                    self.world_to_cell(*points[anchor]),
                    self.world_to_cell(*points[candidate]),
                )
                via = self._polyline_cells(points[anchor:candidate + 1])
                if (direct is not None and via is not None
                        and self._shortcut_preserves_execution_clearance(direct, via)):
                    break
                candidate -= 1
            reduced.append(points[candidate])
            anchor = candidate
        return reduced

    @staticmethod
    def _densify_execution_path(
            points: List[Tuple[float, float]],
            maximum_spacing: float = LIVE_ROUTE_WAYPOINT_SPACING,
    ) -> List[Tuple[float, float]]:
        """Împarte numai segmentele lungi; geometria sigură nu este mutată."""
        if len(points) < 2:
            return points
        spacing = max(NATIVE_WAYPOINT_MIN_DISTANCE + 0.05, float(maximum_spacing))
        dense = [points[0]]
        for start, end in zip(points, points[1:]):
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            pieces = max(1, int(math.ceil(length / spacing)))
            for index in range(1, pieces + 1):
                ratio = index / pieces
                dense.append((
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                ))
        return dense

    def _round_safe_corners(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Rotunjește virajele, dar numai în spațiul hard și de confort verificat."""
        if len(points) < 3:
            return points
        rounded = [points[0]]
        for previous, corner, following in zip(points, points[1:-1], points[2:]):
            incoming = (corner[0] - previous[0], corner[1] - previous[1])
            outgoing = (following[0] - corner[0], following[1] - corner[1])
            incoming_length = math.hypot(*incoming)
            outgoing_length = math.hypot(*outgoing)
            if incoming_length < 0.20 or outgoing_length < 0.20:
                rounded.append(corner)
                continue
            incoming_yaw = math.atan2(incoming[1], incoming[0])
            outgoing_yaw = math.atan2(outgoing[1], outgoing[0])
            turn = abs(wrap_angle(outgoing_yaw - incoming_yaw))
            # Deviațiile mici rămân un singur segment; rotunjirea lor producea
            # grupurile de waypoint-uri albe și mersul „șerpuit”.
            if turn < math.radians(25.0) or turn > math.radians(145.0):
                rounded.append(corner)
                continue

            trim = min(0.32, incoming_length * 0.28, outgoing_length * 0.28)
            entry = (
                corner[0] - incoming[0] / incoming_length * trim,
                corner[1] - incoming[1] / incoming_length * trim,
            )
            exit_point = (
                corner[0] + outgoing[0] / outgoing_length * trim,
                corner[1] + outgoing[1] / outgoing_length * trim,
            )
            curve = [entry]
            # Un singur punct median este suficient pentru o curbă sigură.
            # Trei eșantioane intermediare produceau 20-30 de waypoint-uri și
            # obligau API 1102 să ia aceeași decizie de prea multe ori.
            for t in (0.50,):
                one_minus_t = 1.0 - t
                curve.append((
                    one_minus_t * one_minus_t * entry[0]
                    + 2.0 * one_minus_t * t * corner[0]
                    + t * t * exit_point[0],
                    one_minus_t * one_minus_t * entry[1]
                    + 2.0 * one_minus_t * t * corner[1]
                    + t * t * exit_point[1],
                ))
            curve.append(exit_point)
            rounded.extend(curve)
        rounded.append(points[-1])

        deduplicated = [rounded[0]]
        for point in rounded[1:]:
            if math.hypot(point[0] - deduplicated[-1][0], point[1] - deduplicated[-1][1]) >= 0.04:
                deduplicated.append(point)
        simplified = self._simplify_polyline_preserving_risk(deduplicated)
        original_cells = self._polyline_cells(points)
        rounded_cells = self._polyline_cells(simplified)
        if original_cells is None or rounded_cells is None:
            return points
        if not self._shortcut_is_safer_or_equal(rounded_cells, original_cells):
            return points
        return simplified

    def _nearest_free(self, cell: GridCell, max_radius: int = 6) -> Optional[GridCell]:
        if self._valid(cell):
            return cell
        for radius in range(1, max_radius + 1):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    candidates.append((cell[0] + dx, cell[1] + dy))
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    candidates.append((cell[0] + dx, cell[1] + dy))
            for candidate in candidates:
                if self._valid(candidate):
                    return candidate
        return None

    def plan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        self.expire_dynamic_obstacles()
        requested_start = self.world_to_cell(*start_xy)
        start = self._nearest_free(requested_start)
        goal = self.world_to_cell(*goal_xy)
        if start is None:
            raise ValueError("Poziția curentă este în afara zonei libere a hărții")
        if not self._valid(goal):
            snap_cells = max(
                1, math.ceil(float(self.goal_snap_tolerance) / self.resolution)
            )
            snapped = self._nearest_free(goal, snap_cells)
            snapped_xy = self.cell_to_world(snapped) if snapped is not None else None
            if (snapped_xy is None or math.hypot(
                    snapped_xy[0] - goal_xy[0], snapped_xy[1] - goal_xy[1]
            ) > float(self.goal_snap_tolerance) + 1e-9):
                raise ValueError("Ținta este în afara zonei libere sau prea aproape de un obstacol")
            goal = snapped

        # Cea mai simplă rută are prioritate absolută: dacă între A și B există
        # o linie observată, complet liberă și cu o mică rezervă peste raza hard,
        # nu pornim A* și nu urmărim ondulațiile câmpului de confort. Garda live
        # RealSense + LiDAR continuă să poată opri/replanifica acest segment.
        direct_cells = self._line_cells(start, goal)
        if direct_cells:
            minimum_clearance = min(
                (self.obstacle_distance.get(cell, math.inf) for cell in direct_cells),
                default=math.inf,
            )
            direct_margin = min(0.05, self.resolution * 0.50)
            direct_is_observed = all(cell in self.known_free for cell in direct_cells)
            direct_has_dynamic_risk = any(
                self.dynamic_clearance_cost.get(cell, 0.0) > 0.0
                for cell in direct_cells
            )
            if (direct_is_observed
                    and not direct_has_dynamic_risk
                    and minimum_clearance >= self.robot_radius + direct_margin):
                # O dreaptă liberă are exact A și B. Intersecția cu LiDAR-ul
                # rasterizează oricum întregul segment; punctele coliniare nu
                # aduc siguranță, dar produc marcaje și comenzi redundante.
                return [self.cell_to_world(start), self.cell_to_world(goal)]

        frontier: List[Tuple[float, GridCell]] = [(0.0, start)]
        came_from: Dict[GridCell, Optional[GridCell]] = {start: None}
        cost: Dict[GridCell, float] = {start: 0.0}
        neighbors = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                     (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                     (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dx, dy, step_cost in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._valid(nxt):
                    continue
                # Nu tăiem colțul printre două celule ocupate.
                if dx and dy and (not self._valid((current[0] + dx, current[1]))
                                  or not self._valid((current[0], current[1] + dy))):
                    continue
                unknown_penalty = 0.0 if nxt in self.known_free else self.unknown_space_weight
                new_cost = cost[current] + step_cost*(
                    1.0+self._navigation_penalty(nxt)+unknown_penalty
                )
                previous = came_from.get(current)
                if previous is not None:
                    old_heading = math.atan2(
                        current[1] - previous[1], current[0] - previous[0]
                    )
                    new_heading = math.atan2(dy, dx)
                    turn = abs(wrap_angle(new_heading - old_heading))
                    if turn > 1e-6:
                        nearby_risk = max(
                            self._navigation_penalty(current),
                            self._navigation_penalty(nxt),
                        )
                        # Evită zig-zag-ul și, mai ales, rotația umărului lângă
                        # muchia mesei. În spațiu liber penalizarea rămâne mică.
                        new_cost += (
                            (turn / (math.pi / 4.0)) * (
                                0.10 + 0.55 * nearby_risk
                            )
                            + self._turn_clearance_penalty(current, turn)
                        )
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    priority = new_cost + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current
        if goal not in came_from:
            raise ValueError("Nu există un traseu liber până la țintă")

        cells: List[GridCell] = []
        node: Optional[GridCell] = goal
        while node is not None:
            cells.append(node)
            node = came_from[node]
        cells.reverse()
        # String-pulling păstrează avantajul de clearance găsit de A*: o
        # diagonală este acceptată numai dacă nu este mai scumpă prin zona de confort.
        reduced = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            candidate = len(cells) - 1
            while candidate > anchor + 1:
                direct_cells=self._line_cells(cells[anchor],cells[candidate])
                original_cells=cells[anchor:candidate+1]
                if direct_cells is not None and self._shortcut_is_safer_or_equal(
                    direct_cells, original_cells
                ):
                    break
                candidate-=1
            reduced.append(cells[candidate])
            anchor = candidate
        rounded = self._round_safe_corners([
            self.cell_to_world(cell) for cell in reduced
        ])
        # Păstrăm numai colțurile relevante. Verificarea live rasterizează
        # fiecare segment complet, deci nu are nevoie de puncte artificiale la
        # fiecare 0,85 m pentru a observa un obstacol apărut între două colțuri.
        return self._simplify_polyline_for_execution(rounded)


def plan_pcd_route(map_path: str, start_xy: Tuple[float, float],
                   goal_xy: Tuple[float, float],
                   dynamic_obstacle: Optional[Tuple[float, float, float]] = None,
                   resolution: float = 0.15, robot_radius: float = NORMAL_CLEARANCE_RADIUS,
                   min_obstacle_points: int = 3, obstacle_min_z: float = 0.15,
                   obstacle_max_z: float = 1.60, level_to_floor: bool = True,
                   floor_tolerance: float = 0.08, comfort_radius: Optional[float] = None,
                   clearance_weight: float = 2.5, allow_narrow_fallback: bool = True):
    """Alege ruta sigură, dar permite profilul de coridor când e mult mai simplu.

    Profilul de 0,20 m rămâne o limită hard, nu ignoră obstacole. Îl preferăm
    doar dacă elimină cel puțin două schimbări de direcție și nu lungește ruta;
    astfel un culoar real devine o comandă lungă 1102 în locul unui zig-zag.
    """
    errors = []
    solutions = []
    modes=[("custom",robot_radius)]
    if allow_narrow_fallback:
        if robot_radius < SAFE_CLEARANCE_RADIUS:
            # Profilul de 0,20 m rămâne disponibil pentru uși și culoare reale,
            # dar folosim 0,25 m oriunde există spațiu.
            modes=[("safe",SAFE_CLEARANCE_RADIUS),("narrow",robot_radius)]
        elif robot_radius>NARROW_CLEARANCE_RADIUS:
            modes=[("safe",robot_radius),("narrow",NARROW_CLEARANCE_RADIUS)]
    for clearance_mode, active_radius in modes:
        planner = PCDGridPlanner(
            resolution=resolution,robot_radius=active_radius,min_obstacle_points=min_obstacle_points,
            comfort_radius=comfort_radius,clearance_weight=clearance_weight
        )
        try:
            planner.goal_snap_tolerance = GOAL_SNAP_TOLERANCE
            planner.load(map_path,obstacle_min_z=obstacle_min_z,obstacle_max_z=obstacle_max_z,
                         level_to_floor=level_to_floor,floor_tolerance=floor_tolerance,
                         clear_xy=start_xy,clear_radius=0.40)
            planner.clear_robot_footprint(*start_xy)
            if dynamic_obstacle:
                planner.add_dynamic_obstacle(*dynamic_obstacle)
            path = planner.plan(start_xy, goal_xy)
            solutions.append((planner, path, clearance_mode))
        except Exception as exc:
            errors.append(f"{clearance_mode}: {exc}")
    if not solutions:
        raise ValueError("; ".join(errors))
    if len(solutions) >= 2:
        safe_solution = solutions[0]
        narrow_solution = next(
            (solution for solution in solutions if solution[2] == "narrow"), None
        )
        if narrow_solution is not None:
            safe_path = safe_solution[1]
            narrow_path = narrow_solution[1]

            def route_length(route):
                return sum(
                    math.hypot(end[0] - start[0], end[1] - start[1])
                    for start, end in zip(route, route[1:])
                )

            safe_length = route_length(safe_path)
            narrow_length = route_length(narrow_path)
            if (
                narrow_length <= safe_length * 0.90
                or (
                    safe_length - narrow_length >= 0.60
                    and narrow_length <= safe_length * 1.01
                )
                or (
                    len(safe_path) >= 5
                    and len(narrow_path) <= len(safe_path) - 2
                    and narrow_length <= safe_length * 1.03
                )
            ):
                return narrow_solution
    return solutions[0]


class NativeWaypointNavigator:
    """A* global + replanificare dinamică, executate ca segmente native 1102."""

    def __init__(self, pose_provider: Callable[[], dict], localization_ok: Callable[[], bool],
                 obstacle_guard, send_waypoint: Callable[..., Awaitable[dict]],
                 pause_navigation: Callable[[], Awaitable[dict]],
                 resume_navigation: Callable[[], Awaitable[dict]],
                 event_callback: Callable[[dict], Awaitable[None]],
                 obstacle_wait_before_replan: float = OBSTACLE_WAIT_BEFORE_REPLAN,
                 obstacle_clear_stable: float = OBSTACLE_CLEAR_STABLE,
                 sensor_loss_timeout: float = OBSTACLE_SENSOR_LOSS_TIMEOUT,
                 dynamic_obstacle_ttl: float = DYNAMIC_OBSTACLE_TTL,
                 poll_interval: float = 0.05,
                 navigation_paused: Optional[Callable[[], bool]] = None,
                 lateral_velocity: Optional[Callable[[float], Awaitable[dict]]] = None,
                 stop_locomotion: Optional[Callable[[], Awaitable[dict]]] = None,
                 enable_stagnation_lateral_recovery: bool = False,
                 stagnation_timeout: float = 4.0,
                 max_waypoint_retries: int = 1,
                 waypoint_completed: Optional[
                     Callable[[float, float, float], bool]
                 ] = None):
        self.pose_provider = pose_provider
        self.localization_ok = localization_ok
        self.obstacle_guard = obstacle_guard
        self.send_waypoint = send_waypoint
        self.pause_navigation = pause_navigation
        self.resume_navigation = resume_navigation
        self.event_callback = event_callback
        self.obstacle_wait_before_replan = max(0.0, float(obstacle_wait_before_replan))
        self.obstacle_clear_stable = max(0.0, float(obstacle_clear_stable))
        self.sensor_loss_timeout = max(0.1, float(sensor_loss_timeout))
        self.dynamic_obstacle_ttl = max(0.1, float(dynamic_obstacle_ttl))
        self.poll_interval = max(0.01, float(poll_interval))
        self.navigation_paused = navigation_paused
        self.lateral_velocity = lateral_velocity
        self.stop_locomotion = stop_locomotion
        self.enable_stagnation_lateral_recovery = bool(
            enable_stagnation_lateral_recovery
        )
        self.stagnation_timeout = max(0.1, float(stagnation_timeout))
        self.max_waypoint_retries = max(0, int(max_waypoint_retries))
        self.waypoint_completed = waypoint_completed
        self.last_lateral_escape_at = -math.inf
        self.last_lateral_direction = 0
        self.route_blocked_since: Optional[float] = None
        self.recent_obstacle_info: Optional[dict] = None
        self.recent_obstacle_at = -math.inf
        self.recent_obstacle_pose: Optional[dict] = None
        self.staged_waypoint: Optional[dict] = None
        self.task: Optional[asyncio.Task] = None
        self.cancel_requested = False
        self.planner: Optional[PCDGridPlanner] = None
        self.replan_confirmation_id: Optional[str] = None
        self.replan_confirmation_event: Optional[asyncio.Event] = None
        self.status = {"state": "idle", "path": [], "goal": None, "error": "", "driver": "astar_waypoints"}

    async def _wait_for_localization_recovery(
            self, navigation_deadline: float, already_paused: bool = False,
            timeout: float = 4.0,
    ) -> None:
        """Oprește imediat, apoi separă un gol de telemetrie de pierderea reală.

        Trei verificări consecutive sunt necesare înainte de reluare. Astfel nu
        continuăm orbește pe o singură poziție reapărută și nici nu declarăm
        cursa eșuată din cauza unui singur pachet pos_info lipsă.
        """
        if not already_paused:
            pause_result = await self.pause_navigation()
            if isinstance(pause_result, dict) and pause_result.get("success") is False:
                raise RuntimeError(
                    pause_result.get("error", "API 1201 nu a confirmat oprirea")
                )
        started = time.monotonic()
        stable_samples = 0
        self.status.update({
            "state": "waiting_localization",
            "error": "robot oprit: verific localizarea nativă și odometria de rezervă",
        })
        await self._emit()
        while not self.cancel_requested:
            now = time.monotonic()
            if now > navigation_deadline:
                raise RuntimeError("timeout de navigare în așteptarea localizării")
            if self.localization_ok():
                stable_samples += 1
                if stable_samples >= 3:
                    self.status.update({
                        "state": "navigating",
                        "error": "localizarea a fost reconfirmată; continui ruta",
                        "localization_recoveries": int(
                            self.status.get("localization_recoveries", 0) or 0
                        ) + 1,
                    })
                    await self._emit()
                    return
            else:
                stable_samples = 0
            if now - started >= timeout:
                raise RuntimeError(
                    "localizarea nativă nu a revenit stabil după STOP; "
                    "cursa rămâne oprită în siguranță"
                )
            await asyncio.sleep(self.poll_interval)

    async def start(self, map_path: str, x: float, y: float, yaw: float,
                    speed: float = 0.3, timeout: float = 0.0,
                    prepared_plan: Optional[dict] = None,
                    motion_profile: str = "stable") -> dict:
        if self.task and not self.task.done():
            return {"success": False, "error": "Există deja o navigare activă"}
        if not self.localization_ok():
            return {"success": False, "error": "Poziția SLAM Unitree nu este disponibilă"}
        if not navigation_sensors_ready(self.obstacle_guard):
            return {
                "success": False,
                "error": "Nu există date recente de la RealSense/LiDAR pentru evitarea obstacolelor",
            }
        pose = self.pose_provider()
        if prepared_plan:
            planner = prepared_plan["planner"]
            path = list(prepared_plan["path"])
            clearance_mode = prepared_plan.get("clearance_mode", "normal")
            path_pattern = dict(
                prepared_plan.get("path_pattern") or {"type": "astar"}
            )
        else:
            try:
                planner, path, clearance_mode = await asyncio.to_thread(
                    plan_pcd_route, map_path, (pose["x"], pose["y"]), (x, y)
                )
            except Exception as exc:
                return {"success": False, "error": f"Planificarea A* a eșuat: {exc}"}
            path_pattern = {"type": "astar"}
        self.planner = planner
        self.staged_waypoint = None
        if hasattr(self.obstacle_guard, "configure_navigation_map"):
            self.obstacle_guard.configure_navigation_map(
                planner.floor_plane,
                planner.resolution,
                planner.raw_static_occupied,
                planner.robot_radius,
                planner.obstacle_min_z,
                planner.obstacle_max_z,
            )
        path = self._stabilize_departure_path(path, pose)
        if not prepared_plan:
            path, path_pattern = self._holonomic_direct_pattern(path, pose)
        self.cancel_requested = False
        self.status = {
            "state": "planning", "path": path,
            "goal": {"x": x, "y": y, "yaw": yaw, "speed": speed},
            "pose": pose, "error": "", "driver": "astar_waypoints",
            "waypoint_index": 0, "replans": 0, "clearance_mode": clearance_mode,
            "motion_profile": motion_profile,
            "path_pattern": path_pattern,
            "path_reference_yaw": path_pattern.get("reference_yaw"),
            "native_command_active": False,
        }
        self.task = asyncio.create_task(
            self._run(path, x, y, yaw, speed, timeout, motion_profile)
        )
        await self._emit()
        return {
            "success": True, "state": "planning", "path": path,
            "waypoints": len(path), "clearance_mode": clearance_mode,
            "path_pattern": path_pattern,
        }

    async def stop(self, reason: str = "oprit de utilizator") -> dict:
        self.cancel_requested = True
        if self.replan_confirmation_event:
            self.replan_confirmation_event.set()
        result = await self.pause_navigation()
        if self.task and not self.task.done():
            self.task.cancel()
        if self.planner:
            self.planner.clear_dynamic_costmap()
        self.staged_waypoint = None
        self.status.update({"state": "paused", "error": reason})
        await self._emit()
        return result

    async def confirm_replan(self, confirmation_id: str) -> dict:
        return {
            "success": False,
            "error": "Ocolirile sunt decise și executate automat; confirmarea nu mai este necesară",
        }

    async def reject_replan(self, confirmation_id: str) -> dict:
        return {
            "success": False,
            "error": "Ocolirile sunt automate; folosește STOP navigare pentru oprire",
        }

    async def _emit(self) -> None:
        dynamic_costmap = self.planner.dynamic_costmap_points() if self.planner else []
        await self.event_callback({
            "type": "nav_status", **self.status,
            "dynamic_costmap": dynamic_costmap,
        })

    async def _send_waypoint_guarded(
            self, x: float, y: float, yaw: float, speed: float,
            path: List[Tuple[float, float]], waypoint_index: int,
    ) -> Tuple[dict, Optional[str]]:
        """Monitorizează siguranța și cât timp API 1102 așteaptă feedback.

        Publicarea poate pune robotul în mișcare înainte ca răspunsul 1102 să
        ajungă. Fără această buclă exista o fereastră oarbă de până la 4 secunde.
        """
        dispatch_task = asyncio.create_task(self.send_waypoint(x, y, yaw, speed))
        hazard: Optional[str] = None
        sensor_stale_since: Optional[float] = None
        check_interval = min(0.05, self.poll_interval)
        while not dispatch_task.done() and not self.cancel_requested:
            done, _ = await asyncio.wait({dispatch_task}, timeout=check_interval)
            if done:
                break
            if not self.localization_ok():
                hazard = "localization"
            elif not navigation_sensors_ready(self.obstacle_guard):
                sensor_stale_since = sensor_stale_since or time.monotonic()
                if time.monotonic() - sensor_stale_since >= SENSOR_GLITCH_GRACE:
                    hazard = "sensor"
            else:
                sensor_stale_since = None
                pose = self._pose()
                self.status["pose"] = pose
                observed_at = time.monotonic()
                segment_blocked, _, _, obstacle_info, _ = self._refresh_route_obstacles(
                    path, waypoint_index, pose, observed_at
                )
                if obstacle_info:
                    self.status["dynamic_obstacle"] = obstacle_info
                else:
                    self.status["dynamic_obstacle"] = None
                if self._confirmed_segment_blocked(
                    segment_blocked, obstacle_info, observed_at
                ):
                    hazard = "obstacle"
            if hazard:
                pause_result = await self.pause_navigation()
                if isinstance(pause_result, dict) and pause_result.get("success") is False:
                    dispatch_task.cancel()
                    return pause_result, "pause_failed"
                state = "waiting_obstacle" if hazard == "obstacle" else "waiting_sensor"
                reason = {
                    "obstacle": "OPRIT în timpul confirmării 1102: obstacol detectat",
                    "sensor": "OPRIT în timpul confirmării 1102: flux de siguranță pierdut",
                    "localization": "OPRIT în timpul confirmării 1102: localizare pierdută",
                }[hazard]
                self.status.update({"state": state, "error": reason})
                await self._emit()
                break

        # Nu anulăm taskul după publicare: thread-ul DDS poate continua oricum;
        # colectăm feedback-ul, dar robotul este deja ținut pe pauză de 1201.
        result = await dispatch_task
        return result, hazard

    def _pose(self) -> dict:
        pose = self.pose_provider()
        return {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose.get("yaw", 0.0))}

    def _front_obstacle_position(self, pose: dict) -> Tuple[float, float, float]:
        if hasattr(self.obstacle_guard, "front_obstacle_vector"):
            forward, left = self.obstacle_guard.front_obstacle_vector(default=0.70)
        else:
            # Compatibilitate cu surse/fake-uri care furnizează doar distanța.
            forward = self.obstacle_guard.front_obstacle_distance(default=0.70)
            left = 0.0
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        distance = math.hypot(forward, left)
        return (
            pose["x"] + c * forward - s * left,
            pose["y"] + s * forward + c * left,
            distance,
        )

    def _native_reports_paused(self) -> bool:
        if not self.navigation_paused:
            return False
        try:
            return bool(self.navigation_paused())
        except Exception:
            return False

    def _navigation_blocked(self) -> bool:
        if hasattr(self.obstacle_guard, "is_navigation_blocked"):
            return bool(self.obstacle_guard.is_navigation_blocked())
        return bool(self.obstacle_guard.is_blocked(0.2, 0.0))

    def _lateral_direction_clear(self, direction: int) -> bool:
        if hasattr(self.obstacle_guard, "is_lateral_clear"):
            return bool(self.obstacle_guard.is_lateral_clear(direction))
        # Fake-uri/implementări vechi: verifică strict comanda laterală.
        if hasattr(self.obstacle_guard, "is_blocked"):
            return not bool(self.obstacle_guard.is_blocked(0.0, 0.12 * direction))
        return True

    def _sync_lidar_costmap(self, observed_at: Optional[float] = None):
        if not self.planner or not hasattr(self.obstacle_guard, "front_obstacle_shape"):
            return None
        shape = self.obstacle_guard.front_obstacle_shape()
        if shape and shape.get("points"):
            # Păstrăm uniunea temporală a cadrelor până la TTL. Înlocuirea
            # completă la fiecare scan făcea marginea opusă a unui scaun să
            # dispară când robotul se rotea, iar următorul A* tăia prin ea.
            self.planner.add_dynamic_points(
                shape["points"],
                inflation_radius=max(
                    LIVE_OBSTACLE_MIN_CLEARANCE,
                    self.planner.robot_radius + DYNAMIC_SENSOR_PADDING,
                ),
                observed_at=(time.monotonic() if observed_at is None else observed_at),
                source="lidar",
            )
            return shape
        # Un cadru LiDAR lipsă nu șterge instantaneu scaunul. Celulele rămân
        # până la TTL și sunt eliminate numai dacă observația nu revine.
        return None

    def _current_segment_blocked(self, path: List[Tuple[float, float]],
                                 waypoint_index: int, pose: dict) -> bool:
        if not self.planner or not path or waypoint_index >= len(path):
            return False
        return not self.planner.segment_is_free(
            (pose["x"], pose["y"]), path[waypoint_index]
        )

    def _remaining_route_blocked(self, path: List[Tuple[float, float]],
                                 waypoint_index: int, pose: dict) -> bool:
        """Verifică întreaga rută rămasă față de snapshotul LiDAR curent."""
        if not self.planner or not path or waypoint_index >= len(path):
            return False
        remaining = [(pose["x"], pose["y"]), *path[waypoint_index:]]
        return any(
            not self.planner.segment_is_free(start, end)
            for start, end in zip(remaining, remaining[1:])
        )

    def _confirmed_segment_blocked(self, blocked: bool, obstacle_info: Optional[dict],
                                   observed_at: Optional[float] = None) -> bool:
        """Filtrează zgomotul live fără să întârzie obstacolele apropiate."""
        now = time.monotonic() if observed_at is None else float(observed_at)
        if not blocked:
            self.route_blocked_since = None
            return False
        distance = float((obstacle_info or {}).get("distance", math.inf))
        if distance <= 0.50:
            self.route_blocked_since = now
            return True
        self.route_blocked_since = self.route_blocked_since or now
        return now - self.route_blocked_since >= ROUTE_OBSTACLE_CONFIRMATION

    def _refresh_route_obstacles(
            self, path: List[Tuple[float, float]], waypoint_index: int,
            pose: dict, observed_at: Optional[float] = None,
    ) -> Tuple[bool, bool, Optional[dict], Optional[dict], Optional[tuple]]:
        """Actualizează costmapul și decide după segment, nu global după senzor.

        Un obstacol frontal trebuie să blocheze mersul prin el, dar nu și un
        segment lateral/diagonal deja verificat ca liber de costmap.
        """
        timestamp = time.monotonic() if observed_at is None else observed_at
        # Curățăm înainte de interogarea segmentului. În vechea ordine, o
        # celulă expirată putea declara ruta blocată încă un ciclu și pornea
        # inutil STOP + replănuire.
        self.planner.expire_dynamic_obstacles(
            self.dynamic_obstacle_ttl, now=timestamp
        )
        live_shape = self._sync_lidar_costmap(timestamp)
        sensor_blocked = self._navigation_blocked()
        obstacle_info = None
        fallback = None
        if sensor_blocked:
            obstacle_info, fallback = self._record_obstacle(pose, timestamp)
            if obstacle_info and obstacle_info.get("mode") == "lidar_shape":
                self.recent_obstacle_info = dict(obstacle_info)
                self.recent_obstacle_at = timestamp
                self.recent_obstacle_pose = dict(pose)
        else:
            self.planner.clear_dynamic_source("camera")
        # Nu așteptăm ca obstacolul să ajungă în segmentul imediat următor.
        # Dacă intersectează orice porțiune rămasă, ruta este invalidată și A*
        # o reconstruiește din poziția live.
        segment_blocked = self._remaining_route_blocked(
            path, waypoint_index, pose
        )
        camera_stop_only = bool(
            sensor_blocked and obstacle_info
            and obstacle_info.get("mode") == "camera_stop_only"
        )
        return (
            segment_blocked or camera_stop_only, sensor_blocked, live_shape,
            obstacle_info, fallback,
        )

    def _recent_obstacle_for_recovery(self, pose: dict,
                                      observed_at: Optional[float] = None) -> Optional[dict]:
        """Păstrează scaunul văzut înainte să intre în unghiul mort apropiat."""
        now = time.monotonic() if observed_at is None else float(observed_at)
        if (not self.recent_obstacle_info or not self.recent_obstacle_pose
                or now - self.recent_obstacle_at > RECOVERY_OBSTACLE_MEMORY):
            return None
        if math.hypot(
            pose["x"] - self.recent_obstacle_pose["x"],
            pose["y"] - self.recent_obstacle_pose["y"],
        ) > RECOVERY_OBSTACLE_MAX_ROBOT_TRAVEL:
            return None
        return dict(self.recent_obstacle_info)

    def _record_obstacle(self, pose: dict, observed_at: float):
        """Actualizează costmapul numai din LiDAR sau camera de urgență."""
        shape = (
            self.obstacle_guard.front_obstacle_shape()
            if hasattr(self.obstacle_guard, "front_obstacle_shape") else None
        )
        if shape and shape.get("points"):
            self.planner.clear_dynamic_source("camera")
            points = list(shape["points"])
            self.planner.add_dynamic_points(
                points,
                inflation_radius=max(
                    LIVE_OBSTACLE_MIN_CLEARANCE,
                    self.planner.robot_radius + DYNAMIC_SENSOR_PADDING,
                ),
                observed_at=observed_at,
                source="lidar",
            )
            vector = shape.get("vector")
            if vector:
                forward, left = float(vector[0]), float(vector[1])
                cosine, sine = math.cos(pose["yaw"]), math.sin(pose["yaw"])
                obstacle_x = pose["x"] + cosine * forward - sine * left
                obstacle_y = pose["y"] + sine * forward + cosine * left
                distance = math.hypot(forward, left)
            else:
                obstacle_x = sum(point[0] for point in points) / len(points)
                obstacle_y = sum(point[1] for point in points) / len(points)
                distance = math.hypot(obstacle_x - pose["x"], obstacle_y - pose["y"])
            return {
                "x": obstacle_x, "y": obstacle_y, "distance": distance,
                "mode": "lidar_shape", "shape_points": len(points),
            }, None

        obstacle_x, obstacle_y, distance = self._front_obstacle_position(pose)
        # Camera păstrează rolul de frână redundantă pentru corp și brațe, dar
        # nu furnizează geometrie suficientă pentru ocolire. Nu mai proiectăm
        # un cerc artificial în hartă; numai forma LiDAR intră în A*.
        self.planner.clear_dynamic_source("camera")
        return {
            "x": obstacle_x, "y": obstacle_y, "distance": distance,
            "mode": "camera_stop_only", "shape_points": 0,
        }, None

    @staticmethod
    def _turn_angle(path: List[Tuple[float, float]], waypoint_index: int) -> float:
        if waypoint_index <= 0 or waypoint_index >= len(path) - 1:
            return 0.0
        previous, current, following = (
            path[waypoint_index - 1], path[waypoint_index], path[waypoint_index + 1]
        )
        incoming = math.atan2(current[1] - previous[1], current[0] - previous[0])
        outgoing = math.atan2(following[1] - current[1], following[0] - current[0])
        return abs(wrap_angle(outgoing - incoming))

    def _handoff_distance(self, path: List[Tuple[float, float]],
                          waypoint_index: int, speed: float) -> float:
        """Trimite anticipat următorul 1102 numai pe direcții aproape drepte."""
        if waypoint_index >= len(path) - 1:
            return 0.22
        if self.status.get("path_pattern", {}).get("type") == "forward_then_lateral":
            # Schimbăm ținta înainte de oprirea completă la cotul pattern-ului.
            return 0.32
        turn = self._turn_angle(path, waypoint_index)
        if turn <= math.radians(20.0):
            return min(0.55, max(0.38, 0.28 + speed * 0.55))
        if turn <= math.radians(35.0):
            return 0.30
        return 0.22

    def _can_smooth_handoff(self, path: List[Tuple[float, float]], waypoint_index: int,
                            pose: dict) -> bool:
        if waypoint_index >= len(path) - 1:
            return False
        holonomic = (
            self.status.get("path_pattern", {}).get("type")
            == "forward_then_lateral"
        )
        maximum_turn = math.radians(100.0 if holonomic else 35.0)
        if self._turn_angle(path, waypoint_index) > maximum_turn:
            return False
        turn_cell = self.planner.world_to_cell(*path[waypoint_index])
        desired_clearance = max(
            self.planner.robot_radius + 0.08,
            min(TURN_CLEARANCE_RADIUS, self.planner.comfort_radius),
        )
        if self.planner.obstacle_distance.get(turn_cell, math.inf) < desired_clearance:
            return False
        if holonomic:
            return self.planner.segment_is_free(
                path[waypoint_index], path[waypoint_index + 1]
            )
        return self.planner.smooth_handoff_is_safe(
            (pose["x"], pose["y"]),
            path[waypoint_index],
            path[waypoint_index + 1],
        )

    def _advance_past_native_tolerance(
            self, path: List[Tuple[float, float]], waypoint_index: int,
            pose: dict,
    ) -> int:
        """API 1102 declară uneori FINISHED pentru ținte locale sub ~0,5 m.

        Fuzionăm doar punctele apropiate pentru care scurtătura către următorul
        waypoint păstrează clearance-ul calculat de planner.
        """
        index = waypoint_index
        while index < len(path) - 1:
            distance = math.hypot(
                path[index][0] - pose["x"], path[index][1] - pose["y"]
            )
            if distance >= NATIVE_WAYPOINT_MIN_DISTANCE:
                break
            if not self._can_smooth_handoff(path, index, pose):
                break
            index += 1
        return index

    @staticmethod
    def _native_completion_has_progress(
            initial_distance: float, current_distance: float,
            best_distance: float,
    ) -> bool:
        """Nu acceptă FINISHED-ul unei comenzi vechi pentru o țintă apropiată."""
        closest = min(float(current_distance), float(best_distance))
        return bool(
            current_distance <= 0.30
            or (
                closest <= 0.60
                and initial_distance - closest >= 0.15
            )
        )

    def _select_control_waypoint_index(
            self, path: List[Tuple[float, float]], waypoint_index: int,
            pose: dict,
    ) -> int:
        """Comprimă numai execuția 1102, păstrând polilinia live completă.

        Punctele dese sunt utile pentru afișare și pentru intersecția cu
        obstacole dinamice, dar trimiterea fiecăruia către firmware producea
        STOP + rotație + o nouă estimare yaw. Alegem cel mai îndepărtat punct
        vizibil în fereastra de lookahead numai dacă scurtătura păstrează atât
        spațiul hard, cât și profilul de clearance al traseului A*.
        """
        if (not self.planner or not path
                or waypoint_index >= len(path) - 1):
            return waypoint_index
        start = (float(pose["x"]), float(pose["y"]))
        selected = waypoint_index
        for candidate in range(waypoint_index + 1, len(path)):
            target = path[candidate]
            if math.hypot(target[0] - start[0], target[1] - start[1]) > NATIVE_CONTROL_LOOKAHEAD:
                break
            if not self.planner.segment_is_free(start, target):
                break
            # Planner-ele simplificate din teste/compatibilitate nu expun
            # rasterizarea internă; segment_is_free rămâne fallback-ul sigur.
            if all(hasattr(self.planner, name) for name in (
                    "_line_cells", "_polyline_cells",
                    "_shortcut_preserves_execution_clearance")):
                direct = self.planner._line_cells(
                    self.planner.world_to_cell(*start),
                    self.planner.world_to_cell(*target),
                )
                via = self.planner._polyline_cells(
                    [start, *path[waypoint_index:candidate + 1]]
                )
                if (direct is None or via is None
                        or not self.planner._shortcut_preserves_execution_clearance(
                            direct, via
                        )):
                    break
            selected = candidate
        return selected

    def _stabilize_departure_path(
            self, path: List[Tuple[float, float]], pose: dict,
    ) -> List[Tuple[float, float]]:
        """Prelungește primul segment prea scurt, fără a-i schimba direcția."""
        if not self.planner or len(path) < 3:
            return path
        current = path[1]
        dx, dy = current[0] - pose["x"], current[1] - pose["y"]
        distance = math.hypot(dx, dy)
        if distance >= NATIVE_WAYPOINT_MIN_DISTANCE or distance < 0.08:
            return path
        old_first = self.planner._line_cells(
            self.planner.world_to_cell(pose["x"], pose["y"]),
            self.planner.world_to_cell(*current),
        )
        old_second = self.planner._line_cells(
            self.planner.world_to_cell(*current),
            self.planner.world_to_cell(*path[2]),
        )
        if old_first is None or old_second is None:
            return path
        first_peak, first_mean = self.planner._cells_clearance_risk(old_first)
        second_peak, second_mean = self.planner._cells_clearance_risk(old_second)
        unit_x, unit_y = dx / distance, dy / distance
        for target_distance in (STARTUP_WAYPOINT_DISTANCE, 0.68, 0.65):
            candidate = (
                pose["x"] + unit_x * target_distance,
                pose["y"] + unit_y * target_distance,
            )
            new_first = self.planner._line_cells(
                self.planner.world_to_cell(pose["x"], pose["y"]),
                self.planner.world_to_cell(*candidate),
            )
            new_second = self.planner._line_cells(
                self.planner.world_to_cell(*candidate),
                self.planner.world_to_cell(*path[2]),
            )
            if new_first is None or new_second is None:
                continue
            new_first_peak, new_first_mean = self.planner._cells_clearance_risk(new_first)
            new_second_peak, new_second_mean = self.planner._cells_clearance_risk(new_second)
            if (new_first_peak <= first_peak + 0.08
                    and new_first_mean <= first_mean + 0.08
                    and new_second_peak <= second_peak + 0.08
                    and new_second_mean <= second_mean + 0.08):
                stabilized = list(path)
                stabilized[1] = candidate
                return stabilized
        return path

    def _holonomic_direct_pattern(
            self, path: List[Tuple[float, float]], pose: dict,
    ) -> Tuple[List[Tuple[float, float]], dict]:
        """Transformă o diagonală mică în înainte + translație laterală.

        Se aplică numai unei rute directe și numai dacă ambele laturi ale
        pattern-ului sunt libere. Pentru diagonale mari sau zone strâmte
        păstrăm ruta originală/A*, deoarece o descompunere în L ar fi mai
        lungă și ar putea introduce o apropiere inutilă de obstacole.
        """
        default = {"type": "direct" if len(path) == 2 else "astar"}
        if not self.planner or len(path) != 2:
            return path, default
        start, goal = path
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        cosine, sine = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        forward = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        angle = abs(math.atan2(lateral, max(1e-9, forward)))
        if (forward < 0.80 or abs(lateral) < 0.18 or abs(lateral) > 0.75
                or angle > math.radians(35.0)):
            return path, default
        elbow = (
            start[0] + cosine * forward,
            start[1] + sine * forward,
        )
        if (not self.planner.segment_is_free(start, elbow)
                or not self.planner.segment_is_free(elbow, goal)):
            return path, default
        elbow_cell = self.planner.world_to_cell(*elbow)
        minimum_corner_clearance = self.planner.robot_radius + 0.10
        if self.planner.obstacle_distance.get(elbow_cell, math.inf) < minimum_corner_clearance:
            return path, default
        return [start, elbow, goal], {
            "type": "forward_then_lateral",
            "reference_yaw": float(pose["yaw"]),
            "lateral_direction": "left" if lateral > 0.0 else "right",
            "forward_m": round(forward, 3),
            "lateral_m": round(abs(lateral), 3),
        }

    def _safe_segment_speed(self, path: List[Tuple[float, float]], waypoint_index: int,
                            pose: dict, requested_speed: float) -> float:
        """Viteză adaptivă: rapid în liber, lent la colț, confort sau necunoscut."""
        factor = 1.0
        if self.status.get("clearance_mode") == "narrow":
            # În fallback-ul de culoar permitem apropierea moderată, dar nu și
            # traversarea lui cu viteza folosită în spațiul larg.
            factor = min(factor, 0.55)
        turn = self._turn_angle(path, waypoint_index)
        if turn > math.radians(70.0):
            factor = min(factor, 0.50)
        elif turn > math.radians(35.0):
            factor = min(factor, 0.72)

        target = path[waypoint_index]
        cells = self.planner._line_cells(
            self.planner.world_to_cell(pose["x"], pose["y"]),
            self.planner.world_to_cell(*target),
        )
        if cells:
            maximum_clearance_penalty = max(
                (self.planner._navigation_penalty(cell) for cell in cells),
                default=0.0,
            )
            if maximum_clearance_penalty > 0.0:
                factor = min(factor, max(0.45, 1.0 / (1.0 + 0.22 * maximum_clearance_penalty)))
            if any(cell not in self.planner.known_free for cell in cells):
                factor = min(factor, 0.45)
        return max(0.12, min(float(requested_speed), float(requested_speed) * factor))

    @staticmethod
    def _lateral_motion_mode(path: List[Tuple[float, float]], waypoint_index: int,
                             pose: dict) -> Optional[str]:
        """Detectează un segment sigur care cere pas lateral/diagonal G1."""
        if not path or waypoint_index >= len(path):
            return None
        target = path[waypoint_index]
        dx, dy = target[0] - pose["x"], target[1] - pose["y"]
        distance = math.hypot(dx, dy)
        if distance < 0.08:
            return None
        heading = math.atan2(dy, dx)
        error = wrap_angle(heading - pose["yaw"])
        if math.radians(40.0) <= abs(error) <= math.radians(140.0):
            return "lateral_left" if error > 0.0 else "lateral_right"
        return None

    @staticmethod
    def _departure_lateral_mode(
            path: List[Tuple[float, float]], waypoint_index: int, pose: dict,
    ) -> Optional[str]:
        """Alege o singură repoziționare laterală înaintea unei drepte lungi.

        Nu folosim manevra la colțurile A*: numai o rută start→goal cu două
        puncte poate declanșa pasul. Unghiurile foarte mici merg direct, iar
        cele mari se rezolvă prin orientarea normală a controllerului.
        """
        if len(path) != 2 or waypoint_index != 1:
            return None
        target = path[waypoint_index]
        dx, dy = target[0] - pose["x"], target[1] - pose["y"]
        cosine, sine = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        forward = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        angle = abs(math.atan2(lateral, max(1e-9, forward)))
        if (forward < 0.45 or math.hypot(dx, dy) < 0.85
                or abs(lateral) < 0.30
                or not math.radians(25.0) <= angle <= math.radians(60.0)):
            return None
        return "lateral_left" if lateral > 0.0 else "lateral_right"

    @classmethod
    def _prepare_dynamic_detour(cls, path: List[Tuple[float, float]],
                                pose: dict) -> List[Tuple[float, float]]:
        """A* rămâne singura sursă a geometriei ocolirilor dinamice."""
        return path

    async def _execute_lateral_escape(
            self, path: List[Tuple[float, float]], waypoint_index: int,
            navigation_deadline: float, positioning: bool = False,
    ) -> bool:
        """Execută efectiv un pas lateral prin locomotion, cu SLAM activ.

        API 1102 nu garantează mers holonomic. După 1201 folosim temporar vy,
        verificăm progresul în frame-ul hărții și oprim locomotion înainte să
        revenim la ruta nativă.
        """
        if not self.lateral_velocity or not self.stop_locomotion:
            return False
        pose = self._pose()
        mode = (
            self._departure_lateral_mode(path, waypoint_index, pose)
            if positioning else self._lateral_motion_mode(path, waypoint_index, pose)
        )
        if not positioning and not mode and path and waypoint_index < len(path):
            target = path[waypoint_index]
            heading = math.atan2(target[1] - pose["y"], target[0] - pose["x"])
            error = wrap_angle(heading - pose["yaw"])
            # După un obstacol, și o diagonală moderată începe mai sigur cu
            # o mică degajare laterală, apoi ruta se recalculează.
            if math.radians(20.0) <= abs(error) <= math.radians(160.0):
                mode = "lateral_left" if error > 0.0 else "lateral_right"
        if not mode:
            return False
        direction = 1 if mode == "lateral_left" else -1
        if (not positioning
                and time.monotonic() - self.last_lateral_escape_at < LATERAL_RECOVERY_COOLDOWN):
            return False
        if not self._lateral_direction_clear(direction):
            return False

        left_x, left_y = -math.sin(pose["yaw"]), math.cos(pose["yaw"])
        target = path[waypoint_index]
        requested_lateral = direction * (
            (target[0] - pose["x"]) * left_x + (target[1] - pose["y"]) * left_y
        )
        target_distance = min(0.35, max(0.22, requested_lateral))
        # Verificăm separat segmentul pur lateral; diagonala A* liberă nu
        # garantează automat că și translația cu yaw fix este liberă.
        safe_distance = target_distance
        while safe_distance >= 0.21:
            lateral_target = (
                pose["x"] + left_x * direction * safe_distance,
                pose["y"] + left_y * direction * safe_distance,
            )
            if self.planner.segment_is_free((pose["x"], pose["y"]), lateral_target):
                break
            safe_distance -= 0.05
        if safe_distance < 0.21:
            return False

        start = dict(pose)
        started = time.monotonic()
        # Cooldown-ul începe la prima comandă, inclusiv dacă spațiul se închide
        # pe parcurs. Astfel nu alternăm imediat dreapta-stânga.
        self.last_lateral_escape_at = started
        self.last_lateral_direction = direction
        last_command_at = -math.inf
        best_progress = 0.0
        useful_progress = (
            min(0.31, max(0.18, safe_distance - 0.04))
            if positioning else min(0.14, max(0.10, safe_distance - 0.08))
        )
        lateral_speed = 0.16 if positioning else 0.14
        self.status.update({
            "state": "lateral_positioning" if positioning else "lateral_evading",
            "motion_mode": mode, "command_speed": lateral_speed,
            "error": (
                f"repoziționare laterală unică {'stânga' if direction > 0 else 'dreapta'}"
                if positioning else
                f"execut pas lateral {'stânga' if direction > 0 else 'dreapta'}"
            ),
        })
        await self._emit()
        try:
            while not self.cancel_requested:
                now = time.monotonic()
                if now > navigation_deadline:
                    raise RuntimeError("timeout de navigare în timpul pasului lateral")
                if now - started > 3.2:
                    # SLAM-ul G1 poate publica poziția cu întârziere față de
                    # pasul fizic. O degajare confirmată, chiar mai mică decât
                    # ținta ideală, este suficientă: oprim și replănuim imediat.
                    if best_progress >= 0.07:
                        self.status["lateral_progress"] = best_progress
                        return True
                    raise RuntimeError("pasul lateral nu a progresat util în 3.2s")
                if not self.localization_ok():
                    raise RuntimeError("localizarea s-a pierdut în timpul pasului lateral")
                if not navigation_sensors_ready(self.obstacle_guard):
                    raise RuntimeError("senzorii s-au pierdut în timpul pasului lateral")
                if not self._lateral_direction_clear(direction):
                    self.status["error"] = "direcția laterală s-a blocat; păstrez STOP"
                    await self._emit()
                    return False

                current = self._pose()
                progress = direction * (
                    (current["x"] - start["x"]) * left_x
                    + (current["y"] - start["y"]) * left_y
                )
                best_progress = max(best_progress, progress)
                self.status.update({
                    "pose": current, "lateral_progress": max(0.0, best_progress)
                })
                if best_progress >= useful_progress:
                    return True
                # SetVelocity are durată proprie. Nu blocăm bucla și DDS-ul cu
                # aceeași comandă la fiecare eșantion; o reînnoim la 0.55 s.
                if now - last_command_at >= 0.55:
                    command = await self.lateral_velocity(lateral_speed * direction)
                    if not command.get("success"):
                        raise RuntimeError(command.get(
                            "error", "comanda laterală vy a fost respinsă"
                        ))
                    last_command_at = time.monotonic()
                await asyncio.sleep(max(0.06, min(0.12, self.poll_interval)))
        finally:
            await self.stop_locomotion()

    def _stagnation_recovery_path(self, pose: dict,
                                  obstacle_info: Optional[dict]) -> Optional[List[Tuple[float, float]]]:
        """Alege un pas lateral liber, preferabil opus clusterului observat."""
        if not obstacle_info or obstacle_info.get("mode") != "lidar_shape":
            return None
        max_distance = 0.90
        if float(obstacle_info.get("distance", math.inf)) > max_distance:
            return None
        if time.monotonic() - self.last_lateral_escape_at < LATERAL_RECOVERY_COOLDOWN:
            return None
        preferred = []
        if obstacle_info and all(key in obstacle_info for key in ("x", "y")):
            dx = float(obstacle_info["x"]) - pose["x"]
            dy = float(obstacle_info["y"]) - pose["y"]
            obstacle_left = -math.sin(pose["yaw"]) * dx + math.cos(pose["yaw"]) * dy
            # Obstacol în stânga -> recuperare spre dreapta și invers.
            preferred.append(-1 if obstacle_left >= 0.0 else 1)
        recovery_count = int(self.status.get("stagnation_recoveries", 0) or 0)
        fallback_first = -1 if recovery_count % 2 == 0 else 1
        for direction in (fallback_first, -fallback_first):
            if direction not in preferred:
                preferred.append(direction)

        left_x, left_y = -math.sin(pose["yaw"]), math.cos(pose["yaw"])
        candidates = []
        for preference_rank, direction in enumerate(preferred):
            if not self._lateral_direction_clear(direction):
                continue
            for distance in (0.45, 0.35, 0.25):
                target = (
                    pose["x"] + left_x * direction * distance,
                    pose["y"] + left_y * direction * distance,
                )
                if self.planner.segment_is_free((pose["x"], pose["y"]), target):
                    candidates.append((distance, -preference_rank, direction, target))
                    break
        if candidates:
            _, _, _, target = max(candidates)
            return [(pose["x"], pose["y"]), target]
        return None

    async def _resume_checked(self) -> None:
        result = await self.resume_navigation()
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(result.get("error", "API 1202 nu a confirmat reluarea"))

    def _waypoint_yaw(self, path: List[Tuple[float, float]], waypoint_index: int,
                      pose: dict, goal_yaw: float) -> float:
        pattern = self.status.get("path_pattern", {})
        if pattern.get("type") == "forward_then_lateral":
            reference_yaw = pattern.get("reference_yaw")
            if reference_yaw is not None:
                # Același yaw face al doilea segment o translație laterală,
                # nu o rotație de 90° urmată de încă un mers înainte.
                return float(reference_yaw)
        waypoint = path[waypoint_index]
        dx, dy = waypoint[0] - pose["x"], waypoint[1] - pose["y"]
        if math.hypot(dx, dy) < 0.08:
            return goal_yaw
        incoming_yaw = math.atan2(dy, dx)
        if waypoint_index >= len(path) - 1:
            return incoming_yaw

        # La un colț, orientarea finală a comenzii este tangenta continuă dintre
        # segmentul curent și cel următor. Astfel robotul ia curba o singură
        # dată; nu ajunge orientat pe segmentul vechi ca apoi să se rotească pe
        # loc din nou pentru următorul waypoint.
        next_index = waypoint_index + 1
        while next_index < len(path) - 1 and math.hypot(
                path[next_index][0] - waypoint[0],
                path[next_index][1] - waypoint[1],
        ) < 0.30:
            next_index += 1
        out_dx = path[next_index][0] - waypoint[0]
        out_dy = path[next_index][1] - waypoint[1]
        out_length = math.hypot(out_dx, out_dy)
        in_length = math.hypot(dx, dy)
        if out_length < 0.08 or in_length < 0.08:
            return incoming_yaw
        in_x, in_y = dx / in_length, dy / in_length
        out_x, out_y = out_dx / out_length, out_dy / out_length
        # Pentru o întoarcere aproape completă nu există bisectoare stabilă;
        # ținta curentă trebuie abordată pe direcția de sosire.
        if in_x * out_x + in_y * out_y < -0.50:
            return incoming_yaw
        return math.atan2(in_y + out_y, in_x + out_x)

    async def _wait_for_fresh_sensors(
            self, navigation_deadline: float, already_paused: bool = False,
    ) -> None:
        """Fail-safe: 1201 și nicio mișcare cât RealSense/LiDAR sunt vechi."""
        if not already_paused:
            pause_result = await self.pause_navigation()
            if isinstance(pause_result, dict) and pause_result.get("success") is False:
                raise RuntimeError(pause_result.get("error", "API 1201 nu a confirmat oprirea"))
        lost_at = time.monotonic()
        self.status.update({
            "state": "waiting_sensor",
            "error": "robot oprit: aștept date recente RealSense/LiDAR",
        })
        await self._emit()
        last_status = {}
        while not self.cancel_requested:
            now = time.monotonic()
            if now > navigation_deadline:
                raise RuntimeError("timeout de navigare în așteptarea senzorilor")
            if not self.localization_ok():
                await self._wait_for_localization_recovery(
                    navigation_deadline, already_paused=True,
                )
                self.status.update({
                    "state": "waiting_sensor",
                    "error": "localizare reconfirmată; aștept încă RealSense/LiDAR",
                })
                await self._emit()
                continue
            if navigation_sensors_ready(self.obstacle_guard):
                self.status.update({"state": "navigating", "error": "datele senzorilor au revenit"})
                await self._emit()
                return
            if now - lost_at > self.sensor_loss_timeout:
                if hasattr(self.obstacle_guard, "sensor_status"):
                    last_status = self.obstacle_guard.sensor_status()
                camera_age = last_status.get("camera_age")
                lidar_age = last_status.get("lidar_age")
                lidar_source = last_status.get("lidar_source") or "topic LiDAR necunoscut"
                stale = []
                if not last_status.get("camera_fresh", True):
                    stale.append(
                        "RealSense fără cadre"
                        + (f" de {camera_age:.1f}s" if camera_age is not None else "")
                    )
                if not last_status.get("lidar_fresh", True):
                    stale.append(
                        f"LiDAR {lidar_source} fără cloud valid"
                        + (f" de {lidar_age:.1f}s" if lidar_age is not None else "")
                    )
                detail = "; ".join(stale) or "fluxurile de protecție sunt vechi"
                raise RuntimeError(f"{detail}; robotul rămâne oprit în siguranță")
            await asyncio.sleep(self.poll_interval)

    async def _stage_route_while_paused(
            self, path: List[Tuple[float, float]], waypoint_index: int, pose: dict,
    ) -> dict:
        """Încarcă noul 1102 cât ruta veche este încă ținută pe pauză."""
        if not path or waypoint_index >= len(path):
            raise RuntimeError("ruta nouă nu conține un waypoint executabil")
        waypoint_x, waypoint_y = path[waypoint_index]
        goal = self.status.get("goal") or {}
        requested_speed = float(goal.get("speed", 0.20))
        waypoint_yaw = self._waypoint_yaw(
            path, waypoint_index, pose, float(goal.get("yaw", pose.get("yaw", 0.0)))
        )
        command_speed = self._safe_segment_speed(
            path, waypoint_index, pose, requested_speed
        )
        dispatched_at = time.monotonic()
        result = await self.send_waypoint(
            waypoint_x, waypoint_y, waypoint_yaw, command_speed
        )
        if not result.get("success"):
            raise RuntimeError(result.get(
                "error", "ruta nouă nu a fost acceptată de API 1102 cât robotul era oprit"
            ))
        self.staged_waypoint = {
            "x": waypoint_x, "y": waypoint_y, "yaw": waypoint_yaw,
            "speed": command_speed, "waypoint_index": waypoint_index,
            "result": result, "dispatched_at": dispatched_at,
        }
        return result

    async def _wait_for_clear_or_replan(
            self, path: List[Tuple[float, float]], waypoint_index: int,
            goal_x: float, goal_y: float, replans: int,
            navigation_deadline: float, already_paused: bool = False,
    ) -> Tuple[List[Tuple[float, float]], int, int, float]:
        """Oprește, invalidează ruta veche și validează atomic ocolirea nouă."""
        if not already_paused:
            pause_result = await self.pause_navigation()
            if isinstance(pause_result, dict) and pause_result.get("success") is False:
                raise RuntimeError(pause_result.get("error", "API 1201 nu a confirmat oprirea"))

        self.staged_waypoint = None
        self.status.update({
            "state": "stopped_replanning",
            "path": [],
            "error": "robot oprit; ruta veche a fost invalidată",
            "confirmation_id": None,
        })
        await self._emit()

        obstacle_started = time.monotonic()
        clear_since: Optional[float] = None
        sensor_lost_since: Optional[float] = None
        next_plan_attempt = obstacle_started + self.obstacle_wait_before_replan
        last_emit = 0.0
        candidate_signature = None
        candidate_stable_since: Optional[float] = None
        lateral_unlock_attempted = False

        async def try_lateral_unlock(current_pose: dict,
                                     current_obstacle: Optional[dict]) -> bool:
            """Încearcă o singură degajare laterală în episodul de blocaj."""
            nonlocal lateral_unlock_attempted, next_plan_attempt
            if lateral_unlock_attempted:
                return False
            if time.monotonic() - obstacle_started < DYNAMIC_LATERAL_UNLOCK_DELAY:
                return False
            recovery_path = self._stagnation_recovery_path(
                current_pose, current_obstacle
            )
            if recovery_path is None:
                return False
            lateral_unlock_attempted = True
            direction = self._lateral_motion_mode(recovery_path, 1, current_pose)
            self.status.update({
                "state": "lateral_evading", "path": recovery_path,
                "waypoint_index": 1,
                "error": (
                    "A* nu are ieșire directă; încerc o singură degajare laterală "
                    "verificată, apoi replănuiesc"
                ),
                "motion_mode": direction,
            })
            await self._emit()
            try:
                moved = await self._execute_lateral_escape(
                    recovery_path, 1, navigation_deadline, positioning=False,
                )
            except RuntimeError as exc:
                self.status.update({
                    "state": "waiting_obstacle", "path": [],
                    "error": f"degajarea laterală a fost oprită în siguranță: {exc}",
                })
                await self._emit()
                return False
            if not moved:
                self.status.update({
                    "state": "waiting_obstacle", "path": [],
                    "error": "degajarea laterală nu mai este liberă; păstrez STOP",
                })
                await self._emit()
                return False
            self.status.update({
                "state": "replanning", "path": [],
                "pose": self._pose(),
                "dynamic_lateral_unlocks": int(
                    self.status.get("dynamic_lateral_unlocks", 0) or 0
                ) + 1,
                "error": "pas lateral confirmat; replănuiesc imediat din noua poziție",
            })
            await self._emit()
            next_plan_attempt = time.monotonic()
            return True

        while not self.cancel_requested:
            now = time.monotonic()
            if now > navigation_deadline:
                raise RuntimeError("timeout de navigare în așteptarea obstacolului")
            if not self.localization_ok():
                await self._wait_for_localization_recovery(
                    navigation_deadline, already_paused=True,
                )
                self.status.update({
                    "state": "waiting_obstacle",
                    "error": "localizare reconfirmată; ruta rămâne oprită până la ocolire",
                })
                await self._emit()
                continue

            if not navigation_sensors_ready(self.obstacle_guard):
                candidate_signature = None
                candidate_stable_since = None
                if sensor_lost_since is None:
                    sensor_lost_since = now
                    self.status.update({
                        "state": "waiting_sensor", "path": [],
                        "error": "robot oprit: aștept date recente RealSense/LiDAR",
                    })
                    await self._emit()
                elif now - sensor_lost_since > self.sensor_loss_timeout:
                    raise RuntimeError("datele RealSense/LiDAR s-au pierdut; robotul rămâne oprit")
                await asyncio.sleep(self.poll_interval)
                continue
            sensor_lost_since = None

            pose = self._pose()
            self.status["pose"] = pose
            _, sensor_blocked, live_shape, obstacle_info, _ = (
                self._refresh_route_obstacles(path, waypoint_index, pose, now)
            )

            camera_only = bool(
                sensor_blocked and obstacle_info
                and obstacle_info.get("mode") == "camera_stop_only"
                and not live_shape
            )
            if camera_only:
                clear_since = None
                candidate_signature = None
                candidate_stable_since = None
                self.status.update({
                    "state": "waiting_obstacle", "path": [],
                    "error": (
                        "camera a cerut STOP; aștept confirmarea geometrică LiDAR "
                        "sau eliberarea zonei"
                    ),
                    "dynamic_obstacle": obstacle_info,
                })
                if now - last_emit >= 0.20:
                    await self._emit()
                    last_emit = now
                await asyncio.sleep(self.poll_interval)
                continue

            sensor_confirms_clear = not sensor_blocked and not live_shape
            if sensor_confirms_clear:
                clear_since = clear_since or now
                candidate_signature = None
                candidate_stable_since = None
                if now - clear_since >= self.obstacle_clear_stable:
                    self.planner.clear_dynamic_source("lidar")
                    self.planner.clear_dynamic_source("camera")
                    try:
                        resumed_path = self.planner.plan(
                            (pose["x"], pose["y"]), (goal_x, goal_y)
                        )
                        resumed_path = self._prepare_dynamic_detour(
                            resumed_path, pose
                        )
                        resumed_index = 1 if len(resumed_path) > 1 else 0
                        await self._stage_route_while_paused(
                            resumed_path, resumed_index, pose
                        )
                    except Exception as exc:
                        self.status.update({
                            "state": "waiting_obstacle", "path": [],
                            "error": f"zona pare liberă, dar ruta nu poate fi încărcată: {exc}",
                        })
                        await self._emit()
                        await asyncio.sleep(self.poll_interval)
                        continue
                    self.status.update({
                        "state": "replanning", "path": resumed_path,
                        "waypoint_index": resumed_index,
                        "error": "zona este liberă; noul 1102 este încărcat înainte de reluare",
                        "confirmation_id": None, "dynamic_obstacle": None,
                        "path_pattern": {"type": "astar"},
                        "path_reference_yaw": None,
                    })
                    await self._emit()
                    return resumed_path, resumed_index, replans, 0.0
                await asyncio.sleep(self.poll_interval)
                continue

            clear_since = None
            if obstacle_info is None:
                obstacle_info = {
                    "x": pose["x"], "y": pose["y"], "distance": 0.0,
                    "mode": "route_refresh", "shape_points": 0,
                }

            # Dacă obiectul este deja în spațiul imediat al robotului, un 1102
            # către un punct aflat dincolo de el poate porni prin scaun înainte
            # ca plannerul nativ să respecte colțul A*. Mai întâi degajăm o
            # singură dată lateral, numai pe direcția confirmată liberă.
            if (
                float(obstacle_info.get("distance", math.inf)) <= 0.45
                and await try_lateral_unlock(pose, obstacle_info)
            ):
                clear_since = None
                await asyncio.sleep(self.poll_interval)
                continue

            self.planner.expire_dynamic_obstacles(self.dynamic_obstacle_ttl, now=now)
            if now < next_plan_attempt:
                await asyncio.sleep(self.poll_interval)
                continue

            self.status.update({
                "state": "replanning", "path": [],
                "error": "obstacol LiDAR confirmat; calculez ocolirea",
                "dynamic_obstacle": obstacle_info, "confirmation_id": None,
            })
            if now - last_emit >= 0.80:
                await self._emit()
                last_emit = now

            try:
                candidate_path = self.planner.plan(
                    (pose["x"], pose["y"]), (goal_x, goal_y)
                )
                candidate_path = self._prepare_dynamic_detour(candidate_path, pose)
            except Exception:
                candidate_signature = None
                candidate_stable_since = None
                if await try_lateral_unlock(pose, obstacle_info):
                    clear_since = None
                    await asyncio.sleep(self.poll_interval)
                    continue
                self.status.update({
                    "state": "waiting_obstacle", "path": [],
                    "error": "nu există încă o ocolire sigură; păstrez STOP și încerc din nou",
                })
                if now - last_emit >= 0.80:
                    await self._emit()
                    last_emit = now
                next_plan_attempt = now + 0.35
                await asyncio.sleep(self.poll_interval)
                continue

            candidate_index = 1 if len(candidate_path) > 1 else 0
            candidate_index = self._select_control_waypoint_index(
                candidate_path, candidate_index, pose
            )
            if self._current_segment_blocked(candidate_path, candidate_index, pose):
                candidate_signature = None
                candidate_stable_since = None
                if await try_lateral_unlock(pose, obstacle_info):
                    clear_since = None
                    await asyncio.sleep(self.poll_interval)
                    continue
                self.status.update({
                    "state": "waiting_obstacle", "path": candidate_path,
                    "error": "ocolirea nu are încă o ieșire sigură din poziția curentă",
                })
                if now - last_emit >= 0.80:
                    await self._emit()
                    last_emit = now
                next_plan_attempt = now + 0.35
                await asyncio.sleep(self.poll_interval)
                continue

            signature = tuple(
                (float(point[0]), float(point[1])) for point in candidate_path
            )
            same_route_family = bool(
                candidate_signature
                and len(signature) == len(candidate_signature)
                and all(
                    math.hypot(x1 - x2, y1 - y2) <= 0.20
                    for (x1, y1), (x2, y2) in zip(
                        signature, candidate_signature
                    )
                )
            )
            if not same_route_family:
                candidate_stable_since = now
            # Validăm geometria cea mai nouă, dar nu reluăm cronometrul pentru
            # jitter LiDAR de câțiva centimetri pe aceeași parte a obstacolului.
            candidate_signature = signature
            self.status.update({
                "state": "validating_replan", "path": candidate_path,
                "waypoint_index": candidate_index,
                "error": (
                    f"ocolire liberă; verific stabilitatea {REPLAN_ROUTE_STABLE:.2f} s "
                    "înainte de reluare"
                ),
            })
            if now - last_emit >= 0.20:
                await self._emit()
                last_emit = now
            if now - float(candidate_stable_since or now) < REPLAN_ROUTE_STABLE:
                await asyncio.sleep(self.poll_interval)
                continue

            try:
                await self._stage_route_while_paused(
                    candidate_path, candidate_index, pose
                )
            except Exception as exc:
                candidate_signature = None
                candidate_stable_since = None
                self.status.update({
                    "state": "waiting_obstacle", "path": [],
                    "error": f"ocolirea este sigură, dar 1102 nu a acceptat-o: {exc}",
                })
                await self._emit()
                next_plan_attempt = time.monotonic() + 0.20
                await asyncio.sleep(self.poll_interval)
                continue

            self.replan_confirmation_id = None
            self.replan_confirmation_event = None
            self.status.update({
                "state": "replanning", "path": candidate_path,
                "waypoint_index": candidate_index, "replans": replans + 1,
                "error": "ruta nouă este stabilă și încărcată; continui",
                "confirmation_id": None,
                "path_pattern": {"type": "astar"},
                "path_reference_yaw": None,
            })
            await self._emit()
            return candidate_path, candidate_index, replans + 1, 0.0

        return path, waypoint_index, replans, 0.0

    async def _replan_ahead_while_moving(
            self, path: List[Tuple[float, float]], waypoint_index: int,
            goal_x: float, goal_y: float, replans: int,
    ) -> Optional[Tuple[List[Tuple[float, float]], int, int]]:
        """Calculează ocolirea înainte ca obstacolul să ajungă la robot.

        Comanda 1102 curentă rămâne activă numai cât segmentul imediat este
        liber și senzorii sunt sănătoși. Când planul alternativ este gata,
        bucla exterioară îl trimite ca suprascriere 1102 fără ciclul lent
        1201/1202. Dacă pericolul devine imediat între timp, abandonăm această
        optimizare și ramura fail-safe oprește robotul.
        """
        if self._navigation_blocked() or not navigation_sensors_ready(
                self.obstacle_guard):
            return None
        pose = self._pose()
        if self._current_segment_blocked(path, waypoint_index, pose):
            return None
        self.status.update({
            "state": "navigating", "path": path,
            "error": "obstacol mai în față: calculez ocolirea în mers",
            "replan_in_motion": True,
        })
        await self._emit()
        planning_started = time.monotonic()
        try:
            # Grila implicită de 10 cm păstrează această căutare scurtă. Nu
            # folosim asyncio.to_thread aici: directoarele de deployment care
            # conțin literal `*` pot bloca executorul implicit Python.
            candidate = self.planner.plan(
                (pose["x"], pose["y"]), (goal_x, goal_y)
            )
            candidate = self._prepare_dynamic_detour(candidate, self._pose())
        except Exception:
            self.status["replan_in_motion"] = False
            return None
        live_pose = self._pose()
        if (
            self._navigation_blocked()
            or not navigation_sensors_ready(self.obstacle_guard)
            or self._current_segment_blocked(path, waypoint_index, live_pose)
        ):
            self.status["replan_in_motion"] = False
            return None
        candidate_index = 1 if len(candidate) > 1 else 0
        candidate_index = self._select_control_waypoint_index(
            candidate, candidate_index, live_pose
        )
        if self._current_segment_blocked(candidate, candidate_index, live_pose):
            self.status["replan_in_motion"] = False
            return None
        self.status.update({
            "state": "navigating", "path": candidate,
            "waypoint_index": candidate_index,
            "replans": replans + 1, "replan_in_motion": False,
            "last_replan_time_s": round(
                time.monotonic() - planning_started, 3
            ),
            "error": "ocolire recalculată în mers; continui fără STOP",
            "path_pattern": {"type": "astar"},
            "path_reference_yaw": None,
        })
        await self._emit()
        return candidate, candidate_index, replans + 1

    async def _run(self, path: List[Tuple[float, float]], goal_x: float, goal_y: float,
                   goal_yaw: float, speed: float, timeout: float,
                   motion_profile: str = "stable") -> None:
        started = time.monotonic()
        launch_pose = self._pose()
        launch_complete = False
        departure_lateral_attempted = False
        waypoint_index = 1 if len(path) > 1 else 0
        replans = 0
        replan_grace_until = 0.0
        sensor_stale_since: Optional[float] = None
        navigation_deadline = math.inf if timeout <= 0.0 else started + timeout
        try:
            while not self.cancel_requested:
                if time.monotonic() > navigation_deadline:
                    raise RuntimeError("timeout de navigare")
                if not self.localization_ok():
                    command_was_active = bool(
                        self.status.get("native_command_active", False)
                    )
                    await self._wait_for_localization_recovery(
                        navigation_deadline,
                        already_paused=not command_was_active,
                    )
                    if not self.cancel_requested and command_was_active:
                        await self._resume_checked()
                    continue
                if not navigation_sensors_ready(self.obstacle_guard):
                    sensor_stale_since = sensor_stale_since or time.monotonic()
                    if time.monotonic() - sensor_stale_since < SENSOR_GLITCH_GRACE:
                        await asyncio.sleep(self.poll_interval)
                        continue
                    command_was_active = bool(
                        self.status.get("native_command_active", False)
                    )
                    await self._wait_for_fresh_sensors(
                        navigation_deadline,
                        already_paused=not command_was_active,
                    )
                    if not self.cancel_requested and command_was_active:
                        await self._resume_checked()
                    sensor_stale_since = None
                    continue
                sensor_stale_since = None
                pose = self._pose()
                self.status["pose"] = pose
                if not launch_complete and math.hypot(
                    pose["x"] - launch_pose["x"],
                    pose["y"] - launch_pose["y"],
                ) >= STARTUP_PROGRESS_DISTANCE:
                    launch_complete = True
                advanced_index = self._advance_past_native_tolerance(
                    path, waypoint_index, pose
                )
                advanced_index = self._select_control_waypoint_index(
                    path, advanced_index, pose
                )
                if advanced_index != waypoint_index:
                    waypoint_index = advanced_index
                    self.status["waypoint_index"] = waypoint_index
                observed_at = time.monotonic()
                segment_blocked, _, _, obstacle_info, _ = self._refresh_route_obstacles(
                    path, waypoint_index, pose, observed_at
                )
                segment_blocked = self._confirmed_segment_blocked(
                    segment_blocked, obstacle_info, observed_at
                )
                if obstacle_info:
                    self.status["dynamic_obstacle"] = obstacle_info
                else:
                    self.status["dynamic_obstacle"] = None
                if self.planner.expire_dynamic_obstacles(self.dynamic_obstacle_ttl):
                    await self._emit()
                route_goal = path[-1] if path else (goal_x, goal_y)
                if min(
                    math.hypot(goal_x - pose["x"], goal_y - pose["y"]),
                    math.hypot(route_goal[0] - pose["x"], route_goal[1] - pose["y"]),
                ) <= 0.22:
                    self.status.update({
                        "state": "arrived", "error": "",
                        "waypoint_index": len(path) - 1,
                        "native_command_active": False,
                    })
                    await self.pause_navigation()
                    self.planner.clear_dynamic_costmap()
                    await self._emit()
                    return

                immediate_blocked = self._current_segment_blocked(
                    path, waypoint_index, pose
                )
                if (segment_blocked and not immediate_blocked
                        and time.monotonic() >= replan_grace_until):
                    live_replan = await self._replan_ahead_while_moving(
                        path, waypoint_index, goal_x, goal_y, replans
                    )
                    if live_replan is not None:
                        path, waypoint_index, replans = live_replan
                        continue
                    immediate_blocked = self._current_segment_blocked(
                        path, waypoint_index, self._pose()
                    )
                if (segment_blocked and immediate_blocked
                        and time.monotonic() >= replan_grace_until):
                    path, waypoint_index, replans, grace = await self._wait_for_clear_or_replan(
                        path, waypoint_index, goal_x, goal_y, replans, navigation_deadline
                    )
                    replan_grace_until = time.monotonic() + grace
                    if not self.cancel_requested:
                        await self._resume_checked()
                    continue

                wx, wy = path[waypoint_index]
                distance = math.hypot(wx - pose["x"], wy - pose["y"])
                handoff_distance = self._handoff_distance(path, waypoint_index, speed)
                if (waypoint_index < len(path) - 1
                        and distance <= handoff_distance
                        and self._can_smooth_handoff(path, waypoint_index, pose)):
                    # 1102 acceptă suprascrierea destinației: următoarea comandă
                    # este trimisă înainte ca robotul să frâneze complet aici.
                    waypoint_index += 1
                    continue
                if distance <= 0.22 and waypoint_index < len(path) - 1:
                    waypoint_index += 1
                    continue

                # Pe o rută directă diagonală, o singură translație laterală
                # reduce rotația inițială și aliniază corpul cu drumul lung.
                # Flagul local interzice ciclurile stânga/dreapta.
                departure_mode = (
                    self._departure_lateral_mode(path, waypoint_index, pose)
                    if motion_profile == "adaptive"
                    and not departure_lateral_attempted
                    and not launch_complete
                    and self.lateral_velocity
                    and self.stop_locomotion
                    else None
                )
                if departure_mode:
                    departure_lateral_attempted = True
                    pause_result = await self.pause_navigation()
                    if isinstance(pause_result, dict) and pause_result.get("success") is False:
                        self.status["error"] = (
                            "repoziționarea laterală a fost omisă: API 1201 nu a confirmat STOP"
                        )
                        await self._emit()
                    else:
                        moved = await self._execute_lateral_escape(
                            path, waypoint_index, navigation_deadline,
                            positioning=True,
                        )
                        await self._resume_checked()
                        if moved:
                            pose = self._pose()
                            path = self.planner.plan(
                                (pose["x"], pose["y"]), (goal_x, goal_y)
                            )
                            path = self._prepare_dynamic_detour(path, pose)
                            waypoint_index = 1 if len(path) > 1 else 0
                            launch_pose = dict(pose)
                            self.status.update({
                                "state": "replanning", "path": path,
                                "waypoint_index": waypoint_index,
                                "path_pattern": {"type": "astar"},
                                "path_reference_yaw": None,
                                "error": (
                                    "repoziționare laterală terminată; continui pe linia dreaptă"
                                ),
                            })
                            await self._emit()
                            continue
                waypoint_yaw = self._waypoint_yaw(path, waypoint_index, pose, goal_yaw)
                command_speed = self._safe_segment_speed(
                    path, waypoint_index, pose, speed
                )
                if not launch_complete:
                    command_speed = min(command_speed, STARTUP_SPEED_LIMIT)
                motion_mode = "forward" if launch_complete else "startup_forward"
                pattern = self.status.get("path_pattern", {})
                if (pattern.get("type") == "forward_then_lateral"
                        and waypoint_index == len(path) - 1):
                    motion_mode = f"lateral_{pattern.get('lateral_direction', 'unknown')}"
                self.status.update({
                    "state": "navigating", "waypoint_index": waypoint_index,
                    "command_speed": command_speed, "motion_mode": motion_mode,
                    "error": "",
                })
                await self._emit()
                staged = self.staged_waypoint
                segment_dispatched_at = time.monotonic()
                if (staged and staged.get("waypoint_index") == waypoint_index
                        and math.hypot(staged["x"] - wx, staged["y"] - wy) <= 0.03):
                    # Destinația nouă a fost publicată cât 1201 ținea ruta veche
                    # pe pauză; după 1202 intrăm direct în monitorizarea ei.
                    result = staged.get("result") or {"success": True}
                    dispatch_hazard = None
                    waypoint_yaw = float(staged.get("yaw", waypoint_yaw))
                    command_speed = float(staged.get("speed", command_speed))
                    segment_dispatched_at = float(
                        staged.get("dispatched_at", segment_dispatched_at)
                    )
                    self.staged_waypoint = None
                else:
                    result, dispatch_hazard = await self._send_waypoint_guarded(
                        wx, wy, waypoint_yaw, command_speed, path, waypoint_index
                    )
                if dispatch_hazard == "localization":
                    await self._wait_for_localization_recovery(
                        navigation_deadline, already_paused=True,
                    )
                    if not self.cancel_requested:
                        await self._resume_checked()
                    continue
                if dispatch_hazard == "sensor":
                    await self._wait_for_fresh_sensors(
                        navigation_deadline, already_paused=True
                    )
                    if not self.cancel_requested:
                        await self._resume_checked()
                    continue
                if dispatch_hazard == "obstacle":
                    path, waypoint_index, replans, grace = await self._wait_for_clear_or_replan(
                        path, waypoint_index, goal_x, goal_y, replans,
                        navigation_deadline, already_paused=True,
                    )
                    replan_grace_until = time.monotonic() + grace
                    if not self.cancel_requested:
                        await self._resume_checked()
                    continue
                if dispatch_hazard == "pause_failed":
                    raise RuntimeError(result.get("error", "API 1201 nu a oprit robotul în timpul trimiterii 1102"))
                # Dacă 1102 expiră după ce garda a detectat obstacolul, timeout-ul
                # nu este un eșec de cursă: robotul este deja oprit, iar ramura
                # de mai sus a recalculat ruta. Doar un eșec fără hazard e fatal.
                if not result.get("success"):
                    raise RuntimeError(result.get("error", "waypoint-ul 1102 a fost respins"))
                self.status["native_command_active"] = True
                # Așteptăm atingerea waypoint-ului sau apariția unui obstacol nou.
                segment_started = time.monotonic()
                best_segment_distance=distance
                best_heading_error = abs(wrap_angle(waypoint_yaw - pose["yaw"]))
                last_segment_progress=segment_started
                last_motion_pose = dict(pose)
                segment_retry_count = 0
                segment_sensor_stale_since: Optional[float] = None
                while not self.cancel_requested:
                    await asyncio.sleep(self.poll_interval)
                    if time.monotonic() > navigation_deadline:
                        raise RuntimeError("timeout de navigare")
                    if not self.localization_ok():
                        await self._wait_for_localization_recovery(
                            navigation_deadline, already_paused=False,
                        )
                        if not self.cancel_requested:
                            await self._resume_checked()
                        last_segment_progress = time.monotonic()
                        continue
                    pose = self._pose()
                    self.status["pose"] = pose
                    observed_at = time.monotonic()
                    segment_blocked, _, _, obstacle_info, _ = self._refresh_route_obstacles(
                        path, waypoint_index, pose, observed_at
                    )
                    segment_blocked = self._confirmed_segment_blocked(
                        segment_blocked, obstacle_info, observed_at
                    )
                    if obstacle_info:
                        self.status["dynamic_obstacle"] = obstacle_info
                    else:
                        self.status["dynamic_obstacle"] = None
                    self.planner.expire_dynamic_obstacles(self.dynamic_obstacle_ttl)
                    current_distance=math.hypot(wx-pose["x"],wy-pose["y"])
                    native_finished = False
                    if self.waypoint_completed:
                        try:
                            native_finished = bool(self.waypoint_completed(
                                wx, wy, segment_dispatched_at
                            ))
                        except Exception:
                            native_finished = False
                    if native_finished:
                        if self._native_completion_has_progress(
                                distance, current_distance,
                                best_segment_distance):
                            self.status["native_completion_confirmed"] = True
                            if waypoint_index < len(path) - 1:
                                waypoint_index += 1
                                self.status["waypoint_index"] = waypoint_index
                                break
                            self.status.update({
                                "state": "arrived", "error": "",
                                "waypoint_index": len(path) - 1,
                                "native_command_active": False,
                            })
                            await self.pause_navigation()
                            self.planner.clear_dynamic_costmap()
                            await self._emit()
                            return
                        self.status["native_completion_ignored"] = (
                            "FINISHED fără progres suficient pentru ținta curentă"
                        )
                    if current_distance<=best_segment_distance-0.03:
                        best_segment_distance=current_distance
                        last_segment_progress=time.monotonic()
                    current_heading_error = abs(wrap_angle(waypoint_yaw - pose["yaw"]))
                    if current_heading_error <= best_heading_error - math.radians(3.0):
                        best_heading_error = current_heading_error
                        last_segment_progress = time.monotonic()
                    # Telemetria SLAM ajunge uneori în salturi. Dacă robotul s-a
                    # deplasat fizic cu cel puțin 10 cm, nu îl declarăm blocat
                    # doar fiindcă abaterea temporară nu reduce încă distanța
                    # euclidiană până la waypoint.
                    if math.hypot(
                        pose["x"] - last_motion_pose["x"],
                        pose["y"] - last_motion_pose["y"],
                    ) >= 0.10:
                        last_motion_pose = dict(pose)
                        last_segment_progress = time.monotonic()
                    handoff_distance = self._handoff_distance(path, waypoint_index, speed)
                    if (waypoint_index < len(path) - 1
                            and current_distance <= handoff_distance
                            and self._can_smooth_handoff(path, waypoint_index, pose)):
                        waypoint_index += 1
                        break
                    if current_distance <= 0.22:
                        if waypoint_index < len(path) - 1:
                            waypoint_index += 1
                        break
                    if segment_blocked and time.monotonic() >= replan_grace_until:
                        immediate_blocked = self._current_segment_blocked(
                            path, waypoint_index, pose
                        )
                        if not immediate_blocked:
                            live_replan = await self._replan_ahead_while_moving(
                                path, waypoint_index, goal_x, goal_y, replans
                            )
                            if live_replan is not None:
                                path, waypoint_index, replans = live_replan
                                break
                            immediate_blocked = self._current_segment_blocked(
                                path, waypoint_index, self._pose()
                            )
                        if immediate_blocked:
                            break
                    if not navigation_sensors_ready(self.obstacle_guard):
                        segment_sensor_stale_since = (
                            segment_sensor_stale_since or time.monotonic()
                        )
                        if (time.monotonic() - segment_sensor_stale_since
                                >= SENSOR_GLITCH_GRACE):
                            # Păstrăm aceeași comandă 1102. Vechea ramură ieșea
                            # în bucla exterioară și retrimitea aceeași țintă
                            # după 1202, făcând firmware-ul să reînceapă rotația.
                            await self._wait_for_fresh_sensors(
                                navigation_deadline, already_paused=False
                            )
                            if not self.cancel_requested:
                                await self._resume_checked()
                            segment_sensor_stale_since = None
                            last_segment_progress = time.monotonic()
                        continue
                    segment_sensor_stale_since = None
                    if (time.monotonic() - last_segment_progress >= self.stagnation_timeout
                            and current_distance > 0.22):
                        recovery_obstacle = (
                            obstacle_info or self._recent_obstacle_for_recovery(
                                pose, observed_at
                            )
                        )
                        segment_cells = self.planner._line_cells(
                            self.planner.world_to_cell(pose["x"], pose["y"]),
                            self.planner.world_to_cell(wx, wy),
                        ) or []
                        finite_clearances = [
                            self.planner.obstacle_distance.get(cell, math.inf)
                            for cell in segment_cells
                            if math.isfinite(
                                self.planner.obstacle_distance.get(cell, math.inf)
                            )
                        ]
                        minimum_static_clearance = (
                            min(finite_clearances) if finite_clearances else None
                        )
                        self.status["stagnation_diagnostic"] = {
                            "target_distance_m": round(current_distance, 3),
                            "minimum_static_clearance_m": (
                                round(minimum_static_clearance, 3)
                                if minimum_static_clearance is not None else None
                            ),
                            "recent_lidar_age_s": (
                                round(max(0.0, observed_at - self.recent_obstacle_at), 3)
                                if recovery_obstacle else None
                            ),
                            "command_speed_mps": round(command_speed, 3),
                        }
                        recovery_path = (
                            self._stagnation_recovery_path(pose, recovery_obstacle)
                            if (self.enable_stagnation_lateral_recovery
                                and self.lateral_velocity and self.stop_locomotion)
                            else None
                        )
                        if recovery_path:
                            pause_result = await self.pause_navigation()
                            if isinstance(pause_result, dict) and pause_result.get("success") is False:
                                raise RuntimeError(
                                    pause_result.get("error", "nu am putut opri 1102 pentru recuperare laterală")
                                )
                            moved = await self._execute_lateral_escape(
                                recovery_path, 1, navigation_deadline
                            )
                            if moved:
                                pose = self._pose()
                                path = self.planner.plan(
                                    (pose["x"], pose["y"]), (goal_x, goal_y)
                                )
                                path = self._prepare_dynamic_detour(path, pose)
                                waypoint_index = 1 if len(path) > 1 else 0
                                replans += 1
                                self.status.update({
                                    "state": "replanning", "path": path,
                                    "waypoint_index": waypoint_index, "replans": replans,
                                    "path_pattern": {"type": "astar"},
                                    "path_reference_yaw": None,
                                    "stagnation_recoveries": int(
                                        self.status.get("stagnation_recoveries", 0) or 0
                                    ) + 1,
                                    "error": "recuperare laterală executată; ruta a fost recalculată",
                                })
                                await self._emit()
                                await self._resume_checked()
                                break
                            # Manevra a devenit indisponibilă între plan și
                            # execuție; reluăm controllerul nativ înainte de retry.
                            await self._resume_checked()
                        recent_lidar = bool(
                            recovery_obstacle
                            and recovery_obstacle.get("mode") == "lidar_shape"
                            and observed_at - self.recent_obstacle_at
                                <= self.dynamic_obstacle_ttl
                        )
                        if recent_lidar:
                            self.status.update({
                                "state": "stopped_replanning",
                                "error": (
                                    "fără progres; LiDAR-ul a confirmat recent un obstacol "
                                    "intermitent — opresc și recalculez, nu retrimit ruta veche"
                                ),
                                "dynamic_obstacle": recovery_obstacle,
                            })
                            await self._emit()
                            path, waypoint_index, replans, grace = (
                                await self._wait_for_clear_or_replan(
                                    path, waypoint_index, goal_x, goal_y, replans,
                                    navigation_deadline,
                                )
                            )
                            replan_grace_until = time.monotonic() + grace
                            if not self.cancel_requested:
                                await self._resume_checked()
                            break
                        if segment_retry_count < self.max_waypoint_retries:
                            segment_retry_count += 1
                            self.status.update({
                                "state": "retrying_waypoint",
                                "error": (
                                    f"fără progres {self.stagnation_timeout:.1f}s, dar fără obstacol LiDAR/cameră; "
                                    f"reîncerc waypoint-ul 1102 ({segment_retry_count}/{self.max_waypoint_retries})"
                                ),
                                "dynamic_obstacle": None,
                            })
                            await self._emit()
                            segment_dispatched_at = time.monotonic()
                            retry_result, retry_hazard = await self._send_waypoint_guarded(
                                wx, wy, waypoint_yaw, min(command_speed, 0.18),
                                path, waypoint_index,
                            )
                            if retry_hazard == "localization":
                                await self._wait_for_localization_recovery(
                                    navigation_deadline, already_paused=True,
                                )
                                if not self.cancel_requested:
                                    await self._resume_checked()
                                break
                            if retry_hazard in {"sensor", "obstacle"}:
                                # Bucla exterioară intră în fail-safe/replanificare;
                                # 1201 a fost deja trimis de gardă.
                                break
                            if retry_hazard == "pause_failed":
                                raise RuntimeError(
                                    retry_result.get("error", "API 1201 nu a oprit robotul la reîncercarea 1102")
                                )
                            if not retry_result.get("success"):
                                raise RuntimeError(
                                    retry_result.get("error", "reîncercarea waypoint-ului 1102 a fost respinsă")
                                )
                            if self._native_reports_paused():
                                resume_result = await self.resume_navigation()
                                if (isinstance(resume_result, dict)
                                        and resume_result.get("success") is False):
                                    raise RuntimeError(
                                        resume_result.get("error", "API 1202 nu a confirmat reluarea")
                                    )
                            retry_started = time.monotonic()
                            segment_started = retry_started
                            last_segment_progress = retry_started
                            best_segment_distance = current_distance
                            continue
                        raise RuntimeError(
                            "Unitree nu a progresat după reîncercarea 1102; "
                            "LiDAR-ul și camera nu confirmă niciun obstacol, deci nu inventez o rută de ocolire"
                        )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self.pause_navigation()
            if self.planner:
                self.planner.clear_dynamic_costmap()
            self.status.update({
                "state": "failed", "error": str(exc),
                "native_command_active": False,
            })
            await self._emit()


class AutonomousNavigator:
    def __init__(self, sport_client, pose_provider: Callable[[], dict],
                 localization_ok: Callable[[], bool], obstacle_guard,
                 event_callback: Callable[[dict], Awaitable[None]]):
        self.sport_client = sport_client
        self.pose_provider = pose_provider
        self.localization_ok = localization_ok
        self.obstacle_guard = obstacle_guard
        self.event_callback = event_callback
        self.task: Optional[asyncio.Task] = None
        self.cancel_requested = False
        self.status = {"state": "idle", "path": [], "goal": None, "error": ""}

    async def start(self, map_path: str, x: float, y: float, yaw: float,
                    timeout: float = 120.0) -> dict:
        if self.task and not self.task.done():
            return {"success": False, "error": "Există deja o navigare activă"}
        if not self.localization_ok():
            return {"success": False, "error": "Localizarea nu este activă sau odometria este veche"}
        if not self.sport_client.is_sdk_available():
            return {"success": False, "error": "SDK-ul G1 nu este disponibil pentru comenzi de viteză"}
        if not self.sport_client.is_locomotion_ready():
            return {"success": False, "error": "Robotul nu este în FSM locomotion (500/501/502/801/802)"}
        if not navigation_sensors_ready(self.obstacle_guard):
            return {"success": False, "error": "RealSense depth nu furnizează date recente pentru protecția la obstacole"}
        pose = self.pose_provider()
        planner = PCDGridPlanner()
        try:
            await asyncio.to_thread(planner.load, map_path)
            path = await asyncio.to_thread(planner.plan, (pose["x"], pose["y"]), (x, y))
        except Exception as exc:
            return {"success": False, "error": f"Planificarea a eșuat: {exc}"}
        self.cancel_requested = False
        self.status = {"state": "navigating", "path": path,
                       "goal": {"x": x, "y": y, "yaw": yaw}, "error": ""}
        self.task = asyncio.create_task(self._run(path, x, y, yaw, timeout))
        await self.event_callback({"type": "nav_status", **self.status})
        return {"success": True, "path": path, "waypoints": len(path)}

    async def stop(self, reason: str = "oprit de utilizator") -> dict:
        self.cancel_requested = True
        result = await asyncio.to_thread(self.sport_client.stop)
        if self.task and not self.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=1.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.task.cancel()
        self.status.update({"state": "cancelled", "error": reason})
        await self.event_callback({"type": "nav_status", **self.status})
        return result

    async def _command(self, vx: float, vy: float, wz: float) -> None:
        result = await asyncio.to_thread(self.sport_client.move_to, vx, vy, wz)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "comanda Move a eșuat"))

    async def _run(self, path: List[Tuple[float, float]], goal_x: float,
                   goal_y: float, goal_yaw: float, timeout: float) -> None:
        started = time.monotonic()
        waypoint_index = 1 if len(path) > 1 else 0
        best_distance = float("inf")
        last_progress = time.monotonic()
        try:
            while not self.cancel_requested:
                if time.monotonic() - started > timeout:
                    raise RuntimeError("timeout de navigare")
                if not self.localization_ok():
                    raise RuntimeError("localizarea/odometria s-a pierdut")
                if not navigation_sensors_ready(self.obstacle_guard):
                    raise RuntimeError("datele RealSense depth s-au pierdut")
                pose = self.pose_provider()
                final_distance = math.hypot(goal_x - pose["x"], goal_y - pose["y"])
                if final_distance < 0.22:
                    yaw_error = wrap_angle(goal_yaw - pose["yaw"])
                    if abs(yaw_error) < 0.12:
                        await asyncio.to_thread(self.sport_client.stop)
                        self.status.update({"state": "arrived", "error": ""})
                        await self.event_callback({"type": "nav_status", **self.status})
                        return
                    await self._command(0.0, 0.0, max(-0.35, min(0.35, 0.8 * yaw_error)))
                    await asyncio.sleep(0.1)
                    continue

                while waypoint_index < len(path) - 1:
                    wx, wy = path[waypoint_index]
                    if math.hypot(wx - pose["x"], wy - pose["y"]) > 0.30:
                        break
                    waypoint_index += 1
                wx, wy = path[waypoint_index]
                desired = math.atan2(wy - pose["y"], wx - pose["x"])
                heading_error = wrap_angle(desired - pose["yaw"])
                if final_distance < best_distance - 0.08:
                    best_distance = final_distance
                    last_progress = time.monotonic()
                elif time.monotonic() - last_progress > 8.0:
                    raise RuntimeError("robot blocat: poziția nu progresează")

                if abs(heading_error) > 0.55:
                    vx = 0.0
                else:
                    vx = min(0.28, max(0.08, 0.45 * final_distance)) * max(0.25, math.cos(heading_error))
                wz = max(-0.45, min(0.45, 1.0 * heading_error))
                if self.obstacle_guard.is_blocked(vx, 0.0):
                    raise RuntimeError("obstacol detectat în direcția de mers")
                await self._command(vx, 0.0, wz)
                self.status["pose"] = pose
                self.status["waypoint_index"] = waypoint_index
                await self.event_callback({"type": "nav_status", **self.status})
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.status.update({"state": "failed", "error": str(exc)})
            await self.event_callback({"type": "nav_status", **self.status})
        finally:
            await asyncio.to_thread(self.sport_client.stop)
