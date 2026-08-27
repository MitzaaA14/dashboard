#!/usr/bin/env python3
"""
robot_client.py - Client pentru controlul Unitree G1
Rulează pe Orin (192.168.123.164) și comunică cu MCU-ul robotului
via unitree_sdk2py pe rețeaua internă 192.168.123.x

Fallback SSH: dacă SDK-ul nu e disponibil pe laptop,
trimite comenzi ROS2 prin SSH la Orin.
"""

import subprocess
import json
import math
import time
import threading
import os
from typing import Optional, Callable

# -----------------------------------------------------------------------
# Configurare
# -----------------------------------------------------------------------
ROBOT_IP = "192.168.123.161"          # MCU principal robot
ORIN_IP = "192.168.123.164"           # Orin NX
NETWORK_INTERFACE = "enP8p1s0"        # Interfata Orin spre robot
# Mid360 publică uneori la puțin peste o secundă când serviciul nativ SLAM
# procesează simultan pos_info/ctrl_info. Pragul unic de 2 s evită alternanța
# falsă verde/roșu; după acest interval autonomia rămâne fail-safe.
# Mid360 publică uneori în rafale când serviciul SLAM procesează o comandă
# 1102. Jurnalele reale au arătat pauze de aproape 2 s urmate de cadre valide;
# 2,5 s elimină oscilația verde/roșu, iar camera continuă frâna redundantă.
LIDAR_FRESHNESS_MAX_AGE = 2.5

# API-uri adaugate in clientul G1 oficial dupa versiunea instalata pe robot.
# Firmware-ul nou poate ramane pe controlul intern chiar daca SetVelocity(7105)
# raspunde cu succes; 7110 transfera explicit controlul catre clientul extern.
G1_API_GET_FSM_MODE = 7002
G1_API_SWITCH_TO_USER_CTRL = 7110
G1_API_SWITCH_TO_INTERNAL_CTRL = 7111

ROS_ENV = {
    "ROS_LOCALHOST_ONLY": "0",
    "CYCLONEDDS_URI": "/home/unitree/cyclonedds.xml",
    "PYTHONPATH": "/opt/ros/humble/lib/python3.10/site-packages:"
                  "/opt/ros/humble/local/lib/python3.10/dist-packages:"
                  "/usr/lib/python3/dist-packages",
    "PATH": "/opt/ros/humble/bin:/usr/bin:/bin",
    **os.environ
}

# -----------------------------------------------------------------------
# SLAM Client (prin ROS2 topic publish pe Orin)
# -----------------------------------------------------------------------
class SlamClient:
    """
    Controlează SLAM-ul prin topicul /api/slam_operate/request
    """

    def __init__(self):
        self._publish_callback = None
        self._ros_available = self._check_ros()

    def set_publish_callback(self, callback):
        """Bindează o metodă nativă de publicare (de la server.py)."""
        self._publish_callback = callback

    def _check_ros(self) -> bool:
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                env=ROS_ENV, capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _send_command(self, api_id: int, params: dict = None) -> dict:
        if params is None:
            params = {}

        if self._publish_callback:
            success = self._publish_callback(api_id, params)
            return {"success": success, "output": f"Native pub call triggered (API: {api_id})"}

        # Fallback la subprocess CLI (folosind tipul corect unitree_api/msg/Request)
        return self._ros2_pub_fallback(api_id, params)

    def _ros2_pub_fallback(self, api_id: int, params_dict: dict) -> dict:
        import json
        import shlex
        params_str = json.dumps(params_dict)
        
        setup_cmds = (
            "if [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; fi; "
            "for ws in /home/unitree/workspace/unitree_ros2 /home/unitree/unitree_ros2/cyclonedds_ws /home/unitree/unitree_ros2 /home/unitree/cyclonedds_ws /home/unitree/ros2_ws /home/matei/Desktop/Coduri/unitree_interfaces_ws; do "
            "  if [ -f \"$ws/install/setup.bash\" ]; then source \"$ws/install/setup.bash\"; break; fi; "
            "done; "
            "export ROS_LOCALHOST_ONLY=0; "
            "export CYCLONEDDS_URI=/home/unitree/cyclonedds.xml; "
        )
        
        msg_payload = (
            f"{{header: {{identity: {{id: 99, api_id: {api_id}}}, lease: {{id: 0}}, policy: {{priority: 1, noreply: false}}}}, "
            f"parameter: '{params_str}', binary: []}}"
        )
        
        pub_cmd = f"ros2 topic pub --once -w 1 /api/slam_operate/request unitree_api/msg/Request {shlex.quote(msg_payload)}"
        full_script = setup_cmds + pub_cmd
        
        cmd = f"bash -c {shlex.quote(full_script)}"
        
        try:
            result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=8)
            return {"success": result.returncode == 0, "output": result.stdout, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_mapping(self) -> dict:
        # Payload fix cerut de exemplul SDK Unitree pentru Start Mapping.
        return self._send_command(1801, {"data": {"slam_type": "indoor"}})

    def stop_mapping(self) -> dict:
        # Apelăm 1802 (End Mapping) cu o cale temporară pentru a opri curat SLAM-ul
        return self._send_command(1802, {"data": {"address": "/home/unitree/g1_ws/map/temp_map.pcd"}})

    def save_map(self, map_name: str = "my_map") -> dict:
        if not map_name.endswith(".pcd"):
            map_name = f"{map_name}.pcd"
        if not map_name.startswith("/"):
            # Calea directă este forma documentată de exemplul Unitree 1802.
            # Backendul mută ulterior fișierul în directorul său de hărți.
            map_name = f"/home/unitree/{map_name}"
        return self._send_command(1802, {"data": {"address": map_name}})

    def load_map(self, map_name: str) -> dict:
        if not map_name.endswith(".pcd"):
            map_name = f"{map_name}.pcd"
        if not map_name.startswith("/"):
            import os
            base_dir = "/home/unitree/g1_ws/map"
            if not os.path.exists(base_dir):
                base_dir = "/home/unitree"
            map_name = f"{base_dir}/{map_name}"
        params = {
            "data": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "q_x": 0.0,
                "q_y": 0.0,
                "q_z": 0.0,
                "q_w": 1.0,
                "address": map_name
            }
        }
        return self._send_command(1804, params)

    def set_initial_pose(self, x: float, y: float, yaw: float, map_name: str) -> dict:
        import math
        q_z = math.sin(yaw / 2.0)
        q_w = math.cos(yaw / 2.0)
        params = {
            "data": {
                "x": float(x),
                "y": float(y),
                "z": 0.0,
                "q_x": 0.0,
                "q_y": 0.0,
                "q_z": float(q_z),
                "q_w": float(q_w),
                "address": map_name
            }
        }
        return self._send_command(1804, params)

    def start_relocation(self) -> dict:
        return self._send_command(1804)

    def pose_navigation(self, x: float, y: float, yaw: float, speed: float = 0.3) -> dict:
        """Trimite o destinație absolută planificatorului SLAM Unitree (API 1102)."""
        q_z = math.sin(float(yaw) / 2.0)
        q_w = math.cos(float(yaw) / 2.0)
        params = {
            "data": {
                "targetPose": {
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,
                    "q_x": 0.0,
                    "q_y": 0.0,
                    "q_z": q_z,
                    "q_w": q_w,
                },
                "mode": 1,
                "speed": float(speed),
            }
        }
        return self._send_command(1102, params)

    def pause_navigation(self) -> dict:
        """Pauză pentru navigarea nativă Unitree (API 1201)."""
        return self._send_command(1201)

    def resume_navigation(self) -> dict:
        """Continuă navigarea nativă Unitree (API 1202)."""
        return self._send_command(1202)

    def list_maps(self) -> list:
        """Listează hărțile salvate pe Orin."""
        maps = []
        for search_dir in ["/home/unitree/g1_ws/map", "/home/unitree/maps", "/tmp/slam_maps", "/home/unitree"]:
            try:
                result = subprocess.run(
                    ["find", search_dir, "-name", "*.pcd", "-o", "-name", "*.bt"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    maps.extend([f for f in result.stdout.strip().split("\n") if f])
            except Exception:
                pass
        return maps

    def read_pcd_points(self, filepath: str) -> list:
        """Citește un fișier PCD (ASCII) și returnează punctele ca listă de dict-uri."""
        points = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('DATA'):
                    data_start = i + 1
                    break
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        points.append({
                            'x': float(parts[0]),
                            'y': float(parts[1]),
                            'z': float(parts[2])
                        })
                    except ValueError:
                        pass
        except Exception as e:
            print(f"[SlamClient] read_pcd_points eroare: {e}")
        return points


# -----------------------------------------------------------------------
# Sport Client (mișcare robot) via unitree_sdk2py
# -----------------------------------------------------------------------
class SportClient:
    """
    Controlează mișcarea robotului G1 via unitree_sdk2py.
    Dacă SDK-ul nu e disponibil, folosește ROS2 topic publish.
    """

    def __init__(self):
        self._sdk_client = None
        self._arm_action_client = None
        self._arm_action_map = {}
        self._last_fsm_id = None
        self._last_fsm_check = 0.0
        self._last_fsm_mode = None
        self._last_fsm_mode_check = 0.0
        self._sdk_lock = threading.RLock()
        # Testele și uneltele offline nu trebuie să inițializeze participantul
        # DDS doar prin `import server`. Pe robot variabila nu este setată și
        # inițializarea rămâne identică.
        self._sdk_available = (
            False if os.environ.get("G1_SKIP_SDK_INIT") == "1"
            else self._init_sdk()
        )

    def _init_sdk(self) -> bool:
        try:
            import sys
            sys.path.insert(0, '/home/unitree/unitree_sdk2_python')

            # OBLIGATORIU înainte de a crea orice client SDK: inițializează
            # participantul DDS (ChannelFactory). Fără asta, orice Client().Init()
            # eșuează cu "'NoneType' object has no attribute '_ref'" pentru că
            # SDK-ul ține un singleton global neinițializat.
            try:
                from unitree_sdk2py.core.channel import ChannelFactoryInitialize
                ChannelFactoryInitialize(0, NETWORK_INTERFACE)
                print(f"[SportClient] ChannelFactoryInitialize OK pe interfața {NETWORK_INTERFACE}")
            except Exception as e_cf:
                # Dacă a fost deja inițializat (ex: reload la alt import), ignorăm;
                # altfel logăm explicit ca să fie vizibil în consolă.
                print(f"[SportClient] ChannelFactoryInitialize a semnalat: {e_cf!r} (continuăm)")

            # Încercăm să importăm clientul loco pentru G1. NU mai apelăm
            # SetNetworkInterface() pe client — interfața se dă o singură
            # dată, global, prin ChannelFactoryInitialize de mai sus.
            # Clientul G1LocoClient NU are metoda asta în versiunea ta de SDK
            # (de-aia crăpa cu AttributeError și cădea greșit pe fallback-ul Go2).
            g1_error = None
            go2_error = None
            self._client_kind = None  # "g1" | None
            try:
                from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient as G1LocoClient
                self._sdk_client = G1LocoClient()
                self._sdk_client.SetTimeout(5.0)
                self._sdk_client.Init()
                # Compatibilitate cu firmware-ul G1 nou. SDK-ul Python local
                # este anterior acestor doua metode, dar serviciul sport al
                # robotului le expune cu aceleasi ID-uri ca SDK-ul oficial.
                self._sdk_client._RegistApi(G1_API_SWITCH_TO_USER_CTRL, 0)
                self._sdk_client._RegistApi(G1_API_SWITCH_TO_INTERNAL_CTRL, 0)
                self._client_kind = "g1"
                print("[SportClient] G1LocoClient instanțiat nativ persistent.")

                # Același participant DDS poate găzdui clientul oficial pentru
                # gesturile brațelor G1 (API 7106).
                try:
                    from unitree_sdk2py.g1.arm.g1_arm_action_client import (
                        G1ArmActionClient, action_map,
                    )
                    self._arm_action_client = G1ArmActionClient()
                    self._arm_action_client.SetTimeout(10.0)
                    self._arm_action_client.Init()
                    self._arm_action_map = dict(action_map)
                    print("[SportClient] G1ArmActionClient inițializat (API 7106).")
                except Exception as arm_error:
                    self._arm_action_client = None
                    self._arm_action_map = {}
                    print(f"[SportClient] G1ArmActionClient indisponibil: {arm_error!r}")
            except Exception as e:
                g1_error = e
                # IMPORTANT: acest robot e un G1 (biped). Fallback-ul Go2SportClient
                # e făcut pentru un patruped și NU are metodele FSM (Damp/StandUp/Start)
                # ale G1. Îl păstrăm doar ca diagnostic vizibil, nu îl mai folosim ca
                # client activ, ca să nu ascundem eroarea reală de la G1LocoClient.
                print(f"[SportClient] G1LocoClient a eșuat: {g1_error!r}")
                try:
                    from unitree_sdk2py.go2.sport.sport_client import SportClient as Go2SportClient
                    probe = Go2SportClient()
                    probe.Init()
                    go2_error = None
                    print("[SportClient] Go2SportClient (sondă) s-a inițializat, dar NU e folosit "
                          "— robotul e G1, API-ul Go2 e incompatibil. Repară eroarea G1LocoClient de mai sus.")
                except Exception as e2:
                    go2_error = e2
                    print(f"[SportClient] Go2SportClient (sondă) a eșuat: {go2_error!r}")

            return self._sdk_client is not None and self._client_kind == "g1"
        except Exception as e:
            print(f"[SportClient] Eroare la inițializarea nativă SDK: {e!r}")
            return False

    def damp(self) -> dict:
        """FSM 1 - mod damping (moale, dar rezistent). Prim pas obligatoriu
        înainte de ridicare/Start(), chiar dacă robotul pare deja moale."""
        if self._sdk_available and self._sdk_client:
            try:
                self._sdk_client.Damp()
                return {"success": True, "output": "Damp() trimis (FSM 1)"}
            except Exception as e:
                return {"success": False, "error": f"Damp error: {e}"}
        return {"success": False, "error": "SDK indisponibil"}

    def zero_torque(self) -> dict:
        """FSM 0 - Cuplu Zero."""
        if self._sdk_available and self._sdk_client:
            try:
                ret = self._sdk_client.SetFsmId(0)
                return {"success": True, "output": f"SetFsmId(0) [Zero Torque] trimis -> ret={ret}"}
            except Exception as e:
                return {"success": False, "error": f"SetFsmId(0) error: {e}"}
        return {"success": False, "error": "SDK indisponibil"}

    def get_fsm_id(self) -> dict:
        """Citește starea FSM curentă a robotului (diagnostic)."""
        if self._sdk_available and self._sdk_client:
            try:
                if not hasattr(self._sdk_client, "GetFsmId"):
                    return {"success": False, "error": "GetFsmId indisponibil în această versiune de SDK"}
                ret = self._sdk_client.GetFsmId()
                return {"success": True, "output": f"FSM curent: {ret}"}
            except Exception as e:
                return {"success": False, "error": f"GetFsmId error: {e}"}
        return {"success": False, "error": "SDK indisponibil"}

    def stand_up(self) -> dict:
        """Ridicare în picioare (FSM 4 - Preparation)."""
        if not (self._sdk_available and self._sdk_client):
            return {"success": False, "error": "SDK indisponibil"}
        try:
            ret = self._sdk_client.SetFsmId(4)
            return {"success": True, "output": f"SetFsmId(4, Preparation) trimis -> ret={ret}"}
        except Exception as e:
            return {"success": False, "error": f"SetFsmId(4) error: {e}"}

    def start_locomotion(self) -> dict:
        """FSM 801 - Locomotivă activă (Running G1)."""
        if not (self._sdk_available and self._sdk_client):
            return {"success": False, "error": "SDK indisponibil"}
        try:
            ret = self._sdk_client.SetFsmId(801)
            return {"success": True, "output": f"SetFsmId(801, Running Mode) trimis -> ret={ret}"}
        except Exception as e:
            return {"success": False, "error": f"SetFsmId(801) error: {e}"}

    def set_fsm_id(self, fsm_id: int) -> dict:
        """Setează direct un FSM."""
        if self._sdk_available and self._sdk_client:
            try:
                if not hasattr(self._sdk_client, "SetFsmId"):
                    return {"success": False, "error": "SetFsmId indisponibil în această versiune de SDK"}
                ret = self._sdk_client.SetFsmId(fsm_id)
                return {"success": True, "output": f"SetFsmId({fsm_id}) trimis -> ret={ret}"}
            except Exception as e:
                return {"success": False, "error": f"SetFsmId({fsm_id}) error: {e}"}
        return {"success": False, "error": "SDK indisponibil"}

    def wake_up_sequence(self) -> dict:
        """Secvența completă Damp (1) -> SetFsmId(4) (Preparation) -> HighStand -> SetFsmId(801) (Running)"""
        if not (self._sdk_available and self._sdk_client):
            return {"success": False, "error": "SDK indisponibil - vezi consola serverului pentru detalii de inițializare"}
        try:
            ret_damp = self._sdk_client.Damp()
            print(f"[SportClient] Damp() -> ret={ret_damp}")
            time.sleep(1.5)

            ret_stand = self._sdk_client.SetFsmId(4)
            print(f"[SportClient] SetFsmId(4, Preparation) -> ret={ret_stand}")
            if ret_stand not in (0, None):
                return {
                    "success": False,
                    "error": f"SetFsmId(4) a eșuat cu ret={ret_stand} — Damp() a mers, robotul a rămas jos"
                }
            time.sleep(5.0)

            # Ridică postura la înălțime normală/dreaptă (opțional)
            ret_high = None
            if hasattr(self._sdk_client, "HighStand"):
                try:
                    ret_high = self._sdk_client.HighStand()
                    print(f"[SportClient] HighStand() -> ret={ret_high}")
                except Exception as e_high:
                    print(f"[SportClient] HighStand() a eșuat: {e_high!r} (continuăm oricum)")
            time.sleep(2.0)

            ret_start = self._sdk_client.SetFsmId(801)
            print(f"[SportClient] SetFsmId(801, Running Mode) -> ret={ret_start}")
            time.sleep(1.0)

            return {
                "success": True,
                "output": (
                    f"Damp ret={ret_damp}, SetFsmId(4) ret={ret_stand}, "
                    f"HighStand ret={ret_high}, SetFsmId(801) ret={ret_start}"
                )
            }
        except Exception as e:
            print(f"[SportClient] wake_up_sequence exception: {e!r}")
            return {"success": False, "error": f"Wake-up sequence error: {e}"}

    def move_to(self, x: float, y: float, yaw: float, duration: float = 0.35) -> dict:
        """
        Trimite o comandă de viteză directă prin SDK-ul G1.

        Nu există fallback ROS2 aici: /move_base_simple/goal primește poziții
        absolute, nu viteze. Folosirea lui ca fallback pentru teleop ar putea
        transforma vx/vy/vyaw într-o destinație autonomă și nu ar mai exista
        o oprire sigură la pierderea conexiunii.
        """
        if self._sdk_available and self._sdk_client:
            try:
                with self._sdk_lock:
                    if not self.is_locomotion_ready():
                        return {"success": False, "error": f"FSM {self.get_current_fsm_id()} nu este în mod locomotion"}
                    # Move() din SDK nu returnează codul SetVelocity și ascunde
                    # comenzile respinse. Verificăm direct codul API (0 = succes).
                    command_duration = min(1.2, max(0.10, float(duration)))
                    ret = self._sdk_client.SetVelocity(
                        float(x), float(y), float(yaw), command_duration
                    )
                if ret != 0:
                    return {"success": False, "error": f"SetVelocity respins de robot: ret={ret}"}
                return {
                    "success": True,
                    "output": (
                        f"SetVelocity ret=0 vx={x:.3f} vy={y:.3f} "
                        f"wz={yaw:.3f} duration={command_duration:.2f}"
                    ),
                }
            except Exception as e:
                return {"success": False, "error": f"SetVelocity error: {e}"}
        return {
            "success": False,
            "error": "Teleop refuzat: SDK-ul G1 nu este disponibil; fallback-ul de navigare nu este sigur"
        }

    def is_sdk_available(self) -> bool:
        """Folosit de procesele de teleop pentru a porni numai în mod sigur."""
        return bool(self._sdk_available and self._sdk_client)

    def execute_gesture(self, gesture: str) -> dict:
        """Execută un gest predefinit Unitree; nu acceptă ID-uri arbitrare."""
        gesture_actions = {
            "wave": "high wave",
            "kiss": "two-hand kiss",
            "handshake": "shake hand",
            "clap": "clap",
        }
        action_name = gesture_actions.get(str(gesture or "").strip().lower())
        if not action_name:
            return {"success": False, "error": f"Gest necunoscut: {gesture}"}
        if not self._arm_action_client:
            return {"success": False, "error": "API-ul G1 Arm Action 7106 nu este disponibil"}
        if not self.is_locomotion_ready():
            return {
                "success": False,
                "error": f"Gest refuzat: FSM {self.get_current_fsm_id()} nu este stabil pentru acțiuni",
            }
        action_id = self._arm_action_map.get(action_name)
        if action_id is None:
            return {"success": False, "error": f"Acțiunea '{action_name}' lipsește din firmware"}
        try:
            ret = self._arm_action_client.ExecuteAction(int(action_id))
            if ret not in (0, None):
                return {"success": False, "error": f"API 7106 a respins gestul: ret={ret}"}
            return {
                "success": True,
                "output": f"Gest '{action_name}' pornit prin API 7106 (id={action_id}, ret={ret})",
                "gesture": gesture,
                "action": action_name,
                "action_id": int(action_id),
            }
        except Exception as exc:
            return {"success": False, "error": f"API 7106 gesture error: {exc}"}

    def release_arms(self) -> dict:
        """Revine controlat la postura neutră după un gest programat."""
        if not self._arm_action_client:
            return {"success": False, "error": "API-ul G1 Arm Action 7106 nu este disponibil"}
        action_id = self._arm_action_map.get("release arm", 99)
        try:
            ret = self._arm_action_client.ExecuteAction(int(action_id))
            return {
                "success": ret in (0, None),
                "output": f"Brațe eliberate: ret={ret}",
                "error": "" if ret in (0, None) else f"Release arm respins: ret={ret}",
            }
        except Exception as exc:
            return {"success": False, "error": f"Release arm error: {exc}"}

    def get_current_fsm_id(self, force: bool = False) -> Optional[int]:
        if not (self._sdk_available and self._sdk_client):
            return None
        if not force and time.monotonic() - self._last_fsm_check < 1.0:
            return self._last_fsm_id
        try:
            with self._sdk_lock:
                result = self._sdk_client.GetFsmId()
                if isinstance(result, tuple) and len(result) == 2:
                    code, fsm_id = result
                    value = int(fsm_id) if code == 0 and fsm_id is not None else None
                else:
                    value = int(result) if result is not None else None
                self._last_fsm_id = value
                self._last_fsm_check = time.monotonic()
                return value
        except Exception:
            return None

    def get_current_fsm_mode(self, force: bool = False) -> Optional[int]:
        """Citeste submodul FSM, inclusiv din SDK-urile Python fara wrapper."""
        if not (self._sdk_available and self._sdk_client):
            return None
        if (not force
                and time.monotonic() - self._last_fsm_mode_check < 1.0):
            return self._last_fsm_mode
        try:
            with self._sdk_lock:
                code, payload = self._sdk_client._Call(G1_API_GET_FSM_MODE, "{}")
                value = None
                if code == 0:
                    parsed = json.loads(payload or "{}")
                    raw = parsed.get("data") if isinstance(parsed, dict) else parsed
                    value = int(raw) if raw is not None else None
                self._last_fsm_mode = value
                self._last_fsm_mode_check = time.monotonic()
                return value
        except Exception:
            return None

    def prepare_autonomous_control(self) -> dict:
        """Transfera controlul vitezei de la controllerul intern la SDK.

        Pe firmware-ul nou, FSM 802 singur nu confirma ca 7105 este si
        executat. API 7110 este etapa oficiala care face comenzile externe
        active; fara ea robotul poate raspunde ret=0 si totusi sa ramana pe loc.
        """
        if not (self._sdk_available and self._sdk_client):
            return {"success": False, "error": "SDK-ul G1 nu este disponibil"}
        try:
            with self._sdk_lock:
                fsm_before = self.get_current_fsm_id(force=True)
                if fsm_before not in {500, 501, 502, 801, 802}:
                    return {
                        "success": False,
                        "error": f"Nu preiau controlul: FSM {fsm_before} nu este locomotion",
                    }
                parameter = json.dumps({"data": False})
                ret, _ = self._sdk_client._Call(
                    G1_API_SWITCH_TO_USER_CTRL, parameter
                )
                if ret != 0:
                    return {
                        "success": False,
                        "error": (
                            "Robotul nu a transferat controlul locomotiei catre "
                            f"SDK: SwitchToUserCtrl ret={ret}"
                        ),
                        "fsm_id": fsm_before,
                    }
                self._last_fsm_check = 0.0
                self._last_fsm_mode_check = 0.0
                time.sleep(0.15)
                fsm_after = self.get_current_fsm_id(force=True)
                fsm_mode = self.get_current_fsm_mode(force=True)
                if fsm_after not in {500, 501, 502, 801, 802}:
                    return {
                        "success": False,
                        "error": (
                            "Controlul a fost transferat, dar robotul a iesit din "
                            f"locomotion (FSM {fsm_after})"
                        ),
                        "fsm_id": fsm_after,
                        "fsm_mode": fsm_mode,
                    }
                return {
                    "success": True,
                    "output": (
                        "SwitchToUserCtrl ret=0; control extern activ "
                        f"(FSM {fsm_after}, mode {fsm_mode})"
                    ),
                    "fsm_id": fsm_after,
                    "fsm_mode": fsm_mode,
                    "api": G1_API_SWITCH_TO_USER_CTRL,
                }
        except Exception as exc:
            return {
                "success": False,
                "error": f"SwitchToUserCtrl a esuat: {exc}",
            }

    def is_locomotion_ready(self) -> bool:
        # Acest G1 raportează 802 după tranziția Run 801.
        return self.get_current_fsm_id() in {500, 501, 502, 801, 802}

    def navigate_to(self, x: float, y: float, yaw: float) -> dict:
        """
        Navigare autonomă (Coordonate pe hartă).
        Trimite o coordonată absolută (x, y, yaw) către planificatorul ROS 2
        al robotului (pe topicul /move_base_simple/goal), ocolind SDK-ul de teleoperare.
        """
        return self._ros2_move_to(x, y, yaw)

    def _ros2_move_to(self, x: float, y: float, yaw: float) -> dict:
        """Navigare via ROS2 (fallback)."""
        import math
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        msg = (
            f'{{"header": {{"frame_id": "map"}}, '
            f'"pose": {{"position": {{"x": {x}, "y": {y}, "z": 0.0}}, '
            f'"orientation": {{"x": 0.0, "y": 0.0, "z": {qz:.4f}, "w": {qw:.4f}}}}}}}'
        )
        try:
            result = subprocess.run(
                ["ros2", "topic", "pub", "--once",
                 "/move_base_simple/goal",
                 "geometry_msgs/msg/PoseStamped", msg],
                env=ROS_ENV, capture_output=True, text=True, timeout=10
            )
            return {"success": result.returncode == 0, "output": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop(self) -> dict:
        """Oprește robotul."""
        if self._sdk_available and self._sdk_client:
            try:
                with self._sdk_lock:
                    if hasattr(self._sdk_client, "SetVelocity"):
                        ret = self._sdk_client.SetVelocity(0.0, 0.0, 0.0, 0.2)
                    elif hasattr(self._sdk_client, "StopMove"):
                        ret = self._sdk_client.StopMove()
                    else:
                        ret = self._sdk_client.Stop()
                if ret not in (0, None):
                    return {"success": False, "error": f"Stop respins: ret={ret}"}
                return {"success": True, "output": f"Stop direct: ret={ret}"}
            except Exception as e:
                return {"success": False, "error": f"Stop direct error: {e}"}
        else:
            return {"success": False, "error": "SDK indisponibil, oprirea nativă nu e implementată"}

    def get_status(self) -> dict:
        """Obține statusul curent al robotului."""
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", ROBOT_IP],
                capture_output=True, timeout=3
            )
            robot_reachable = result.returncode == 0
        except Exception:
            robot_reachable = False

        return {
            "robot_reachable": robot_reachable,
            "robot_ip": ROBOT_IP,
            "orin_ip": ORIN_IP,
            "sdk_available": self._sdk_available
        }


# -----------------------------------------------------------------------
# Odometry Reader (poziția robotului din ROS2)
# -----------------------------------------------------------------------
class OdometryReader:
    """Citește odometria robotului din topicurile ROS2 sau din shared state."""

    ODOM_TOPICS = [
        "/state_estimator/fusion_odom",
        "/unitree/slam_mapping/odom",
        "/unitree_slam/high_rate_odometry",
        "/odom"
    ]

    def __init__(self):
        self._state_provider = None

    def set_state_provider(self, callback):
        """Bindează un furnizor de stare dinamic pentru a evita ros2 topic echo subprocesses."""
        self._state_provider = callback

    def get_latest_pose(self) -> Optional[dict]:
        """Încearcă să citească poziția din shared state sau din topicurile active."""
        if self._state_provider:
            return self._state_provider()

        for topic in self.ODOM_TOPICS:
            pose = self._read_topic(topic)
            if pose:
                return pose
        return None

    def _read_topic(self, topic: str) -> Optional[dict]:
        try:
            result = subprocess.run(
                ["ros2", "topic", "echo", "--once", topic],
                env=ROS_ENV, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout:
                return self._parse_odom_yaml(result.stdout, topic)
        except Exception:
            pass
        return None

    def _parse_odom_yaml(self, yaml_str: str, source_topic: str) -> Optional[dict]:
        """Parsează output-ul de la ros2 topic echo."""
        import re
        try:
            px = re.search(r'position:.*?x: ([+-]?\d+\.?\d*)', yaml_str, re.DOTALL)
            py = re.search(r'position:.*?y: ([+-]?\d+\.?\d*)', yaml_str, re.DOTALL)
            ox = re.search(r'orientation:.*?x: ([+-]?\d+\.?\d*)', yaml_str, re.DOTALL)
            oy = re.search(r'orientation:.*?y: ([+-]?\d+\.?\d*)', yaml_str, re.DOTALL)
            oz = re.search(r'orientation:.*?z: ([+-]?\d+\.?\d*)', yaml_str, re.DOTALL)
            ow = re.search(r'orientation:.*?w: ([+-]?\d+\.?\d*)', yaml_str, re.DOTALL)

            if px and py:
                import math
                qz = float(oz.group(1)) if oz else 0.0
                qw = float(ow.group(1)) if ow else 1.0
                yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
                return {
                    "x": float(px.group(1)),
                    "y": float(py.group(1)),
                    "yaw": yaw,
                    "yaw_deg": math.degrees(yaw),
                    "source": source_topic
                }
        except Exception:
            pass
        return None


# -----------------------------------------------------------------------
# Obstacle Guard (blocare mișcare pe baza detectării de obstacole)
# -----------------------------------------------------------------------
class ObstacleGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._camera_blocked = set()
        self._camera_warning = set()
        self._camera_distances = {}
        self._lidar_blocked = set()
        self._camera_last_update = 0.0
        self._lidar_last_update = 0.0
        self._camera_center_distance = None
        self._lidar_center_distance = None
        self._lidar_center_vector = None
        self._lidar_zone_distances = {}
        self._lidar_obstacle_shape = []
        self._lidar_self_filtered_points = 0
        self._lidar_source = None
        self._lidar_tracks = {}
        self._next_lidar_track_id = 1
        self._floor_plane = None
        self._static_resolution = None
        self._static_raw_cells = set()
        self._static_match_offsets = ((0, 0),)
        self._robot_radius = 0.42
        self._obstacle_min_z = 0.10
        self._obstacle_max_z = 1.85

    def _clear_lidar_locked(self) -> None:
        self._lidar_blocked = set()
        self._lidar_center_distance = None
        self._lidar_center_vector = None
        self._lidar_zone_distances = {}
        self._lidar_obstacle_shape = []
        self._lidar_self_filtered_points = 0
        self._lidar_source = None
        self._lidar_tracks = {}
        self._next_lidar_track_id = 1
        self._lidar_last_update = 0.0

    def set_floor_plane(self, plane) -> None:
        """Primește planul z=a*x+b*y+c detectat de costmapul hărții încărcate."""
        with self._lock:
            self._floor_plane = dict(plane) if plane else None
            self._clear_lidar_locked()

    def floor_plane(self):
        """Snapshot thread-safe al planului podelei folosit de filtrul LiDAR."""
        with self._lock:
            return dict(self._floor_plane) if self._floor_plane else None

    def configure_navigation_map(self, plane, resolution: float, raw_static_cells,
                                 robot_radius: float, obstacle_min_z: float = 0.10,
                                 obstacle_max_z: float = 1.85) -> None:
        """Configurează filtrul live în exact același frame și la aceleași cote ca A*."""
        resolution = max(0.03, float(resolution))
        match_radius = max(0.10, min(0.18, resolution))
        span = max(1, math.ceil(match_radius / resolution))
        offsets = []
        for offset_x in range(-span, span + 1):
            for offset_y in range(-span, span + 1):
                if math.hypot(offset_x * resolution, offset_y * resolution) <= match_radius + resolution * 0.35:
                    offsets.append((offset_x, offset_y))
        with self._lock:
            self._floor_plane = dict(plane) if plane else None
            self._static_resolution = resolution
            self._static_raw_cells = set(raw_static_cells or ())
            self._static_match_offsets = tuple(offsets) or ((0, 0),)
            # Pentru costmapul local folosim aceeași rază hard ca plannerul;
            # pragurile de urgență ale camerei rămân independente mai jos.
            self._robot_radius = max(0.10, float(robot_radius))
            self._obstacle_min_z = float(obstacle_min_z)
            self._obstacle_max_z = float(obstacle_max_z)
            self._clear_lidar_locked()

    def update(self, zones: dict):
        blocked = {name for name, zone in zones.items() if zone.get("level") == "danger"}
        warning = {name for name, zone in zones.items() if zone.get("level") == "warning"}
        distances = {
            name: float(zone["dist"])
            for name, zone in zones.items()
            if zone.get("dist") is not None
        }
        with self._lock:
            self._camera_blocked = blocked
            self._camera_warning = warning
            self._camera_distances = distances
            center = zones.get("center", {}).get("dist")
            self._camera_center_distance = float(center) if center is not None else None
            self._camera_last_update = time.time()

    def update_lidar_points(self, points: list, pose: dict,
                            source: str = "slam_global_points") -> None:
        """Extrage clustere 2D temporare din cloud-ul live în frame map."""
        yaw = float(pose.get("yaw", 0.0))
        pose_x = float(pose.get("x", 0.0))
        pose_y = float(pose.get("y", 0.0))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        with self._lock:
            floor_plane = dict(self._floor_plane) if self._floor_plane else None
            static_resolution = self._static_resolution
            static_cells = set(self._static_raw_cells)
            static_offsets = tuple(self._static_match_offsets)
            robot_radius = self._robot_radius
            obstacle_min_z = self._obstacle_min_z
            obstacle_max_z = self._obstacle_max_z

        grid_resolution = max(0.08, min(0.12, static_resolution or 0.10))
        grid_samples = {}
        self_filtered_points = 0
        for point in points:
            x = float(point["x"])
            y = float(point["y"])
            z = float(point.get("z", 0.0))
            relative_z = z
            if floor_plane:
                relative_z = z - (
                    float(floor_plane.get("a", 0.0)) * x
                    + float(floor_plane.get("b", 0.0)) * y
                    + float(floor_plane.get("c", 0.0))
                )
            if not obstacle_min_z <= relative_z <= obstacle_max_z:
                continue

            delta_x, delta_y = x - pose_x, y - pose_y
            forward = cosine * delta_x + sine * delta_y
            left = -sine * delta_x + cosine * delta_y
            # Pelvisul, picioarele și carcasa G1 apar uneori în Mid360 la
            # 8-20 cm de originea base_link. Aceste puncte sunt în interiorul
            # corpului, deci nu pot reprezenta un obstacol evitabil. Fără
            # filtrul de amprentă formau un inel dinamic și blocau robotul în
            # prima milisecundă, deși RealSense vedea culoarul liber.
            if math.hypot(forward, left) <= max(0.28, robot_radius + 0.08):
                self_filtered_points += 1
                continue
            # Costmap local pe aproximativ 2 m: suficient pentru a păstra un
            # scaun deja văzut când apare un al doilea obiect, fără întregul cloud.
            if forward <= -0.35 or math.hypot(forward, left) > 2.0 or abs(left) > 1.50:
                continue

            if static_resolution and static_cells:
                static_cell = (round(x / static_resolution), round(y / static_resolution))
                if any(
                    (static_cell[0] + offset_x, static_cell[1] + offset_y) in static_cells
                    for offset_x, offset_y in static_offsets
                ):
                    # Punctul corespunde unui perete/mobilier deja prezent în PCD.
                    continue

            grid_cell = (round(x / grid_resolution), round(y / grid_resolution))
            grid_samples.setdefault(grid_cell, []).append((x, y, forward, left))

        # Clustering pe grilă: zgomotul izolat nu devine obstacol, dar suprafețele
        # și picioarele unui scaun rămân reprezentate prin forma lor reală.
        remaining = set(grid_samples)
        clusters = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            queue = [seed]
            while queue:
                cell_x, cell_y = queue.pop()
                for offset_x in (-1, 0, 1):
                    for offset_y in (-1, 0, 1):
                        neighbor = (cell_x + offset_x, cell_y + offset_y)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
            samples = [sample for cell in component for sample in grid_samples[cell]]
            if len(samples) >= 4:
                clusters.append({"cells": component, "samples": samples})

        observed_at = time.time()
        with self._lock:
            tracks = self._lidar_tracks
            unmatched_track_ids = set(tracks)

            for cluster in clusters:
                samples = cluster["samples"]
                centroid = (
                    sum(sample[0] for sample in samples) / len(samples),
                    sum(sample[1] for sample in samples) / len(samples),
                )
                nearest_id = None
                nearest_distance = 0.36
                for track_id in unmatched_track_ids:
                    track = tracks[track_id]
                    distance = math.hypot(
                        centroid[0] - track["centroid"][0],
                        centroid[1] - track["centroid"][1],
                    )
                    if distance < nearest_distance:
                        nearest_id = track_id
                        nearest_distance = distance
                if nearest_id is None:
                    nearest_id = self._next_lidar_track_id
                    self._next_lidar_track_id += 1
                    tracks[nearest_id] = {
                        "centroid": centroid, "cells": set(cluster["cells"]),
                        "points": [(sample[0], sample[1]) for sample in samples],
                        "hits": 1, "last_seen": observed_at,
                    }
                else:
                    unmatched_track_ids.discard(nearest_id)
                    track = tracks[nearest_id]
                    track.update({
                        "centroid": centroid, "cells": set(cluster["cells"]),
                        "points": [(sample[0], sample[1]) for sample in samples],
                        "hits": min(20, int(track.get("hits", 1)) + 1),
                        "last_seen": observed_at,
                    })

            # Un track aflat încă în câmpul frontal se șterge după 0,8 s de
            # spațiu observat liber. În unghi mort îl păstrăm 2,5 s, astfel încât
            # un scaun să nu dispară doar fiindcă robotul a ajuns lângă el.
            for track_id in list(tracks):
                track = tracks[track_id]
                age = observed_at - float(track.get("last_seen", 0.0))
                delta_x = track["centroid"][0] - pose_x
                delta_y = track["centroid"][1] - pose_y
                forward = cosine * delta_x + sine * delta_y
                left = -sine * delta_x + cosine * delta_y
                visible = 0.05 < forward < 1.80 and abs(left) < 1.20
                if age > 2.5 or (visible and age > 0.8):
                    tracks.pop(track_id, None)

            confirmed_clusters = []
            for track in tracks.values():
                transformed_samples = []
                for world_x, world_y in track.get("points", ()):
                    world_x = float(world_x)
                    world_y = float(world_y)
                    delta_x, delta_y = world_x - pose_x, world_y - pose_y
                    forward = cosine * delta_x + sine * delta_y
                    left = -sine * delta_x + cosine * delta_y
                    transformed_samples.append((world_x, world_y, forward, left))
                nearest_forward = min(
                    (sample[2] for sample in transformed_samples if sample[2] > 0.0),
                    default=math.inf,
                )
                if int(track.get("hits", 0)) >= 2 or nearest_forward <= 0.60:
                    confirmed_clusters.append({
                        "cells": set(track["cells"]),
                        "samples": transformed_samples,
                    })

            shape_cells = {
                cell for cluster in confirmed_clusters for cell in cluster["cells"]
            }
            obstacle_shape = [
                (cell_x * grid_resolution, cell_y * grid_resolution)
                for cell_x, cell_y in sorted(shape_cells)
            ][:320]

            blocking_half_width = robot_radius + 0.06
            blocking_clusters = [
                cluster for cluster in confirmed_clusters
                if any(0.08 < sample[2] < 0.90
                       and abs(sample[3]) <= blocking_half_width
                       for sample in cluster["samples"])
            ]
            center_vector = None
            center_distance = None
            blocked = set()
            zone_distances = {}

            if blocking_clusters:
                blocked.add("center")
                primary = min(
                    blocking_clusters,
                    key=lambda cluster: min(
                        sample[2] for sample in cluster["samples"]
                        if 0.08 < sample[2] < 0.90
                        and abs(sample[3]) <= blocking_half_width
                    ),
                )
                crossing = [
                    sample for sample in primary["samples"]
                    if 0.08 < sample[2] < 0.90
                    and abs(sample[3]) <= blocking_half_width
                ]
                crossing.sort(key=lambda sample: sample[2])
                sample_count = max(1, math.ceil(len(crossing) * 0.35))
                nearest = crossing[:sample_count]
                forward_values = sorted(sample[2] for sample in nearest)
                left_values = sorted(sample[3] for sample in nearest)
                middle = len(nearest) // 2
                center_vector = (forward_values[middle], left_values[middle])
                center_distance = math.hypot(*center_vector)
                zone_distances["center"] = center_distance

            side_limit = blocking_half_width + 0.35
            left_samples = [
                sample
                for cluster in confirmed_clusters
                for sample in cluster["samples"]
                if 0.05 < sample[2] < 0.95
                and blocking_half_width < sample[3] <= side_limit
            ]
            right_samples = [
                sample
                for cluster in confirmed_clusters
                for sample in cluster["samples"]
                if 0.05 < sample[2] < 0.95
                and -side_limit <= sample[3] < -blocking_half_width
            ]
            if left_samples:
                blocked.add("left")
                zone_distances["left"] = min(
                    math.hypot(sample[2], sample[3]) for sample in left_samples
                )
            if right_samples:
                blocked.add("right")
                zone_distances["right"] = min(
                    math.hypot(sample[2], sample[3]) for sample in right_samples
                )

            self._lidar_blocked = blocked
            self._lidar_center_distance = center_distance
            self._lidar_center_vector = center_vector
            self._lidar_zone_distances = zone_distances
            self._lidar_obstacle_shape = obstacle_shape
            self._lidar_self_filtered_points = self_filtered_points
            self._lidar_source = str(source)
            self._lidar_last_update = observed_at

    def is_blocked(self, vx: float, vy: float) -> bool:
        """Protecție generală pentru teleoperare: combină camera și LiDAR-ul."""
        with self._lock:
            now = time.time()
            blocked = set()
            if now - self._camera_last_update <= 2.0:
                blocked.update(self._camera_blocked)
            if now - self._lidar_last_update <= LIDAR_FRESHNESS_MAX_AGE:
                blocked.update(self._lidar_blocked)

        if not blocked:
            return False
        # La mers înainte protejăm și mâinile/umerii, nu doar axa pelvisului.
        if vx > 0.05 and blocked.intersection({"left", "center", "right"}):
            return True
        if vy > 0.05 and blocked.intersection({"left", "center"}):
            return True
        if vy < -0.05 and blocked.intersection({"right", "center"}):
            return True
        return False

    def is_navigation_blocked(self) -> bool:
        """Protejează axa corpului și anvelopa apropiată a mâinilor.

        Clusterele laterale depărtate rămân doar în costmap și în
        ``is_lateral_clear``. Un cluster lateral sub 0,56 m oprește însă mersul
        înainte: pelvisul poate avea culoar liber în timp ce mâna sau umărul
        ating colțul unui scaun.
        """
        with self._lock:
            now = time.time()
            lidar_blocked = False
            if now - self._lidar_last_update <= LIDAR_FRESHNESS_MAX_AGE:
                lidar_blocked = "center" in self._lidar_blocked
                if not lidar_blocked:
                    lidar_blocked = any(
                        zone in self._lidar_blocked
                        and self._lidar_zone_distances.get(zone, math.inf) <= 0.56
                        for zone in ("left", "right")
                    )
            camera_emergency = False
            if now - self._camera_last_update <= 2.0:
                for zone in self._camera_blocked:
                    distance = self._camera_distances.get(zone)
                    # Camera depth nu oferă geometrie mapabilă suficientă. Un
                    # scaun aflat lateral la 0,6-0,7 m bloca înainte întreaga
                    # direcție frontală, chiar dacă LiDAR-ul confirma culoarul.
                    # Păstrăm frâna centrală și una laterală numai la distanță
                    # imediată; protecția teleop/SetVelocity rămâne mai strictă
                    # în `is_blocked`, iar LiDAR-ul continuă să blocheze forma.
                    limit = 0.58 if zone == "center" else 0.44
                    if distance is not None and distance <= limit:
                        camera_emergency = True
                        break
                if not camera_emergency:
                    # Pentru autonomie reacționăm predictiv la obiectele văzute
                    # de RealSense înainte de contact. Pragurile vechi de
                    # 0,82/0,90 m transformau mobilierul lateral, aflat în afara
                    # traseului, într-un obstacol circular fals foarte mare.
                    for zone in self._camera_warning:
                        distance = self._camera_distances.get(zone)
                        limit = 0.66 if zone == "center" else 0.42
                        if distance is not None and distance <= limit:
                            camera_emergency = True
                            break
            return lidar_blocked or camera_emergency

    def is_lateral_clear(self, direction: int) -> bool:
        """Verifică numai spațiul în direcția pasului, nu scaunul din față."""
        side = "left" if int(direction) > 0 else "right"
        with self._lock:
            now = time.time()
            if now - self._camera_last_update <= 2.0 and side in self._camera_blocked:
                return False
            if (now - self._camera_last_update <= 2.0
                    and side in self._camera_warning
                    and self._camera_distances.get(side, math.inf) <= 0.60):
                return False
            if (now - self._lidar_last_update <= LIDAR_FRESHNESS_MAX_AGE
                    and side in self._lidar_blocked):
                return False
        return True

    def has_fresh_data(self, max_age: float = 2.0) -> bool:
        """Compatibilitate pentru teleop/diagnostic: cel puțin un flux recent."""
        status = self.sensor_status(max_age)
        return bool(status["camera_fresh"] or status["lidar_fresh"])

    def sensor_status(self, max_age: float = 2.0) -> dict:
        """Prospețimea separată a surselor complementare de evitare."""
        with self._lock:
            now = time.time()
            camera_fresh = self._camera_last_update > 0.0 and now - self._camera_last_update <= max_age
            lidar_fresh = (
                self._lidar_last_update > 0.0
                and now - self._lidar_last_update
                    <= min(max_age, LIDAR_FRESHNESS_MAX_AGE)
            )
            camera_zones = {
                zone: "danger" for zone in sorted(self._camera_blocked)
            }
            camera_zones.update({
                zone: "warning"
                for zone in sorted(self._camera_warning - self._camera_blocked)
            })
            return {
                "camera_fresh": camera_fresh,
                "lidar_fresh": lidar_fresh,
                "camera_age": None if self._camera_last_update <= 0.0 else max(0.0, now - self._camera_last_update),
                "lidar_age": None if self._lidar_last_update <= 0.0 else max(0.0, now - self._lidar_last_update),
                "camera_zones": camera_zones,
                "camera_distances": dict(self._camera_distances),
                "lidar_zones": sorted(self._lidar_blocked),
                "lidar_zone_distances": dict(self._lidar_zone_distances),
                "lidar_center_distance": self._lidar_center_distance,
                "lidar_self_filtered_points": self._lidar_self_filtered_points,
                "lidar_source": self._lidar_source,
            }

    def navigation_sensors_ready(self, max_age: float = 2.0) -> bool:
        """Autonomia cere simultan camera pentru mâini și LiDAR pentru geometrie."""
        status = self.sensor_status(max_age)
        return bool(status["camera_fresh"] and status["lidar_fresh"])

    def front_obstacle_shape(self):
        """Forma 2D curentă a obiectului necunoscut care intersectează coridorul robotului."""
        with self._lock:
            if (time.time() - self._lidar_last_update > LIDAR_FRESHNESS_MAX_AGE
                    or not self._lidar_obstacle_shape):
                return None
            return {
                "points": list(self._lidar_obstacle_shape),
                "distance": self._lidar_center_distance,
                "vector": self._lidar_center_vector,
            }

    def front_obstacle_distance(self, default: float = 0.70) -> float:
        forward, left = self.front_obstacle_vector(default)
        return math.hypot(forward, left)

    def front_obstacle_vector(self, default: float = 0.70):
        """Poziția obstacolului (înainte, stânga) în cadrul robotului."""
        with self._lock:
            now = time.time()
            if (now - self._lidar_last_update <= LIDAR_FRESHNESS_MAX_AGE
                    and self._lidar_center_vector is not None):
                forward, left = self._lidar_center_vector
            elif (now - self._camera_last_update <= 2.0
                  and (self._camera_blocked or self._camera_warning)):
                candidates = []
                active_zones = self._camera_blocked or self._camera_warning
                for zone in active_zones:
                    distance = self._camera_distances.get(zone)
                    if distance is None:
                        continue
                    lateral = 0.0
                    if zone == "left":
                        lateral = max(0.32, self._robot_radius + 0.08)
                    elif zone == "right":
                        lateral = -max(0.32, self._robot_radius + 0.08)
                    candidates.append((float(distance), lateral))
                forward, left = min(candidates, key=lambda item: item[0]) if candidates else (float(default), 0.0)
            else:
                forward, left = float(default), 0.0
        distance = math.hypot(forward, left)
        if distance < 0.25:
            scale = 0.25 / max(distance, 1e-6)
            return forward * scale, left * scale
        if distance > 1.50:
            scale = 1.50 / distance
            return forward * scale, left * scale
        return float(forward), float(left)


# -----------------------------------------------------------------------
# Instanțe globale (singleton)
# -----------------------------------------------------------------------
slam_client = SlamClient()
sport_client = SportClient()
odom_reader = OdometryReader()
obstacle_guard = ObstacleGuard()
