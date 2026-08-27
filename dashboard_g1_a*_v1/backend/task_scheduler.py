#!/usr/bin/env python3
"""
task_scheduler.py - Motor de execuție a task-urilor programate
Suportă: waypoint navigation, comenzi custom, wait, repeat
Persistă task-urile în JSON.
"""

import json
import asyncio
import math
import time
import uuid
import os
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Toleranțe pentru navigarea în buclă închisă.
# Un task NAVIGATE se consideră terminat doar când poziția RAPORTATĂ de robot
# ajunge în toleranță — nu după un sleep fix. Fără asta, orice înlănțuire de
# waypoint-uri e doar noroc cronometrat.
# ---------------------------------------------------------------------------
NAV_ARRIVAL_DIST = 0.25       # m — cât de aproape de țintă înseamnă "ajuns"
NAV_ARRIVAL_YAW = 0.20        # rad (~11°) — toleranță de orientare finală
NAV_POLL_INTERVAL = 0.2       # s — frecvența de verificare a poziției
NAV_STUCK_TIMEOUT = 10.0      # s fără progres real -> robot blocat
NAV_STUCK_MIN_PROGRESS = 0.05 # m — sub atât nu se consideră progres


class TaskType(str, Enum):
    NAVIGATE   = "navigate"    # Merge la coordonate
    COMMAND    = "command"     # Rulează o comandă custom
    WAIT       = "wait"        # Pauză N secunde
    SLAM_START = "slam_start"  # Pornește mapping
    SLAM_STOP  = "slam_stop"   # Oprește mapping
    SLAM_SAVE  = "slam_save"   # Salvează harta


class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    type: TaskType = TaskType.WAIT
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    order: int = 0

    def to_dict(self):
        d = asdict(self)
        d['type'] = self.type.value
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Task':
        d = d.copy()
        d['type'] = TaskType(d.get('type', 'wait'))
        d['status'] = TaskStatus(d.get('status', 'pending'))
        return cls(**d)


class TaskScheduler:
    TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.json")
    COMMANDS_FILE = os.path.join(os.path.dirname(__file__), "commands.json")

    def __init__(self):
        self.tasks: List[Task] = []
        self.custom_commands: List[Dict] = []
        self._running = False
        self._current_task_id: Optional[str] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._status_callback = None
        self._pose_provider = None
        self._load()

    def set_status_callback(self, cb):
        """Callback apelat la fiecare schimbare de status."""
        self._status_callback = cb

    def set_pose_provider(self, cb):
        """
        Bindează sursa de poziție folosită pentru verificarea sosirii.

        server.py ar trebui să apeleze:
            scheduler.set_pose_provider(lambda: map_state["pose"])
        Callback-ul returnează un dict {"x": float, "y": float, "yaw": float}
        (yaw în radiani) sau None dacă poziția nu e disponibilă.
        """
        self._pose_provider = cb

    def _get_pose(self) -> Optional[dict]:
        """Citește poziția curentă; None dacă nu există sursă validă."""
        if self._pose_provider:
            try:
                pose = self._pose_provider()
            except Exception:
                pose = None
            if pose and "x" in pose and "y" in pose:
                return pose

        # Fallback: cititorul de odometrie din robot_client
        try:
            from robot_client import odom_reader
            return odom_reader.get_latest_pose()
        except Exception:
            return None

    def _load(self):
        """Încarcă task-urile și comenzile din fișiere JSON."""
        try:
            if os.path.exists(self.TASKS_FILE):
                with open(self.TASKS_FILE) as f:
                    data = json.load(f)
                    self.tasks = [Task.from_dict(t) for t in data]
                    # Reset toate task-urile running la pending la restart
                    for t in self.tasks:
                        if t.status == TaskStatus.RUNNING:
                            t.status = TaskStatus.PENDING
        except Exception:
            self.tasks = []

        try:
            if os.path.exists(self.COMMANDS_FILE):
                with open(self.COMMANDS_FILE) as f:
                    self.custom_commands = json.load(f)
        except Exception:
            self.custom_commands = []

    def _save(self):
        """Salvează task-urile în fișier JSON."""
        try:
            with open(self.TASKS_FILE, 'w') as f:
                json.dump([t.to_dict() for t in self.tasks], f, indent=2)
        except Exception as e:
            print(f"Eroare salvare tasks: {e}")

    def _save_commands(self):
        try:
            with open(self.COMMANDS_FILE, 'w') as f:
                json.dump(self.custom_commands, f, indent=2)
        except Exception as e:
            print(f"Eroare salvare commands: {e}")

    # -----------------------------------------------------------------------
    # CRUD Tasks
    # -----------------------------------------------------------------------
    def add_task(self, name: str, type: str, params: dict) -> Task:
        task = Task(
            name=name,
            type=TaskType(type),
            params=params,
            order=len(self.tasks)
        )
        self.tasks.append(task)
        self._save()
        return task

    def remove_task(self, task_id: str) -> bool:
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.id != task_id]
        self._save()
        return len(self.tasks) < before

    def reorder_tasks(self, ordered_ids: List[str]):
        """Reordonează task-urile după lista de ID-uri."""
        id_to_task = {t.id: t for t in self.tasks}
        reordered = []
        for i, tid in enumerate(ordered_ids):
            if tid in id_to_task:
                id_to_task[tid].order = i
                reordered.append(id_to_task[tid])
        # Adaugă task-urile care nu erau în lista de reordonare
        for t in self.tasks:
            if t.id not in [r.id for r in reordered]:
                reordered.append(t)
        self.tasks = reordered
        self._save()

    def reset_tasks(self):
        """Resetează toate task-urile la pending."""
        for t in self.tasks:
            t.status = TaskStatus.PENDING
            t.result = None
            t.started_at = None
            t.finished_at = None
        self._save()

    def clear_tasks(self):
        self.tasks = []
        self._save()

    def get_tasks(self) -> List[dict]:
        return [t.to_dict() for t in sorted(self.tasks, key=lambda t: t.order)]

    # -----------------------------------------------------------------------
    # CRUD Custom Commands  
    # -----------------------------------------------------------------------
    def add_command(self, name: str, type: str, command: str, description: str = "") -> dict:
        cmd = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "type": type,        # "ros2_topic", "ros2_service", "shell", "python"
            "command": command,
            "description": description,
            "created_at": time.time()
        }
        self.custom_commands.append(cmd)
        self._save_commands()
        return cmd

    def remove_command(self, cmd_id: str) -> bool:
        before = len(self.custom_commands)
        self.custom_commands = [c for c in self.custom_commands if c['id'] != cmd_id]
        self._save_commands()
        return len(self.custom_commands) < before

    def get_commands(self) -> List[dict]:
        return self.custom_commands

    async def run_command(self, cmd_id: str) -> dict:
        """Execută imediat o comandă custom."""
        cmd = next((c for c in self.custom_commands if c['id'] == cmd_id), None)
        if not cmd:
            return {"success": False, "error": "Comanda nu a fost găsită"}
        return await self._execute_command(cmd)

    # -----------------------------------------------------------------------
    # Scheduler Engine
    # -----------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._running

    def get_current_task(self) -> Optional[str]:
        return self._current_task_id

    async def start(self) -> dict:
        """Pornește execuția secvențială a task-urilor."""
        if self._running:
            return {"success": False, "error": "Scheduler-ul rulează deja"}
        
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        if not pending:
            return {"success": False, "error": "Nu există task-uri pending"}
        
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        return {"success": True, "message": f"Pornit {len(pending)} task-uri"}

    async def stop(self) -> dict:
        """Oprește execuția."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        return {"success": True, "message": "Scheduler oprit"}

    async def _run_loop(self):
        """Loop-ul principal de execuție a task-urilor."""
        from robot_client import slam_client, sport_client
        
        try:
            sorted_tasks = sorted(self.tasks, key=lambda t: t.order)
            for task in sorted_tasks:
                if not self._running:
                    break
                if task.status != TaskStatus.PENDING:
                    continue
                    
                # Marchează ca running
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                self._current_task_id = task.id
                self._save()
                await self._notify()

                try:
                    result = await self._execute_task(task, slam_client, sport_client)
                    task.status = TaskStatus.DONE if result.get("success") else TaskStatus.FAILED
                    task.result = result.get("output") or result.get("error") or "OK"
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.result = str(e)
                
                task.finished_at = time.time()
                self._current_task_id = None
                self._save()
                await self._notify()
                
                # Pauză scurtă între task-uri
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            # Oprire forțată
            for t in self.tasks:
                if t.status == TaskStatus.RUNNING:
                    t.status = TaskStatus.CANCELLED
            self._save()
        finally:
            self._running = False
            self._current_task_id = None
            await self._notify()

    async def _execute_task(self, task: Task, slam_client, sport_client) -> dict:
        """Execută un singur task."""
        if task.type == TaskType.WAIT:
            secs = task.params.get("seconds", 1)
            await asyncio.sleep(float(secs))
            return {"success": True, "output": f"Așteptat {secs}s"}

        elif task.type == TaskType.NAVIGATE:
            x = float(task.params.get("x", 0.0))
            y = float(task.params.get("y", 0.0))
            yaw = float(task.params.get("yaw", 0.0))
            timeout = float(task.params.get("timeout", 30))
            # navigate_to (goal absolut), NU move_to (comandă de viteză).
            return await self._navigate_and_wait(sport_client, x, y, yaw, timeout)

        elif task.type == TaskType.SLAM_START:
            return slam_client.start_mapping()

        elif task.type == TaskType.SLAM_STOP:
            return slam_client.stop_mapping()

        elif task.type == TaskType.SLAM_SAVE:
            map_name = task.params.get("map_name", f"map_{int(time.time())}")
            return slam_client.save_map(map_name)

        elif task.type == TaskType.COMMAND:
            cmd_id = task.params.get("command_id")
            if cmd_id:
                cmd = next((c for c in self.custom_commands if c['id'] == cmd_id), None)
                if cmd:
                    return await self._execute_command(cmd)
            # Sau comandă inline
            inline_cmd = task.params.get("command", "")
            if inline_cmd:
                return await self._run_shell(inline_cmd)
            return {"success": False, "error": "Nicio comandă specificată"}

        return {"success": False, "error": f"Tip task necunoscut: {task.type}"}

    async def _navigate_and_wait(self, sport_client, x: float, y: float,
                                 yaw: float, timeout: float) -> dict:
        """
        Trimite un goal absolut și așteaptă confirmarea sosirii din odometrie.

        Se termină în exact patru feluri, toate explicite:
          - ajuns    : distanța și yaw-ul sunt în toleranță
          - timeout  : a expirat bugetul de timp
          - blocat   : nu s-a apropiat de țintă NAV_STUCK_TIMEOUT secunde
          - anulat   : scheduler-ul a fost oprit între timp
        În ultimele trei cazuri robotul primește STOP înainte de a raporta eșec.
        """
        send = await asyncio.to_thread(sport_client.navigate_to, x, y, yaw)
        if not send.get("success"):
            return {"success": False,
                    "error": f"Goal respins: {send.get('error') or send.get('output')}"}

        start = time.time()
        best_dist = float("inf")
        last_progress = start

        while True:
            await asyncio.sleep(NAV_POLL_INTERVAL)
            now = time.time()

            if not self._running:
                await asyncio.to_thread(sport_client.stop)
                return {"success": False, "error": "Navigare anulată (scheduler oprit)"}

            pose = self._get_pose()
            if not pose:
                # Fără sursă de poziție nu putem confirma sosirea. Nu pretindem
                # succes: lăsăm robotul să meargă bugetul de timp și raportăm
                # explicit că rezultatul e neverificat.
                if now - start >= timeout:
                    return {"success": False,
                            "error": "Fără odometrie: sosirea nu a putut fi verificată"}
                continue

            dx = x - float(pose.get("x", 0.0))
            dy = y - float(pose.get("y", 0.0))
            dist = math.hypot(dx, dy)

            dyaw = abs(math.atan2(math.sin(yaw - float(pose.get("yaw", 0.0))),
                                  math.cos(yaw - float(pose.get("yaw", 0.0)))))

            if dist <= NAV_ARRIVAL_DIST and dyaw <= NAV_ARRIVAL_YAW:
                return {"success": True,
                        "output": f"Ajuns la ({x:.2f}, {y:.2f}) în {now - start:.1f}s "
                                  f"(eroare {dist:.2f}m, {math.degrees(dyaw):.1f}°)"}

            if dist < best_dist - NAV_STUCK_MIN_PROGRESS:
                best_dist = dist
                last_progress = now

            if now - last_progress > NAV_STUCK_TIMEOUT:
                await asyncio.to_thread(sport_client.stop)
                return {"success": False,
                        "error": f"Robot blocat: fără progres {NAV_STUCK_TIMEOUT:.0f}s "
                                 f"la {dist:.2f}m de țintă"}

            if now - start >= timeout:
                await asyncio.to_thread(sport_client.stop)
                return {"success": False,
                        "error": f"Timeout după {timeout:.0f}s, oprit la {dist:.2f}m de țintă"}

    async def _execute_command(self, cmd: dict) -> dict:
        """Execută o comandă custom."""
        import subprocess as sp
        cmd_type = cmd.get("type", "shell")
        command = cmd.get("command", "")
        
        if cmd_type == "shell":
            return await self._run_shell(command)
        elif cmd_type == "ros2_topic":
            return await self._run_shell(f"ros2 topic pub --once {command}")
        elif cmd_type == "ros2_service":
            return await self._run_shell(f"ros2 service call {command}")
        elif cmd_type == "python":
            return await self._run_shell(f"python3 -c '{command}'")
        else:
            return await self._run_shell(command)

    async def _run_shell(self, command: str) -> dict:
        """Rulează o comandă shell async."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return {
                "success": proc.returncode == 0,
                "output": stdout.decode(errors='replace')[:500],
                "error": stderr.decode(errors='replace')[:200]
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _notify(self):
        """Notifică frontend-ul despre schimbări de status."""
        if self._status_callback:
            try:
                await self._status_callback()
            except Exception:
                pass


# Singleton global
scheduler = TaskScheduler()