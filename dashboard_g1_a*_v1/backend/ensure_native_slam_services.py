#!/usr/bin/env python3
"""Restart confirmat pentru LiDAR + SLAM Unitree, într-un proces separat.

Backendul principal folosește rclpy/CycloneDDS. RobotStateClient rămâne
izolat aici pentru a evita inițializarea a două implementări DDS în același
proces. Helperul nu trimite comenzi de locomoție.
"""

from __future__ import annotations

import argparse
import sys
import time


def _client(interface: str, timeout: float):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

    ChannelFactoryInitialize(0, interface)
    client = RobotStateClient()
    client.SetTimeout(min(max(timeout, 1.0), 10.0))
    client.Init()
    return client


def _wait_status(client, name: str, enabled: bool, timeout: float) -> None:
    expected = 0 if enabled else 1
    deadline = time.monotonic() + max(1.0, timeout)
    last_status = None
    while time.monotonic() < deadline:
        code, services = client.ServiceList()
        if code == 0 and services is not None:
            for service in services:
                if service.name == name:
                    last_status = service.status
                    if last_status == expected:
                        return
                    break
        time.sleep(0.2)
    state = "ON" if enabled else "OFF"
    raise RuntimeError(f"{name} nu a confirmat {state} (status={last_status!r})")


def _set_and_wait(client, name: str, enabled: bool, timeout: float) -> None:
    code = client.ServiceSwitch(name, enabled)
    if code != 0:
        state = "ON" if enabled else "OFF"
        raise RuntimeError(f"ServiceSwitch({name}, {state}) a eșuat: {code}")
    _wait_status(client, name, enabled, timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--timeout", type=float, default=20.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stop-native-sensors", action="store_true")
    mode.add_argument("--restart-slam", action="store_true")
    args = parser.parse_args()

    try:
        client = _client(args.interface, args.timeout)
        if args.stop_native_sensors:
            _set_and_wait(client, "unitree_slam", False, args.timeout)
            _set_and_wait(client, "lidar_driver", False, args.timeout)
            print("[native-slam] unitree_slam + lidar_driver OFF confirmate", flush=True)
        else:
            # Un simplu ON este no-op dacă serviciile apar deja pornite.
            # Restartul real le obligă să redeschidă atât LiDAR-ul, cât și IMU-ul.
            _set_and_wait(client, "unitree_slam", False, args.timeout)
            _set_and_wait(client, "lidar_driver", False, args.timeout)
            time.sleep(1.0)
            _set_and_wait(client, "lidar_driver", True, args.timeout)
            time.sleep(2.0)
            _set_and_wait(client, "unitree_slam", True, args.timeout)
            time.sleep(2.0)
            print(
                "[native-slam] lidar_driver + unitree_slam restartate și "
                "confirmate; LiDAR/IMU ON",
                flush=True,
            )
    except Exception as exc:
        print(f"[native-slam] EROARE: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
