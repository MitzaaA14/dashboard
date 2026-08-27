#!/usr/bin/env python3
"""
cmd_vel_bridge.py - Bridge ROS2 -> SportClient pentru teleop_twist_keyboard

Ascultă topicul /cmd_vel (geometry_msgs/msg/Twist), publicat de
teleop_twist_keyboard, și trimite comenzile direct la robot prin
sport_client.move_to() din robot_client.py (folosește unitree_sdk2py
dacă e disponibil, altfel fallback ROS2).

IMPORTANT - SIGURANȚĂ:
teleop_twist_keyboard trimite mesaje doar CÂND apeși o tastă, nu la un
ritm constant. Fără watchdog, dacă se blochează terminalul sau se pierde
conexiunea, robotul ar putea continua să se miște cu ultima comandă
primită. Watchdog-ul de mai jos oprește robotul automat dacă nu a mai
primit niciun mesaj de mișcare în WATCHDOG_TIMEOUT secunde.

Rulare (pe Orin, într-un terminal separat de server.py):
    python3 cmd_vel_bridge.py

Apoi, într-un alt terminal:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

import time
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from robot_client import sport_client

WATCHDOG_TIMEOUT = 0.5  # secunde fără mesaj nou -> STOP automat
WATCHDOG_POLL_INTERVAL = 0.1  # cât de des verificăm watchdog-ul


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.on_twist, 10)
        self._last_msg_time = time.time()
        self._last_was_stop = True
        self._lock = threading.Lock()

        self.get_logger().info(
            f"cmd_vel_bridge pornit. Ascult /cmd_vel, watchdog={WATCHDOG_TIMEOUT}s"
        )

        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def on_twist(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y
        vyaw = msg.angular.z

        with self._lock:
            self._last_msg_time = time.time()

            # Dacă toate componentele sunt ~0, tratăm ca stop explicit
            # (teleop_twist_keyboard trimite Twist gol la "orice altă tastă").
            if abs(vx) < 1e-4 and abs(vy) < 1e-4 and abs(vyaw) < 1e-4:
                if not self._last_was_stop:
                    result = sport_client.stop()
                    self.get_logger().info(f"STOP: {result}")
                    self._last_was_stop = True
                return

            self._last_was_stop = False

        result = sport_client.move_to(vx, vy, vyaw)
        if not result.get("success"):
            self.get_logger().warn(f"move_to a eșuat: {result}")

    def _watchdog_loop(self):
        while True:
            time.sleep(WATCHDOG_POLL_INTERVAL)
            with self._lock:
                idle_time = time.time() - self._last_msg_time
                already_stopped = self._last_was_stop

            if idle_time > WATCHDOG_TIMEOUT and not already_stopped:
                self.get_logger().warn(
                    f"Watchdog: {idle_time:.2f}s fără comandă nouă -> STOP de siguranță"
                )
                result = sport_client.stop()
                self.get_logger().info(f"Watchdog STOP: {result}")
                with self._lock:
                    self._last_was_stop = True


def main():
    if not sport_client.is_sdk_available():
        print(
            "[cmd_vel_bridge] REFUZ PORNIRE: SDK-ul G1 nu este disponibil. "
            "Nu există fallback sigur pentru comenzi de viteză."
        )
        return

    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Oprire finală de siguranță la ieșire
        try:
            sport_client.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()