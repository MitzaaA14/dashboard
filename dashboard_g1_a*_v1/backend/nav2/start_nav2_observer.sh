#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_FILE="${SCRIPT_DIR}/nav2_observer_params.yaml"
PID_FILE="/tmp/dashboard_g1_v24_nav2_observer.pids"

stop_process_group() {
  local leader="${1:-}"
  [[ -z "${leader}" ]] && return 0
  kill -TERM -- "-${leader}" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 -- "-${leader}" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  # Unele executabile ROS 2 blocate într-un transition callback ignoră TERM.
  # Grupul este sigur de țintit: a fost creat de noi cu setsid și liderul este
  # PID-ul memorat mai jos, nu un pattern sau un PID descoperit aproximativ.
  kill -KILL -- "-${leader}" 2>/dev/null || true
}

stop_recorded_group() {
  local leader="${1:-}"
  local expected="${2:-}"
  [[ ! "${leader}" =~ ^[0-9]+$ ]] && return 0
  local command
  command="$(ps -o args= -p "${leader}" 2>/dev/null || true)"
  # PID-urile se pot reutiliza după un restart. Oprim numai grupurile care
  # aparțin exact executabilelor observerului v24, niciodată un PID arbitrar.
  if [[ -n "${command}" && "${command}" == *"${expected}"* \
        && "${command}" == *"${PARAMS_FILE}"* ]]; then
    echo "Oprire proces Nav2 v24 rămas din sesiunea anterioară (PID ${leader})."
    stop_process_group "${leader}"
  fi
}

if [[ -r "${PID_FILE}" ]]; then
  read -r OLD_PLANNER_PID OLD_LIFECYCLE_PID < "${PID_FILE}" || true
  stop_recorded_group "${OLD_LIFECYCLE_PID:-}" "nav2_lifecycle_manager lifecycle_manager"
  stop_recorded_group "${OLD_PLANNER_PID:-}" "nav2_planner planner_server"
fi

cleanup() {
  trap - EXIT INT TERM
  # `ros2 run` pornește executabilul Nav2 ca proces copil. Oprirea numai a
  # wrapperului lasă planner_server orfan și următorul dashboard pornește două
  # noduri cu același nume. Fiecare comandă rulează într-o sesiune proprie, iar
  # cleanup-ul oprește întregul grup asociat, fără să atingă alte noduri ROS.
  stop_process_group "${LIFECYCLE_PID:-}"
  stop_process_group "${PLANNER_PID:-}"
  : > "${PID_FILE}"
}
trap cleanup EXIT INT TERM

echo "Nav2 OBSERVER: planner + costmap, fără controller_server și fără cmd_vel."
echo "Deschide harta 2D din dashboard înainte sau după pornire pentru publicarea /map."

setsid ros2 run nav2_planner planner_server \
  --ros-args --params-file "${PARAMS_FILE}" &
PLANNER_PID=$!

setsid ros2 run nav2_lifecycle_manager lifecycle_manager \
  --ros-args \
  -r __node:=lifecycle_manager_nav2_observer \
  --params-file "${PARAMS_FILE}" &
LIFECYCLE_PID=$!

printf '%s %s\n' "${PLANNER_PID}" "${LIFECYCLE_PID}" > "${PID_FILE}"

wait "${PLANNER_PID}"
