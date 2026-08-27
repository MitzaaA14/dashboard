/**
 * app.js - G1 Robot Dashboard Frontend Logic
 * Handles: tabs, WebSocket, map canvas, nav goals, scheduler, custom commands
 */

// =========================================================================
// CONFIG
// =========================================================================
const API_BASE = `http://${window.location.hostname}:8080`;
const WS_URL   = `ws://${window.location.hostname}:8080/ws`;

// =========================================================================
// STATE
// =========================================================================
const state = {
  ws: null,
  wsReconnectTimer: null,
  activeTab: 'mapping',
  pose: { x: 0, y: 0, yaw: 0 },
  tasks: [],
  commands: [],
  waypoints: JSON.parse(localStorage.getItem('g1_waypoints') || '[]'),
  schedulerRunning: false,
  slamActive: false,
  robotReachable: false,

  // Map canvas state
  map: {
    points: [],      // array of {x, y, z, intensity}
    mapPoints: [],   // harta SLAM
    goalPending: null,
    goalActive: null,
    transform: { x: 0, y: 0, scale: 40 }, // px per meter
    dragging: false,
    lastMouse: null,
    tool: 'pan',     // 'pan' | 'navigate'
    width: 0, height: 0,
    hasData: false,
  }
};

// =========================================================================
// DOM REFS
// =========================================================================
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// =========================================================================
// INIT
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initMap();
  initSlamControls();
  initNavControls();
  initScheduler();
  initCommands();
  initWebSocket();
  startStatusPolling();
  loadWaypoints();

  // Quick commands
  $$('.quick-cmd-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $('cmd-type').value = btn.dataset.cmd;
      $('cmd-command').value = btn.dataset.command;
      $('cmd-name').value = btn.dataset.name;
    });
  });
});

// =========================================================================
// TABS
// =========================================================================
function initTabs() {
  const titles = {
    mapping: '🗺️ Mapping SLAM',
    navigation: '🧭 Navigare',
    scheduler: '📋 Task Scheduler',
    commands: '⚙️ Comenzi Custom'
  };

  $$('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      state.activeTab = tab;

      $$('.nav-item').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      $$('.tab-content').forEach(c => c.classList.remove('active'));
      $(`tab-${tab}`).classList.add('active');

      $('page-title').textContent = titles[tab] || tab;

      if (tab === 'navigation') {
        requestAnimationFrame(() => resizeCanvas());
      }
    });
  });
}

// =========================================================================
// WEBSOCKET
// =========================================================================
function initWebSocket() {
  if (state.ws) {
    state.ws.close();
  }

  try {
    state.ws = new WebSocket(WS_URL);

    state.ws.onopen = () => {
      updateWsStatus(true);
      addLog('WebSocket conectat la backend.', 'success');
      clearTimeout(state.wsReconnectTimer);
    };

    state.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    state.ws.onclose = () => {
      updateWsStatus(false);
      state.wsReconnectTimer = setTimeout(initWebSocket, 3000);
    };

    state.ws.onerror = () => {
      updateWsStatus(false);
    };
  } catch (e) {
    console.error('WS init error:', e);
    state.wsReconnectTimer = setTimeout(initWebSocket, 3000);
  }
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'init':
      if (msg.tasks) renderTasks(msg.tasks);
      if (msg.commands) renderCommands(msg.commands);
      state.schedulerRunning = msg.running || false;
      updateSchedulerUI();
      break;

    case 'pose_update':
      if (msg.pose) {
        state.pose = msg.pose;
        updatePoseDisplay(msg.pose);
        drawMap();
      }
      break;

    case 'tasks_update':
    case 'scheduler_update':
      if (msg.tasks) renderTasks(msg.tasks);
      if (typeof msg.running !== 'undefined') {
        state.schedulerRunning = msg.running;
        updateSchedulerUI();
      }
      break;

    case 'slam_event':
      const ev = msg.event;
      if (ev === 'mapping_started') {
        state.slamActive = true;
        updateSlamUI(true);
        addLog('SLAM Mapping pornit!', 'success');
        showToast('SLAM Mapping pornit ✓', 'success');
      } else if (ev === 'mapping_stopped') {
        state.slamActive = false;
        updateSlamUI(false);
        addLog('SLAM Mapping oprit.', 'info');
      } else if (ev === 'map_saved') {
        addLog(`Harta salvată: ${msg.map_name}`, 'success');
        showToast(`Harta "${msg.map_name}" salvată ✓`, 'success');
        fetchMaps();
      }
      break;

    case 'nav_event':
      if (msg.event === 'goal_sent') {
        const g = msg.goal;
        state.map.goalActive = g;
        addLog(`Goal trimis: X=${g.x.toFixed(2)}, Y=${g.y.toFixed(2)}, Yaw=${g.yaw.toFixed(0)}°`, 'success');
        showToast('Goal de navigare trimis ✓', 'success');
        drawMap();
      } else if (msg.event === 'stopped') {
        state.map.goalActive = null;
        showToast('Robot oprit.', 'info');
        drawMap();
      }
      break;

    case 'pong':
      break;
  }
}

function sendWs(data) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(data));
  }
}

// =========================================================================
// STATUS POLLING
// =========================================================================
let statusInterval = null;
function startStatusPolling() {
  fetchStatus();
  statusInterval = setInterval(fetchStatus, 5000);
}

async function fetchStatus() {
  try {
    const resp = await fetch(`${API_BASE}/api/status`);
    if (!resp.ok) throw new Error(resp.status);
    const data = await resp.json();

    state.robotReachable = data.robot_reachable;
    state.slamActive = data.slam_mapping_active;

    // Update sidebar
    $('robot-dot').className = 'status-dot ' + (data.robot_reachable ? 'ok' : 'err');
    $('robot-status-text').textContent = data.robot_reachable ? 'Conectat' : 'Deconectat';

    $('slam-dot').className = 'status-dot ' + (data.slam_mapping_active ? 'ok' : data.slam_relocation_active ? 'warn' : '');
    $('slam-status-text').textContent = data.slam_mapping_active ? 'Mapping' : data.slam_relocation_active ? 'Relocalizare' : 'Oprit';

    $('stat-topics').textContent = (data.ros_topics || []).length;
    $('stat-slam-mapping').textContent = data.slam_mapping_active ? '✓ Activ' : '✗';
    $('stat-slam-reloc').textContent = data.slam_relocation_active ? '✓ Activ' : '✗';

    updateSlamUI(data.slam_mapping_active);
  } catch (e) {
    $('robot-dot').className = 'status-dot err';
    $('robot-status-text').textContent = 'Eroare conexiune';
  }
}

$('btn-refresh-status').addEventListener('click', () => {
  fetchStatus();
  addLog('Status actualizat manual.', 'info');
});

// =========================================================================
// SLAM CONTROLS
// =========================================================================
function initSlamControls() {
  $('btn-start-mapping').addEventListener('click', async () => {
    addLog('Trimit comanda Start Mapping...', 'info');
    const result = await apiPost('/api/slam/start');
    addLog(result.success ? 'Start Mapping OK' : `Eroare: ${result.error}`,
           result.success ? 'success' : 'error');
  });

  $('btn-stop-mapping').addEventListener('click', async () => {
    addLog('Trimit comanda Stop Mapping...', 'info');
    const result = await apiPost('/api/slam/stop');
    addLog(result.success ? 'Stop Mapping OK' : `Eroare: ${result.error}`,
           result.success ? 'success' : 'error');
  });

  $('btn-save-map').addEventListener('click', async () => {
    const name = $('map-name-input').value.trim() || 'my_map';
    addLog(`Salvez harta: "${name}"...`, 'info');
    const result = await apiPost('/api/slam/save', { map_name: name });
    addLog(result.success !== false ? `Hartă salvată: ${name}` : `Eroare: ${result.error}`,
           result.success !== false ? 'success' : 'error');
  });

  $('btn-load-map').addEventListener('click', async () => {
    const mapName = $('map-select').value;
    if (!mapName) { showToast('Selectează o hartă mai întâi!', 'warning'); return; }
    const result = await apiPost('/api/slam/load', { map_name: mapName });
    showToast(result.success !== false ? 'Hartă încărcată ✓' : 'Eroare la încărcare', result.success !== false ? 'success' : 'error');
  });

  $('btn-start-reloc').addEventListener('click', async () => {
    addLog('Pornesc relocalizarea...', 'info');
    const result = await apiPost('/api/slam/relocate');
    addLog(result.success ? 'Relocalizare pornită ✓' : `Eroare: ${result.error}`,
           result.success ? 'success' : 'error');
  });

  $('btn-clear-log').addEventListener('click', () => {
    $('log-container').innerHTML = '';
    addLog('Log șters.', 'info');
  });

  fetchMaps();
}

async function fetchMaps() {
  try {
    const data = await apiFetch('/api/slam/maps');
    const select = $('map-select');
    const maps = data.maps || [];
    select.innerHTML = '<option value="">-- Selectează o hartă --</option>';
    maps.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m.split('/').pop();
      select.appendChild(opt);
    });
  } catch (e) { /* ignore */ }
}

function updateSlamUI(active) {
  state.slamActive = active;
  $('slam-state-badge').textContent = active ? 'ACTIV' : 'OPRIT';
  $('slam-state-badge').className = 'slam-state-badge' + (active ? ' active' : '');
  $('btn-start-mapping').disabled = active;
  $('btn-stop-mapping').disabled = !active;
  if (active) {
    $('map-no-data').style.display = 'none';
    state.map.hasData = true;
  }
}

// =========================================================================
// MAP CANVAS
// =========================================================================
function initMap() {
  const canvas = $('map-canvas');
  const wrapper = $('map-wrapper');

  // Resize observer
  const ro = new ResizeObserver(resizeCanvas);
  ro.observe(wrapper);

  // Mouse events
  canvas.addEventListener('mousedown', onMapMouseDown);
  canvas.addEventListener('mousemove', onMapMouseMove);
  canvas.addEventListener('mouseup', onMapMouseUp);
  canvas.addEventListener('wheel', onMapWheel, { passive: false });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('mouseleave', () => { state.map.dragging = false; });

  // Tools
  $('tool-pan').addEventListener('click', () => setMapTool('pan'));
  $('tool-navigate').addEventListener('click', () => setMapTool('navigate'));
  $('btn-zoom-in').addEventListener('click', () => zoomMap(1.3));
  $('btn-zoom-out').addEventListener('click', () => zoomMap(0.77));
  $('btn-zoom-reset').addEventListener('click', resetMapView);
  $('btn-clear-markers').addEventListener('click', () => {
    state.map.goalActive = null;
    state.map.goalPending = null;
    $('map-tooltip').style.display = 'none';
    drawMap();
  });

  // Goal tooltip
  $('btn-confirm-goal').addEventListener('click', confirmNavGoal);
  $('btn-cancel-goal').addEventListener('click', () => {
    state.map.goalPending = null;
    $('map-tooltip').style.display = 'none';
  });

  resizeCanvas();

  // Generate demo point cloud grid for testing when no real data
  generateDemoMap();
  drawMap();
}

function generateDemoMap() {
  // Generate a simple demo environment for visualization testing
  const points = [];
  // Outer walls
  for (let i = -5; i <= 5; i += 0.1) {
    points.push({x: i, y: -4, z: 0.5});
    points.push({x: i, y:  4, z: 0.5});
    points.push({x: -5, y: i, z: 0.5});
    points.push({x:  5, y: i, z: 0.5});
  }
  // Some obstacles
  for (let i = -1; i <= 1; i += 0.1) {
    for (let j = -1; j <= 1; j += 0.1) {
      if (Math.random() > 0.6) points.push({x: 2 + i, y: 1 + j, z: 0.3});
      if (Math.random() > 0.6) points.push({x: -2 + i, y: -1 + j, z: 0.3});
    }
  }
  state.map.points = points;
}

function resizeCanvas() {
  const canvas = $('map-canvas');
  const wrapper = $('map-wrapper');
  const rect = wrapper.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  state.map.width = rect.width;
  state.map.height = rect.height;

  if (state.map.transform.x === 0 && state.map.transform.y === 0) {
    state.map.transform.x = rect.width / 2;
    state.map.transform.y = rect.height / 2;
  }
  drawMap();
}

function worldToCanvas(wx, wy) {
  const t = state.map.transform;
  return {
    cx: t.x + wx * t.scale,
    cy: t.y - wy * t.scale
  };
}

function canvasToWorld(cx, cy) {
  const t = state.map.transform;
  return {
    wx: (cx - t.x) / t.scale,
    wy: -(cy - t.y) / t.scale
  };
}

function drawMap() {
  const canvas = $('map-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  // Background
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#060a12';
  ctx.fillRect(0, 0, w, h);

  // Grid
  drawGrid(ctx, w, h);

  // Point cloud (LiDAR brut)
  if (state.map.points.length > 0) {
    ctx.save();
    state.map.points.forEach(pt => {
      const {cx, cy} = worldToCanvas(pt.x, pt.y);
      if (cx < -5 || cx > w+5 || cy < -5 || cy > h+5) return;
      const alpha = Math.min(1, 0.6 + (pt.z || 0) * 0.2);
      ctx.fillStyle = `rgba(79, 142, 247, ${alpha})`;
      ctx.beginPath();
      ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  // SLAM map points
  if (state.map.mapPoints.length > 0) {
    ctx.save();
    state.map.mapPoints.forEach(pt => {
      const {cx, cy} = worldToCanvas(pt.x, pt.y);
      if (cx < -5 || cx > w+5 || cy < -5 || cy > h+5) return;
      ctx.fillStyle = 'rgba(34, 211, 238, 0.7)';
      ctx.beginPath();
      ctx.arc(cx, cy, 1.8, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  // Active goal marker
  if (state.map.goalActive) {
    const g = state.map.goalActive;
    const {cx, cy} = worldToCanvas(g.x, g.y);
    drawGoalMarker(ctx, cx, cy, g.yaw || 0, '#4ade80');
  }

  // Pending goal marker
  if (state.map.goalPending) {
    const g = state.map.goalPending;
    const {cx, cy} = worldToCanvas(g.x, g.y);
    drawGoalMarker(ctx, cx, cy, 0, 'rgba(74,222,128,0.5)', true);
  }

  // Robot position
  const {cx: rx, cy: ry} = worldToCanvas(state.pose.x, state.pose.y);
  drawRobot(ctx, rx, ry, state.pose.yaw || 0);
}

function drawGrid(ctx, w, h) {
  const t = state.map.transform;
  const gridSize = t.scale; // 1m grid
  
  ctx.strokeStyle = 'rgba(99, 140, 255, 0.06)';
  ctx.lineWidth = 1;

  const startX = t.x % gridSize;
  const startY = t.y % gridSize;

  for (let x = startX; x < w; x += gridSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let x = startX - gridSize; x >= 0; x -= gridSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = startY; y < h; y += gridSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let y = startY - gridSize; y >= 0; y -= gridSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Origin cross
  const {cx: ox, cy: oy} = worldToCanvas(0, 0);
  if (ox >= 0 && ox <= w && oy >= 0 && oy <= h) {
    ctx.strokeStyle = 'rgba(99, 140, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ox-10, oy); ctx.lineTo(ox+10, oy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox, oy-10); ctx.lineTo(ox, oy+10); ctx.stroke();
  }
}

function drawRobot(ctx, cx, cy, yaw) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-yaw); // ROS yaw is CCW, canvas is CW

  // Glow
  const grd = ctx.createRadialGradient(0, 0, 2, 0, 0, 16);
  grd.addColorStop(0, 'rgba(250, 204, 21, 0.4)');
  grd.addColorStop(1, 'rgba(250, 204, 21, 0)');
  ctx.fillStyle = grd;
  ctx.beginPath(); ctx.arc(0, 0, 16, 0, Math.PI * 2); ctx.fill();

  // Body
  ctx.fillStyle = '#facc15';
  ctx.beginPath();
  ctx.arc(0, 0, 8, 0, Math.PI * 2);
  ctx.fill();

  // Direction arrow
  ctx.fillStyle = '#1a1a1a';
  ctx.beginPath();
  ctx.moveTo(10, 0); ctx.lineTo(3, -4); ctx.lineTo(3, 4);
  ctx.closePath(); ctx.fill();

  ctx.restore();
}

function drawGoalMarker(ctx, cx, cy, yawDeg, color, dashed = false) {
  ctx.save();
  ctx.translate(cx, cy);

  if (dashed) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 12, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    // Pulse ring
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 14, 0, Math.PI * 2);
    ctx.stroke();

    // Filled dot
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(0, 0, 6, 0, Math.PI * 2);
    ctx.fill();

    // Direction
    const rad = (yawDeg || 0) * Math.PI / 180;
    ctx.rotate(-rad);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(14, 0);
    ctx.stroke();
  }
  ctx.restore();
}

// Map interaction
function onMapMouseDown(e) {
  if (e.button === 1 || (e.button === 0 && state.map.tool === 'pan')) {
    state.map.dragging = true;
    state.map.lastMouse = { x: e.clientX, y: e.clientY };
  }
}

function onMapMouseMove(e) {
  const rect = $('map-canvas').getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const {wx, wy} = canvasToWorld(cx, cy);

  $('map-cursor-pos').textContent = `X: ${wx.toFixed(2)}m  Y: ${wy.toFixed(2)}m`;

  if (state.map.dragging) {
    const dx = e.clientX - state.map.lastMouse.x;
    const dy = e.clientY - state.map.lastMouse.y;
    state.map.transform.x += dx;
    state.map.transform.y += dy;
    state.map.lastMouse = { x: e.clientX, y: e.clientY };
    drawMap();
  }
}

function onMapMouseUp(e) {
  if (state.map.dragging) {
    state.map.dragging = false;
    return;
  }

  if (e.button === 0 && state.map.tool === 'navigate') {
    const rect = $('map-canvas').getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const {wx, wy} = canvasToWorld(cx, cy);
    showNavTooltip(cx, cy, wx, wy);
  }
}

function onMapWheel(e) {
  e.preventDefault();
  const rect = $('map-canvas').getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const factor = e.deltaY > 0 ? 0.85 : 1.18;
  zoomMapAt(factor, mx, my);
}

function zoomMap(factor) {
  const cx = state.map.width / 2;
  const cy = state.map.height / 2;
  zoomMapAt(factor, cx, cy);
}

function zoomMapAt(factor, mx, my) {
  const t = state.map.transform;
  t.x = mx + (t.x - mx) * factor;
  t.y = my + (t.y - my) * factor;
  t.scale = Math.max(5, Math.min(500, t.scale * factor));
  drawMap();
}

function resetMapView() {
  state.map.transform = {
    x: state.map.width / 2,
    y: state.map.height / 2,
    scale: 40
  };
  drawMap();
}

function setMapTool(tool) {
  state.map.tool = tool;
  $$('.map-tool').forEach(b => b.classList.remove('active'));
  $(`tool-${tool}`).classList.add('active');
  $('map-canvas').style.cursor = tool === 'navigate' ? 'crosshair' : 'grab';
}

function showNavTooltip(canvasX, canvasY, wx, wy) {
  state.map.goalPending = { x: wx, y: wy };
  const tooltip = $('map-tooltip');
  $('tooltip-coords').textContent = `X: ${wx.toFixed(2)}m  Y: ${wy.toFixed(2)}m`;
  $('tooltip-yaw').value = 0;

  const wrapper = $('map-wrapper');
  const wRect = wrapper.getBoundingClientRect();
  let left = canvasX + 15;
  let top = canvasY - 80;
  if (left + 210 > wrapper.clientWidth) left = canvasX - 225;
  if (top < 0) top = canvasY + 15;

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
  tooltip.style.display = 'block';

  drawMap();
}

async function confirmNavGoal() {
  const g = state.map.goalPending;
  if (!g) return;
  const yawDeg = parseFloat($('tooltip-yaw').value || '0');
  const yawRad = yawDeg * Math.PI / 180;

  $('map-tooltip').style.display = 'none';
  state.map.goalPending = null;
  state.map.goalActive = { x: g.x, y: g.y, yaw: yawRad };

  // Sincronizează cu input-urile din panoul de navigare
  $('nav-x').value = g.x.toFixed(2);
  $('nav-y').value = g.y.toFixed(2);
  $('nav-yaw').value = yawDeg;

  const result = await apiPost('/api/nav/goal', { x: g.x, y: g.y, yaw: yawRad });
  $('nav-active-goal').textContent = `(${g.x.toFixed(2)}, ${g.y.toFixed(2)})`;
  $('nav-state-chip').textContent = 'Navigând';
  $('nav-state-chip').className = 'status-chip active';

  if (result.success !== false) {
    showToast(`Navigare spre (${g.x.toFixed(1)}, ${g.y.toFixed(1)}) ✓`, 'success');
  } else {
    showToast(`Eroare navigare: ${result.error || result.stderr || 'Necunoscută'}`, 'error');
  }
  drawMap();
}

// =========================================================================
// NAV CONTROLS
// =========================================================================
function initNavControls() {
  $('btn-send-goal').addEventListener('click', async () => {
    const x = parseFloat($('nav-x').value || '0');
    const y = parseFloat($('nav-y').value || '0');
    const yawDeg = parseFloat($('nav-yaw').value || '0');
    const yaw = yawDeg * Math.PI / 180;
    const timeout = parseFloat($('nav-timeout').value || '30');

    const result = await apiPost('/api/nav/goal', { x, y, yaw, timeout });
    state.map.goalActive = { x, y, yaw };
    $('nav-active-goal').textContent = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
    $('nav-state-chip').textContent = 'Navigând';
    $('nav-state-chip').className = 'status-chip active';
    drawMap();

    if (result.success !== false) {
      showToast(`Goal trimis: (${x.toFixed(1)}, ${y.toFixed(1)}) ✓`, 'success');
    } else {
      showToast(`Eroare: ${result.error || 'Navigare eșuată'}`, 'error');
    }
  });

  $('btn-stop-robot').addEventListener('click', async () => {
    const result = await apiPost('/api/nav/stop');
    state.map.goalActive = null;
    $('nav-state-chip').textContent = 'Oprit';
    $('nav-state-chip').className = 'status-chip';
    $('nav-active-goal').textContent = '--';
    drawMap();
    showToast('Robot oprit ✓', 'info');
  });

  $('btn-save-waypoint').addEventListener('click', () => {
    const x = parseFloat($('nav-x').value || state.pose.x);
    const y = parseFloat($('nav-y').value || state.pose.y);
    const yawDeg = parseFloat($('nav-yaw').value || '0');
    const name = prompt('Numele waypoint-ului:', `WP ${state.waypoints.length + 1}`);
    if (!name) return;
    const wp = { id: Date.now(), name, x, y, yaw: yawDeg };
    state.waypoints.push(wp);
    saveWaypoints();
    renderWaypoints();
    showToast(`Waypoint "${name}" salvat ✓`, 'success');
  });
}

function loadWaypoints() {
  state.waypoints = JSON.parse(localStorage.getItem('g1_waypoints') || '[]');
  renderWaypoints();
}

function saveWaypoints() {
  localStorage.setItem('g1_waypoints', JSON.stringify(state.waypoints));
}

function renderWaypoints() {
  const list = $('waypoints-list');
  if (state.waypoints.length === 0) {
    list.innerHTML = '<div class="empty-state-sm">Niciun waypoint salvat.</div>';
    return;
  }
  list.innerHTML = state.waypoints.map(wp => `
    <div class="waypoint-item" data-id="${wp.id}">
      <span class="waypoint-label">${escapeHtml(wp.name)}</span>
      <span class="waypoint-coords">(${wp.x.toFixed(1)}, ${wp.y.toFixed(1)})</span>
      <div class="waypoint-actions">
        <button class="btn btn-ghost btn-sm" onclick="navigateToWaypoint(${wp.id})" title="Navighează">➜</button>
        <button class="btn btn-ghost btn-sm danger" onclick="deleteWaypoint(${wp.id})" title="Șterge">✕</button>
      </div>
    </div>
  `).join('');
}

window.navigateToWaypoint = async (id) => {
  const wp = state.waypoints.find(w => w.id === id);
  if (!wp) return;
  const yawRad = wp.yaw * Math.PI / 180;
  await apiPost('/api/nav/goal', { x: wp.x, y: wp.y, yaw: yawRad });
  showToast(`Navighează spre "${wp.name}" ✓`, 'success');
  state.map.goalActive = { x: wp.x, y: wp.y, yaw: yawRad };
  drawMap();
};

window.deleteWaypoint = (id) => {
  state.waypoints = state.waypoints.filter(w => w.id !== id);
  saveWaypoints();
  renderWaypoints();
};

// =========================================================================
// SCHEDULER
// =========================================================================
function initScheduler() {
  $('task-type-select').addEventListener('change', updateTaskParamsUI);
  updateTaskParamsUI();

  $('btn-add-task').addEventListener('click', addTask);
  $('btn-run-scheduler').addEventListener('click', startScheduler);
  $('btn-stop-scheduler').addEventListener('click', stopScheduler);
  $('btn-reset-tasks').addEventListener('click', async () => {
    await apiPost('/api/tasks/reset');
    showToast('Task-uri resetate la Pending', 'info');
  });
  $('btn-clear-tasks').addEventListener('click', () => {
    if (!confirm('Ștergi toate task-urile?')) return;
    apiDelete('/api/tasks').then(() => {
      renderTasks([]);
      showToast('Task-uri șterse', 'info');
    });
  });

  // Load initial tasks
  apiFetch('/api/tasks').then(data => {
    if (data.tasks) renderTasks(data.tasks);
    if (typeof data.running !== 'undefined') {
      state.schedulerRunning = data.running;
      updateSchedulerUI();
    }
  });
}

function updateTaskParamsUI() {
  const type = $('task-type-select').value;
  $$('.task-params').forEach(el => el.style.display = 'none');
  const el = $(`task-params-${type}`);
  if (el) el.style.display = 'flex';
}

async function addTask() {
  const type = $('task-type-select').value;
  const name = $('task-name-input').value.trim() || typeLabel(type);
  let params = {};

  if (type === 'navigate') {
    params = {
      x: parseFloat($('tp-x').value || '0'),
      y: parseFloat($('tp-y').value || '0'),
      yaw: parseFloat($('tp-yaw').value || '0') * Math.PI / 180,
      timeout: parseFloat($('tp-timeout').value || '30')
    };
  } else if (type === 'wait') {
    params = { seconds: parseFloat($('tp-wait-secs').value || '3') };
  } else if (type === 'slam_save') {
    params = { map_name: $('tp-map-name').value.trim() || `map_${Date.now()}` };
  } else if (type === 'command') {
    params = {
      type: $('tp-cmd-type').value,
      command: $('tp-cmd-text').value.trim()
    };
  }

  const result = await apiPost('/api/tasks', { name, type, params });
  $('task-name-input').value = '';
  showToast(`Task "${name}" adăugat ✓`, 'success');
}

async function startScheduler() {
  const result = await apiPost('/api/tasks/start');
  if (result.success) {
    state.schedulerRunning = true;
    updateSchedulerUI();
    showToast('Scheduler pornit ✓', 'success');
  } else {
    showToast(result.error || 'Eroare la pornire', 'error');
  }
}

async function stopScheduler() {
  await apiPost('/api/tasks/stop');
  state.schedulerRunning = false;
  updateSchedulerUI();
  showToast('Scheduler oprit.', 'info');
}

function updateSchedulerUI() {
  const running = state.schedulerRunning;
  $('scheduler-state').textContent = running ? 'RULEAZĂ' : 'Oprit';
  $('scheduler-state').className = 'scheduler-state' + (running ? ' running' : '');
  $('btn-run-scheduler').disabled = running;
  $('btn-stop-scheduler').disabled = !running;
}

function renderTasks(tasks) {
  state.tasks = tasks;
  const list = $('tasks-list');
  const empty = $('tasks-empty-state');

  if (!tasks || tasks.length === 0) {
    list.innerHTML = '';
    list.appendChild(empty);
    empty.style.display = 'flex';
    updateProgress(0, 0);
    updateBadge('scheduler', 0);
    return;
  }

  empty.style.display = 'none';
  const icons = { navigate: '🧭', wait: '⏱', slam_start: '▶', slam_stop: '⏹', slam_save: '💾', command: '⚙' };

  list.innerHTML = tasks.map(t => {
    const detail = getTaskDetail(t);
    const statusClass = t.status !== 'pending' ? `status-${t.status}` : '';
    const isRunning = t.status === 'running';
    return `
      <div class="task-item ${statusClass}" data-id="${t.id}">
        <span class="task-drag">⠿</span>
        <span class="task-icon">${icons[t.type] || '•'}</span>
        <div class="task-info">
          <div class="task-name">${escapeHtml(t.name)}</div>
          ${detail ? `<div class="task-detail">${detail}</div>` : ''}
          ${t.result ? `<div class="task-detail" style="color:var(--text-muted)">${escapeHtml(t.result.substring(0,80))}</div>` : ''}
        </div>
        ${isRunning ? '<div class="task-spinner"></div>' : ''}
        <span class="task-status-chip ${t.status !== 'pending' ? t.status : ''}">${statusLabel(t.status)}</span>
        ${t.status === 'pending' ? `<button class="btn btn-ghost btn-sm danger" onclick="deleteTask('${t.id}')">✕</button>` : ''}
      </div>
    `;
  }).join('');

  // Progress
  const done = tasks.filter(t => t.status === 'done').length;
  updateProgress(done, tasks.length);
  updateBadge('scheduler', tasks.filter(t => t.status === 'pending').length);
}

function getTaskDetail(t) {
  if (t.type === 'navigate') {
    const p = t.params;
    const yawDeg = p.yaw ? (p.yaw * 180 / Math.PI).toFixed(0) : 0;
    return `X:${(p.x||0).toFixed(2)} Y:${(p.y||0).toFixed(2)} Yaw:${yawDeg}°`;
  }
  if (t.type === 'wait') return `${t.params.seconds || 1}s`;
  if (t.type === 'slam_save') return t.params.map_name || '';
  if (t.type === 'command') return (t.params.command || '').substring(0, 40);
  return '';
}

window.deleteTask = async (id) => {
  await apiDelete(`/api/tasks/${id}`);
};

function updateProgress(done, total) {
  const pct = total > 0 ? (done / total * 100) : 0;
  $('progress-bar').style.width = `${pct}%`;
  $('progress-text').textContent = `${done} / ${total} task-uri completate`;
}

function statusLabel(s) {
  return { pending: 'Pending', running: 'Rulează', done: 'Done', failed: 'Eșuat', cancelled: 'Anulat' }[s] || s;
}
function typeLabel(t) {
  return { navigate: 'Navigare', wait: 'Așteptare', slam_start: 'Start Mapping', slam_stop: 'Stop Mapping', slam_save: 'Salvare Hartă', command: 'Comandă' }[t] || t;
}

// =========================================================================
// COMMANDS
// =========================================================================
function initCommands() {
  $('btn-create-cmd').addEventListener('click', async () => {
    const name = $('cmd-name').value.trim();
    const type = $('cmd-type').value;
    const command = $('cmd-command').value.trim();
    const description = $('cmd-description').value.trim();

    if (!name || !command) {
      showToast('Completează numele și comanda!', 'warning');
      return;
    }

    await apiPost('/api/commands', { name, type, command, description });
    $('cmd-name').value = '';
    $('cmd-command').value = '';
    $('cmd-description').value = '';
    showToast(`Comanda "${name}" creată ✓`, 'success');

    const data = await apiFetch('/api/commands');
    renderCommands(data.commands || []);
  });

  // Load initial commands
  apiFetch('/api/commands').then(data => {
    if (data.commands) renderCommands(data.commands);
  });
}

function renderCommands(commands) {
  state.commands = commands;
  const list = $('commands-list');
  const empty = $('commands-empty-state');
  $('commands-count').textContent = `${commands.length} comenzi`;

  if (!commands || commands.length === 0) {
    list.innerHTML = '';
    list.appendChild(empty);
    empty.style.display = 'flex';
    return;
  }

  empty.style.display = 'none';
  const typeIcons = { shell: '🖥', ros2_topic: '📡', ros2_service: '🔧', python: '🐍' };

  list.innerHTML = commands.map(cmd => `
    <div class="command-item">
      <div class="cmd-icon">${typeIcons[cmd.type] || '⚙'}</div>
      <div class="cmd-info">
        <div class="cmd-name">${escapeHtml(cmd.name)}</div>
        ${cmd.description ? `<div class="cmd-desc">${escapeHtml(cmd.description)}</div>` : ''}
        <div class="cmd-preview">${escapeHtml(cmd.command)}</div>
      </div>
      <div class="cmd-actions">
        <span class="cmd-type-badge">${cmd.type}</span>
        <div style="display:flex;gap:4px;margin-top:6px">
          <button class="btn btn-success btn-sm" onclick="runCommand('${cmd.id}')">▶ Run</button>
          <button class="btn btn-ghost btn-sm" onclick="addCmdToTask('${cmd.id}')">+ Task</button>
          <button class="btn btn-ghost btn-sm danger" onclick="deleteCommand('${cmd.id}')">✕</button>
        </div>
      </div>
    </div>
  `).join('');
}

window.runCommand = async (id) => {
  showToast('Execut comanda...', 'info');
  const result = await apiPost(`/api/commands/${id}/run`);
  if (result.success) {
    showToast(`Comanda executată ✓\n${(result.output || '').substring(0, 60)}`, 'success');
  } else {
    showToast(`Eroare: ${result.error || 'Necunoscută'}`, 'error');
  }
};

window.deleteCommand = async (id) => {
  if (!confirm('Ștergi această comandă?')) return;
  await apiDelete(`/api/commands/${id}`);
  const data = await apiFetch('/api/commands');
  renderCommands(data.commands || []);
};

window.addCmdToTask = (id) => {
  const cmd = state.commands.find(c => c.id === id);
  if (!cmd) return;
  // Switch la scheduler și pre-completează formularul
  $$('.nav-item').forEach(b => { if (b.dataset.tab === 'scheduler') b.click(); });
  $('task-type-select').value = 'command';
  updateTaskParamsUI();
  $('tp-cmd-type').value = cmd.type;
  $('tp-cmd-text').value = cmd.command;
  $('task-name-input').value = cmd.name;
  showToast(`Comanda "${cmd.name}" pregătită pentru scheduler ✓`, 'info');
};

// =========================================================================
// UI HELPERS
// =========================================================================
function updatePoseDisplay(pose) {
  $('pose-x').textContent = pose.x.toFixed(2);
  $('pose-y').textContent = pose.y.toFixed(2);
  $('pose-yaw').textContent = (pose.yaw_deg || (pose.yaw * 180 / Math.PI) || 0).toFixed(1);
  $('nav-current-pos').textContent = `(${pose.x.toFixed(2)}, ${pose.y.toFixed(2)})`;
}

function updateWsStatus(connected) {
  $('ws-dot').className = 'status-dot ' + (connected ? 'ok' : 'err');
  $('ws-status-text').textContent = connected ? 'Conectat' : 'Deconectat';
}

function updateBadge(tab, count) {
  const badge = $(`badge-${tab}`);
  if (!badge) return;
  badge.textContent = count;
  badge.style.display = count > 0 ? 'inline-block' : 'none';
}

function addLog(message, level = 'info') {
  const container = $('log-container');
  const entry = document.createElement('div');
  entry.className = `log-entry log-${level}`;
  const now = new Date();
  const time = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
  entry.innerHTML = `<span class="log-time">${time}</span><span>${escapeHtml(message)}</span>`;
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;

  // Keep max 100 entries
  while (container.children.length > 100) {
    container.removeChild(container.firstChild);
  }
}

function showToast(message, type = 'info') {
  const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ'}</span><span class="toast-text">${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('fade-out');
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

// =========================================================================
// API HELPERS
// =========================================================================
async function apiFetch(path) {
  try {
    const resp = await fetch(`${API_BASE}${path}`);
    return await resp.json();
  } catch (e) {
    console.error('API fetch error:', path, e);
    return {};
  }
}

async function apiPost(path, body = {}) {
  try {
    const resp = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    return await resp.json();
  } catch (e) {
    console.error('API post error:', path, e);
    return { success: false, error: e.message };
  }
}

async function apiDelete(path) {
  try {
    const resp = await fetch(`${API_BASE}${path}`, { method: 'DELETE' });
    return await resp.json();
  } catch (e) {
    return { success: false };
  }
}

function escapeHtml(str) {
  if (typeof str !== 'string') return String(str || '');
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// =========================================================================
// KEYBOARD SHORTCUTS
// =========================================================================
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  switch (e.key.toLowerCase()) {
    case 'p': setMapTool('pan'); break;
    case 'n': setMapTool('navigate'); break;
    case 's': if (state.activeTab === 'navigation') apiPost('/api/nav/stop'); break;
    case '1': $$('.nav-item')[0]?.click(); break;
    case '2': $$('.nav-item')[1]?.click(); break;
    case '3': $$('.nav-item')[2]?.click(); break;
    case '4': $$('.nav-item')[3]?.click(); break;
  }
});
