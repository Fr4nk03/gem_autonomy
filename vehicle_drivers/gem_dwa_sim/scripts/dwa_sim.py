#!/usr/bin/env python3

#================================================================
# File name: dwa_sim.py
# Description: Dynamic Window Approach (DWA) local planner for the
#              GEM vehicle.  Drives to a fixed goal while avoiding
#              static obstacles (detected via Ouster LiDAR) and a
#              single dynamic agent (tracked via LiDAR clusters +
#              known constant velocity).  The DWA rollout uses a
#              time-indexed collision check for the dynamic agent so
#              the planner anticipates its future position.
#
# Usage:
#   roslaunch gem_dwa_sim dwa_sim.launch
# Python version: 3.8
#================================================================

import math
import os
import threading
import yaml

import cv2
import numpy as np
from numpy import linalg as la

import rospy
import sensor_msgs.point_cloud2 as pc2

from ackermann_msgs.msg     import AckermannDrive
from cv_bridge              import CvBridge, CvBridgeError
from gazebo_msgs.msg        import ModelState
from gazebo_msgs.srv        import GetModelState, SetModelState
from geometry_msgs.msg      import Twist
from sensor_msgs.msg        import Image, PointCloud2
from tf.transformations     import euler_from_quaternion


# ══════════════════════════════════════════════════════════════
#  DWA Configuration
# ══════════════════════════════════════════════════════════════

class DWAConfig:
    # ── Vehicle ──────────────────────────────────────────────
    wheelbase   = 1.75   # m  (GEM e4)
    robot_radius = 1.3   # m  collision / safety radius

    # ── Speed limits ─────────────────────────────────────────
    max_speed   = 4.0    # m/s  (~14.4 km/h, GEM e4 top speed)
    min_speed   = 0.0    # m/s  (no reversing)
    max_steer   = 0.50   # rad  (~28.6°)
    # ω_max at max_speed: v·tan(δ_max)/L = 4.0·tan(0.5)/1.75 ≈ 1.25
    max_yaw_rate = 1.3   # rad/s

    # ── Acceleration limits ───────────────────────────────────
    max_accel          = 1.0   # m/s²
    max_delta_yaw_rate = 0.5   # rad/s²

    # ── Sampling resolution ───────────────────────────────────
    v_resolution        = 0.10  # m/s
    yaw_rate_resolution = 0.05  # rad/s

    # ── Rollout ───────────────────────────────────────────────
    dt           = 0.10  # s
    predict_time = 3.0   # s

    # ── Dynamic agent ─────────────────────────────────────────
    # Larger safety bubble than robot_radius — planner starts
    # steering away from the pedestrian well before a hard collision.
    dynamic_agent_radius = 2.5   # m

    # ── Cost weights ─────────────────────────────────────────
    to_goal_cost_gain  = 0.15
    speed_cost_gain    = 2.0   # stronger push toward max_speed on clear paths
    obstacle_cost_gain = 1.0
    dynamic_cost_gain  = 3.0   # higher than static — pedestrian avoidance priority

    # ── Terminal condition ────────────────────────────────────
    goal_tolerance = 0.5  # m  → hard brake issued when inside this radius


# ══════════════════════════════════════════════════════════════
#  Pure DWA functions  (no ROS dependency)
# ══════════════════════════════════════════════════════════════

def _motion(state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Kinematic model: state=[x,y,yaw,v,ω], u=[v,ω]."""
    s = state.copy()
    s[2] += u[1] * dt                   # yaw
    s[0] += u[0] * math.cos(s[2]) * dt  # x
    s[1] += u[0] * math.sin(s[2]) * dt  # y
    s[3]  = u[0]                         # v
    s[4]  = u[1]                         # ω
    return s


def _dynamic_window(state: np.ndarray, cfg: DWAConfig):
    Vs = [cfg.min_speed,    cfg.max_speed,
          -cfg.max_yaw_rate, cfg.max_yaw_rate]
    Vd = [state[3] - cfg.max_accel          * cfg.dt,
          state[3] + cfg.max_accel          * cfg.dt,
          state[4] - cfg.max_delta_yaw_rate * cfg.dt,
          state[4] + cfg.max_delta_yaw_rate * cfg.dt]
    return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]


def _rollout(state_init: np.ndarray, v: float, omega: float,
             cfg: DWAConfig) -> np.ndarray:
    s = state_init.copy()
    traj = [s.copy()]
    t = 0.0
    while t <= cfg.predict_time:
        s = _motion(s, np.array([v, omega]), cfg.dt)
        traj.append(s.copy())
        t += cfg.dt
    return np.array(traj)


def _static_cost(traj: np.ndarray,
                 static_obs: np.ndarray,
                 cfg: DWAConfig) -> float:
    """
    Cost from static LiDAR obstacles.
    Returns inf on collision (within robot_radius), else 1 / min_clearance.
    """
    min_clearance = float('inf')
    for pt in traj:
        if static_obs.size == 0:
            break
        dx = static_obs[:, 0] - pt[0]
        dy = static_obs[:, 1] - pt[1]
        d  = float(np.hypot(dx, dy).min())
        if d <= cfg.robot_radius:
            return float('inf')
        if d < min_clearance:
            min_clearance = d
    return 0.0 if math.isinf(min_clearance) else 1.0 / min_clearance


def _dynamic_cost(traj: np.ndarray,
                  agent_pos,
                  agent_vel: np.ndarray,
                  cfg: DWAConfig) -> float:
    """
    Cost from the single dynamic agent.
    Uses dynamic_agent_radius (larger than robot_radius) so the planner
    starts steering away well before a hard collision.
    Agent position is time-indexed: agent_pos + agent_vel * (i * dt).
    Returns inf on violation, else 1 / min_clearance.
    """
    if agent_pos is None:
        return 0.0
    min_clearance = float('inf')
    for i, pt in enumerate(traj):
        t  = i * cfg.dt
        ax = agent_pos[0] + agent_vel[0] * t
        ay = agent_pos[1] + agent_vel[1] * t
        d  = math.hypot(pt[0] - ax, pt[1] - ay)
        if d <= cfg.dynamic_agent_radius:
            return float('inf')
        if d < min_clearance:
            min_clearance = d
    return 0.0 if math.isinf(min_clearance) else 1.0 / min_clearance


def _goal_cost(traj: np.ndarray, goal: np.ndarray) -> float:
    return math.hypot(goal[0] - traj[-1, 0], goal[1] - traj[-1, 1])


def dwa_control(state:      np.ndarray,
                goal:       np.ndarray,
                static_obs: np.ndarray,
                agent_pos,
                agent_vel:  np.ndarray,
                cfg:        DWAConfig):
    """
    Returns (best_u, best_traj).
    best_u = [v, omega].
    """
    dw = _dynamic_window(state, cfg)

    best_u    = np.array([0.0, 0.0])
    best_traj = _rollout(state, 0.0, 0.0, cfg)
    min_cost  = float('inf')

    for v in np.arange(dw[0], dw[1] + cfg.v_resolution, cfg.v_resolution):
        for omega in np.arange(dw[2], dw[3] + cfg.yaw_rate_resolution,
                               cfg.yaw_rate_resolution):
            traj = _rollout(state, v, omega, cfg)

            sc     = cfg.obstacle_cost_gain * _static_cost(traj, static_obs, cfg)
            dc     = cfg.dynamic_cost_gain  * _dynamic_cost(traj, agent_pos, agent_vel, cfg)
            goal_c = cfg.to_goal_cost_gain  * _goal_cost(traj, goal)
            spd    = cfg.speed_cost_gain    * (cfg.max_speed - traj[-1, 3])

            cost = sc + dc + goal_c + spd
            if cost < min_cost:
                min_cost  = cost
                best_u    = np.array([v, omega])
                best_traj = traj

    return best_u, best_traj


# ══════════════════════════════════════════════════════════════
#  LiDAR helpers
# ══════════════════════════════════════════════════════════════

def _voxel_filter(pts: np.ndarray, res: float = 0.5) -> np.ndarray:
    """Downsample 2-D points to a voxel grid of size res."""
    if len(pts) == 0:
        return pts
    cells = np.round(pts / res).astype(int)
    _, idx = np.unique(cells, axis=0, return_index=True)
    return pts[idx]


def _cluster(pts: np.ndarray,
             radius: float = 1.5,
             min_pts: int  = 3):
    """
    Simple greedy DBSCAN-like clustering for small point sets.
    Returns list of (centroid_xy, n_members).
    """
    if len(pts) < min_pts:
        return []
    assigned = np.zeros(len(pts), dtype=bool)
    clusters = []
    for i in range(len(pts)):
        if assigned[i]:
            continue
        dists   = la.norm(pts - pts[i], axis=1)
        members = np.where((dists < radius) & ~assigned)[0]
        if len(members) < min_pts:
            assigned[i] = True
            continue
        assigned[members] = True
        clusters.append((pts[members].mean(axis=0), len(members)))
    return clusters


# ══════════════════════════════════════════════════════════════
#  Constant-velocity agent tracker
# ══════════════════════════════════════════════════════════════

class AgentTracker:
    """
    Tracks a single dynamic agent whose LiDAR points are pre-filtered
    to a known bounding box (derived from the agent's trajectory).

    Because only points inside the bounding box are passed in, the
    largest cluster is unambiguously the agent — no nearest-neighbour
    association needed.

    Velocity is estimated from consecutive detections using EMA:
        raw_vel = (new_pos - prev_pos) / dt
        vel     = alpha * raw_vel + (1 - alpha) * vel_prev
    """

    def __init__(self,
                 init_vel=None,
                 alpha: float = 0.4):
        """
        init_vel: optional (vx, vy) seed used before the first detection pair
        alpha:    EMA factor — higher = faster adaptation, more noise
        """
        self.pos    = None
        self.vel    = np.asarray(init_vel, dtype=float).copy() if init_vel is not None \
                      else np.zeros(2)
        self.t_last = None
        self.alpha  = alpha

    def update(self, clusters, now: float):
        """
        clusters: list of (centroid_xy, n_pts) already filtered to agent region.
        Picks the largest cluster as the agent detection.
        """
        if not clusters:
            return

        best    = max(clusters, key=lambda c: c[1])
        new_pos = best[0].copy()

        if self.pos is not None and self.t_last is not None:
            dt = now - self.t_last
            if dt >= 0.02:                              # guard against tiny dt
                raw_vel  = (new_pos - self.pos) / dt
                self.vel = self.alpha * raw_vel + (1.0 - self.alpha) * self.vel

        if self.pos is None:
            rospy.loginfo(f"[Tracker] First detection at "
                          f"({new_pos[0]:.2f}, {new_pos[1]:.2f})")

        self.pos    = new_pos
        self.t_last = now

    @property
    def position(self):
        return self.pos.copy() if self.pos is not None else None

    @property
    def velocity(self):
        return self.vel.copy()


# ══════════════════════════════════════════════════════════════
#  ROS Node
# ══════════════════════════════════════════════════════════════

class GemDwaNode:

    def __init__(self):
        cfg = DWAConfig()
        self.cfg = cfg

        # ── Goal (fixed at launch) ────────────────────────────
        gx = float(rospy.get_param('~goal_x', -20.0))
        gy = float(rospy.get_param('~goal_y', -20.0))
        self.goal = np.array([gx, gy])
        rospy.loginfo(f"[DWA] Goal: ({gx:.2f}, {gy:.2f})")

        # ── Agent tracker (velocity estimated from LiDAR) ─────
        avx = float(rospy.get_param('~agent_vel_x', 0.0))
        avy = float(rospy.get_param('~agent_vel_y', 0.0))
        self.tracker = AgentTracker(init_vel=np.array([avx, avy]))

        # Bounding box around agent trajectory (+ margin) for LiDAR filtering
        yaml_path        = rospy.get_param('~yaml_path', '')
        bbox_margin      = float(rospy.get_param('~agent_bbox_margin', 2.0))
        self._agent_bbox = self._agent_bbox_from_yaml(yaml_path, margin=bbox_margin)
        if self._agent_bbox is not None:
            rospy.loginfo(f"[DWA] Agent bbox: "
                          f"x=[{self._agent_bbox[0]:.1f}, {self._agent_bbox[1]:.1f}]  "
                          f"y=[{self._agent_bbox[2]:.1f}, {self._agent_bbox[3]:.1f}]")

        # ── Thread-safe shared state ──────────────────────────
        # _vehicle_pose: (x, y, z, yaw) updated by main loop, read by lidar callback
        self._lock         = threading.Lock()
        self._vehicle_pose = None                  # (x, y, z, yaw)
        self._static_obs   = np.empty((0, 2))
        self._agent_pos    = None                  # updated by tracker in lidar callback
        self._agent_vel    = np.array([avx, avy])  # updated by tracker in lidar callback

        # ── Ouster sensor offset in body (base_link) frame ────
        # Chain: base_link -[xyz=(-0.17,0,1.66) rpy=(π/2,0,0)]-> top_rack_link
        #              -[xyz=(0.02,0.42,0) rpy=(π/2,0,π)]-> ouster frame
        # Computed once; R_sensor_to_base = diag(-1,-1,1) (180° about z)
        self._sensor_in_base  = np.array([-0.15, 0.0, 2.08])   # metres
        self._R_sensor_to_base = np.array([[-1., 0., 0.],
                                            [ 0.,-1., 0.],
                                            [ 0., 0., 1.]])

        # ── Publishers ────────────────────────────────────────
        # self.ackermann_pub = rospy.Publisher('/ackermann_cmd',   AckermannDrive, queue_size=1)
        self.ackermann_pub = rospy.Publisher('/dwa_cmd', AckermannDrive, queue_size=1)
        self.map_pub       = rospy.Publisher('/dwa/map_view',    Image,          queue_size=1)

        self.ackermann_msg = AckermannDrive()
        self.bridge        = CvBridge()

        # ── GNSS map image ────────────────────────────────────
        # Path: POLARIS_GEM_Simulator/images/gnss_map.png
        _root = os.path.abspath(__file__).split('vehicle_drivers')[0]
        _map_path = os.path.join(_root, 'images', 'gnss_map.png')
        self.map_image = cv2.imread(_map_path)
        if self.map_image is None:
            rospy.logwarn(f"[DWA] GNSS map not found at {_map_path}; using blank canvas.")
        else:
            rospy.loginfo(f"[DWA] Loaded GNSS map: {self.map_image.shape}")

        # Image dimensions (from gem_gnss_image.py)
        self.img_w = 2107
        self.img_h = 1313

        # GPS bounding box of the GNSS map image (from gem_gnss_image.py)
        self.lat_start_bt = 40.092722    # latitude at image bottom edge
        self.lon_start_l  = -88.236365   # longitude at image left edge
        self.lat_scale    = 0.00062      # degrees latitude image spans (bottom→top)
        self.lon_scale    = 0.00136      # degrees longitude image spans (left→right)

        # Gazebo world GPS origin (from highbay_track.world spherical_coordinates)
        # Gazebo (0, 0) ↔ (lat_ref, lon_ref)
        # Gazebo x → East, Gazebo y → North
        self.lat_ref  = 40.093013541202175
        self.lon_ref  = -88.23576464901934
        self.m_per_lat = 111111.0
        self.m_per_lon = 111111.0 * math.cos(math.radians(self.lat_ref))
        # Pixels per metre (≈ 18 px/m at this scale)
        self.px_per_m = (self.img_w / self.lon_scale) / self.m_per_lon

        # ── LiDAR subscriber ─────────────────────────────────
        rospy.Subscriber('/ouster/points', PointCloud2,
                         self._lidar_cb, queue_size=1, buff_size=2**24)

        self.rate = rospy.Rate(10)
        rospy.loginfo("[DWA] Node ready.")

    # ── YAML helpers ──────────────────────────────────────────────────────────

    def _agent_bbox_from_yaml(self, path: str, margin: float = 2.0):
        """
        Compute an (x_min, x_max, y_min, y_max) bounding box from the first
        agent's trajectory waypoints plus a safety margin.
        Returns None if YAML is missing or has no agents.
        """
        if not path:
            return None
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            agents = (data or {}).get('agents', [])
            if not agents:
                return None
            traj = agents[0].get('trajectory', [])
            if not traj:
                return None
            xs = [float(wp[1]) for wp in traj]
            ys = [float(wp[2]) for wp in traj]
            bbox = (min(xs) - margin, max(xs) + margin,
                    min(ys) - margin, max(ys) + margin)
            return bbox          # (x_min, x_max, y_min, y_max)
        except Exception as e:
            rospy.logwarn(f"[DWA] Could not compute agent bbox from YAML: {e}")
        return None

    # ── LiDAR callback ────────────────────────────────────────────────────────

    def _lidar_cb(self, msg: PointCloud2):
        """
        Transform Ouster points to world frame using cached Gazebo vehicle
        pose + precomputed URDF sensor offset — no TF required.

        Pipeline:
          1. Read every 4th point for speed.
          2. Rotate sensor → base_link (fixed URDF rotation).
          3. Rotate + translate base_link → world (from Gazebo pose).
          4. Ground-filter: 0.10 m < z_world < 2.50 m.
          5. Self-filter: 2.5 m < r_xy < 30 m from sensor.
          6. Voxel downsample → cluster.
          7. Associate nearest cluster to agent tracker; rest → static obstacles.
        """
        with self._lock:
            pose = self._vehicle_pose   # (x, y, z, yaw) or None

        if pose is None:
            return   # vehicle state not yet received

        vx, vy, vz, vyaw = pose

        # ── Rotation matrices ─────────────────────────────────
        cy, sy = math.cos(vyaw), math.sin(vyaw)
        R_base_world = np.array([[ cy, -sy, 0.],
                                  [ sy,  cy, 0.],
                                  [ 0.,  0., 1.]])

        # sensor → world  =  base→world  ×  sensor→base
        R_s2w = R_base_world @ self._R_sensor_to_base

        # sensor origin in world frame
        sensor_world = R_base_world @ self._sensor_in_base + np.array([vx, vy, vz])
        sensor_xy    = sensor_world[:2]

        # ── Read point cloud (every 4th point for speed) ──────
        raw = []
        gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        for i, pt in enumerate(gen):
            if i % 4 == 0:
                raw.append(pt)

        if not raw:
            return

        pts_s = np.array(raw, dtype=float)   # Nx3 in sensor frame

        # ── Transform to world frame (vectorised) ─────────────
        pts_w = (R_s2w @ pts_s.T).T + sensor_world   # Nx3

        # ── Ground filter ─────────────────────────────────────
        # 0.30 m floor (not 0.10) — absorbs small z-errors from the
        # approximate sensor transform so ground points don't bleed through.
        mask  = (pts_w[:, 2] > 0.30) & (pts_w[:, 2] < 2.20)
        pts_w = pts_w[mask]
        if len(pts_w) == 0:
            return

        # ── Self-filter ───────────────────────────────────────
        # 3.0 m inner radius covers the full GEM body length.
        # 20 m outer radius cuts far-field sparse returns.
        r_xy  = la.norm(pts_w[:, :2] - sensor_xy, axis=1)
        pts_w = pts_w[(r_xy > 3.0) & (r_xy < 20.0)]
        if len(pts_w) == 0:
            return

        pts_2d = pts_w[:, :2]

        now = rospy.Time.now().to_sec()

        # ── Split points: agent region vs. static region ──────
        if self._agent_bbox is not None:
            x_min, x_max, y_min, y_max = self._agent_bbox
            in_box = (
                (pts_2d[:, 0] >= x_min) & (pts_2d[:, 0] <= x_max) &
                (pts_2d[:, 1] >= y_min) & (pts_2d[:, 1] <= y_max)
            )
            agent_pts  = pts_2d[in_box]
            static_pts = pts_2d[~in_box]
        else:
            # No bbox configured — treat everything as static
            agent_pts  = np.empty((0, 2))
            static_pts = pts_2d

        # ── Agent: cluster points inside bbox → tracker ───────
        if len(agent_pts) > 0:
            agent_clusters = _cluster(_voxel_filter(agent_pts, res=0.3),
                                      radius=1.0, min_pts=2)
            self.tracker.update(agent_clusters, now)

        # ── Static obstacles: cluster everything outside bbox ─
        # min_pts=8 requires a substantial cluster — filters out single
        # beams, thin props, and other sparse false positives.
        static_clusters = _cluster(_voxel_filter(static_pts, res=0.5),
                                   radius=1.5, min_pts=8)
        static_obs = (np.array([c[0] for c in static_clusters])
                      if static_clusters else np.empty((0, 2)))

        with self._lock:
            self._static_obs = static_obs
            self._agent_pos  = self.tracker.position
            self._agent_vel  = self.tracker.velocity

    # ── Gazebo state ──────────────────────────────────────────────────────────

    def _get_gem_state(self):
        """
        Returns (np.array([x, y, yaw, v, omega]), raw_resp) for the DWA planner.
        Also caches (x, y, z, yaw) in self._vehicle_pose for the LiDAR callback.
        Returns (None, None) on failure.
        """
        try:
            svc  = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = svc(model_name='gem_e4')
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, f"[DWA] get_model_state: {e}")
            return None, None

        x   = resp.pose.position.x
        y   = resp.pose.position.y
        z   = resp.pose.position.z
        q   = resp.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        v     = math.hypot(resp.twist.linear.x, resp.twist.linear.y)
        omega = resp.twist.angular.z

        with self._lock:
            self._vehicle_pose = (x, y, z, yaw)

        return np.array([x, y, yaw, v, omega]), resp

    # ── Hard brake ────────────────────────────────────────────────────────────

    def _hard_brake(self, gazebo_resp):
        """
        Two-step hard stop:
        1. Publish speed=0 on /ackermann_cmd so the wheel velocity controllers
           hold zero (accel=0 → else branch in gem_control → direct assignment).
        2. Call /gazebo/set_model_state with Twist()=zero to immediately zero
           the body's momentum in the physics engine — this is what actually
           stops the vehicle from coasting past the goal.
        """
        # Step 1: zero wheel commands
        self.ackermann_msg.speed          = 0.0
        self.ackermann_msg.steering_angle = 0.0
        self.ackermann_msg.acceleration   = 0.0
        self.ackermann_pub.publish(self.ackermann_msg)

        # Step 2: zero physics velocity directly
        try:
            svc   = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state = ModelState()
            state.model_name      = 'gem_e4'
            state.pose            = gazebo_resp.pose   # keep current pose
            state.twist           = Twist()             # zero linear + angular
            state.reference_frame = 'world'
            svc(state)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, f"[DWA] set_model_state hard-brake failed: {e}")

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _w2px(self, wx: float, wy: float):
        """Convert Gazebo world (x=East, y=North) to image pixel (col, row)."""
        lat = self.lat_ref + wy / self.m_per_lat
        lon = self.lon_ref + wx / self.m_per_lon
        col = int(self.img_w * (lon - self.lon_start_l) / self.lon_scale)
        row = int(self.img_h - self.img_h * (lat - self.lat_start_bt) / self.lat_scale)
        return col, row

    def _in_img(self, col: int, row: int, margin: int = 5) -> bool:
        return (margin <= col < self.img_w - margin and
                margin <= row < self.img_h - margin)

    # ── Map-view publisher ────────────────────────────────────────────────────

    def _publish_map_view(self,
                          state:      np.ndarray,
                          traj:       np.ndarray,
                          static_obs: np.ndarray,
                          agent_pos,
                          agent_vel:  np.ndarray):
        """
        Draw everything on a copy of the GNSS map and publish as Image.

        Layers (back → front):
          1. GNSS map background
          2. DWA planned trajectory        — blue line
          3. Agent predicted trajectory    — yellow dashed line
          4. Static LiDAR obstacles        — red circles
          5. Tracked dynamic agent         — orange filled circle + velocity arrow
          6. Goal                          — green ring
          7. Vehicle                       — white filled circle + heading arrow
          8. Status text overlay
        """
        if self.map_image is not None:
            img = self.map_image.copy()
        else:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        r = int(max(3, self.px_per_m))  # 1 m in pixels

        # ── 2. DWA planned trajectory (blue) ─────────────────
        if traj is not None and len(traj) > 1:
            pts = []
            for pt in traj[::2]:
                c, rw = self._w2px(pt[0], pt[1])
                if self._in_img(c, rw):
                    pts.append((c, rw))
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (255, 120, 0), 2)

        # ── 3. Agent predicted path (yellow dashes) ──────────
        if agent_pos is not None:
            horizon_steps = int(self.cfg.predict_time / self.cfg.dt)
            prev_pt = None
            draw = True   # toggle for dashed effect
            for i in range(0, horizon_steps + 1, 2):
                t  = i * self.cfg.dt
                ax = agent_pos[0] + agent_vel[0] * t
                ay = agent_pos[1] + agent_vel[1] * t
                c, rw = self._w2px(ax, ay)
                if self._in_img(c, rw):
                    if prev_pt is not None and draw:
                        cv2.line(img, prev_pt, (c, rw), (0, 230, 255), 2)
                    prev_pt = (c, rw)
                draw = not draw

        # ── 4. Static LiDAR obstacles (red circles) ──────────
        for ox, oy in static_obs:
            c, rw = self._w2px(ox, oy)
            if self._in_img(c, rw):
                cv2.circle(img, (c, rw), max(4, r), (0, 0, 220), 2)

        # ── 5. Dynamic agent (orange filled + velocity arrow) ─
        if agent_pos is not None:
            ac, ar = self._w2px(agent_pos[0], agent_pos[1])
            if self._in_img(ac, ar):
                cv2.circle(img, (ac, ar), max(6, r + 2), (0, 140, 255), -1)
                cv2.circle(img, (ac, ar), max(6, r + 2), (0,  80, 200),  2)
                # Velocity arrow: show 2 s ahead
                vc, vr = self._w2px(agent_pos[0] + agent_vel[0] * 2.0,
                                    agent_pos[1] + agent_vel[1] * 2.0)
                if self._in_img(vc, vr):
                    cv2.arrowedLine(img, (ac, ar), (vc, vr), (0, 80, 200), 2,
                                    tipLength=0.3)
                cv2.putText(img, "agent", (ac + r + 3, ar - r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1,
                            cv2.LINE_AA)

        # ── 6. Goal (green ring) ──────────────────────────────
        gc, gr = self._w2px(self.goal[0], self.goal[1])
        if self._in_img(gc, gr, margin=-30):
            goal_r = max(8, int(self.cfg.goal_tolerance * self.px_per_m))
            cv2.circle(img, (gc, gr), goal_r, (0, 210, 0), 2)
            cv2.circle(img, (gc, gr), 4,      (0, 210, 0), -1)
            cv2.putText(img, "GOAL", (gc + goal_r + 3, gr),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 0), 2,
                        cv2.LINE_AA)

        # ── 7. Vehicle (white circle + heading arrow) ─────────
        vc, vr = self._w2px(state[0], state[1])
        if self._in_img(vc, vr, margin=-30):
            cv2.circle(img, (vc, vr), max(6, r + 3), (255, 255, 255), -1)
            cv2.circle(img, (vc, vr), max(6, r + 3), (160, 160, 160),  2)
            # Heading arrow: Gazebo yaw=0 → East (+col); yaw=π/2 → North (-row)
            alen = int(self.px_per_m * 2.5)
            hdx  =  int(alen * math.cos(state[2]))
            hdy  = -int(alen * math.sin(state[2]))
            cv2.arrowedLine(img, (vc, vr), (vc + hdx, vr + hdy),
                            (30, 30, 30), 2, tipLength=0.35)
            cv2.putText(img, "GEM", (vc + r + 4, vr - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # ── 8. Status text ────────────────────────────────────
        dist_g = math.hypot(self.goal[0] - state[0], self.goal[1] - state[1])
        lines = [
            f"v = {state[3]:.2f} m/s",
            f"dist to goal = {dist_g:.1f} m",
            f"static obs = {len(static_obs)}",
            f"agent tracked = {'yes' if agent_pos is not None else 'no'}",
        ]
        for i, line in enumerate(lines):
            y_txt = 28 + i * 22
            cv2.putText(img, line, (12, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),       2, cv2.LINE_AA)
            cv2.putText(img, line, (12, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Publish ───────────────────────────────────────────
        try:
            self.map_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr_throttle(5.0, f"[DWA] map publish error: {e}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        while not rospy.is_shutdown():

            state, gazebo_resp = self._get_gem_state()
            if state is None:
                self.rate.sleep()
                continue

            dist = math.hypot(self.goal[0] - state[0],
                              self.goal[1] - state[1])

            if dist < self.cfg.goal_tolerance:
                rospy.loginfo_throttle(2.0, "[DWA] Goal reached — hard brake.")
                self._hard_brake(gazebo_resp)
                with self._lock:
                    static_obs = self._static_obs.copy()
                    agent_pos  = self._agent_pos.copy() if self._agent_pos is not None else None
                    agent_vel  = self._agent_vel.copy()
                self._publish_map_view(state, None, static_obs, agent_pos, agent_vel)
                self.rate.sleep()
                continue

            with self._lock:
                static_obs = self._static_obs.copy()
                agent_pos  = self._agent_pos.copy() if self._agent_pos is not None else None
                agent_vel  = self._agent_vel.copy()

            # ── DWA ───────────────────────────────────────────
            u, traj = dwa_control(state, self.goal, static_obs,
                                  agent_pos, agent_vel, self.cfg)
            v, omega = float(u[0]), float(u[1])

            # Convert ω → steering angle (Ackermann relation)
            if abs(v) > 0.05:
                steer = math.atan2(omega * self.cfg.wheelbase, v)
            else:
                steer = 0.0
            steer = float(np.clip(steer, -self.cfg.max_steer, self.cfg.max_steer))

            self.ackermann_msg.speed          = v
            self.ackermann_msg.steering_angle = steer
            self.ackermann_msg.acceleration   = 0.0
            self.ackermann_pub.publish(self.ackermann_msg)

            speed_est = la.norm(agent_vel)
            rospy.loginfo_throttle(
                1.0,
                f"[DWA] v={v:.2f} m/s  δ={math.degrees(steer):+.1f}°  "
                f"dist={dist:.1f} m  obs={len(static_obs)}  "
                f"agent_vel=({agent_vel[0]:.2f},{agent_vel[1]:.2f}) |{speed_est:.2f}| m/s"
            )

            # ── Map-view visualization ────────────────────────
            self._publish_map_view(state, traj, static_obs, agent_pos, agent_vel)

            self.rate.sleep()


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def main():
    rospy.init_node('gem_dwa_sim', anonymous=True)
    node = GemDwaNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
