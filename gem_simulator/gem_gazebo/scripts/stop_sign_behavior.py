#!/usr/bin/env python3

#================================================================
# File name: stop_sign_behavior.py
# Description: Fast-Slow Brain. State machine that fuses the Fast
#              System (YOLO hazard flag) with the Slow System (VLM
#              decision) and commands the GEM vehicle to STOP /
#              SLOW / PROCEED in a straight line.
#
# Fast path:  /yolo/hazard  ──► BRAKING ──► STOPPED_WAIT
# Slow path:  STOPPED_WAIT + persist ─► /vlm/trigger
#             /vlm/decision { action: STOP|SLOW|PROCEED }
#
# Outputs:
#   /ackermann_cmd      (ackermann_msgs/AckermannDrive)
#   /vlm/trigger        (std_msgs/Empty)
#   /fast_slow/state    (std_msgs/String)
#================================================================

import json
import math

import rospy
from std_msgs.msg          import Bool, Empty, String
from ackermann_msgs.msg    import AckermannDrive
from gazebo_msgs.msg       import ModelStates


# ── speed setpoints (m/s) — straight-line only ──────────────────────────
SPEED_PROCEED = 0.5
SPEED_SLOW    = 0.2
SPEED_STOP    = 0.0

# ── timing constants (s) ────────────────────────────────────────────────
STOP_VEL_EPS         = 0.10   # |v| below this counts as stopped
PERSIST_TO_VLM_S     = 1.0    # hazard must persist this long after stop to trigger VLM
VLM_REQUERY_S        = 5.0    # while held by VLM STOP/SLOW, re-query at this interval
VLM_TIMEOUT_S        = 8.0    # if no /vlm/decision arrives, retry trigger
HAZARD_CLEAR_HOLD_S  = 0.5    # SLOW → DRIVING requires hazard clear for this long
COMMAND_RATE_HZ      = 20.0


class FastSlowBrain:

    def __init__(self):
        rospy.init_node('stop_sign_behavior')

        rospy.loginfo("========================================")
        rospy.loginfo("FAST-SLOW BRAIN (YOLO + Qwen2.5-VL)")
        rospy.loginfo("========================================")

        # ── inputs ────────────────────────────────────────────────────
        self.hazard          = False
        self.hazard_info_raw = "{}"
        self.current_speed   = 0.0
        self.vehicle_name    = rospy.get_param('~vehicle_substring', 'gem')

        # ── state ─────────────────────────────────────────────────────
        self.state             = "DRIVING"   # DRIVING|BRAKING|STOPPED_WAIT|VLM_QUERY|SLOW|RESUMING
        self.state_start_time  = rospy.get_time()
        self.last_hazard_true  = 0.0         # last time hazard was True
        self.last_hazard_false = rospy.get_time()
        self.vlm_pending       = False
        self.vlm_last_sent_t   = 0.0
        self.vlm_last_action   = None        # 'STOP' | 'SLOW' | 'PROCEED'
        self.vlm_last_reason   = ""

        # ── ROS I/O ───────────────────────────────────────────────────
        rospy.Subscriber('/yolo/hazard',         Bool,        self.hazard_cb,    queue_size=1)
        rospy.Subscriber('/yolo/hazard_info',    String,      self.info_cb,      queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gz_cb,        queue_size=1)
        rospy.Subscriber('/vlm/decision',        String,      self.vlm_cb,      queue_size=1)

        self.pub_cmd     = rospy.Publisher('/ackermann_cmd',   AckermannDrive, queue_size=10)
        self.pub_trigger = rospy.Publisher('/vlm/trigger',     Empty,          queue_size=1)
        self.pub_state   = rospy.Publisher('/fast_slow/state', String,         queue_size=1)

        self.rate = rospy.Rate(COMMAND_RATE_HZ)

    # ── callbacks ────────────────────────────────────────────────────
    def hazard_cb(self, msg):
        now = rospy.get_time()
        self.hazard = bool(msg.data)
        if self.hazard:
            self.last_hazard_true = now
        else:
            self.last_hazard_false = now

    def info_cb(self, msg):
        self.hazard_info_raw = msg.data

    def gz_cb(self, msg):
        for i, name in enumerate(msg.name):
            if self.vehicle_name in name:
                vx = msg.twist[i].linear.x
                vy = msg.twist[i].linear.y
                self.current_speed = math.hypot(vx, vy)
                return

    def vlm_cb(self, msg):
        try:
            obj = json.loads(msg.data)
        except Exception:
            rospy.logwarn_throttle(5, "[brain] cannot parse /vlm/decision: %s", msg.data[:120])
            return
        action = str(obj.get("action", "")).strip().upper()
        if action not in {"STOP", "SLOW", "PROCEED"}:
            rospy.logwarn_throttle(5, "[brain] VLM action invalid: %s", action)
            return
        self.vlm_last_action = action
        self.vlm_last_reason = str(obj.get("reasoning", ""))[:200]
        self.vlm_pending     = False
        rospy.loginfo("[brain] VLM → %s  (%s)", action, self.vlm_last_reason)

    # ── helpers ──────────────────────────────────────────────────────
    def _enter(self, new_state):
        if new_state != self.state:
            rospy.loginfo("[brain] %s → %s", self.state, new_state)
            self.state = new_state
            self.state_start_time = rospy.get_time()

    def _publish(self, speed):
        cmd = AckermannDrive()
        cmd.speed          = float(speed)
        cmd.steering_angle = 0.0     # straight-line only
        self.pub_cmd.publish(cmd)

    def _send_vlm_trigger(self):
        self.pub_trigger.publish(Empty())
        self.vlm_pending     = True
        self.vlm_last_action = None
        self.vlm_last_sent_t = rospy.get_time()
        rospy.loginfo("[brain] → VLM trigger sent")

    def _is_stopped(self):
        return self.current_speed < STOP_VEL_EPS

    # ── main loop ────────────────────────────────────────────────────
    def run(self):
        while not rospy.is_shutdown():
            self._step()
            self.pub_state.publish(String(data=self.state))
            self.rate.sleep()

    def _step(self):
        now      = rospy.get_time()
        in_state = now - self.state_start_time

        if self.state == "DRIVING":
            self._publish(SPEED_PROCEED)
            if self.hazard:
                self._enter("BRAKING")

        elif self.state == "BRAKING":
            self._publish(SPEED_STOP)
            if self._is_stopped():
                self._enter("STOPPED_WAIT")

        elif self.state == "STOPPED_WAIT":
            self._publish(SPEED_STOP)
            if not self.hazard:
                # detection cleared while waiting → resume directly
                if (now - self.last_hazard_true) > HAZARD_CLEAR_HOLD_S:
                    self._enter("DRIVING")
                    return
            elif in_state >= PERSIST_TO_VLM_S:
                self._send_vlm_trigger()
                self._enter("VLM_QUERY")

        elif self.state == "VLM_QUERY":
            self._publish(SPEED_STOP)
            # if hazard cleared while we're waiting, just go
            if not self.hazard and (now - self.last_hazard_true) > HAZARD_CLEAR_HOLD_S:
                self._enter("DRIVING")
                return
            if self.vlm_last_action is not None:
                act = self.vlm_last_action
                self.vlm_last_action = None
                if act == "PROCEED":
                    self._enter("RESUMING")
                elif act == "SLOW":
                    self._enter("SLOW")
                else:  # STOP
                    # stay stopped; will re-query
                    self._enter("STOPPED_WAIT")
                return
            # no decision yet — retry trigger if timed out
            if self.vlm_pending and (now - self.vlm_last_sent_t) > VLM_TIMEOUT_S:
                rospy.logwarn("[brain] VLM timeout, re-triggering")
                self._send_vlm_trigger()

        elif self.state == "SLOW":
            self._publish(SPEED_SLOW)
            # fast override: hazard appears again → BRAKE
            if self.hazard:
                self._enter("BRAKING")
                return
            # hazard clear for long enough → resume full speed
            if (now - self.last_hazard_true) > HAZARD_CLEAR_HOLD_S * 4:
                self._enter("DRIVING")
                return
            # periodically re-query VLM in case its verdict changes
            if (now - self.vlm_last_sent_t) > VLM_REQUERY_S and not self.vlm_pending:
                self._send_vlm_trigger()

        elif self.state == "RESUMING":
            # short cooldown to give the actor time to fully leave the lane
            self._publish(SPEED_SLOW)
            if in_state >= 1.5:
                self._enter("DRIVING")
            if self.hazard:
                self._enter("BRAKING")

        else:
            rospy.logerr_throttle(5, "[brain] unknown state %s, forcing STOP", self.state)
            self._publish(SPEED_STOP)


if __name__ == '__main__':
    try:
        FastSlowBrain().run()
    except rospy.ROSInterruptException:
        pass
