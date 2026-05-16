#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mock_perception_publisher.py
─────────────────────────────
Simulates YOLO object detection and VLM scene labeling for test scenarios.

Gazebo Classic cannot animate custom gestures, so this node injects fake
perceptions on a time-based schedule to stand in for the real YOLO + VLM
pipeline during simulator testing.

ROS Parameters
~~~~~~~~~~~~~~
  ~scenario_type  (str)   : stop_sign | pedestrian_crossing | worker_wave
                            | worker_stop | circular_loop
  ~trigger_delay  (float) : seconds before first publish  [default: 5.0]
  ~loop           (bool)  : loop the scenario indefinitely [default: true]
  ~publish_rate   (float) : Hz for detection/label pubs   [default: 2.0]

Topics Published
~~~~~~~~~~~~~~~~
  /yolo/detections      (std_msgs/String) – JSON list of detection dicts
  /vlm/scene_label      (std_msgs/String) – STOP | SLOW | YIELD | PROCEED
  /test/scenario_phase  (std_msgs/String) – human-readable phase label

JSON detection schema
~~~~~~~~~~~~~~~~~~~~~
  {
    "detections": [
      {
        "class":       "person" | "stop_sign" | ...,
        "confidence":  0.0-1.0,
        "distance_m":  float,
        "bbox":        [cx, cy, w, h]   (pixel coords, 640×480 frame assumed)
      }
    ],
    "scenario": "...",
    "phase":    "..."
  }
"""

import json
import rospy
from std_msgs.msg import String


# ── detection helper ──────────────────────────────────────────────────────────
def _det(cls: str, conf: float, dist: float, bbox: list) -> dict:
    return {"class": cls, "confidence": conf, "distance_m": dist, "bbox": bbox}


# ── scenario phase tables ─────────────────────────────────────────────────────
# Each entry: (phase_duration_s, phase_label, [detections], vlm_label)
SCENARIOS = {
    "stop_sign": [
        (4.0, "approaching",  [],
         "PROCEED"),
        (6.0, "sign_visible", [_det("stop_sign", 0.92, 20.0, [320, 240, 80, 80])],
         "STOP"),
        (8.0, "at_sign",      [_det("stop_sign", 0.97,  4.0, [320, 240, 180, 180])],
         "STOP"),
        (3.0, "resuming",     [],
         "PROCEED"),
    ],

    "pedestrian_crossing": [
        (3.0, "approaching",     [],
         "PROCEED"),
        (4.0, "ped_appearing",   [_det("person", 0.88, 18.0, [520, 240, 40, 90])],
         "YIELD"),
        (5.0, "ped_in_path",     [_det("person", 0.95,  7.0, [320, 240, 80, 160])],
         "STOP"),
        (6.0, "ped_still_close", [_det("person", 0.93,  5.0, [320, 240, 90, 180])],
         "STOP"),
        (4.0, "ped_clearing",    [_det("person", 0.82, 14.0, [120, 240, 50, 110])],
         "YIELD"),
        (3.0, "path_clear",      [],
         "PROCEED"),
    ],

    "worker_wave": [
        (3.0, "approaching",    [],
         "PROCEED"),
        (4.0, "worker_far",     [_det("person", 0.82, 20.0, [340, 200, 35, 90])],
         "PROCEED"),
        (7.0, "wave_detected",  [_det("person", 0.91, 12.0, [320, 200, 65, 140])],
         "SLOW"),
        (6.0, "still_waving",   [_det("person", 0.93,  8.0, [320, 200, 80, 160])],
         "SLOW"),
        (3.0, "worker_cleared", [],
         "PROCEED"),
    ],

    "worker_stop": [
        (3.0, "approaching",      [],
         "PROCEED"),
        (4.0, "worker_roadside",  [_det("person", 0.85, 20.0, [360, 200, 35, 90])],
         "PROCEED"),
        (7.0, "stop_gesture",     [_det("person", 0.94, 10.0, [320, 200, 80, 160])],
         "STOP"),
        (8.0, "gesture_held",     [_det("person", 0.96,  6.0, [320, 200, 95, 190])],
         "STOP"),
        (4.0, "gesture_released", [],
         "PROCEED"),
    ],

    "circular_loop": [
        (3.0, "straight_clear",    [],
         "PROCEED"),
        (4.0, "entering_workzone", [_det("person", 0.80, 18.0, [340, 200, 40, 100])],
         "SLOW"),
        (5.0, "worker_wave_zone",  [_det("person", 0.89, 12.0, [320, 200, 65, 140])],
         "SLOW"),
        (5.0, "stop_sign_zone",    [_det("stop_sign", 0.93, 20.0, [320, 240, 80, 80])],
         "STOP"),
        (3.0, "stopped_at_sign",   [_det("stop_sign", 0.97,  4.0, [320, 240, 180, 180])],
         "STOP"),
        (5.0, "pedestrian_zone",   [_det("person", 0.92,  8.0, [320, 240, 80, 160])],
         "STOP"),
        (4.0, "ped_clearing",      [_det("person", 0.78, 15.0, [160, 240, 55, 120])],
         "YIELD"),
        (4.0, "clear_section",     [],
         "PROCEED"),
        (5.0, "ambiguous_zone",    [_det("person", 0.62, 22.0, [300, 200, 35, 90])],
         "SLOW"),
        (3.0, "back_to_start",     [],
         "PROCEED"),
    ],
}


def main():
    rospy.init_node("mock_perception_publisher", anonymous=False)

    scenario_type = rospy.get_param("~scenario_type", "stop_sign")
    trigger_delay = float(rospy.get_param("~trigger_delay", 5.0))
    loop_scenario = rospy.get_param("~loop", True)
    pub_rate_hz   = float(rospy.get_param("~publish_rate", 2.0))

    if scenario_type not in SCENARIOS:
        rospy.logfatal(
            "[mock_perception] Unknown scenario_type '%s'. Valid: %s",
            scenario_type, list(SCENARIOS.keys())
        )
        return

    pub_yolo  = rospy.Publisher("/yolo/detections",     String, queue_size=10)
    pub_vlm   = rospy.Publisher("/vlm/scene_label",     String, queue_size=10)
    pub_phase = rospy.Publisher("/test/scenario_phase", String, queue_size=10)

    rospy.loginfo("[mock_perception] Waiting %.1f s before scenario start …", trigger_delay)
    rospy.sleep(trigger_delay)
    rospy.loginfo("[mock_perception] Starting scenario: %s", scenario_type)

    phases = SCENARIOS[scenario_type]
    rate   = rospy.Rate(pub_rate_hz)

    while not rospy.is_shutdown():
        for duration, phase_label, detections, vlm_label in phases:
            if rospy.is_shutdown():
                break
            rospy.loginfo(
                "[mock_perception] phase=%-22s  vlm=%-8s  dets=%d",
                phase_label, vlm_label, len(detections)
            )
            end_time = rospy.Time.now() + rospy.Duration(duration)
            while rospy.Time.now() < end_time and not rospy.is_shutdown():
                payload = json.dumps({
                    "detections": detections,
                    "scenario":   scenario_type,
                    "phase":      phase_label,
                })
                pub_yolo.publish(payload)
                pub_vlm.publish(vlm_label)
                pub_phase.publish(phase_label)
                rate.sleep()

        if not loop_scenario:
            rospy.loginfo("[mock_perception] Scenario complete.")
            break

    rospy.loginfo("[mock_perception] Shutting down.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
