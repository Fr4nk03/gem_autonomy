#!/usr/bin/env python3

#================================================================
# File name: yolo_detector.py
# Description: Fast-System YOLO node for the Fast-Slow perception
#              pipeline. Runs YOLOv8 on the front camera, estimates
#              distance from bbox height, and flags hazards
#              (person/cyclist) inside a configurable Danger Zone.
# Author: Henry Che
# Usage: rosrun gem_gazebo yolo_detector.py
#        roslaunch gem_gazebo yolo_detector.launch
# Python version: 3.8
#================================================================

import json
import math

import rospy
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg   import Bool, String, ColorRGBA
from cv_bridge      import CvBridge, CvBridgeError

try:
    from jsk_rviz_plugins.msg import OverlayText
    JSK_AVAILABLE = True
except ImportError:
    JSK_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from vision_msgs.msg import (
        Detection2DArray, Detection2D, ObjectHypothesisWithPose,
    )
    VISION_MSGS_AVAILABLE = True
except ImportError:
    VISION_MSGS_AVAILABLE = False


# Real-world heights (m) used for monocular distance estimation.
# distance = real_h * focal_px / bbox_h_px
_REAL_HEIGHTS = {
    0:  1.70,   # person
    1:  1.10,   # bicycle
    2:  1.50,   # car
    3:  1.20,   # motorcycle
    11: 0.75,   # stop sign
}


class YoloDetectorNode:

    def __init__(self):
        if not YOLO_AVAILABLE:
            rospy.logfatal("ultralytics is not installed. Run: pip install ultralytics")
            raise SystemExit(1)

        # ── model / inference params ──────────────────────────────────────
        self.model_name  = rospy.get_param('~model',       'yolov8n.pt')
        self.conf_thresh = rospy.get_param('~confidence',  0.4)
        self.img_size    = rospy.get_param('~img_size',    320)
        self.device      = rospy.get_param('~device',      'cpu')
        self.image_topic = rospy.get_param('~image_topic', '/oak/rgb/image_raw')

        # ── danger zone params ─────────────────────────────────────────────
        # hazard classes: 0=person, 1=bicycle (COCO)
        self.hazard_classes  = set(rospy.get_param('~hazard_classes', [0, 1]))
        self.danger_dist_m   = float(rospy.get_param('~danger_distance_m', 10.0))
        # camera intrinsics for monocular distance estimation
        self.image_w_full    = int(rospy.get_param('~image_w',  1280))
        self.image_h_full    = int(rospy.get_param('~image_h',  720))
        self.hfov_rad        = float(rospy.get_param('~hfov_rad', math.radians(72.0)))
        self.focal_px        = (self.image_w_full / 2.0) / math.tan(self.hfov_rad / 2.0)
        # in-lane strike zone in pixels (bbox center must lie inside)
        self.strike_left     = float(rospy.get_param('~strike_zone_left',  300.0))
        self.strike_right    = float(rospy.get_param('~strike_zone_right', 980.0))

        rospy.loginfo(
            "[yolo] model=%s img_size=%d device=%s focal_px=%.1f danger<%.1fm strike=[%.0f,%.0f]",
            self.model_name, self.img_size, self.device,
            self.focal_px, self.danger_dist_m, self.strike_left, self.strike_right,
        )

        self.model = YOLO(self.model_name)
        # warm-up
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.img_size, device=self.device, verbose=False)
        rospy.loginfo("[yolo] model ready")

        self.bridge = CvBridge()

        # ── publishers ─────────────────────────────────────────────────────
        self.pub_image   = rospy.Publisher('/yolo/image_annotated', Image,  queue_size=1)
        self.pub_hazard  = rospy.Publisher('/yolo/hazard',          Bool,   queue_size=1)
        self.pub_info    = rospy.Publisher('/yolo/hazard_info',     String, queue_size=1)
        self.pub_overlay = (
            rospy.Publisher('/yolo/detection_info', OverlayText, queue_size=1)
            if JSK_AVAILABLE else None
        )
        self.pub_detections = (
            rospy.Publisher('/yolo/detections', Detection2DArray, queue_size=1)
            if VISION_MSGS_AVAILABLE else None
        )

        self.sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback,
            queue_size=1, buff_size=2 ** 24,
        )
        rospy.loginfo("[yolo] subscribed to %s", self.image_topic)

    # ------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("[yolo] cv_bridge error: %s", e)
            return

        results = self.model(
            cv_image,
            imgsz=self.img_size, conf=self.conf_thresh,
            device=self.device, verbose=False,
        )
        boxes = results[0].boxes
        names = results[0].names

        hazards   = []      # detections inside the danger zone
        all_dets  = []      # everything (for debug)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1

                dist = self._estimate_distance(cls_id, bh)
                in_lane = (self.strike_left <= cx <= self.strike_right)
                is_hazard_cls = cls_id in self.hazard_classes
                in_zone = is_hazard_cls and in_lane and (dist <= self.danger_dist_m)

                entry = {
                    "class_id":  cls_id,
                    "class":     names.get(cls_id, str(cls_id)),
                    "confidence": round(conf, 3),
                    "distance_m": round(dist, 2),
                    "bbox":       [round(cx, 1), round(cy, 1), round(bw, 1), round(bh, 1)],
                    "in_lane":    bool(in_lane),
                    "in_danger":  bool(in_zone),
                }
                all_dets.append(entry)
                if in_zone:
                    hazards.append(entry)

        # ── hazard topics (Fast System primary output) ─────────────────────
        is_hazard = len(hazards) > 0
        self.pub_hazard.publish(Bool(data=is_hazard))
        info_payload = {
            "stamp":      msg.header.stamp.to_sec(),
            "hazard":     is_hazard,
            "hazards":    hazards,
            "detections": all_dets,
        }
        self.pub_info.publish(String(data=json.dumps(info_payload)))

        # ── annotated image ────────────────────────────────────────────────
        try:
            annotated = results[0].plot()
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header = msg.header
            self.pub_image.publish(ann_msg)
        except CvBridgeError as e:
            rospy.logerr("[yolo] failed to publish annotated image: %s", e)

        # ── overlay text (optional jsk_rviz) ───────────────────────────────
        if self.pub_overlay is not None:
            self.pub_overlay.publish(self._build_overlay(all_dets, hazards))

        # ── structured Detection2DArray (optional vision_msgs) ─────────────
        if self.pub_detections is not None and boxes is not None:
            det_array = Detection2DArray()
            det_array.header = msg.header
            for box in boxes:
                det = Detection2D()
                det.header = msg.header
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det.bbox.center.x = (x1 + x2) / 2.0
                det.bbox.center.y = (y1 + y2) / 2.0
                det.bbox.size_x   = x2 - x1
                det.bbox.size_y   = y2 - y1
                hyp = ObjectHypothesisWithPose()
                hyp.id    = int(box.cls[0].item())
                hyp.score = float(box.conf[0].item())
                det.results.append(hyp)
                det_array.detections.append(det)
            self.pub_detections.publish(det_array)

        if is_hazard:
            rospy.loginfo_throttle(
                1.0, "[yolo] HAZARD: %s",
                ", ".join(f"{h['class']}@{h['distance_m']}m" for h in hazards),
            )

    # ------------------------------------------------------------------
    def _estimate_distance(self, cls_id, bbox_h_px):
        real_h = _REAL_HEIGHTS.get(cls_id, 1.5)
        return (real_h * self.focal_px) / max(1.0, bbox_h_px)

    # ------------------------------------------------------------------
    def _build_overlay(self, all_dets, hazards):
        text = OverlayText()
        text.width, text.height = 320, 420
        text.left, text.top     = 10, 10
        text.text_size          = 11
        text.line_width         = 2
        text.font               = "DejaVu Sans Mono"
        text.fg_color = ColorRGBA(0.1, 1.0, 0.95, 1.0)
        text.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.35)

        lines = [
            "---- YOLO (Fast System) ----",
            f"Model: {self.model_name}",
            f"Danger zone: <{self.danger_dist_m:.1f} m",
            f"Hazards in zone: {len(hazards)}",
            "----------------------------",
        ]
        if hazards:
            lines.append("HAZARDS:")
            for h in hazards:
                lines.append(f"  {h['class']:<10} {h['distance_m']:>5.2f}m  conf={h['confidence']:.2f}")
        if all_dets:
            lines.append("ALL:")
            for d in all_dets[:8]:
                tag = "!" if d["in_danger"] else " "
                lines.append(f" {tag}{d['class']:<10} {d['distance_m']:>5.2f}m")
        text.text = "\n".join(lines)
        return text


# ----------------------------------------------------------------------
def main():
    rospy.init_node('yolo_detector', anonymous=True)
    try:
        YoloDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
