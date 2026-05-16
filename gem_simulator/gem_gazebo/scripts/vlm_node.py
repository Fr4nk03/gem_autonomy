#!/usr/bin/env python3

#================================================================
# File name: vlm_node.py
# Description: Slow-System VLM node. Holds a rolling window of the
#              last N camera frames. On /vlm/trigger, asks
#              Qwen2.5-VL-3B-Instruct to reason about the scene
#              with Chain-of-Thought and produce a JSON action
#              decision (STOP / SLOW / PROCEED). Publishes the
#              parsed result on /vlm/decision.
#
# Python version: 3.8
# GPU is strongly recommended (3B params, fp16/bf16).
#================================================================

import json
import re
import threading
from collections import deque

import rospy
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg    import Empty, String
from cv_bridge       import CvBridge, CvBridgeError


# ── Chain-of-Thought prompt (single unified prompt, JSON output) ──────────
_PROMPT_COT = (
    "Act as an autonomous driving safety system. [json caption en] "
    "You are observing a short sequence of frames from the vehicle's "
    "front-facing camera. The vehicle is currently stopped. "
    "Reason step by step before answering:\n"
    "1. Identify every person, cyclist or hazard in the scene.\n"
    "2. Describe each person's pose, gesture, or motion across the frames "
    "(e.g., raising a STOP sign, waving the vehicle forward, walking across, "
    "standing still on the shoulder).\n"
    "3. Infer their intent toward the vehicle (block, wave-through, "
    "indifferent, transient crossing).\n"
    "4. Decide the safest action.\n"
    "Then output a single JSON object with exactly these keys:\n"
    "  'hazards_detected' (list of short strings),\n"
    "  'reasoning' (string capturing steps 1-4 succinctly),\n"
    "  'action' (one of 'STOP', 'SLOW', 'PROCEED').\n"
    "Output only the JSON object, no extra prose."
)

_VALID_ACTIONS = {"STOP", "SLOW", "PROCEED"}
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_decision(raw):
    """Return (action, parsed_dict_or_None). Defaults to STOP on failure
    (the *safe* fallback in the Slow path is to keep the vehicle stationary)."""
    if not raw:
        return "STOP", None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        act = str(obj.get("action", "")).strip().upper()
        if act in _VALID_ACTIONS:
            return act, obj
    for kw in ("STOP", "SLOW", "PROCEED"):
        if re.search(rf"\b{kw}\b", text, re.IGNORECASE):
            return kw, obj
    return "STOP", obj


# ── Backends ──────────────────────────────────────────────────────────────

class _QwenVLBackend:
    """Qwen2.5-VL-3B-Instruct backend with multi-image input."""

    def __init__(self, model_id, max_new_tokens):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        self.torch  = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo("[vlm] loading %s on %s ...", model_id, self.device)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = int(max_new_tokens)
        rospy.loginfo("[vlm] qwen ready (max_new_tokens=%d)", self.max_new_tokens)

    def query(self, pil_images, prompt):
        from qwen_vl_utils import process_vision_info
        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()


class _MockBackend:
    """No-GPU fallback. Always says STOP — the safe default when
    the real VLM is unavailable."""
    def query(self, pil_images, prompt):
        return ('{"hazards_detected": ["unknown"], '
                '"reasoning": "Mock VLM (no model loaded). Defaulting to STOP for safety.", '
                '"action": "STOP"}')


# ── Node ──────────────────────────────────────────────────────────────────

class VlmNode:

    def __init__(self):
        rospy.init_node('vlm_node')

        # ── params ────────────────────────────────────────────────────
        self.image_topic   = rospy.get_param('~image_topic',   '/oak/rgb/image_raw')
        self.backend_name  = rospy.get_param('~backend',       'qwen')   # 'qwen' | 'mock'
        self.model_id      = rospy.get_param('~model_id',      'Qwen/Qwen2.5-VL-3B-Instruct')
        self.max_new_tokens = int(rospy.get_param('~max_new_tokens', 256))
        self.window_size   = int(rospy.get_param('~window_size',     3))
        self.window_dt_s   = float(rospy.get_param('~window_dt_s',  0.7))
        self.resize_short  = int(rospy.get_param('~resize_short',  448))   # short side px

        self.bridge = CvBridge()
        self._buffer_lock = threading.Lock()
        # bounded deque holding (stamp_sec, np.ndarray RGB)
        self.frames = deque(maxlen=max(8, self.window_size * 4))

        # ── load backend ──────────────────────────────────────────────
        self.backend = self._load_backend()

        # one-at-a-time inference guard (the model call is heavy)
        self._busy = threading.Lock()

        # ── ROS I/O ───────────────────────────────────────────────────
        self.pub_decision = rospy.Publisher('/vlm/decision', String, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self._image_cb,
                         queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber('/vlm/trigger', Empty, self._trigger_cb, queue_size=2)

        rospy.loginfo("[vlm] ready (backend=%s, window=%d frames, dt=%.2fs)",
                      self.backend_name, self.window_size, self.window_dt_s)

    # ── backend factory ───────────────────────────────────────────────
    def _load_backend(self):
        if self.backend_name == "mock":
            rospy.loginfo("[vlm] using mock backend")
            return _MockBackend()
        try:
            return _QwenVLBackend(self.model_id, self.max_new_tokens)
        except Exception as e:
            rospy.logwarn("[vlm] failed to load qwen (%s) — falling back to mock", e)
            self.backend_name = "mock"
            return _MockBackend()

    # ── camera callback: rolling window ───────────────────────────────
    def _image_cb(self, msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            rospy.logwarn_throttle(5, "[vlm] cv_bridge: %s", e)
            return
        with self._buffer_lock:
            self.frames.append((msg.header.stamp.to_sec(), rgb))

    # ── trigger callback: run inference in a worker thread ────────────
    def _trigger_cb(self, _msg):
        if not self._busy.acquire(blocking=False):
            rospy.loginfo_throttle(2, "[vlm] busy, dropping trigger")
            return
        t = threading.Thread(target=self._run_inference, daemon=True)
        t.start()

    # ── pick N frames at roughly window_dt_s spacing ──────────────────
    def _select_window(self):
        from PIL import Image as PILImage
        with self._buffer_lock:
            snap = list(self.frames)
        if not snap:
            return []
        latest_t = snap[-1][0]
        chosen = []
        # walk the deque from newest to oldest; greedily keep frames
        # at >= window_dt_s spacing
        last_kept_t = None
        for t, frame in reversed(snap):
            if last_kept_t is None or (last_kept_t - t) >= self.window_dt_s:
                chosen.append((t, frame))
                last_kept_t = t
                if len(chosen) >= self.window_size:
                    break
        chosen.reverse()
        pil_list = []
        for _, frame in chosen:
            pil = PILImage.fromarray(frame)
            # resize keeping aspect ratio to limit VRAM
            w, h = pil.size
            short = min(w, h)
            if short > self.resize_short:
                scale = self.resize_short / float(short)
                pil = pil.resize((int(w * scale), int(h * scale)))
            pil_list.append(pil)
        return pil_list

    # ── core inference (runs off the ROS callback thread) ─────────────
    def _run_inference(self):
        try:
            pil_images = self._select_window()
            if not pil_images:
                rospy.logwarn("[vlm] trigger received but frame buffer empty")
                self._publish_safe_fallback("no frames available")
                return
            rospy.loginfo("[vlm] querying with %d frame(s)", len(pil_images))
            t0 = rospy.get_time()
            raw = self.backend.query(pil_images, _PROMPT_COT)
            dt = rospy.get_time() - t0
            action, obj = _extract_decision(raw)
            if obj is None:
                obj = {"hazards_detected": [], "reasoning": raw[:300], "action": action}
            obj["action"]  = action
            obj["latency_s"] = round(dt, 3)
            obj["n_frames"]  = len(pil_images)
            payload = json.dumps(obj)
            self.pub_decision.publish(String(data=payload))
            rospy.loginfo("[vlm] → %s  (%.2fs, %d frames)", action, dt, len(pil_images))
        except Exception as e:
            rospy.logerr("[vlm] inference error: %s", e)
            self._publish_safe_fallback(str(e)[:200])
        finally:
            self._busy.release()

    def _publish_safe_fallback(self, reason):
        payload = json.dumps({
            "hazards_detected": ["vlm_error"],
            "reasoning": "VLM error or empty buffer; defaulting to STOP. " + reason,
            "action": "STOP",
        })
        self.pub_decision.publish(String(data=payload))


def main():
    try:
        VlmNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
