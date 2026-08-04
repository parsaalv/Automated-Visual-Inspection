"""Vision inference pipeline: mock classifier and the real YOLO + PatchCore pipeline."""

import os
import random
from enum import Enum

import cv2
import numpy as np

from logging_config import logger


class VisionClass(Enum):
    """Quality classification outcome for an inspected part."""
    GOOD   = 0
    REPAIR = 1
    SCRAP  = 2


class MockVisionModule:
    """Stand-in vision module used before the real model pipeline is available.

    Supports three modes: a fixed per-object ``scenario`` mapping, or a
    ``random`` classification drawn from a seeded RNG (results are cached
    per object so repeated inference calls are stable).
    """

    def __init__(self, mode="random", scenario=None, seed=None):
        self.mode = mode
        self.scenario = scenario or {}
        self._rng = random.Random(seed)
        self._cache = {}

    def set_mode(self, mode):
        """Switch between "random" and "scenario" classification modes."""
        self.mode = mode
        print(f"[MockVision] Mode set to: {self.mode}")

    def set_scenario(self, scenario):
        """Replace the fixed object-label -> class_id scenario mapping."""
        self.scenario = scenario

    def _resolve_class(self, obj_id, obj_label):
        """Resolve (and cache) the classification result for a given object."""
        cache_key = obj_label if obj_label is not None else obj_id
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.mode == "scenario":
            class_id = self.scenario.get(obj_label, self.scenario.get(obj_id, VisionClass.GOOD.value))
        else:
            class_id = self._rng.choice([VisionClass.GOOD.value, VisionClass.REPAIR.value, VisionClass.SCRAP.value])
        self._cache[cache_key] = class_id
        return class_id

    def infer(self, obj_id, obj_label, frame=None):
        """Return a mock inference result dict for the given object."""
        class_id = self._resolve_class(obj_id, obj_label)
        return {
            "object_id": obj_id,
            "class_id": class_id,
            "confidence": 1.0
        }

    def reset(self):
        """Clear the cached per-object classification results."""
        self._cache = {}


class RealVisionModule:
    """Real two-stage vision pipeline: YOLOv8 part detection followed by PatchCore anomaly detection.

    For each captured frame, YOLO first locates and crops the highest-
    confidence part; the crop is then passed to a per-class PatchCore
    anomaly model to obtain a defect score and (if applicable) defect
    contours, which together determine the final GOOD/REPAIR/SCRAP
    classification.
    """

    def __init__(self, config):
        self.config = config
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"
        self._yolo_model = None
        self._anomaly_inferencers = {}
        self._cache = {}
        print(f"[RealVision] Initialized. Device: {self.device}")

    def preload_all_models(self):
        """Load the YOLO model and every per-class anomaly model up front, before the simulation starts."""
        print("[RealVision] Preloading models... (one-time only, may take a few seconds)")
        self._load_yolo()
        for class_name in self.config.SELECTED_CLASSES:
            self._load_anomaly_model(class_name)
        print("[RealVision] All models loaded. The system is ready for real-time inference.")

    def _load_yolo(self):
        """Lazily load (and cache) the YOLO part-detection model."""
        if self._yolo_model is None:
            from ultralytics import YOLO
            print(f"[RealVision] Loading YOLO weights from: {self.config.YOLO_WEIGHTS_PATH}")
            if not os.path.exists(self.config.YOLO_WEIGHTS_PATH):
                raise FileNotFoundError(
                    f"[RealVision] YOLO weights file not found: {self.config.YOLO_WEIGHTS_PATH}\n"
                    f"The path is resolved relative to the current working directory ({os.getcwd()}); "
                    f"update YOLO_WEIGHTS_PATH in Config to the correct absolute path."
                )
            self._yolo_model = YOLO(self.config.YOLO_WEIGHTS_PATH)
        return self._yolo_model

    def _load_anomaly_model(self, class_name):
        """Lazily load (and cache) the anomaly-detection model for a given part class."""
        if class_name not in self._anomaly_inferencers:
            import glob
            from anomalib.deploy import TorchInferencer
            pattern = os.path.join(self.config.ANOMALY_BASE_DIR, class_name, '**', '*.pt')
            pt_paths = glob.glob(pattern, recursive=True)
            if not pt_paths:
                print(f"[RealVision] Warning: no anomaly model found for class '{class_name}' (pattern: {pattern})")
                self._anomaly_inferencers[class_name] = None
            else:
                best_pt = max(pt_paths, key=os.path.getmtime)
                print(f"[RealVision] Loading anomaly model for '{class_name}' from: {best_pt}")
                self._anomaly_inferencers[class_name] = TorchInferencer(path=best_pt, device=self.device)
        return self._anomaly_inferencers[class_name]

    def _detect_and_crop(self, frame_bgr):
        """Run YOLO on a frame and crop out the highest-confidence detected part.

        Returns
        -------
        tuple
            ``(class_name, cropped_rgb, crop_box, yolo_bbox, confidence)``,
            or ``(None, None, None, None, None)`` if nothing was detected.
        """
        model = self._load_yolo()
        results = model.predict(source=frame_bgr, conf=self.config.YOLO_CONF_THRESHOLD, verbose=False)
        res = results[0]
        if len(res.boxes) == 0:
            return None, None, None, None, None
        confs = res.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        box = res.boxes.xyxy[best_idx].cpu().numpy()
        cls_idx = int(res.boxes.cls[best_idx].cpu().numpy())
        class_name = model.names[cls_idx]
        h, w = frame_bgr.shape[:2]
        pad = self.config.CROP_PADDING
        x1, y1, x2, y2 = box
        x1_pad, y1_pad = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2_pad, y2_pad = min(w, int(x2) + pad), min(h, int(y2) + pad)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cropped_rgb = frame_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
        crop_box = (x1_pad, y1_pad, x2_pad, y2_pad)
        yolo_bbox = (int(x1), int(y1), int(x2), int(y2))
        return class_name, cropped_rgb, crop_box, yolo_bbox, float(confs[best_idx])

    def _extract_defect_contours(self, pred_mask, crop_w, crop_h):
        """Convert a raw anomaly prediction mask into contours and a defect-area percentage.

        Returns
        -------
        tuple
            ``(contours, defect_percentage)``. ``contours`` is ``None`` if
            no mask was provided or it was empty.
        """
        if pred_mask is None:
            return None, 0.0
        if hasattr(pred_mask, "cpu"):
            mask_np = pred_mask.squeeze().cpu().numpy()
        else:
            mask_np = np.array(pred_mask)
        if mask_np.size == 0:
            return None, 0.0
        if mask_np.dtype == bool:
            mask_uint8 = mask_np.astype(np.uint8) * 255
        elif mask_np.max() <= 1.0:
            mask_uint8 = (mask_np * 255).astype(np.uint8)
        else:
            mask_uint8 = mask_np.astype(np.uint8)

        pred_mask_resized = cv2.resize(mask_uint8, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        defect_pixels = np.count_nonzero(pred_mask_resized)
        total_pixels = crop_w * crop_h
        defect_percentage = (defect_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, defect_percentage

    def _score_to_class(self, pred_score, defect_percentage):
        """Map an anomaly score and defect area percentage to a final VisionClass value."""
        if pred_score < self.config.ANOMALY_CONFIDENCE_THRESHOLD:
            return VisionClass.GOOD.value
        else:
            # defect_percentage is 0-100. Divide by 100 to compare with 0-1 threshold (like 0.15)
            if (defect_percentage / 100.0) < self.config.DEFECT_AREA_SCRAP_THRESHOLD:
                return VisionClass.REPAIR.value
            else:
                return VisionClass.SCRAP.value

    def infer(self, obj_id, obj_label, frame=None):
        """Run the full YOLO + PatchCore pipeline on a captured frame for one object.

        Falls back to a fail-safe GOOD classification whenever a required
        input is missing (no frame, no YOLO detection, or no anomaly model
        for the detected class), so that a pipeline gap never blocks the
        conveyor. Results are cached per object label.
        """
        cache_key = obj_label if obj_label is not None else obj_id
        if cache_key in self._cache:
            return self._cache[cache_key]
        if frame is None:
            logger.warning(f"[RealVision] No frame received for {obj_label} -> GOOD (fail-safe)")
            res = {"object_id": obj_id, "class_id": VisionClass.GOOD.value, "confidence": 0.0, "defect_contours_frame": None, "defect_percentage": 0.0, "yolo_class": None, "yolo_bbox": None, "yolo_conf": 0.0}
            self._cache[cache_key] = res
            return res
        class_name, cropped_rgb, crop_box, yolo_bbox, yolo_conf = self._detect_and_crop(frame)
        if class_name is None:
            logger.info(f"[RealVision] {obj_label}: YOLO detected nothing -> GOOD")
            res = {"object_id": obj_id, "class_id": VisionClass.GOOD.value, "confidence": 0.0, "defect_contours_frame": None, "defect_percentage": 0.0, "yolo_class": None, "yolo_bbox": None, "yolo_conf": 0.0}
            self._cache[cache_key] = res
            return res
        inferencer = self._load_anomaly_model(class_name)
        if inferencer is None:
            res = {"object_id": obj_id, "class_id": VisionClass.GOOD.value, "confidence": 0.0, "defect_contours_frame": None, "defect_percentage": 0.0, "yolo_class": class_name, "yolo_bbox": yolo_bbox, "yolo_conf": yolo_conf}
            self._cache[cache_key] = res
            return res
        prediction = inferencer.predict(image=cropped_rgb)
        pred_score = prediction.pred_score
        if hasattr(pred_score, "item"):
            pred_score = pred_score.item()
        defect_contours_local = None
        defect_contours_frame = None
        defect_percentage = 0.0

        if pred_score >= getattr(self.config, 'ANOMALY_CONFIDENCE_THRESHOLD', 0.5):
            crop_h, crop_w = cropped_rgb.shape[:2]
            defect_contours_local, defect_percentage = self._extract_defect_contours(
                getattr(prediction, "pred_mask", None), crop_w, crop_h
            )
            if defect_contours_local and crop_box is not None:
                cx1, cy1, _, _ = crop_box
                defect_contours_frame = [cnt + np.array([cx1, cy1]) for cnt in defect_contours_local]

        class_id = self._score_to_class(pred_score, defect_percentage)

        class_name_fa = {0: "GOOD", 1: "REPAIR", 2: "SCRAP"}.get(class_id, str(class_id))
        if defect_contours_frame is not None:
            logger.info(
                f"[RealVision] {obj_label} -> YOLO: {class_name} ({yolo_conf:.2f}) | Anomaly Score: {pred_score:.3f} | Area: {defect_percentage:.2f}% -> {class_name_fa}"
            )
        else:
            logger.info(f"[RealVision] {obj_label} -> YOLO: {class_name} ({yolo_conf:.2f}) | Anomaly Score: {pred_score:.3f} -> {class_name_fa}")

        result_dict = {
            "object_id": obj_id,
            "class_id": class_id,
            "confidence": float(pred_score),
            "defect_contours_frame": defect_contours_frame,
            "defect_percentage": float(defect_percentage),
            "yolo_class": class_name,
            "yolo_bbox": yolo_bbox,
            "yolo_conf": float(yolo_conf) if yolo_conf else 0.0,
        }
        self._cache[cache_key] = result_dict
        return result_dict

    def reset(self):
        """Clear the cached per-object inference results."""
        self._cache = {}
