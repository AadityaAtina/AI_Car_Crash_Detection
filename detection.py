"""
detection.py — Phase 1 + Phase 2 YOLO Detection Pipeline
AI-Powered Emergency Response System
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime


# ── Model Path ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Models", "best.pt"
)
model = YOLO(MODEL_PATH)


# ── Storage Folders ──────────────────────────────────────────────────────────
BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR    = BASE_DIR / "temp"      # annotated detection images (cleared each run)
UPLOADS_DIR = BASE_DIR / "uploads"   # original images (persisted, timestamped)


# ── Class → Canonical Label Mapping ──────────────────────────────────────────
CLASS_LABEL_MAP = {
    "Glass Brack":      "glass_break",
    "Side Door Damage": "side_damage",
    "Front Damage":     "front_damage",
    "Back Damage":      "rear_damage",
    "Dent":             "dent",
    "Pillar Damage":    "pillar_damage",
    "Roof Damage":      "roof_damage",
}


# ── Severity Map ──────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    "front_damage":   "HIGH",
    "side_damage":    "HIGH",
    "rear_damage":    "MEDIUM",
    "roof_damage":    "CRITICAL",
    "pillar_damage":  "CRITICAL",
    "glass_break":    "MEDIUM",
    "dent":           "LOW",
}


# ── Battery Labels ────────────────────────────────────────────────────────────
BATTERY_LABELS = {"battery", "ev_battery", "hybrid_battery", "battery_warning"}


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
SEVERITY_COLOURS = {
    "CRITICAL": (0, 0, 255),
    "HIGH":     (0, 128, 255),
    "MEDIUM":   (0, 200, 200),
    "LOW":      (0, 200, 0),
}

BATTERY_COLOUR = (0, 0, 255)


def _severity_from_conf(label: str, conf: float) -> str:
    """
    Combine the base severity with confidence to potentially upgrade/downgrade.
    conf >= 0.80 → upgrade one step
    conf < 0.40  → downgrade one step
    """
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    base = SEVERITY_MAP.get(label, "MEDIUM")
    idx = order.index(base)

    if conf >= 0.80 and idx < len(order) - 1:
        idx += 1
    elif conf < 0.40 and idx > 0:
        idx -= 1

    return order[idx]


def _draw_box(img, x1, y1, x2, y2, label_text, colour, is_battery=False):
    """Draw a styled bounding box with label badge."""
    thickness = 3 if is_battery else 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), colour, 2)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

    badge_y1 = max(y1 - th - baseline - 6, 0)
    badge_y2 = y1
    cv2.rectangle(img, (x1, badge_y1), (x1 + tw + 8, badge_y2), colour, -1)

    cv2.putText(
        img,
        label_text,
        (x1 + 4, badge_y2 - baseline),
        font,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA
    )

    if is_battery:
        h, w = y2 - y1, x2 - x1
        stripe_overlay = img[y1:y2, x1:x2].copy()
        for i in range(0, w + h, 20):
            cv2.line(stripe_overlay, (i, 0), (i - h, h), (0, 0, 200), 2)
        cv2.addWeighted(
            stripe_overlay, 0.25,
            img[y1:y2, x1:x2], 0.75,
            0,
            img[y1:y2, x1:x2]
        )


def _clear_temp_folder():
    """Remove all files from temp folder (keeps only latest annotated image)."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for item in TEMP_DIR.iterdir():
        if item.is_file():
            item.unlink()


def _save_images(original_img, annotated_img) -> dict:
    """
    • Original image  → uploads/<timestamp>.jpg  (permanent, one file per request)
    • Annotated image → temp/annotated.jpg        (temp folder cleared each run)

    Returns:
        {
            "original_image_path": str,   # full path inside uploads/
            "upload_filename":     str,   # basename, e.g. "20260506_144730_123456.jpg"
            "annotated_image_path": str,  # full path inside temp/
        }
    """
    # ── uploads/ : save original with timestamp filename ──────────────────────
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")   # e.g. 20260506_144730_123456
    upload_filename = f"{ts}.jpg"
    original_path   = UPLOADS_DIR / upload_filename

    ok1 = cv2.imwrite(str(original_path), original_img)
    if not ok1:
        raise RuntimeError(f"Failed to save original image: {original_path}")

    # ── temp/ : save only the annotated detection image ───────────────────────
    _clear_temp_folder()
    annotated_path = TEMP_DIR / "annotated.jpg"

    ok2 = cv2.imwrite(str(annotated_path), annotated_img)
    if not ok2:
        raise RuntimeError(f"Failed to save annotated image: {annotated_path}")

    return {
        "original_image_path":  str(original_path),
        "upload_filename":      upload_filename,
        "annotated_image_path": str(annotated_path),
    }


def analyze_image(image_bytes: bytes) -> dict:
    """
    Full Phase 1 + Phase 2 analysis pipeline.

    Returns:
    {
        "detections": [
            {
                "label": str,
                "canonical_label": str,
                "confidence": float,
                "severity": str,
                "is_battery": bool,
                "bbox": [x1, y1, x2, y2]
            }, ...
        ],
        "battery_hazard": bool,
        "critical_found": bool,
        "annotated_image": str,   # file path → temp/annotated.jpg  (use as a link)
        "original_image_path": str,   # uploads/<timestamp>.jpg
        "upload_filename": str,        # basename of uploaded original
        "annotated_image_path": str,  # same as annotated_image (full path)
        "summary": {
            "damage_types": [str],
            "severity_levels": [str],
            "has_battery_hazard": bool
        }
    }
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_img is None:
        raise ValueError("Invalid image: unable to decode.")

    img = original_img.copy()

    results = model(img, conf=0.25)
    result = results[0]

    detections = []
    battery_hazard = False
    critical_found = False

    for box in (result.boxes or []):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        raw_name = model.names[class_id]

        canonical = CLASS_LABEL_MAP.get(raw_name, raw_name.lower().replace(" ", "_"))
        is_batt = canonical in BATTERY_LABELS
        severity = "BATTERY HAZARD" if is_batt else _severity_from_conf(canonical, conf)

        if is_batt:
            battery_hazard = True
        if severity == "CRITICAL":
            critical_found = True

        colour = BATTERY_COLOUR if is_batt else SEVERITY_COLOURS.get(severity, (200, 200, 200))

        if is_batt:
            label_text = f"BATTERY HAZARD {conf:.0%}"
        else:
            label_text = f"{canonical.upper()}  {severity}  {conf:.0%}"

        _draw_box(img, x1, y1, x2, y2, label_text, colour, is_battery=is_batt)

        detections.append({
            "label": raw_name,
            "canonical_label": canonical,
            "confidence": round(conf, 3),
            "severity": severity,
            "is_battery": is_batt,
            "bbox": [x1, y1, x2, y2],
        })

    saved_paths = _save_images(original_img, img)

    # annotated_image_path already saved to temp/ — return it as a file link
    annotated_image_link = saved_paths["annotated_image_path"]

    damage_types = [d["canonical_label"] for d in detections if not d["is_battery"]]
    sev_levels = list({d["severity"] for d in detections})

    return {
        "detections": detections,
        "battery_hazard": battery_hazard,
        "critical_found": critical_found,
        "annotated_image":      annotated_image_link,   # file path in temp/
        "original_image_path":  saved_paths["original_image_path"],
        "upload_filename":       saved_paths["upload_filename"],
        "annotated_image_path": saved_paths["annotated_image_path"],
        "summary": {
            "damage_types": damage_types,
            "severity_levels": sev_levels,
            "has_battery_hazard": battery_hazard,
        },
    }