"""
app.py — AI-Powered Emergency Response System
Pure REST API: POST image → JSON (detections + tools + annotated image)
"""

import logging
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from detection import analyze_image
from recommender import recommend_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("emergency_api")

app = FastAPI(
    title="AI Emergency Response API",
    description=(
        "POST a vehicle accident image → receive YOLO detections, "
        "severity levels, rescue tool recommendations, and the annotated image."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Serve saved images over HTTP ──────────────────────────────────────
_BASE = Path(__file__).parent
for _folder in ("temp", "uploads"):
    (_BASE / _folder).mkdir(parents=True, exist_ok=True)
    app.mount(f"/{_folder}", StaticFiles(directory=str(_BASE / _folder)), name=_folder)

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "2.1.0"}

@app.post("/analyze", tags=["Detection"])
async def analyze(file: UploadFile = File(...)):
    ct = (file.content_type or "").lower()
    allowed = {"image/jpeg", "image/jpg", "image/png"}
    if ct not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type '{ct}'. Send JPEG or PNG.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # ── Image size validation ─────────────────────────────────────
    nparr = np.frombuffer(image_bytes, np.uint8)
    _img_check = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if _img_check is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Send a valid JPEG or PNG.")

    h, w = _img_check.shape[:2]
    if w != 640 or h != 640:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Image size must be exactly 640 × 640 pixels. "
                f"Received: {w} × {h} px. "
                f"Please Ensure the image dimensions are at least 640 × 640 pixels."
            ),
        )

    logger.info(f"Image received: '{file.filename}'  {len(image_bytes):,} bytes  [{w}×{h}px]")

    try:
        detection_result = analyze_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

    logger.info(
        f"Phase 1 — {len(detection_result['detections'])} detections | "
        f"battery={detection_result['battery_hazard']} | "
        f"critical={detection_result['critical_found']}"
    )

    try:
        recommendation = recommend_tools(detection_result)
    except Exception as e:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {e}")

    logger.info(
        f"Phase 2 — tools={len(recommendation['recommended_tools'])} | "
        f"battery={recommendation['battery_hazard']} | "
        f"critical={recommendation['critical_found']}"
    )

    return JSONResponse({
        "detected_damages": recommendation["detected_damages"],
        "severity_levels": recommendation["severity_levels"],
        "battery_hazard": recommendation["battery_hazard"],
        "critical_found": recommendation["critical_found"],
        "recommended_tools": recommendation["recommended_tools"],
        "summary_message": recommendation["summary_message"],
        "annotated_image_url": "/temp/annotated.jpg",      # view via GET /temp/annotated.jpg
        #"upload_filename":     detection_result["upload_filename"],
        #"original_image_url":  f"/uploads/{detection_result['upload_filename']}",
        "damage_detected": [
            {
                "label": d["label"],
                "canonical_label": d["canonical_label"],
                "severity": d["severity"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
                "is_battery": d["is_battery"],
            }
            for d in detection_result["detections"]
        ],
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
