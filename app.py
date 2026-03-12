import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_CANDIDATES = [
    BASE_DIR / "index_realtime_mesh_ai_bones_joints_extended.html",
    BASE_DIR / "index_realtime_mesh_ai_bodywide_fixed.html",
    BASE_DIR / "index_realtime_mesh_ai_organ_split.html",
    BASE_DIR / "index_realtime_mesh_ai_fixed_heart_mapping.html",
    BASE_DIR / "index_realtime_mesh_ai.html",
    BASE_DIR / "index.html",
]

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
CLOUD_DB_PATH = Path(os.getenv("CLOUD_DB_PATH", str(BASE_DIR / "cloud_snapshots.db")))
try:
    CLOUD_DB_MAX_BYTES = int(os.getenv("CLOUD_DB_MAX_BYTES", "20000000"))
except ValueError:
    CLOUD_DB_MAX_BYTES = 20000000

app = Flask(__name__, static_folder=None)
_client = genai.Client(api_key=API_KEY) if API_KEY else None
_cloud_lock = threading.Lock()

ALLOWED_ORGANS = [
    "skull",
    "tmj",
    "hyoid_laryngeal",
    "spine",
    "cervical_spine",
    "thoracic_spine",
    "lumbar_spine",
    "sacrum_coccyx",
    "ribcage",
    "pelvis",
    "sacroiliac_pubic_joint",
    "shoulder_girdle",
    "sternoclavicular_joint",
    "acromioclavicular_joint",
    "shoulder_joint",
    "humerus",
    "upper_limb",
    "elbow_joint",
    "forearm_bones",
    "wrist_joint",
    "wrist_hand",
    "hand_bones",
    "hand_joints",
    "femur",
    "hip_joint",
    "knee_joint",
    "tibia_fibula",
    "ankle_joint",
    "ankle_foot",
    "foot_bones",
    "foot_joints",
    "brain",
    "spinal_cord",
    "meninges",
    "peripheral_nerves",
    "thyroid",
    "parathyroid",
    "heart",
    "coronary_vessels",
    "aorta",
    "pulmonary_vessels",
    "carotid_vessels",
    "cerebral_vessels",
    "upper_limb_vessels",
    "lower_limb_vessels",
    "lung_left",
    "lung_right",
    "trachea",
    "salivary_glands",
    "tonsils",
    "tongue",
    "esophagus",
    "stomach",
    "small_intestine",
    "large_intestine",
    "liver",
    "gallbladder",
    "pancreas",
    "spleen",
    "kidney_left",
    "kidney_right",
    "bladder",
    "prostate",
    "testis_left",
    "testis_right",
    "penis",
    "seminal_vesicles",
    "uterus",
    "ovary_left",
    "ovary_right",
    "thymus",
    "lymph_nodes_head_neck",
    "lymph_nodes_thoracic",
    "lymph_nodes_abdominal",
    "lymph_nodes_pelvic",
    "lymph_nodes_upper_limb",
    "lymph_nodes_lower_limb",
    "head_neck_muscles",
    "torso_muscles",
    "upper_limb_muscles",
    "lower_limb_muscles",
    "skin_head_neck",
    "skin_torso",
    "skin_upper_limb",
    "skin_lower_limb",
    # legacy aliases
    "lymph_nodes_cervical",
    "lymph_nodes_inguinal",
]
ALLOWED_STATUSES = ["sick", "warning", "healthy"]

ORGAN_ID_ALIASES = {
    "lymph_nodes_cervical": "lymph_nodes_head_neck",
    "lymph_nodes_inguinal": "lymph_nodes_lower_limb",
    "temporomandibular_joint": "tmj",
    "jaw_joint": "tmj",
    "sternoclavicular": "sternoclavicular_joint",
    "acromioclavicular": "acromioclavicular_joint",
    "glenohumeral_joint": "shoulder_joint",
    "shoulder": "shoulder_joint",
    "humerus_bone": "humerus",
    "forearm": "forearm_bones",
    "hand": "hand_bones",
    "hand_joint": "hand_joints",
    "wrist": "wrist_joint",
    "cervical": "cervical_spine",
    "thoracic": "thoracic_spine",
    "lumbar": "lumbar_spine",
    "sacrococcygeal": "sacrum_coccyx",
    "sacroiliac_joint": "sacroiliac_pubic_joint",
    "pubic_symphysis": "sacroiliac_pubic_joint",
    "lower_leg": "tibia_fibula",
    "ankle": "ankle_joint",
    "foot": "foot_bones",
    "foot_joint": "foot_joints",
}


def canonicalize_organ_id(oid: str) -> str:
    key = str(oid or "").strip()
    return ORGAN_ID_ALIASES.get(key, key)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_doctor_key(raw: str) -> str:
    key = str(raw or "").strip().lower()
    if not key:
        raise ValueError("doctor key is required")
    if len(key) > 190:
        key = key[:190]
    return key


def _init_cloud_store() -> None:
    CLOUD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CLOUD_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doctor_snapshots (
                doctor_key TEXT PRIMARY KEY,
                db_b64 TEXT NOT NULL,
                db_sha256 TEXT DEFAULT '',
                size_bytes INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


_init_cloud_store()


_PROMPT_FILE = BASE_DIR / "gemini_prompt_humapper.txt"


def _load_prompt_template() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8")
    return (
        "Верни СТРОГО JSON без markdown.\n"
        "Разрешённые organ ids: [<ALLOWED_ORGANS>]\n"
        "Разрешённые statuses: [<ALLOWED_STATUSES>]\n"
        "Формат: {{\"results\":[{{\"disease\":\"...\",\"severity\":\"sick\","
        "\"confidence\":0.9,\"organs\":[{{\"id\":\"heart\",\"status\":\"sick\"}}]}}]}}\n"
        "Медицинский текст:\n<MEDICAL_TEXT>"
    )


def _build_prompt(text: str) -> str:
    organs_str = ", ".join(o for o in ALLOWED_ORGANS if o not in ORGAN_ID_ALIASES)
    template = _load_prompt_template()
    return (
        template.replace("<ALLOWED_ORGANS>", organs_str)
        .replace("<ALLOWED_STATUSES>", ", ".join(ALLOWED_STATUSES))
        .replace("<MEDICAL_TEXT>", text)
    )


def _index_file() -> Path:
    for path in INDEX_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Не найден HTML интерфейса рядом с сервером.")


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Пустой ответ модели")

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        text = fence_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON не найден в ответе модели")
        text = text[start : end + 1]
    return json.loads(text)


def _validate_results(payload: dict) -> dict:
    results = payload.get("results", [])
    clean_results = []

    for item in results:
        if not isinstance(item, dict):
            continue

        disease = str(item.get("disease", "")).strip()
        severity = str(item.get("severity", "")).strip()
        confidence = item.get("confidence", 0.0)
        organs = item.get("organs", [])

        if not disease or severity not in ALLOWED_STATUSES:
            continue

        clean_organs = []
        seen = set()

        for organ_item in organs:
            if not isinstance(organ_item, dict):
                continue

            oid = canonicalize_organ_id(str(organ_item.get("id", "")).strip())
            status = str(organ_item.get("status", "")).strip()

            if oid not in ALLOWED_ORGANS or status not in ALLOWED_STATUSES:
                continue

            key = (oid, status)
            if key in seen:
                continue
            seen.add(key)
            clean_organs.append({"id": oid, "status": status})

        if not clean_organs:
            continue

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        clean_results.append(
            {
                "disease": disease,
                "severity": severity,
                "confidence": max(0.0, min(1.0, confidence)),
                "source": "gemini",
                "organs": clean_organs,
            }
        )

    return {"results": clean_results}


@app.get("/health")
def healthcheck():
    index_exists = _index_file().exists() if any(path.exists() for path in INDEX_CANDIDATES) else False
    return jsonify(
        {
            "ok": True,
            "service": "humapper",
            "index_found": index_exists,
            "gemini_configured": bool(API_KEY),
            "model": MODEL,
        }
    )


@app.get("/")
def serve_index():
    index_file = _index_file()
    return send_from_directory(index_file.parent, index_file.name)


@app.get("/3d/<path:filename>")
def serve_3d(filename: str):
    folder = BASE_DIR / "3d"
    return send_from_directory(folder, filename)


@app.get("/api/cloud-db/<path:doctor_key>")
def cloud_db_get(doctor_key: str):
    try:
        key = _normalize_doctor_key(doctor_key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _cloud_lock, sqlite3.connect(CLOUD_DB_PATH) as conn:
        row = conn.execute(
            "SELECT doctor_key, db_b64, db_sha256, size_bytes, updated_at FROM doctor_snapshots WHERE doctor_key=?",
            (key,),
        ).fetchone()

    if not row:
        return jsonify({"error": "not_found"}), 404

    return jsonify(
        {
            "doctor_key": row[0],
            "db_b64": row[1],
            "db_sha256": row[2] or "",
            "size_bytes": int(row[3] or 0),
            "updated_at": row[4],
        }
    )


@app.put("/api/cloud-db/<path:doctor_key>")
def cloud_db_put(doctor_key: str):
    try:
        key = _normalize_doctor_key(doctor_key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    data = request.get_json(silent=True) or {}
    db_b64 = str(data.get("db_b64", "")).strip()
    if not db_b64:
        return jsonify({"error": "db_b64 is required"}), 400

    approx_size = (len(db_b64) * 3) // 4
    if approx_size <= 0:
        return jsonify({"error": "empty payload"}), 400
    if approx_size > CLOUD_DB_MAX_BYTES:
        return jsonify({"error": f"payload too large (>{CLOUD_DB_MAX_BYTES} bytes)"}), 413

    db_sha256 = str(data.get("db_sha256", "")).strip().lower()[:128]
    updated_at = _utc_now_iso()

    with _cloud_lock, sqlite3.connect(CLOUD_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO doctor_snapshots(doctor_key, db_b64, db_sha256, size_bytes, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(doctor_key) DO UPDATE SET
              db_b64=excluded.db_b64,
              db_sha256=excluded.db_sha256,
              size_bytes=excluded.size_bytes,
              updated_at=excluded.updated_at
            """,
            (key, db_b64, db_sha256, approx_size, updated_at),
        )
        conn.commit()

    return jsonify(
        {
            "ok": True,
            "doctor_key": key,
            "db_sha256": db_sha256,
            "size_bytes": approx_size,
            "updated_at": updated_at,
        }
    )


@app.post("/api/cloud-db/<path:doctor_key>")
def cloud_db_post(doctor_key: str):
    return cloud_db_put(doctor_key)


@app.get("/<path:filename>")
def serve_misc(filename: str):
    return send_from_directory(BASE_DIR, filename)


@app.post("/api/analyze")
def analyze():
    if not _client:
        return jsonify({"error": "GEMINI_API_KEY не задан"}), 503

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if len(text) < 4:
        return jsonify({"results": []})

    try:
        response = _client.models.generate_content(
            model=MODEL,
            contents=_build_prompt(text),
        )
        payload = _extract_json(response.text or "")
        return jsonify(_validate_results(payload))
    except Exception as exc:
        return jsonify({"error": f"AI analysis failed: {exc}"}), 500


@app.get("/api/prompt")
def get_prompt():
    return jsonify({"prompt_template": _build_prompt("{ТВОЙ_ТЕКСТ}")})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
