#!/usr/bin/env python3
"""Convert Synthea CSV export to HuMapper dataset format.

Input directory must contain at least:
- patients.csv
- encounters.csv (optional, but strongly recommended)
- conditions.csv (optional)

Output format:
{
  "schema_version": "humapper-patientdb/v1",
  "patients": [...],
  "timeline_items": [...],
  "organ_data_current": [...]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "humapper-patientdb/v1"


ORGAN_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("heart", ("heart", "cardiac", "myocard", "coronary", "arrhythm", "angina")),
    ("lung_left", ("left lung", "left lower lobe", "left upper lobe", "pulmonary left")),
    ("lung_right", ("right lung", "right lower lobe", "right upper lobe", "pulmonary right")),
    ("lung_right", ("lung", "pulmonary", "pneumonia", "asthma", "copd", "bronch")),
    ("liver", ("liver", "hepatic", "hepatitis", "cirrhosis", "steatosis")),
    ("kidney_left", ("left kidney",)),
    ("kidney_right", ("right kidney",)),
    ("kidney_right", ("kidney", "renal", "nephro")),
    ("brain", ("brain", "cerebr", "stroke", "intracran", "epilep", "migraine", "neuro")),
    ("thyroid", ("thyroid", "hypothy", "hyperthy")),
    ("pancreas", ("pancrea", "diabetes", "insulin")),
    ("stomach", ("stomach", "gastr", "ulcer", "duoden")),
    ("small_intestine", ("small intestine", "ileum", "jejunum")),
    ("large_intestine", ("colon", "large intestine", "colitis", "divertic")),
    ("bladder", ("bladder", "cystitis", "urinary")),
    ("prostate", ("prostate", "prostatic")),
    ("uterus", ("uterus", "endometri", "pregnan", "cervix")),
    ("ovary_left", ("ovary", "ovarian")),
    ("spine", ("spine", "vertebra", "lumbar", "thoracic", "cervical pain")),
]


RED_HINTS = (
    "cancer",
    "malignant",
    "infarct",
    "failure",
    "sepsis",
    "hemorrhage",
    "stroke",
    "embol",
    "severe",
    "critical",
)
ORANGE_HINTS = (
    "suspicion",
    "suspected",
    "nodule",
    "lesion",
    "chronic",
    "monitor",
    "follow-up",
    "follow up",
    "abnormal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Synthea CSV files into HuMapper dataset.json format."
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing Synthea CSV files (patients.csv, encounters.csv, conditions.csv).",
    )
    parser.add_argument(
        "--out",
        default="dataset.json",
        help="Output JSON path. Default: dataset.json",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="Optional limit of processed patients. 0 means all.",
    )
    parser.add_argument(
        "--default-source",
        default="Synthea",
        help="Timeline source label. Default: Synthea",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_iso(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    dt = None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def display_date_from_iso(iso_value: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
        return dt.strftime("%d.%m.%Y")
    except ValueError:
        return iso_value


def infer_organ_id(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered)
    for organ, keywords in ORGAN_KEYWORDS:
        if any(k in lowered for k in keywords):
            return organ
    return "heart"


def infer_color(text: str) -> str:
    lowered = (text or "").lower()
    if any(k in lowered for k in RED_HINTS):
        return "red"
    if any(k in lowered for k in ORANGE_HINTS):
        return "orange"
    return "orange"


def infer_ttl_days(color: str) -> int:
    if color == "red":
        return 180
    if color == "orange":
        return 90
    if color == "yellow":
        return 30
    if color == "blue":
        return 365
    return 365


def infer_status_from_color(color: str) -> str:
    if color == "red":
        return "sick"
    if color in ("orange", "yellow"):
        return "warning"
    return "none"


def patient_full_name(row: Dict[str, str]) -> str:
    first = (row.get("FIRST") or "").strip()
    last = (row.get("LAST") or "").strip()
    if first or last:
        return f"{last} {first}".strip()
    return (row.get("NAME") or row.get("Id") or "Unknown").strip()


@dataclass
class Event:
    patient_id: str
    event_id: str
    date_iso: str
    study_type: str
    source: str
    conclusion: str
    organ_id: str
    color: str
    subtype: str
    ai_confidence: float
    ttl_days: int
    notes: str
    status: str
    payload: Dict[str, object]


def build_events(
    encounters: List[Dict[str, str]],
    conditions: List[Dict[str, str]],
    default_source: str,
) -> List[Event]:
    events: List[Event] = []
    encounter_by_id = {e.get("Id", ""): e for e in encounters if e.get("Id")}

    # Prefer condition-driven timeline when available.
    for idx, c in enumerate(conditions):
        patient = (c.get("PATIENT") or "").strip()
        if not patient:
            continue
        desc = (c.get("DESCRIPTION") or c.get("CODE") or "").strip()
        encounter_id = (c.get("ENCOUNTER") or "").strip()
        encounter = encounter_by_id.get(encounter_id, {})
        date_iso = normalize_iso(c.get("START") or encounter.get("START") or "")
        organ_id = infer_organ_id(desc)
        color = infer_color(desc)
        events.append(
            Event(
                patient_id=patient,
                event_id=f"cond_{encounter_id or idx}_{organ_id}",
                date_iso=date_iso,
                study_type=(encounter.get("ENCOUNTERCLASS") or "Condition").strip() or "Condition",
                source=default_source,
                conclusion=desc,
                organ_id=organ_id,
                color=color,
                subtype="suspicion" if color == "orange" else "",
                ai_confidence=0.72 if color == "orange" else 0.9 if color == "red" else 0.5,
                ttl_days=infer_ttl_days(color),
                notes=(encounter.get("REASONDESCRIPTION") or "").strip(),
                status="completed",
                payload={
                    "condition_code": c.get("CODE", ""),
                    "condition_description": desc,
                    "encounter_id": encounter_id,
                },
            )
        )

    if events:
        return events

    # Fallback: encounter-driven timeline.
    for idx, e in enumerate(encounters):
        patient = (e.get("PATIENT") or "").strip()
        if not patient:
            continue
        desc = (e.get("REASONDESCRIPTION") or e.get("DESCRIPTION") or e.get("ENCOUNTERCLASS") or "").strip()
        date_iso = normalize_iso(e.get("START") or "")
        organ_id = infer_organ_id(desc)
        color = infer_color(desc)
        events.append(
            Event(
                patient_id=patient,
                event_id=f"enc_{e.get('Id', idx)}_{organ_id}",
                date_iso=date_iso,
                study_type=(e.get("ENCOUNTERCLASS") or "Encounter").strip() or "Encounter",
                source=default_source,
                conclusion=desc,
                organ_id=organ_id,
                color=color,
                subtype="suspicion" if color == "orange" else "",
                ai_confidence=0.68 if color == "orange" else 0.88 if color == "red" else 0.5,
                ttl_days=infer_ttl_days(color),
                notes="",
                status="completed",
                payload={"encounter_id": e.get("Id", "")},
            )
        )
    return events


def build_dataset(
    patients_rows: List[Dict[str, str]],
    events: List[Event],
    max_patients: int,
) -> Dict[str, object]:
    selected_patients = patients_rows
    if max_patients and max_patients > 0:
        selected_patients = patients_rows[:max_patients]
    selected_ids = {(p.get("Id") or "").strip() for p in selected_patients if (p.get("Id") or "").strip()}

    patients_out = []
    for row in selected_patients:
        patient_id = (row.get("Id") or "").strip()
        if not patient_id:
            continue
        patients_out.append(
            {
                "id": patient_id,
                "name": patient_full_name(row),
                "gender": "female" if (row.get("GENDER") or "").strip().upper() == "F" else "male",
                "dob": (row.get("BIRTHDATE") or "").strip(),
                "address": " ".join(
                    part.strip()
                    for part in (row.get("ADDRESS"), row.get("CITY"), row.get("STATE"))
                    if (part or "").strip()
                ),
                "phone": "",
            }
        )

    timeline_items = []
    for ev in sorted(events, key=lambda x: (x.date_iso, x.event_id)):
        if ev.patient_id not in selected_ids:
            continue
        timeline_items.append(
            {
                "id": ev.event_id,
                "patientid": ev.patient_id,
                "date": ev.date_iso,
                "study_type": ev.study_type,
                "source": ev.source,
                "dicom_refs": [],
                "affected_organs": [
                    {
                        "organid": ev.organ_id,
                        "color": ev.color,
                        "subtype": ev.subtype,
                        "ai_confidence": ev.ai_confidence,
                        "ttl_days": ev.ttl_days,
                        "notes": ev.notes,
                    }
                ],
                "conclusion": ev.conclusion,
                "status": ev.status,
                "payload": ev.payload,
            }
        )

    latest_by_patient_organ: Dict[Tuple[str, str], Event] = {}
    for ev in timeline_items:
        pid = ev["patientid"]
        for affected in ev.get("affected_organs", []):
            organ = affected.get("organid")
            if not organ:
                continue
            key = (pid, organ)
            existing = latest_by_patient_organ.get(key)
            if existing is None or str(existing["date"]) <= str(ev["date"]):
                latest_by_patient_organ[key] = ev

    organ_data_current = []
    for (pid, organ), ev in sorted(latest_by_patient_organ.items(), key=lambda x: (x[0][0], x[0][1])):
        affected = next((a for a in ev.get("affected_organs", []) if a.get("organid") == organ), {})
        color = affected.get("color", "gray")
        organ_data_current.append(
            {
                "patientid": pid,
                "organ_id": organ,
                "status": infer_status_from_color(color),
                "study_type": ev.get("study_type", ""),
                "study_date": display_date_from_iso(ev.get("date", "")),
                "findings": ev.get("conclusion", ""),
                "notes": affected.get("notes", ""),
                "color": color,
                "subtype": affected.get("subtype", ""),
                "ai_confidence": affected.get("ai_confidence", 0.0),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "patients": patients_out,
        "timeline_items": timeline_items,
        "organ_data_current": organ_data_current,
    }


def main() -> int:
    args = parse_args()
    csv_dir = Path(args.csv_dir).resolve()
    if not csv_dir.exists():
        raise SystemExit(f"CSV directory does not exist: {csv_dir}")

    patients_rows = read_csv_rows(csv_dir / "patients.csv")
    encounters = read_csv_rows(csv_dir / "encounters.csv")
    conditions = read_csv_rows(csv_dir / "conditions.csv")

    if not patients_rows:
        raise SystemExit(f"patients.csv not found or empty in: {csv_dir}")

    events = build_events(encounters, conditions, args.default_source)
    dataset = build_dataset(patients_rows, events, args.max_patients)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Patients: {len(dataset['patients'])}")
    print(f"Timeline items: {len(dataset['timeline_items'])}")
    print(f"Current organ rows: {len(dataset['organ_data_current'])}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

