import json
import os
import re
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

app = Flask(__name__, static_folder=None)
_client = genai.Client(api_key=API_KEY) if API_KEY else None

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

PROMPT_TEMPLATE = """Ты преобразуешь медицинский текст в JSON для 3D-анатомического приложения.

Задача:
- выделить только явно упомянутые или прямо диагностируемые анатомические структуры;
- вернуть их в виде organ ids из списка ниже;
- не переносить патологию на соседние органы автоматически;
- если в тексте названа часть органа, а отдельного id для этой части нет — выбрать ближайший родительский organ id.

Разрешённые organ ids:
[{organs}]

Разрешённые statuses:
[{statuses}]

Правила:
1. Верни СТРОГО JSON. Без markdown, без комментариев, без текста до и после JSON.
2. Если данных недостаточно — верни {{"results":[]}}.
3. Используй только organ ids из списка.
4. Не придумывай диагнозы, осложнения и вторичные поражения.
5. Не расширяй находку на всю систему организма, если в тексте указана только одна зона.
6. "sick" — явная патология, травма, опухоль, выраженное нарушение функции, воспаление, острое состояние.
7. "warning" — умеренные изменения, подозрение, контроль, компенсированное состояние, неполная уверенность.
8. "healthy" — явная норма или отсутствие патологии по конкретной зоне.
9. confidence — число от 0 до 1.
10. Одно состояние = один объект в results.
11. В organs перечисляй только реально затронутые зоны.

Жёсткие правила привязки:
- Сердечная недостаточность, ФВ, дилатация ЛЖ, гипертрофия ЛЖ, клапанная регургитация, миокардит, перикардит без прямого упоминания сосудов -> heart.
- Коронарный стеноз, инфаркт миокарда из-за коронарного русла, стентирование/тромбоз коронарных артерий -> coronary_vessels, и только если явно есть поражение миокарда можно добавить heart.
- Аневризма/диссекция/коарктация/стеноз аорты как сосуда -> aorta.
- ТЭЛА, тромбоз лёгочной артерии, патология pulmonary trunk / pulmonary artery / pulmonary vein -> pulmonary_vessels.
- Сонные, лицевые, щитовидные, челюстные, глазничные и другие сосуды головы/шеи -> carotid_vessels.
- Мозговые артерии и сосуды Виллизиева круга -> cerebral_vessels.
- Сосуды руки -> upper_limb_vessels.
- Сосуды ноги/таза -> lower_limb_vessels.
- Пневмония, фиброз, ателектаз, очаг, инфильтрат, опухоль лёгкого -> lung_left / lung_right / оба лёгких по тексту.
- Трахеит, стеноз трахеи, поражение бронха, обструкция крупного бронха -> trachea.

- Перелом черепа, верхней/нижней челюсти, скуловой, носовой, височной, лобной, теменной кости -> skull.
- Дисфункция ВНЧС, артроз ВНЧС, вывих нижней челюсти -> tmj.
- Поражение подъязычной/гортанной связочной зоны -> hyoid_laryngeal.

- Перелом или травма шейных позвонков, C1–C7, атланта, аксиса, шейный спондилоз, шейная грыжа -> cervical_spine.
- Перелом или травма грудных позвонков, грудной спондилоз, грудная грыжа -> thoracic_spine.
- Перелом или травма поясничных позвонков, люмбаго, поясничная грыжа, спондилолистез -> lumbar_spine.
- Перелом крестца / копчика -> sacrum_coccyx.
- Если отдел позвоночника не уточнён -> spine.

- Перелом рёбер, грудины, ушиб грудной клетки -> ribcage.
- Перелом ключицы / лопатки -> shoulder_girdle.
- Поражение грудино-ключичного сочленения -> sternoclavicular_joint.
- Поражение акромиально-ключичного сочленения -> acromioclavicular_joint.
- Вывих плеча, артроз/артрит плечевого сустава, капсулит, нестабильность плеча -> shoulder_joint.
- Перелом плечевой кости -> humerus.
- Если указана просто травма верхней конечности без уточнения -> upper_limb.
- Вывих/артрит/артроз локтя, эпикондилит -> elbow_joint.
- Перелом лучевой или локтевой кости, перелом обеих костей предплечья -> forearm_bones.
- Поражение лучезапястного сустава, вывих/артроз запястья -> wrist_joint.
- Если указаны просто запястье/кисть без детализации костей против суставов -> wrist_hand.
- Перелом пястных костей, костей пальцев руки, ладьевидной и других костей кисти -> hand_bones.
- Метакарпофаланговые, межфаланговые, межзапястные и другие суставы кисти -> hand_joints.

- Перелом таза, вертлужной впадины -> pelvis.
- Сакроилеит, повреждение крестцово-подвздошного сустава, симфизит -> sacroiliac_pubic_joint.
- Артроз / вывих / дисплазия тазобедренного сустава, повреждение вертлужной губы -> hip_joint.
- Перелом бедренной кости, шейки бедра -> femur.
- Артроз / травма / мениски / крестообразные связки колена, вывих надколенника -> knee_joint.
- Перелом большеберцовой или малоберцовой кости -> tibia_fibula.
- Поражение голеностопного сустава, разрыв ATFL/CFL/deltoid, вывих голеностопа -> ankle_joint.
- Если указаны стопа/голеностоп без разделения на кость и сустав -> ankle_foot.
- Перелом таранной, пяточной, кубовидной, ладьевидной, клиновидных, плюсневых костей и фаланг стопы -> foot_bones.
- Подтаранный, таранно-ладьевидный, пяточно-кубовидный, плюснефаланговые, межфаланговые и другие суставы стопы -> foot_joints.

- Поражение слюнных желёз -> salivary_glands.
- Тонзиллит -> tonsils.
- Язык -> tongue.
- Пищевод -> esophagus.
- Желудок -> stomach.
- Тонкая кишка -> small_intestine.
- Толстая кишка / аппендикс / прямая кишка -> large_intestine.
- Печень -> liver.
- Жёлчный пузырь и жёлчные пути -> gallbladder.
- Поджелудочная железа -> pancreas.
- Селезёнка -> spleen.
- Левая/правая почка -> kidney_left / kidney_right.
- Мочевой пузырь -> bladder.
- Простата -> prostate.
- Яичко/придаток яичка -> testis_left / testis_right.
- Семенные пузырьки / семявыносящие пути -> seminal_vesicles.
- Половой член -> penis.
- Матка -> uterus.
- Левый/правый яичник -> ovary_left / ovary_right.
- Щитовидная железа -> thyroid.
- Паращитовидные железы -> parathyroid.
- Тимус -> thymus.
- Лимфоузлы головы и шеи -> lymph_nodes_head_neck.
- Грудные лимфоузлы -> lymph_nodes_thoracic.
- Брюшные лимфоузлы -> lymph_nodes_abdominal.
- Тазовые лимфоузлы -> lymph_nodes_pelvic.
- Подмышечные / локтевые / подключичные лимфоузлы -> lymph_nodes_upper_limb.
- Паховые / подколенные / большеберцовые лимфоузлы -> lymph_nodes_lower_limb.
- Мышцы головы и шеи -> head_neck_muscles.
- Мышцы туловища -> torso_muscles.
- Мышцы руки -> upper_limb_muscles.
- Мышцы ноги -> lower_limb_muscles.
- Кожа головы и шеи -> skin_head_neck.
- Кожа туловища -> skin_torso.
- Кожа руки -> skin_upper_limb.
- Кожа ноги -> skin_lower_limb.

Формат ответа:
{{
  "results": [
    {{
      "disease": "Краткое название состояния",
      "severity": "sick",
      "confidence": 0.93,
      "organs": [
        {{"id":"heart","status":"sick"}}
      ]
    }}
  ]
}}

Медицинский текст:
{text}"""


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
        text = text[start:end + 1]
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

        clean_results.append({
            "disease": disease,
            "severity": severity,
            "confidence": max(0.0, min(1.0, confidence)),
            "source": "gemini",
            "organs": clean_organs,
        })

    return {"results": clean_results}


def _build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(
        organs=", ".join(ALLOWED_ORGANS),
        statuses=", ".join(ALLOWED_STATUSES),
        text=text,
    )


@app.get("/")
def serve_index():
    index_file = _index_file()
    return send_from_directory(index_file.parent, index_file.name)


@app.get("/3d/<path:filename>")
def serve_3d(filename: str):
    folder = BASE_DIR / "3d"
    return send_from_directory(folder, filename)


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
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
