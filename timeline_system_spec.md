# Система Таймлайна (HuMapper)

## 1. Цель
Интерактивный клинический таймлайн, синхронизированный с 3D-моделью пациента:
- точка исследования по дате -> состояние органов на эту дату;
- TTL-старение сигналов;
- приоритизация и фильтры для снижения перегрузки врача;
- подготовка к PACS/DICOM и EMR интеграции.

## 2. Цветовая модель
- `red`: подтвержденная патология (высокий приоритет)
- `orange`: подозрение / наблюдение (средний)
- `yellow`: устаревание/overdue (низко-средний)
- `blue`: техническая проблема/артефакт (низкий)
- `gray`: нет актуальных данных (нулевой)

## 3. TTL и фазы
- `0 .. TTL` -> `bright`
- `TTL .. 1.33*TTL` -> `dim`
- `1.33*TTL .. 2*TTL` -> `overdue` (серый + желтый контур)
- `>2*TTL` -> `stale` (серый)

Дефолт:
- `red = 180 дн`
- `orange = 90 дн`
- `yellow = 30 дн`
- `blue = 365 дн`
- `gray = 365 дн`

Допускаются organ-specific overrides (например, `heart.red = 150`).

## 4. Приоритизация
`priorityScore = colorWeight + aiConfidence + clinicalSeverity - agePenalty + overdueBoost`

Рекомендуемые веса:
- `colorWeight`: red=5, orange=4, yellow=2, blue=1, gray=0
- `aiConfidence`: 0..1
- `clinicalSeverity`: 0..1
- `agePenalty`: рост по времени с шагом 30 дней
- `overdueBoost`: +20 (если overdue)

## 5. Данные
## 5.1 TimelineItem
```json
{
  "id": "tl_2026_01_15_ct_001",
  "patient_id": "pat123",
  "date": "2026-01-15T10:30:00Z",
  "study_type": "CT",
  "source": "PACS_A",
  "dicom_refs": ["dicom://..."],
  "affected_organs": [
    {
      "organ_id": "lung_right",
      "signal": "orange",
      "subtype": "suspicion",
      "ai_confidence": 0.72,
      "ttl_days": 90,
      "notes": "nodular opacity 8mm"
    }
  ],
  "conclusion": "Suspicious nodule in right lower lobe",
  "status": "completed"
}
```

## 5.2 OrganState
```json
{
  "organ_id": "liver",
  "current_signal": "blue",
  "last_updated": "2026-01-15T10:30:00Z",
  "phase": "dim",
  "priority_score": 246,
  "history": [
    {"date":"2025-06-01","signal":"orange","reason":"steatosis AI 0.6"}
  ]
}
```

## 6. API (рекомендуемый контракт)
- `GET /api/patients/{id}/timeline?from=&to=&study_type=&priority_min=&page=`
- `GET /api/patients/{id}/timeline/{item_id}`
- `POST /api/patients/{id}/timeline` (webhook PACS/AI)
- `POST /api/patients/{id}/organ/{organ_id}/action` (`confirm|dismiss|schedule_repeat|mark_irrelevant`)
- `GET /api/patients/{id}/organ_states`
- `GET /api/config/ttl`
- `PUT /api/config/ttl`

## 7. WebSocket события
- `timeline.updated`
- `organstate.changed`
- `task.created`

## 8. UI сценарии
- Клик по точке таймлайна -> 3D модель на дату + панель деталей.
- Две выбранные точки -> compare mode (`side-by-side` / `overlay`).
- Фильтры: `all`, `critical`, `overdue`, группировка по системам.
- Кластеризация при плотных датах (сжатие до ограниченного числа точек).
- Задачи: overdue и scheduled_pending (иконка часов + приоритет).

## 9. Текущее внедрение в `index.html`
- TTL/фазы/приоритет реализованы в:
  - `buildClinicalTimelineModel(...)`
  - `computeTimelinePhase(...)`
  - `computeTimelinePriority(...)`
  - `compressTimelineEntries(...)`
- Визуальные фильтры и счётчик задач реализованы в:
  - `EnhancedTimeline(...)`

## 10. Roadmap
1. Перевести timeline storage из `organ_data` в отдельные таблицы `timeline_items`, `timeline_item_organs`, `organ_states`, `tasks`.
2. Добавить полноценные `dicom_refs` и PACS webhook ingestion.
3. Добавить compare engine (overlay delta heatmap).
4. Внедрить RBAC + аудит действий врача.
