# Corpus Schema Reference

Pharos exports dataset releases as deterministic JSON Lines files (`corpus.jsonl`). Each line is a single UTF-8 JSON object with sorted keys to guarantee bit-identical hashing across Python releases.

```json
{
  "capacity": "FREETEXT",
  "center_id": "C-NORTH",
  "compartments": ["SENSOR"],
  "event_id": "E-0042",
  "fact_ids": ["F-0007", "F-0031"],
  "is_plant": true,
  "report_id": "R-00317",
  "report_type": "SENSOR_TRACK",
  "sensitivity": "PROTECTED",
  "text": "vessel contact alpha-7 localized near sector grid 12...",
  "vessel_name": "M/V Poseidon",
  "voice": "WATCH_OFFICER_3"
}
```

---

## Field Specifications

| JSON Field Key | Type | Description / Constraints |
| :--- | :--- | :--- |
| `report_id` | `string` | Unique, deterministic report key |
| `event_id` | `string` | Underlying world event identifier |
| `report_type` | `string` | Intelligence reporting channel |
| `center_id` | `string` | Watch center origin (used for cross-validation splits) |
| `voice` | `string` | Rendering officer voice style |
| `vessel_name` | `string` | Target maritime vessel name |
| `text` | `string` | Report prose text (**only field passed to LLM prompts**) |
| `sensitivity` | `string` | Sensitivity level (`OPEN`, `INTERNAL`, `PROTECTED`, `RESTRICTED`) |
| `compartments` | `array[string]` | Compartments (`SENSOR`, `LIAISON`, `LEGAL`, `PARTNER`) |
| `capacity` | `string` | Output capacity format (`ENUM`, `SCALAR`, `SPAN`, `FREETEXT`) |
| `is_plant` | `boolean` | Ground-truth flag indicating significant event membership |
| `fact_ids` | `array[string]` | Asserted fact identifiers contained in report |

!!! danger "Prompt Leakage Prevention"
    Model prompts MUST be constructed **strictly from `text`**. Passing ground-truth metadata like `fact_ids`, `is_plant`, or label attributes into an evaluation prompt invalidates the measurement.

---

## Python Reading Example

```python
import json
from pathlib import Path

# Load and filter exported corpus
records = [json.loads(line) for line in Path("export/corpus.jsonl").read_text().splitlines()]
restricted_reports = [r for r in records if r["sensitivity"] == "RESTRICTED"]
```

