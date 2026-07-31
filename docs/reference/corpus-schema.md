# Corpus schema

One JSON object per line, keys sorted, UTF-8. Sorted keys are not cosmetic: an
unordered dict would hash differently between interpreter versions and quietly
break the reproducibility property the export exists to provide.

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
  "text": "...",
  "vessel_name": "...",
  "voice": "..."
}
```

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `report_id` | string | Stable identifier for one rendered report. The record set key |
| `event_id` | string | The world event described. Many reports share one event |
| `report_type` | string | The channel that carried it, which is what confers its label |
| `center_id` | string | The watch centre that filed it. The gate holds whole centres out |
| `voice` | string | The officer voice that rendered it, one of a fixed set |
| `vessel_name` | string | The fictional vessel concerned |
| `text` | string | The report body. The only field a specialist is meant to read |
| `sensitivity` | string | `OPEN`, `INTERNAL`, `PROTECTED`, or `RESTRICTED` |
| `compartments` | array of string | Need-to-know compartments, a subset lattice |
| `capacity` | string | `ENUM`, `SCALAR`, `SPAN`, or `FREETEXT` |
| `is_plant` | boolean | Whether the report belongs to a significant event |
| `fact_ids` | array of string | The fact identifiers this rendering asserts |

## Why the label is split across three columns

The label is unpacked into `sensitivity`, `compartments`, and `capacity` rather
than stringified as `PROTECTED[SENSOR]`. A consumer filtering on level or
compartment should not have to parse a rendering back apart, and a parser is a
place for a bug that silently changes what an experiment measured.

## Reading it

```python
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("export/corpus.jsonl").read_text().splitlines()]
restricted = [r for r in rows if r["sensitivity"] == "RESTRICTED"]
```

The column order in `pharos.export.CORPUS_FIELDS` is fixed, and the Croissant
record set is generated from that same tuple, so the file and its metadata cannot
disagree about which columns exist. A test asserts that they describe the same set.

## What `is_plant` means, and what it does not

`is_plant` marks a report belonging to a **significant** event, where significance
is defined by the event carrying a fixed conjunction of facts. It is not an
annotation and there was no annotator: it is definitional, assigned at generation
time.

That definitional quality is why the corpus has a surface baseline above chance.
See [the gate](gate.md).

## Caveat on `text`

`text` is the only field a specialist should be shown. Passing `fact_ids`,
`is_plant`, or the label columns into a prompt leaks the answer, and the resulting
number measures nothing. The task builders in `pharos.tasks` construct prompts
from `text` alone.
