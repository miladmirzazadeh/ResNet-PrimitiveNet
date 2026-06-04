"""ArchCAD-400K semantic class schema (30 classes + Others), with robust
string->id normalization.

ArchCAD stores each primitive's `semantic` as a NAME (e.g. "single_door",
"wall"). The official id table (from the dataset card) skips from 29 to 100
("Others"); we remap 100 -> 30 so labels are contiguous [0..30] = 31 classes.

Optionally we add `duplicate` = 31 for the synthetic "agent should delete this"
flaw class (ArchCAD has no such class).

>>> VERIFY the exact `semantic` strings with scripts/kaggle_inspect.py output and
adjust ALIASES if any name differs. The mapping is intentionally forgiving
(unknown -> Others) so an unseen name never crashes training.
"""
from __future__ import annotations
import re

# id -> canonical name (ArchCAD card; 100 remapped to 30)
ID2NAME = {
    0: "axis_grid", 1: "single_door", 2: "double_door", 3: "parent_child_door",
    4: "other_door", 5: "elevator", 6: "staircase", 7: "sink", 8: "urinal",
    9: "toilet", 10: "bathtub", 11: "squat_toilet", 12: "other_fixtures",
    13: "drain", 14: "table", 15: "chair", 16: "bed", 17: "sofa", 18: "hole",
    19: "glass", 20: "wall", 21: "concrete_column", 22: "steel_column",
    23: "concrete_beam", 24: "steel_beam", 25: "parking_space", 26: "foundation",
    27: "pile", 28: "rebar", 29: "fire_hydrant", 30: "others",
}
NAME2ID = {v: k for k, v in ID2NAME.items()}
RAW_OTHERS_ID = 100          # ArchCAD's literal "Others" id
OTHERS_ID = 30
DUPLICATE_ID = 31            # synthetic-only extra class
NUM_CLASSES = 31             # set to 32 if you train with the duplicate class

# countable (instance-able) classes — informational, used by the grouping step
COUNTABLE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 25, 27, 29}

# tolerant aliases: normalized string -> id. Extend after Kaggle inspection.
ALIASES = {
    "axis": 0, "grid": 0, "axis_and_grid": 0, "axisgrid": 0,
    "parentchild_door": 3, "parent_child": 3,
    "stair": 6, "stairs": 6, "staircase": 6,
    "squat": 11, "fixture": 12, "fixtures": 12,
    "column_concrete": 21, "column_steel": 22,
    "beam_concrete": 23, "beam_steel": 24,
    "hydrant": 29, "fire_hydrant": 29,
    "other": 30, "misc": 30,
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def semantic_to_id(value) -> int:
    """Map ArchCAD's `semantic` (string name OR integer id) -> contiguous class id."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = int(value)
        return OTHERS_ID if v == RAW_OTHERS_ID else (v if 0 <= v <= 29 else OTHERS_ID)
    key = _norm(value)
    if key in NAME2ID:
        return NAME2ID[key]
    if key in ALIASES:
        return ALIASES[key]
    if key.isdigit():
        v = int(key)
        return OTHERS_ID if v == RAW_OTHERS_ID else (v if 0 <= v <= 29 else OTHERS_ID)
    return OTHERS_ID
