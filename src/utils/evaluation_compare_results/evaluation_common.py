"""Shared configuration and helpers for agent-evaluation scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANNOTATION_DIR = PROJECT_ROOT / "src" / "evaluation" / "annotation_json"
RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "results"
TABLES_DIR = RESULTS_DIR / "tables"
IMAGES_DIR = RESULTS_DIR / "images"
MATRIX_DIR = PROJECT_ROOT / "src" / "evaluation" / "matrix"


@dataclass(frozen=True)
class Evaluator:
    key: str
    filename: str
    display_name: str
    kind: str


EVALUATORS = [
    Evaluator("E5", "IA_Evaluation_E5_yoloV8s.json", "E5", "ai"),
    Evaluator("E6", "IA_Evaluation_E6_yoloV8m.json", "E6", "ai"),
    Evaluator("E7", "IA_Evaluation_E7_yoloV11n.json", "E7", "ai"),
    Evaluator(
        "ModeloEspecialista",
        "IA_Evaluation_specialist_agents.json",
        "ModeloEspecialista",
        "ai",
    ),
    Evaluator(
        "Usuario_Control",
        "Control_User_Evaluation_Yasmina_Moreira.json",
        "Usuario Control",
        "human",
    ),
    Evaluator("R1_Radiologia", "R1_User_Evaluation_Marina.json", "R1 Radiologia", "human"),
    Evaluator(
        "Experto_Radiologo",
        "Catedratico_User_Evaluation_Jose_Carlos.json",
        "Experto Radiologo",
        "human",
    ),
]

EVALUATOR_BY_KEY = {e.key: e for e in EVALUATORS}

ZONE_PREFIXES = {"NoF_UR": "UR", "UR": "UR", "WRI": "WRI", "SHF": "SHF"}
ZONE_LABELS = {"UR": "Radio/cubito", "WRI": "Muneca", "SHF": "Humero"}
ZONE_ORDER = ["WRI", "UR", "SHF"]


def ensure_output_dirs() -> None:
    for directory in (RESULTS_DIR, TABLES_DIR, IMAGES_DIR, MATRIX_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_evaluator_json(evaluator: Evaluator):
    path = ANNOTATION_DIR / evaluator.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation JSON for {evaluator.display_name}: {path}")
    return load_json(path)


def normalize_image_name(item: dict) -> str:
    raw = (
        item.get("file_upload")
        or item.get("data", {}).get("image")
        or item.get("data", {}).get("Image")
        or str(item.get("id", ""))
    )
    raw = unquote(str(raw)).replace("\\", "/")
    parsed = urlparse(raw)
    path = parsed.path if parsed.scheme else raw
    return Path(path).name


def extract_boxes(data) -> dict[str, list[dict[str, float]]]:
    result = {}
    for item in data:
        image_name = normalize_image_name(item)
        boxes = []
        for annotation in item.get("annotations", []):
            for output in annotation.get("result", []):
                value = output.get("value", {})
                if not {"x", "y", "width", "height"}.issubset(value):
                    continue

                original_width = output.get("original_width") or 1
                original_height = output.get("original_height") or 1
                boxes.append(
                    {
                        "x": value["x"] / 100 * original_width,
                        "y": value["y"] / 100 * original_height,
                        "w": value["width"] / 100 * original_width,
                        "h": value["height"] / 100 * original_height,
                    }
                )
        result[image_name] = boxes
    return result


def iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y2 = min(box1["y"] + box1["h"], box2["y"] + box2["h"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def best_iou_match(boxes1: list[dict[str, float]], boxes2: list[dict[str, float]]) -> float | None:
    if not boxes1 and not boxes2:
        return None
    if not boxes1 or not boxes2:
        return 0.0

    total = 0.0
    used = set()
    for box1 in boxes1:
        best_value = 0.0
        best_index = -1
        for index, box2 in enumerate(boxes2):
            if index in used:
                continue
            value = iou(box1, box2)
            if value > best_value:
                best_value = value
                best_index = index
        if best_index >= 0:
            total += best_value
            used.add(best_index)

    return total / max(len(boxes1), len(boxes2))


def optimal_iou_match(boxes1: list[dict[str, float]], boxes2: list[dict[str, float]]) -> float | None:
    """Calcula un matching IoU óptimo y simétrico entre dos listas de cajas."""
    if not boxes1 and not boxes2:
        return None
    if not boxes1 or not boxes2:
        return 0.0

    if len(boxes1) <= len(boxes2):
        smaller, larger = boxes1, boxes2
    else:
        smaller, larger = boxes2, boxes1

    iou_matrix = [
        [iou(small_box, large_box) for large_box in larger]
        for small_box in smaller
    ]
    memo = {}

    def search(row: int, used_mask: int) -> float:
        if row == len(smaller):
            return 0.0
        key = (row, used_mask)
        if key in memo:
            return memo[key]

        best = 0.0
        for col in range(len(larger)):
            if used_mask & (1 << col):
                continue
            candidate = iou_matrix[row][col] + search(row + 1, used_mask | (1 << col))
            if candidate > best:
                best = candidate

        memo[key] = best
        return best

    return search(0, 0) / max(len(boxes1), len(boxes2))


def is_ai_pair(key1: str, key2: str) -> bool:
    return EVALUATOR_BY_KEY[key1].kind == "ai" and EVALUATOR_BY_KEY[key2].kind == "ai"


def valid_pair(key1: str, key2: str) -> bool:
    return key1 == key2 or not is_ai_pair(key1, key2)


def pair_order() -> list[tuple[str, str]]:
    humans = [e.key for e in EVALUATORS if e.kind == "human"]
    ais = [e.key for e in EVALUATORS if e.kind == "ai"]
    expert_order = ["Experto_Radiologo", "R1_Radiologia", "Usuario_Control"]

    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(expert_order):
        for second in expert_order[index + 1 :]:
            pairs.append((first, second))
        for ai_key in ais:
            pairs.append((first, ai_key))

    return [(a, b) for a, b in pairs if a in humans and b in humans + ais]


def display_name(key: str) -> str:
    return EVALUATOR_BY_KEY[key].display_name


def pair_label(key1: str, key2: str) -> str:
    return f"{display_name(key1)} vs {display_name(key2)}"


def extract_zone(filename: str) -> str:
    for prefix, zone in ZONE_PREFIXES.items():
        if filename.startswith(prefix):
            return zone
    return "OTHER"
