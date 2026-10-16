from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


# =========================================================
# Project constants
# =========================================================

DATASET_ID = "typhoon-ai/thai-dialect-isan-dataset"

MODELS = {
    "wav2vec2": "airesearch/wav2vec2-large-xlsr-53-th",
    "normalizer": "google/byt5-small",
}

PREFIX = "isan-wav2vec2"

MAX_AUDIO_SECONDS = 20.0

SAMPLE_RATE = 16_000


# Characters that wav2vec2 CTC tokenizer cannot represent.
# This is intentionally conservative.
CTC_UNKNOWN_CHARS = {
    "฿",
}


# =========================================================
# Text normalization
# =========================================================

def normalize_text(text: Any) -> str:
    """
    Normalize Thai/Isan text for training and evaluation.

    Steps:
      1. Convert to string
      2. Unicode NFC
      3. Lowercase
      4. Normalize whitespace
      5. Remove common punctuation/control symbols
    """

    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Replace newlines/tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Keep Thai characters, Latin characters, digits,
    # spaces and basic combining marks.
    text = re.sub(
        r"[^\u0E00-\u0E7Fa-zA-Z0-9ก-๙\s]",
        " ",
        text,
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# Validation helpers
# =========================================================

def contains_ctc_unknown_character(text: str) -> bool:
    """
    Return True if text contains characters that should
    not be passed to the wav2vec2 CTC tokenizer.
    """

    if not text:
        return False

    return any(ch in CTC_UNKNOWN_CHARS for ch in text)


def valid_target(text: Any) -> bool:
    """
    Basic target validation.
    """

    if text is None:
        return False

    return bool(str(text).strip())


# =========================================================
# Hashing
# =========================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# =========================================================
# JSON helpers
# =========================================================

def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def read_json(path: str | Path) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


# =========================================================
# Fingerprints
# =========================================================

def fingerprint_rows(rows: list[dict]) -> str:
    """
    Deterministic fingerprint of manifest contents.

    Only stable identifying fields are included.
    """

    normalized = []

    for row in rows:
        normalized.append(
            {
                "id": row.get("id"),
                "audio_sha256": row.get("audio_sha256"),
                "speaker": row.get("speaker"),
                "source_split": row.get("source_split"),
                "raw": row.get("raw"),
                "thai_spelling": row.get("thai_spelling"),
                "isan_spelling": row.get("isan_spelling"),
            }
        )

    normalized.sort(
        key=lambda x: (
            str(x.get("id")),
            str(x.get("audio_sha256")),
        )
    )

    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return sha256_bytes(payload)


def fingerprint_file(path: str | Path) -> str:
    return sha256_file(path)


# =========================================================
# Speaker extraction
# =========================================================

def infer_speaker_from_filename(filename: str) -> str:
    """
    Infer speaker ID from the Typhoon filename convention.

    Example:
        opentyphoon;is;x_061;gen;0049.wav

    The third semicolon-separated field is treated as
    the speaker identifier.

    This follows the project guide's convention.
    """

    name = Path(filename).name

    parts = name.split(";")

    if len(parts) >= 3:
        return parts[2]

    # Fallback: filename stem
    return Path(filename).stem


# =========================================================
# Misc
# =========================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path