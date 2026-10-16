from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from datasets import Audio, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# Allow:
#     python scripts\prepare.py
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (
    DATASET_ID,
    MODELS,
    MAX_AUDIO_SECONDS,
    SAMPLE_RATE,
    contains_ctc_unknown_character,
    fingerprint_rows,
    infer_speaker_from_filename,
    normalize_text,
    valid_target,
    write_json,
    write_jsonl,
)


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--out",
    type=str,
    default="data/official",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

parser.add_argument(
    "--max-seconds",
    type=float,
    default=MAX_AUDIO_SECONDS,
)

args = parser.parse_args()


out = ROOT / args.out

if out.exists():
    existing = list(out.iterdir())

    if existing:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out}\n"
            f"Please remove it or choose another --out directory."
        )

out.mkdir(
    parents=True,
    exist_ok=True,
)

audio_out = out / "audio"
audio_out.mkdir(parents=True, exist_ok=True)

# =========================================================
# Lock dataset/model revisions
# =========================================================

api = HfApi()

dataset_info = api.dataset_info(DATASET_ID)

dataset_revision = (
    getattr(dataset_info, "sha", None)
    or "main"
)

lock = {
    "dataset": {
        "id": DATASET_ID,
        "revision": dataset_revision,
    },
    "models": MODELS,
}

write_json(
    out / "sources.lock.json",
    lock,
)

# Also store root-level lock for consistency with train.py
write_json(
    ROOT / "sources.lock.json",
    lock,
)


# =========================================================
# Load official dataset
# =========================================================

print()
print("=" * 70)
print("Loading dataset")
print("=" * 70)
print(f"Dataset : {DATASET_ID}")
print(f"Revision: {dataset_revision}")
print()


raw = load_dataset(
    DATASET_ID,
    revision=dataset_revision,
)


print("Official splits:")

for split_name, ds in raw.items():
    print(
        f"  {split_name}: {len(ds)} examples"
    )

print()


# =========================================================
# Audio preparation
# =========================================================

for split_name in raw:
    raw[split_name] = raw[split_name].cast_column(
        "audio",
        Audio(
            sampling_rate=SAMPLE_RATE
        ),
    )


# =========================================================
# Process rows
# =========================================================

accepted = []
excluded = []


def reject(
    row_id: str,
    source_split: str,
    reason: str,
):
    excluded.append(
        {
            "id": row_id,
            "source_split": source_split,
            "reason": reason,
        }
    )


for original_split, ds in raw.items():

    print(
        f"Processing official split: {original_split}"
    )

    for idx, row in enumerate(ds):

        row_id = str(
            row.get(
                "id",
                f"{original_split}_{idx}"
            )
        )

        audio = row.get("audio")

        if audio is None:
            reject(
                row_id,
                original_split,
                "missing_audio",
            )
            continue

        # -------------------------------------------------
        # Targets
        # -------------------------------------------------

        thai = normalize_text(
            row.get("thai_spelling", "")
        )

        isan = normalize_text(
            row.get("isan_spelling", "")
        )

        if not valid_target(thai):
            reject(
                row_id,
                original_split,
                "empty_thai_target",
            )
            continue

        if not valid_target(isan):
            reject(
                row_id,
                original_split,
                "empty_isan_target",
            )
            continue

        # -------------------------------------------------
        # CTC character check
        # -------------------------------------------------

        if contains_ctc_unknown_character(thai):
            reject(
                row_id,
                original_split,
                "ctc_unknown_character_thai",
            )
            continue

        if contains_ctc_unknown_character(isan):
            reject(
                row_id,
                original_split,
                "ctc_unknown_character_isan",
            )
            continue

        # -------------------------------------------------
        # Audio
        # -------------------------------------------------

        try:
            array = audio["array"]

            sampling_rate = int(
                audio["sampling_rate"]
            )

            duration = (
                len(array) / sampling_rate
            )

        except Exception:
            reject(
                row_id,
                original_split,
                "audio_decode_error",
            )
            continue

        if duration < 1.0:
            reject(
                row_id,
                original_split,
                "duration_too_short",
            )
            continue

        if duration > args.max_seconds:
            reject(
                row_id,
                original_split,
                "duration_too_long",
            )
            continue

        # -------------------------------------------------
        # Speaker
        # -------------------------------------------------

        name = str(
            row.get(
                "name",
                row_id
            )
        )

        speaker = infer_speaker_from_filename(
            name
        )

        sf.write(
            str(audio_out / name),
            array,
            sampling_rate,
        )

        # -------------------------------------------------
        # Audio fingerprint
        #
        # We hash the decoded PCM bytes so duplicate
        # recordings can be detected.
        # -------------------------------------------------

        pcm = np.asarray(
            array,
            dtype=np.float32,
        )

        audio_sha256 = __import__(
            "hashlib"
        ).sha256(
            pcm.tobytes()
        ).hexdigest()

        # -------------------------------------------------
        # Store accepted record
        # -------------------------------------------------

        accepted.append(
            {
                "id": row_id,

                "audio_sha256": audio_sha256,

                "source_split": original_split,

                "speaker": speaker,

                "name": name,

                "district": row.get(
                    "district"
                ),

                "province": row.get(
                    "province"
                ),

                "age": row.get(
                    "age"
                ),

                "gender": row.get(
                    "gender"
                ),

                "question_id": row.get(
                    "question_id"
                ),

                "question": row.get(
                    "question"
                ),

                "duration": float(
                    duration
                ),

                "raw": normalize_text(
                    row.get("raw", "")
                ),

                "thai_spelling": thai,

                "isan_spelling": isan,
            }
        )


# =========================================================
# Duplicate detection
# =========================================================

seen_audio = set()
unique_accepted = []

for row in accepted:

    audio_hash = row["audio_sha256"]

    if audio_hash in seen_audio:

        excluded.append(
            {
                "id": row["id"],
                "source_split": row[
                    "source_split"
                ],
                "reason": "duplicate_audio",
            }
        )

        continue

    seen_audio.add(audio_hash)

    unique_accepted.append(row)


accepted = unique_accepted


# =========================================================
# Separate official splits
# =========================================================

source_rows = {
    "train": [
        r for r in accepted
        if r["source_split"] == "train"
    ],

    "test": [
        r for r in accepted
        if r["source_split"] == "test"
    ],
}


if not source_rows["train"]:
    raise ValueError(
        "No accepted examples from official train split."
    )

if not source_rows["test"]:
    raise ValueError(
        "No accepted examples from official test split."
    )


# =========================================================
# Official Typhoon split protocol
#
# IMPORTANT:
#   - Original Typhoon train remains train/validation
#   - Original Typhoon test remains FINAL TEST
#   - We NEVER combine original train and test.
#
# Because train.py requires validation.jsonl,
# ONLY official train is divided:
#
#   90% -> train
#   10% -> validation
#
# Official test is untouched.
# =========================================================

train_source = source_rows["train"]

groups = np.array(
    [
        r["speaker"]
        for r in train_source
    ]
)

indices = np.arange(
    len(train_source)
)


train_idx, val_idx = next(
    GroupShuffleSplit(
        n_splits=1,
        test_size=0.10,
        random_state=args.seed,
    ).split(
        indices,
        groups=groups,
    )
)


split_rows = {
    "train": [
        train_source[int(i)]
        for i in train_idx
    ],

    "validation": [
        train_source[int(i)]
        for i in val_idx
    ],

    "test": source_rows["test"],
}


# =========================================================
# Deterministic shuffle
# =========================================================

rng = np.random.default_rng(
    args.seed
)

for split_name in split_rows:

    rng.shuffle(
        split_rows[split_name]
    )

    write_jsonl(
        out / f"{split_name}.jsonl",
        split_rows[split_name],
    )


# =========================================================
# Safety checks
# =========================================================

# ---------------------------------------------------------
# Audio must never overlap
# ---------------------------------------------------------

for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:

    audio_a = {
        r["audio_sha256"]
        for r in split_rows[a]
    }

    audio_b = {
        r["audio_sha256"]
        for r in split_rows[b]
    }

    overlap = audio_a & audio_b

    assert not overlap, (
        f"Audio overlap detected between "
        f"{a} and {b}: {len(overlap)}"
    )


# ---------------------------------------------------------
# Train/validation speaker separation
# ---------------------------------------------------------

train_speakers = {
    r["speaker"]
    for r in split_rows["train"]
}

val_speakers = {
    r["speaker"]
    for r in split_rows["validation"]
}

train_val_overlap = (
    train_speakers &
    val_speakers
)

assert not train_val_overlap, (
    "Speaker overlap detected between "
    f"train and validation: {train_val_overlap}"
)


# ---------------------------------------------------------
# Official train/test speaker overlap
#
# IMPORTANT:
# We DO NOT assert zero overlap here.
# The official dataset split may contain the
# same speaker in both official splits.
# ---------------------------------------------------------

test_speakers = {
    r["speaker"]
    for r in split_rows["test"]
}

official_train_speakers = {
    r["speaker"]
    for r in (
        source_rows["train"]
    )
}

official_train_test_speaker_overlap = (
    official_train_speakers &
    test_speakers
)


# =========================================================
# Target overlap
# =========================================================

train_targets = {
    r["thai_spelling"]
    for r in split_rows["train"]
}

test_targets = {
    r["thai_spelling"]
    for r in split_rows["test"]
}


target_overlap = (
    train_targets &
    test_targets
)


# =========================================================
# Report
# =========================================================

report = {

    "protocol":
        "official Typhoon train/test split; "
        "official train split into train/validation only",

    "dataset_id":
        DATASET_ID,

    "dataset_revision":
        dataset_revision,

    "speaker_key_status":
        "inferred from filename; "
        "third semicolon-separated field",

    "seed":
        args.seed,

    "max_seconds":
        args.max_seconds,

    "excluded":
        dict(
            Counter(
                x["reason"]
                for x in excluded
            )
        ),

    "original_splits":
        {
            split_name: {
                "accepted":
                    len(rows),

                "speakers":
                    len({
                        r["speaker"]
                        for r in rows
                    }),

                "hours":
                    sum(
                        r["duration"]
                        for r in rows
                    ) / 3600,
            }

            for split_name, rows
            in source_rows.items()
        },

    "splits":
        {
            split_name: {
                "clips":
                    len(rows),

                "speakers":
                    len({
                        r["speaker"]
                        for r in rows
                    }),

                "hours":
                    sum(
                        r["duration"]
                        for r in rows
                    ) / 3600,
            }

            for split_name, rows
            in split_rows.items()
        },

    "official_test_preserved":
        True,

    "official_train_test_speaker_overlap":
        {
            "count":
                len(
                    official_train_test_speaker_overlap
                ),

            "speakers":
                sorted(
                    official_train_test_speaker_overlap
                ),
        },

    "train_validation_speaker_overlap":
        len(train_val_overlap),

    "exact_target_overlap_train_test":
        len(target_overlap),

    "accepted_total":
        len(accepted),

    "excluded_total":
        len(excluded),
}


# =========================================================
# Manifest fingerprints
# =========================================================

manifest_fingerprints = {}

for split_name in split_rows:

    manifest_fingerprints[split_name] = (
        fingerprint_rows(
            split_rows[split_name]
        )
    )


report[
    "manifest_fingerprints"
] = manifest_fingerprints


write_json(
    out / "report.json",
    report,
)


# =========================================================
# Print report
# =========================================================

print()
print("=" * 70)
print("PREPARE COMPLETE")
print("=" * 70)

print(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
)

print()
print(f"Output: {out}")
print()