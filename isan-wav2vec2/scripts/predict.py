from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset
from jiwer import cer, wer
from transformers import (
    AutoModelForCTC,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (
    fingerprint_rows,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--kind",
    required=True,
    choices=[
        "wav2vec2",
        "normalizer",
    ],
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
)

parser.add_argument(
    "--data",
    default="data/official",
)

parser.add_argument(
    "--split",
    choices=[
        "validation",
        "test",
    ],
    default="test",
)

parser.add_argument(
    "--out",
    required=True,
)

parser.add_argument(
    "--input",
    type=str,
    default=None,
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
)

parser.add_argument(
    "--fp32",
    action="store_true",
)

args = parser.parse_args()


# =========================================================
# CUDA
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required."
    )

device = torch.device("cuda")


# =========================================================
# Paths
# =========================================================

data_dir = ROOT / args.data
out_path = ROOT / args.out

out_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Normalizer input validation
# =========================================================

if args.kind == "normalizer":

    if not args.checkpoint:
        raise ValueError(
            "Normalizer prediction requires "
            "--checkpoint."
        )

    if not args.input:
        raise ValueError(
            "Normalizer prediction requires "
            "--input JSONL."
        )


# =========================================================
# Load source lock
# =========================================================

data_lock = read_json(
    data_dir /
    "sources.lock.json"
)


# =========================================================
# Load selected manifest
# =========================================================

rows = read_jsonl(
    data_dir /
    f"{args.split}.jsonl"
)

manifest_fingerprint = fingerprint_rows(
    rows
)

if args.limit is not None:
    rows = rows[:args.limit]


print()
print("=" * 70)
print("PREDICTION")
print("=" * 70)

print(
    f"Kind       : {args.kind}"
)

print(
    f"Split      : {args.split}"
)

print(
    f"Examples   : {len(rows)}"
)

print(
    f"Checkpoint : {args.checkpoint or 'pretrained'}"
)

print()


# =========================================================
# NORMALIZER
# =========================================================

if args.kind == "normalizer":

    checkpoint = (
        ROOT /
        args.checkpoint
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint)
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(checkpoint)
    )

    model.to(device)

    model.eval()


    input_rows = read_jsonl(
        ROOT /
        args.input
    )


    # -----------------------------------------------------
    # Map references by ID
    # -----------------------------------------------------

    ref_by_id = {
        r["id"]: r
        for r in rows
    }


    results = []

    start = time.time()


    for i, pred_row in enumerate(
        input_rows
    ):

        row_id = pred_row["id"]

        if row_id not in ref_by_id:
            raise ValueError(
                f"Prediction ID {row_id} "
                f"does not exist in selected "
                f"{args.split} manifest."
            )

        source_text = pred_row[
            "prediction"
        ]

        source_text = str(
            source_text
        )


        inputs = tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }


        with torch.no_grad():

            generated = model.generate(
                **inputs,
                max_length=256,
            )


        prediction = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )


        reference = ref_by_id[
            row_id
        ]["thai_spelling"]


        results.append(
            {
                "id":
                    row_id,

                "reference":
                    reference,

                "isan_reference":
                    ref_by_id[
                        row_id
                    ]["isan_spelling"],

                "input_prediction":
                    source_text,

                "prediction":
                    prediction,

                "duration":
                    ref_by_id[
                        row_id
                    ]["duration"],
            }
        )


    runtime = time.time() - start


    references = [
        r["reference"]
        for r in results
    ]

    predictions = [
        r["prediction"]
        for r in results
    ]


    metrics = {

        "kind":
            args.kind,

        "split":
            args.split,

        "checkpoint":
            str(checkpoint),

        "examples":
            len(results),

        "cer":
            cer(
                references,
                predictions,
            ),

        "wer":
            wer(
                references,
                predictions,
            ),

        "runtime_seconds":
            runtime,

        "rtf":
            runtime /
            max(
                sum(
                    r["duration"]
                    for r in results
                ),
                1e-9,
            ),

        "manifest_fingerprint":
            manifest_fingerprint,
    }


# =========================================================
# WAV2VEC2
# =========================================================

else:

    if args.checkpoint:

        model_path = (
            ROOT /
            args.checkpoint
        )

    else:

        model_path = (
            data_lock
            .get("models", {})
            .get(
                "wav2vec2",
                "airesearch/wav2vec2-large-xlsr-53-th",
            )
        )


    processor = AutoProcessor.from_pretrained(
        str(model_path)
    )

    model = AutoModelForCTC.from_pretrained(
        str(model_path)
    )

    model.to(device)

    model.eval()


    # -----------------------------------------------------
    # Load original HF dataset audio
    # -----------------------------------------------------

    dataset_id = data_lock[
        "dataset"
    ]["id"]

    revision = data_lock[
        "dataset"
    ]["revision"]


    raw = load_dataset(
        dataset_id,
        revision=revision,
    )


    raw = raw.cast_column(
        "audio",
        Audio(
            sampling_rate=16_000
        ),
    )


    lookup = {}


    for split_name, ds in raw.items():

        for idx, row in enumerate(ds):

            row_id = str(
                row.get(
                    "id",
                    f"{split_name}_{idx}"
                )
            )

            lookup[row_id] = row


    results = []

    total_audio_seconds = 0.0

    start = time.time()


    for i, manifest_row in enumerate(rows):

        row_id = manifest_row["id"]


        if row_id not in lookup:
            raise KeyError(
                f"Audio ID not found: {row_id}"
            )


        audio = lookup[row_id][
            "audio"
        ]


        inputs = processor(
            audio["array"],
            sampling_rate=
                audio["sampling_rate"],
            return_tensors="pt",
        )


        input_values = inputs[
            "input_values"
        ].to(device)


        attention_mask = inputs.get(
            "attention_mask"
        )


        if attention_mask is not None:
            attention_mask = (
                attention_mask.to(device)
            )


        with torch.no_grad():

            output = model(
                input_values,
                attention_mask=
                    attention_mask,
            )


        predicted_ids = torch.argmax(
            output.logits,
            dim=-1,
        )


        prediction = processor.batch_decode(
            predicted_ids
        )[0]


        prediction = prediction.strip()


        results.append(
            {
                "id":
                    row_id,

                "reference":
                    manifest_row[
                        "thai_spelling"
                    ],

                "isan_reference":
                    manifest_row[
                        "isan_spelling"
                    ],

                "prediction":
                    prediction,

                "duration":
                    manifest_row[
                        "duration"
                    ],
            }
        )


        total_audio_seconds += (
            manifest_row["duration"]
        )


    runtime = time.time() - start


    references = [
        r["reference"]
        for r in results
    ]

    predictions = [
        r["prediction"]
        for r in results
    ]


    metrics = {

        "kind":
            args.kind,

        "split":
            args.split,

        "checkpoint":
            str(model_path),

        "examples":
            len(results),

        "cer":
            cer(
                references,
                predictions,
            ),

        "wer":
            wer(
                references,
                predictions,
            ),

        "runtime_seconds":
            runtime,

        "rtf":
            runtime /
            max(
                total_audio_seconds,
                1e-9,
            ),

        "manifest_fingerprint":
            manifest_fingerprint,
    }


# =========================================================
# Save results
# =========================================================

result_jsonl = (
    out_path.with_suffix(".jsonl")
)

metrics_path = (
    out_path.with_suffix(".metrics.json")
)


write_jsonl(
    result_jsonl,
    results,
)

write_json(
    metrics_path,
    metrics,
)


print()
print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print(
    json.dumps(
        metrics,
        ensure_ascii=False,
        indent=2,
    )
)

print()
print(
    f"Predictions: {result_jsonl}"
)

print(
    f"Metrics    : {metrics_path}"
)