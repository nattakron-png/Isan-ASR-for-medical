"""
Benchmark an external HF ASR model (default: typhoon-ai/typhoon-isan-asr-whisper)
on the same official test split used by predict.py, so results are directly
comparable to XD_PD / XD_FD.

v2: reads local WAV files from data/<...>/audio/ one at a time instead of
loading the entire HF dataset into a RAM lookup dict. The v1 approach held
~6 GB of decoded audio in memory, which combined with the 3 GB Whisper-medium
weights caused the process to be killed silently by Windows.

CAVEAT: typhoon-isan-asr-whisper was fine-tuned by the Typhoon team, likely
on (a version of) the same typhoon-ai/thai-dialect-isan-dataset. If its
training data overlaps with this test split, its score here is optimistic
relative to XD_full, which never saw this test split. Treat this as a
reference number, not a strictly fair comparison, unless you've confirmed
their train/test split from the model card.

Usage:
    python scripts\\predict_external_whisper_v2.py --data data\\official --split test --out results\\test\\ext_typhoon_whisper
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from jiwer import cer, wer
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import fingerprint_rows, read_jsonl, write_json, write_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/official")
parser.add_argument("--split", default="test")
parser.add_argument("--out", required=True)
parser.add_argument("--model", default="typhoon-ai/typhoon-isan-asr-whisper")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--fp32", action="store_true")
parser.add_argument("--language", default="th")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

data_dir = ROOT / args.data
audio_dir = data_dir / "audio"
out_path = ROOT / args.out
out_path.parent.mkdir(parents=True, exist_ok=True)

rows = read_jsonl(data_dir / f"{args.split}.jsonl")
manifest_fingerprint = fingerprint_rows(rows)

if args.limit is not None:
    rows = rows[: args.limit]

print()
print("=" * 70)
print("EXTERNAL MODEL PREDICTION")
print("=" * 70)
print(f"Model      : {args.model}")
print(f"Split      : {args.split}")
print(f"Examples   : {len(rows)}")
print(f"Audio dir  : {audio_dir}")
print()

if not audio_dir.is_dir():
    raise FileNotFoundError(
        f"Audio directory not found: {audio_dir}\n"
        "Run scripts/prepare.py first so local WAV files exist."
    )

print("Loading pipeline...", flush=True)

pipe = pipeline(
    "automatic-speech-recognition",
    model=args.model,
    device=0,
    dtype=torch.float32 if args.fp32 else torch.float16,
)

print("Pipeline loaded.", flush=True)

results = []
start = time.time()

for i, manifest_row in enumerate(rows):
    row_id = manifest_row["id"]
    wav_path = audio_dir / manifest_row["name"]

    if not wav_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {wav_path}\n"
            f"Manifest ID: {row_id}"
        )

    audio_array, sampling_rate = sf.read(str(wav_path), dtype="float32")

    output = pipe(
        {"array": audio_array, "sampling_rate": sampling_rate},
        generate_kwargs={"language": args.language},
    )

    prediction = output["text"].strip()

    if i == 0:
        print(f"First prediction OK: {prediction[:40]!r}", flush=True)

    results.append(
        {
            "id": row_id,
            "reference": manifest_row["thai_spelling"],
            "isan_reference": manifest_row["isan_spelling"],
            "prediction": prediction,
            "duration": manifest_row["duration"],
        }
    )

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(rows)}", flush=True)

runtime = time.time() - start

references = [r["reference"] for r in results]
predictions = [r["prediction"] for r in results]

metrics = {
    "model": args.model,
    "split": args.split,
    "examples": len(results),
    "cer": cer(references, predictions),
    "wer": wer(references, predictions),
    "runtime_seconds": runtime,
    "rtf": runtime / max(sum(r["duration"] for r in results), 1e-9),
    "manifest_fingerprint": manifest_fingerprint,
}

result_jsonl = out_path.with_suffix(".jsonl")
metrics_path = out_path.with_suffix(".metrics.json")

write_jsonl(result_jsonl, results)
write_json(metrics_path, metrics)

print()
print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print()
print(f"Predictions: {result_jsonl}")
print(f"Metrics    : {metrics_path}")