"""
Print side-by-side sample predictions from two result JSONL files
produced by predict.py / predict_external_whisper_v2.py.

Usage:
    python scripts\\compare_samples.py --a results\\test\\XD_FD.jsonl --b results\\test\\ext_typhoon_whisper.jsonl --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jiwer import cer

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--a", required=True, help="First results jsonl")
parser.add_argument("--b", required=True, help="Second results jsonl")
parser.add_argument("--a-name", default="wav2vec2 (XD_FD)")
parser.add_argument("--b-name", default="typhoon-whisper")
parser.add_argument("--n", type=int, default=3)
args = parser.parse_args()


def load(path):
    out = {}
    with (ROOT / path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row
    return out


a_rows = load(args.a)
b_rows = load(args.b)

shared = [rid for rid in a_rows if rid in b_rows]
print(f"Shared examples: {len(shared)}\n")

for n, rid in enumerate(shared[: args.n]):
    ra = a_rows[rid]
    rb = b_rows[rid]
    ref = ra["reference"]

    print("=" * 70)
    print(f"[{n + 1}] id: {rid}   duration: {ra.get('duration', '?'):.2f}s")
    print("=" * 70)
    print(f"  reference (thai) : {ref}")
    print(f"  isan reference   : {ra.get('isan_reference', '')}")
    print()
    print(f"  {args.a_name}")
    print(f"    prediction     : {ra['prediction']}")
    print(f"    CER            : {cer(ref, ra['prediction']):.4f}")
    print()
    print(f"  {args.b_name}")
    print(f"    prediction     : {rb['prediction']}")
    print(f"    CER            : {cer(ref, rb['prediction']):.4f}")
    print()