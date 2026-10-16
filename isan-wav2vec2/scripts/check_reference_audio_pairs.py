#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review possible audio/reference mismatches in a Thai/Isan wav2vec2 test set.

This script DOES NOT delete rows or change the official test set.
It reads the existing audio_review.json produced by review_xd_direct_audio.py,
ranks items for manual listening, and writes a review CSV/JSON.

Run from the project root:
    python scripts\check_reference_audio_pairs.py
Optional:
    python scripts\check_reference_audio_pairs.py --top 60
    python scripts\check_reference_audio_pairs.py --open-first
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any


def first_value(obj: dict, *keys: str, default=None):
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_float(value: Any):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("items", "results", "examples", "records", "data", "audio_review"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        # A dict keyed by example ID
        if data and all(isinstance(v, dict) for v in data.values()):
            records = []
            for key, value in data.items():
                row = dict(value)
                row.setdefault("id", key)
                records.append(row)
            return records
    return []


def normalize_row(row: dict) -> dict:
    ref = norm_text(first_value(
        row, "reference", "ref", "ground_truth", "target", "thai_spelling",
        "reference_text", "original_thai_spelling", default=""
    ))
    pred = norm_text(first_value(
        row, "prediction", "pred", "hypothesis", "transcription", "predicted_text",
        "text", default=""
    ))
    wav = first_value(row, "wav_path", "audio_path", "path", "audio", default="")
    if isinstance(wav, dict):
        wav = first_value(wav, "path", "file", "wav_path", default="")
    id_ = str(first_value(row, "id", "utterance_id", "sample_id", default=""))
    return {
        "id": id_,
        "reference": ref,
        "prediction": pred,
        "wav_path": str(wav or ""),
        "speaker": str(first_value(row, "speaker", "speaker_id", default="")),
        "duration": to_float(first_value(row, "duration", "duration_seconds", default=None)),
        "cer": to_float(first_value(row, "cer", "CER", "cer_percent", default=None)),
        "wer_newmm": to_float(first_value(row, "wer_newmm", "wer_newmm_percent", "WER newmm", "wer", default=None)),
        "source": row,
    }


def mismatch_review_score(row: dict) -> tuple[float, list[str]]:
    """
    Heuristic for prioritizing manual review only, not automatic mismatch detection.
    High text divergence can be either a model error or a wrong reference/audio pair.
    """
    ref = row["reference"]
    pred = row["prediction"]
    reasons = []
    score = 0.0

    cer = row.get("cer")
    wer = row.get("wer_newmm")
    # Metrics in review_xd_direct_audio.py are usually fractions, but tolerate percentages.
    if cer is not None:
        cer_fraction = cer / 100 if cer > 1 else cer
        if cer_fraction >= 0.70:
            score += 5
            reasons.append("CER สูงมาก (>=70%)")
        elif cer_fraction >= 0.40:
            score += 3
            reasons.append("CER สูง (>=40%)")
        elif cer_fraction >= 0.25:
            score += 1.5
            reasons.append("CER ค่อนข้างสูง (>=25%)")

    if wer is not None:
        wer_fraction = wer / 100 if wer > 1 else wer
        if wer_fraction >= 0.90:
            score += 3
            reasons.append("WER newmm สูงมาก (>=90%)")
        elif wer_fraction >= 0.65:
            score += 1.5
            reasons.append("WER newmm สูง (>=65%)")

    if not ref:
        score += 5
        reasons.append("Reference ว่าง")
    if not pred:
        score += 2
        reasons.append("Prediction ว่าง")
    if "[UNK]" in pred.upper():
        score += 1
        reasons.append("Prediction มี [UNK]")

    # Very different text lengths are a useful review signal, but not proof of mismatch.
    if ref and pred:
        ratio = max(len(ref), len(pred)) / max(1, min(len(ref), len(pred)))
        if ratio >= 2.5 and abs(len(ref) - len(pred)) >= 20:
            score += 2
            reasons.append("ความยาว Reference/Prediction ต่างกันมาก")

    if not reasons:
        reasons.append("ตรวจตามลำดับคะแนนปกติ")
    return score, reasons


def open_audio(path: str):
    p = Path(path)
    if not p.exists():
        print(f"ไม่พบไฟล์เสียง: {p}")
        return
    try:
        if os.name == "nt":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as exc:
        print(f"เปิดไฟล์เสียงไม่สำเร็จ: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Rank suspicious audio/reference pairs for manual review.")
    parser.add_argument("--input", default="results/test/XD_FD_audio_review/audio_review.json",
                        help="Path to audio_review.json")
    parser.add_argument("--outdir", default="results/test/XD_FD_pair_review",
                        help="Output directory")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of ranked examples to print (default: 50)")
    parser.add_argument("--open-first", action="store_true",
                        help="Open the first ranked existing WAV")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ไม่พบไฟล์: {input_path}")
        print("ให้รัน scripts\\review_xd_direct_audio.py ก่อน แล้วค่อยรันสคริปต์นี้")
        sys.exit(1)

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"อ่าน JSON ไม่สำเร็จ: {exc}")
        sys.exit(1)

    raw_records = get_records(data)
    if not raw_records:
        print("ไม่พบรายการตัวอย่างใน JSON")
        print("เปิด JSON ดูโครงสร้างข้อมูล แล้วส่งข้อความ error หรือส่วนต้นของไฟล์มาให้ตรวจได้")
        sys.exit(1)

    rows = []
    for raw in raw_records:
        row = normalize_row(raw)
        score, reasons = mismatch_review_score(row)
        row["review_score"] = score
        row["review_reasons"] = "; ".join(reasons)
        # Make relative paths relative to project root (current working directory).
        wav = Path(row["wav_path"]) if row["wav_path"] else None
        if wav and not wav.is_absolute():
            wav = Path.cwd() / wav
        row["wav_exists"] = bool(wav and wav.exists())
        row["wav_path_resolved"] = str(wav) if wav else ""
        rows.append(row)

    # Prioritize heuristic flags, then larger CER and WER.
    def sort_key(r):
        cer = r["cer"] if r["cer"] is not None else -1
        wer = r["wer_newmm"] if r["wer_newmm"] is not None else -1
        # tolerate percentages
        return (r["review_score"], cer / 100 if cer > 1 else cer, wer / 100 if wer > 1 else wer)

    rows.sort(key=sort_key, reverse=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "pair_review_candidates.json"
    csv_path = outdir / "pair_review_candidates.csv"

    serializable = []
    for rank, row in enumerate(rows, start=1):
        serializable.append({
            "rank": rank,
            "id": row["id"],
            "review_score": row["review_score"],
            "review_reasons": row["review_reasons"],
            "cer": row["cer"],
            "wer_newmm": row["wer_newmm"],
            "duration": row["duration"],
            "speaker": row["speaker"],
            "reference": row["reference"],
            "prediction": row["prediction"],
            "wav_path": row["wav_path_resolved"],
            "wav_exists": row["wav_exists"],
        })

    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "rank", "id", "review_score", "review_reasons", "cer", "wer_newmm",
        "duration", "speaker", "reference", "prediction", "wav_path", "wav_exists"
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable)

    print(f"จำนวนตัวอย่างที่อ่านได้: {len(rows)}")
    print(f"พบไฟล์เสียง: {sum(1 for r in rows if r['wav_exists'])}/{len(rows)}")
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV : {csv_path.resolve()}")
    print()
    print("ตัวอย่างที่ควรฟังตรวจสอบก่อน (คะแนนนี้ใช้จัดลำดับเท่านั้น ไม่ได้ยืนยันว่าข้อมูลผิด)")
    print("=" * 78)

    shown = rows[:max(0, args.top)]
    for rank, row in enumerate(shown, start=1):
        print(f"\nอันดับ {rank} | ID: {row['id']} | review score: {row['review_score']:.1f}")
        print(f"สาเหตุที่จัดเข้ารายการ: {row['review_reasons']}")
        print(f"CER: {row['cer']} | WER newmm: {row['wer_newmm']} | ระยะเวลา: {row['duration']}")
        print(f"REFERENCE : {row['reference']}")
        print(f"PREDICTION: {row['prediction']}")
        print(f"WAV       : {row['wav_path_resolved']}")
        print(f"พบไฟล์เสียง: {'ใช่' if row['wav_exists'] else 'ไม่พบ'}")

    print("\nข้อควรจำ:")
    print("- สคริปต์นี้จัดลำดับรายการที่ควรฟัง ไม่ได้ตัดสินอัตโนมัติว่า Reference ผิด")
    print("- อย่าลบตัวอย่างออกจาก test set เดิมจากคะแนนผิดพลาดเพียงอย่างเดียว")
    print("- หากพบว่าเสียงกับ Reference ไม่ตรงจริง ให้จด ID และหลักฐานไว้ แล้วรายงานผลทั้งชุดเดิมและชุดที่ตรวจสอบแล้วแยกกัน")

    if args.open_first and rows:
        print("\nกำลังเปิดเสียงอันดับ 1...")
        open_audio(rows[0]["wav_path_resolved"])


if __name__ == "__main__":
    main()
