import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON ผิดรูปแบบใน {path} บรรทัด {line_no}: {exc}")
    return rows


def resolve_audio_path(row, manifest_path, project_root):
    """Resolve common audio_path/name layouts without assuming one fixed layout."""
    candidates = []
    audio_value = row.get("audio_path")
    if audio_value:
        p = Path(str(audio_value))
        candidates.append(p if p.is_absolute() else project_root / p)
        candidates.append(p if p.is_absolute() else manifest_path.parent / p)

    name_value = row.get("name")
    if name_value:
        name = Path(str(name_value)).name
        candidates.extend([
            project_root / "data" / "official" / "audio" / name,
            project_root / "data" / "audio" / name,
            project_root / "data" / "typhoon" / "audio" / name,
            manifest_path.parent / "audio" / name,
        ])

    # Remove duplicates while preserving candidate order.
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            if candidate.is_file():
                return candidate
    return None


def open_audio(path):
    """Open the WAV in the user's default Windows audio application."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception as exc:
        print(f"เปิดไฟล์เสียงอัตโนมัติไม่สำเร็จ: {path}\nรายละเอียด: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Join XD Direct errors to the test manifest to locate WAV files. No training or inference."
    )
    parser.add_argument(
        "--predictions",
        default="results/test/XD_FD.jsonl",
        help="Existing prediction JSONL",
    )
    parser.add_argument(
        "--manifest",
        default="data/official/test.jsonl",
        help="Manifest used for the test split",
    )
    parser.add_argument(
        "--analysis",
        default="results/test/XD_FD_error_analysis/error_analysis.json",
        help="Error-analysis JSON from analyze_xd_direct_errors.py",
    )
    parser.add_argument(
        "--out-dir",
        default="results/test/XD_FD_audio_review",
        help="Output folder for review CSV/JSON",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of worst examples to print (default 30; 0 prints all)",
    )
    parser.add_argument(
        "--open-first",
        action="store_true",
        help="Open the highest-ranked example's WAV in the default audio player",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    predictions_path = (project_root / args.predictions).resolve()
    manifest_path = (project_root / args.manifest).resolve()
    analysis_path = (project_root / args.analysis).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    if not predictions_path.is_file():
        raise SystemExit(f"ไม่พบไฟล์ผลทำนาย: {predictions_path}")
    if not manifest_path.is_file():
        raise SystemExit(
            f"ไม่พบ manifest: {manifest_path}\n"
            "ตรวจสอบว่ามี data\\official\\test.jsonl หรือระบุ --manifest ให้ตรงกับโปรเจกต์"
        )

    prediction_rows = read_jsonl(predictions_path)
    manifest_rows = read_jsonl(manifest_path)

    manifest_by_id = {str(row.get("id", "")): row for row in manifest_rows}
    analysis_by_id = {}
    if analysis_path.is_file():
        try:
            analysis_rows = json.loads(analysis_path.read_text(encoding="utf-8"))
            analysis_by_id = {str(row.get("id", "")): row for row in analysis_rows}
        except (json.JSONDecodeError, TypeError):
            print(f"คำเตือน: อ่านไฟล์ analysis ไม่ได้ จะใช้ลำดับจาก predictions แทน: {analysis_path}")

    joined = []
    missing_manifest = 0
    missing_audio = 0

    # Prefer the already-ranked analysis list; predictions supply the authoritative ref/pred fields.
    if analysis_by_id:
        ordered_ids = [str(row.get("id", "")) for row in analysis_rows]
        prediction_by_id = {str(row.get("id", "")): row for row in prediction_rows}
        ordered_predictions = [
            prediction_by_id[row_id]
            for row_id in ordered_ids
            if row_id in prediction_by_id
        ]
        # Include any prediction rows not present in the analysis file.
        seen_ids = {str(row.get("id", "")) for row in ordered_predictions}
        ordered_predictions.extend(
            row for row in prediction_rows if str(row.get("id", "")) not in seen_ids
        )
    else:
        ordered_predictions = prediction_rows

    for rank, pred_row in enumerate(ordered_predictions, 1):
        row_id = str(pred_row.get("id", ""))
        manifest_row = manifest_by_id.get(row_id)
        if manifest_row is None:
            missing_manifest += 1
            manifest_row = {}

        audio_path = resolve_audio_path(manifest_row, manifest_path, project_root)
        if audio_path is None:
            missing_audio += 1

        stats = analysis_by_id.get(row_id, {})
        joined.append({
            "rank": rank,
            "id": row_id,
            "cer_percent": stats.get("cer_percent", ""),
            "wer_newmm_percent": stats.get("wer_newmm_percent", ""),
            "char_edits": stats.get("char_edits", ""),
            "word_edits": stats.get("word_edits", ""),
            "duration_seconds": manifest_row.get("duration", pred_row.get("duration", "")),
            "speaker": manifest_row.get("speaker", ""),
            "source_split": manifest_row.get("source_split", ""),
            "question_id": manifest_row.get("question_id", ""),
            "reference": pred_row.get("reference", ""),
            "prediction": pred_row.get("prediction", ""),
            "audio_path": str(audio_path) if audio_path else "",
            "audio_exists": bool(audio_path),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "audio_review.json"
    csv_out = out_dir / "audio_review.csv"

    json_out.write_text(json.dumps(joined, ensure_ascii=False, indent=2), encoding="utf-8")
    if joined:
        with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(joined[0].keys()))
            writer.writeheader()
            writer.writerows(joined)

    print(f"\nจำนวนผลทำนาย: {len(prediction_rows)}")
    print(f"จำนวนที่ไม่มี ID ใน manifest: {missing_manifest}")
    print(f"จำนวนที่หาไฟล์เสียงไม่พบ: {missing_audio}")
    print(f"JSON: {json_out}")
    print(f"CSV : {csv_out}")
    print("\nตัวอย่างเรียงตามลำดับความผิดพลาดจาก error_analysis.json")

    shown = joined if args.top == 0 else joined[:max(args.top, 0)]
    for row in shown:
        cer = row["cer_percent"]
        wer = row["wer_newmm_percent"]
        cer_text = f"{float(cer):.2f}%" if cer != "" else "N/A"
        wer_text = f"{float(wer):.2f}%" if wer != "" else "N/A"
        print("\n" + "-" * 78)
        print(f'อันดับ {row["rank"]} | ID: {row["id"]}')
        print(f'CER: {cer_text} | WER newmm: {wer_text} | ระยะเวลา: {row["duration_seconds"]} วินาที')
        print("REFERENCE :", row["reference"])
        print("PREDICTION:", row["prediction"])
        print("WAV       :", row["audio_path"] or "[ไม่พบไฟล์เสียง]")
        if row["speaker"]:
            print("SPEAKER   :", row["speaker"])

    if args.open_first:
        first_with_audio = next((row for row in joined if row["audio_path"]), None)
        if first_with_audio:
            print(f'\nกำลังเปิดไฟล์เสียงอันดับ {first_with_audio["rank"]}: {first_with_audio["audio_path"]}')
            open_audio(Path(first_with_audio["audio_path"]))
        else:
            print("\nไม่มีไฟล์ WAV ที่เปิดได้ กรุณาตรวจ audio_path ใน manifest")

    print("\nหมายเหตุ: สคริปต์นี้อ่านผลทำนายและ manifest เท่านั้น ไม่โหลดโมเดล ไม่ทำนายซ้ำ และไม่ฝึกโมเดล")


if __name__ == "__main__":
    main()
