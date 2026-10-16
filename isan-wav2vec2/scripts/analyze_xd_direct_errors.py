import argparse
import csv
import json
import unicodedata
from pathlib import Path

from pythainlp.tokenize import word_tokenize


def normalize(text):
    """Normalize Unicode and trim outer whitespace without rewriting the transcript."""
    return unicodedata.normalize("NFC", str(text or "")).strip()


def levenshtein_counts(reference, prediction):
    """
    Levenshtein distance plus substitution/deletion/insertion counts.
    Tie-breaking prefers substitution, then deletion, then insertion.
    """
    n, m = len(reference), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "M"
            else:
                options = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], back[i][j] = min(options, key=lambda item: item[0])

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1
        else:
            break

    return {
        "distance": dp[n][m],
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_units": n,
        "prediction_units": m,
    }


def tokenize_thai(text):
    return word_tokenize(normalize(text), engine="newmm")


def rate(distance, denominator):
    # Empty references have undefined rate; report 0 if both are empty, else None.
    if denominator == 0:
        return 0.0 if distance == 0 else None
    return distance / denominator


def main():
    parser = argparse.ArgumentParser(
        description="Rank XD Direct ASR errors from an existing prediction JSONL; no model inference/training."
    )
    parser.add_argument(
        "--input",
        default="results/test/XD_FD.jsonl",
        help="Input prediction JSONL (default: results/test/XD_FD.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/test/XD_FD_error_analysis",
        help="Directory for CSV and JSON reports",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of worst examples to print (default: 30; use 0 for all)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        raise SystemExit(f"ไม่พบไฟล์ผลทำนาย: {input_path}")

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON ผิดรูปแบบที่บรรทัด {line_number}: {exc}")
            if "reference" not in row or "prediction" not in row:
                raise SystemExit(
                    f"บรรทัด {line_number} ไม่มี field 'reference' หรือ 'prediction'"
                )

            ref = normalize(row["reference"])
            pred = normalize(row["prediction"])

            char_stats = levenshtein_counts(list(ref), list(pred))
            ref_tokens = tokenize_thai(ref)
            pred_tokens = tokenize_thai(pred)
            word_stats = levenshtein_counts(ref_tokens, pred_tokens)

            cer_value = rate(char_stats["distance"], char_stats["reference_units"])
            wer_value = rate(word_stats["distance"], word_stats["reference_units"])

            # Average of the two normalized error rates, used only for ranking.
            ranking_score = (
                (cer_value if cer_value is not None else 0.0)
                + (wer_value if wer_value is not None else 0.0)
            ) / 2.0

            rows.append({
                "id": row.get("id", ""),
                "reference": ref,
                "prediction": pred,
                "reference_tokens": " | ".join(ref_tokens),
                "prediction_tokens": " | ".join(pred_tokens),
                "cer": cer_value,
                "cer_percent": None if cer_value is None else cer_value * 100,
                "char_edits": char_stats["distance"],
                "char_substitutions": char_stats["substitutions"],
                "char_deletions": char_stats["deletions"],
                "char_insertions": char_stats["insertions"],
                "reference_characters": char_stats["reference_units"],
                "wer_newmm": wer_value,
                "wer_newmm_percent": None if wer_value is None else wer_value * 100,
                "word_edits": word_stats["distance"],
                "word_substitutions": word_stats["substitutions"],
                "word_deletions": word_stats["deletions"],
                "word_insertions": word_stats["insertions"],
                "reference_words_newmm": word_stats["reference_units"],
                "prediction_words_newmm": word_stats["prediction_units"],
                "ranking_score": ranking_score,
            })

    # Highest combined normalized CER/WER first, then highest CER and WER.
    rows.sort(
        key=lambda r: (
            r["ranking_score"],
            r["cer"] if r["cer"] is not None else -1,
            r["wer_newmm"] if r["wer_newmm"] is not None else -1,
            r["char_edits"],
        ),
        reverse=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "error_analysis.json"
    csv_path = out_dir / "error_analysis.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nจำนวนตัวอย่างที่วิเคราะห์: {len(rows)}")
    print("การเรียงลำดับ: ค่าเฉลี่ย (CER + WER newmm) สูงไปต่ำ")
    print("หมายเหตุ: CER/WER รายตัวอย่างเป็นอัตราความผิดพลาด ไม่ใช่ความแม่นยำ")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")

    display_rows = rows if args.top == 0 else rows[:max(args.top, 0)]
    for rank, row in enumerate(display_rows, 1):
        cer_text = "N/A" if row["cer_percent"] is None else f'{row["cer_percent"]:.2f}%'
        wer_text = "N/A" if row["wer_newmm_percent"] is None else f'{row["wer_newmm_percent"]:.2f}%'
        print("\n" + "=" * 78)
        print(f'อันดับ {rank} | ID: {row["id"]}')
        print(
            f'CER: {cer_text} ({row["char_edits"]} edits / '
            f'{row["reference_characters"]} ตัวอักษร) | '
            f'WER newmm: {wer_text} ({row["word_edits"]} edits / '
            f'{row["reference_words_newmm"]} คำอ้างอิง)'
        )
        print(
            f'ตัวอักษร: แทนที่ {row["char_substitutions"]}, '
            f'ขาด {row["char_deletions"]}, เพิ่ม {row["char_insertions"]}'
        )
        print(
            f'คำ: แทนที่ {row["word_substitutions"]}, '
            f'ขาด {row["word_deletions"]}, เพิ่ม {row["word_insertions"]}'
        )
        print("REFERENCE :", row["reference"])
        print("PREDICTION:", row["prediction"])


if __name__ == "__main__":
    main()
