
import json
import unicodedata
from pathlib import Path

from jiwer import wer
from pythainlp.tokenize import word_tokenize

path = Path("results/test/XD_FD.jsonl")

rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

def clean(text):
    return unicodedata.normalize("NFC", str(text)).strip()

references = [clean(r["reference"]) for r in rows]
predictions = [clean(r["prediction"]) for r in rows]

# วิธีเดิม: แบ่งหน่วยด้วยช่องว่าง
wer_space = wer(references, predictions)

# วิธีทดลอง: ตัดคำภาษาไทยก่อนคำนวณ WER
ref_tokenized = [
    " ".join(word_tokenize(s, engine="newmm"))
    for s in references
]
pred_tokenized = [
    " ".join(word_tokenize(s, engine="newmm"))
    for s in predictions
]
wer_thai = wer(ref_tokenized, pred_tokenized)

print(f"จำนวนตัวอย่าง: {len(rows)}")
print(f"WER แบบเดิม (space-delimited): {wer_space:.4%}")
print(f"WER แบบตัดคำไทย (newmm):      {wer_thai:.4%}")
print("\n--- ตัวอย่าง 10 รายการแรก ---")

for i, row in enumerate(rows[:10], 1):
    print(f"\n[{i}] ID: {row['id']}")
    print("REF :", row["reference"])
    print("PRED:", row["prediction"])
    print("REF tokens :", ref_tokenized[i - 1])
    print("PRED tokens:", pred_tokenized[i - 1])
