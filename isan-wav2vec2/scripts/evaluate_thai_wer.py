
import json
import unicodedata
from pathlib import Path

from jiwer import cer, wer
from pythainlp.tokenize import word_tokenize

INPUT = Path("results/test/XD_FD.jsonl")
OUTPUT = Path("results/test/XD_FD.thai_metrics.json")


def clean(text):
    return unicodedata.normalize("NFC", str(text)).strip()


def tokenize_thai(text):
    # ตัดคำไทยและใส่ช่องว่างระหว่างคำ
    return " ".join(word_tokenize(clean(text), engine="newmm"))


with INPUT.open("r", encoding="utf-8") as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

references = [clean(row["reference"]) for row in rows]
predictions = [clean(row["prediction"]) for row in rows]

ref_tokens = [tokenize_thai(text) for text in references]
pred_tokens = [tokenize_thai(text) for text in predictions]

metrics = {
    "examples": len(rows),
    "cer_original_text": cer(references, predictions),
    "wer_space_delimited": wer(references, predictions),
    "wer_pythainlp_newmm": wer(ref_tokens, pred_tokens),
    "tokenizer": "PyThaiNLP newmm",
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n=== Thai ASR Evaluation ===")
print(f"จำนวนตัวอย่าง: {metrics['examples']}")
print(f"CER: {metrics['cer_original_text']:.4%}")
print(f"WER แบบเดิม: {metrics['wer_space_delimited']:.4%}")
print(f"WER ตัดคำไทย newmm: {metrics['wer_pythainlp_newmm']:.4%}")
print(f"บันทึกผลที่: {OUTPUT}")
