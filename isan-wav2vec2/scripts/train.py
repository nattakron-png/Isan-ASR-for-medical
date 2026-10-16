from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from jiwer import cer, wer
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCTC,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (
    MODELS,
    fingerprint_rows,
    read_json,
    read_jsonl,
    write_json,
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
    "--target",
    required=True,
    choices=[
        "thai_spelling",
        "isan_spelling",
    ],
)

parser.add_argument(
    "--data",
    default="data/official",
)

parser.add_argument(
    "--out",
    required=True,
)

parser.add_argument(
    "--epochs",
    type=int,
    default=3,
)

parser.add_argument(
    "--batch",
    type=int,
    default=1,
)

parser.add_argument(
    "--accum",
    type=int,
    default=16,
)

parser.add_argument(
    "--lr",
    type=float,
    default=None,
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

parser.add_argument(
    "--train-limit",
    type=int,
    default=None,
)

parser.add_argument(
    "--validation-limit",
    type=int,
    default=None,
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=-1,
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
)

parser.add_argument(
    "--fp32",
    action="store_true",
)

args = parser.parse_args()


# =========================================================
# Paths
# =========================================================

data_dir = ROOT / args.data
out_dir = ROOT / args.out

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CUDA check
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this training script."
    )

device = torch.device("cuda")

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

print(
    f"kind       : {args.kind}"
)

print(
    f"target     : {args.target}"
)

print(
    f"data       : {data_dir}"
)

print(
    f"output     : {out_dir}"
)

print(
    f"GPU        : {torch.cuda.get_device_name(0)}"
)

print(
    f"PyTorch    : {torch.__version__}"
)

print()


# =========================================================
# Validate target
# =========================================================

if args.kind == "normalizer":

    if args.target != "thai_spelling":
        raise ValueError(
            "Normalizer target must be thai_spelling."
        )


# =========================================================
# Load source lock
# =========================================================

data_lock_path = (
    data_dir /
    "sources.lock.json"
)

root_lock_path = (
    ROOT /
    "sources.lock.json"
)

if not data_lock_path.exists():
    raise FileNotFoundError(
        f"Missing {data_lock_path}. "
        f"Run prepare.py first."
    )

if not root_lock_path.exists():
    raise FileNotFoundError(
        f"Missing {root_lock_path}. "
        f"Run prepare.py first."
    )


data_lock = read_json(
    data_lock_path
)

root_lock = read_json(
    root_lock_path
)

if data_lock != root_lock:
    raise RuntimeError(
        "Dataset/model lock mismatch between "
        "data directory and project root."
    )


# =========================================================
# Load manifests
# =========================================================

train_rows = read_jsonl(
    data_dir / "train.jsonl"
)

val_rows = read_jsonl(
    data_dir / "validation.jsonl"
)


if args.train_limit is not None:
    train_rows = train_rows[
        :args.train_limit
    ]


if args.validation_limit is not None:
    val_rows = val_rows[
        :args.validation_limit
    ]


print(
    f"Train examples      : {len(train_rows)}"
)

print(
    f"Validation examples : {len(val_rows)}"
)


# =========================================================
# Manifest fingerprints
# =========================================================

train_fingerprint = fingerprint_rows(
    train_rows
)

val_fingerprint = fingerprint_rows(
    val_rows
)


# =========================================================
# Model
# =========================================================

model_name = MODELS[
    args.kind
]

print(
    f"Model: {model_name}"
)


# =========================================================
# WAV2VEC2
# =========================================================

if args.kind == "wav2vec2":

    processor = AutoProcessor.from_pretrained(
        model_name
    )

    model = AutoModelForCTC.from_pretrained(
        model_name
    )

    model.freeze_feature_encoder()

    model.to(device)

    # -----------------------------------------------------
    # Memory-safe audio loading
    #
    # prepare.py already saves every accepted waveform as a local WAV
    # file in the manifest's audio_path field. Use those files directly.
    # This avoids decoding the entire Hugging Face dataset into RAM.
    # -----------------------------------------------------

    from datasets import Audio

    def make_wav_dataset(rows):
        examples = []
        audio_dir = data_dir / "audio"

        for r in rows:
            # Manifest stores the original filename in `name`; accepted
            # audio files are written under data/<split>/audio/.
            filename = Path(r["name"]).name
            audio_path = audio_dir / filename

            if not audio_path.exists():
                # Fallback: some prepare.py versions may have written the
                # filename directly relative to the project root.
                alt = ROOT / "data" / filename
                if alt.exists():
                    audio_path = alt

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}\\n"
                    f"Manifest ID: {r['id']}\\n"
                    f"Manifest name: {r['name']}"
                )

            examples.append({
                "id": r["id"],
                "audio": str(audio_path),
                "text": r[args.target],
            })

        ds = Dataset.from_list(examples)
        ds = ds.cast_column(
            "audio",
            Audio(sampling_rate=16_000),
        )
        return ds

    # -----------------------------------------------------
    # Data collator
    # Process audio lazily per batch so input tensors are not
    # materialized for the entire training/validation set.
    # -----------------------------------------------------

    def data_collator(features):
        audio_arrays = [
            f["audio"]["array"]
            for f in features
        ]

        audio_sampling_rate = features[0]["audio"]["sampling_rate"]

        batch = processor(
            audio_arrays,
            sampling_rate=audio_sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        labels_batch = processor.tokenizer(
            [f["text"] for f in features],
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        batch["labels"] = labels
        return batch

    train_ds = make_wav_dataset(train_rows)
    val_ds = make_wav_dataset(val_rows)

    if args.lr is None:
        lr = 1e-5
    else:
        lr = args.lr


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    def compute_metrics(pred):

        pred_ids = np.argmax(
            pred.predictions,
            axis=-1,
        )

        pred_text = processor.batch_decode(
            pred_ids
        )

        label_ids = pred.label_ids.copy()

        label_ids[
            label_ids == -100
        ] = processor.tokenizer.pad_token_id

        ref_text = processor.batch_decode(
            label_ids,
            group_tokens=False,
        )

        return {
            "cer":
                cer(
                    ref_text,
                    pred_text,
                ),

            "wer":
                wer(
                    ref_text,
                    pred_text,
                ),
        }


    training_args = TrainingArguments(
        output_dir=str(out_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=
            args.batch,

        per_device_eval_batch_size=
            args.batch,

        gradient_accumulation_steps=
            args.accum,

        learning_rate=lr,

        warmup_ratio=0.1,

        logging_steps=50,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="cer",

        greater_is_better=False,

        gradient_checkpointing=True,

        fp16=(
            torch.cuda.is_available()
            and not args.fp32
        ),

        report_to="none",

        seed=args.seed,

        max_steps=args.max_steps,

        remove_unused_columns=False,

        optim="adafactor",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,
    )


    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )


# =========================================================
# BYT5 NORMALIZER
# =========================================================

elif args.kind == "normalizer":

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    model.to(device)


    def make_normalizer_dataset(rows):

        return Dataset.from_list(
            [
                {
                    "input_text":
                        r["isan_spelling"],

                    "target_text":
                        r["thai_spelling"],
                }

                for r in rows
            ]
        )


    train_ds = make_normalizer_dataset(
        train_rows
    )

    val_ds = make_normalizer_dataset(
        val_rows
    )


    def tokenize_batch(batch):

        inputs = tokenizer(
            batch["input_text"],
            max_length=256,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=256,
            truncation=True,
        )

        inputs["labels"] = labels[
            "input_ids"
        ]

        return inputs


    train_ds = train_ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            "input_text",
            "target_text",
        ],
    )

    val_ds = val_ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            "input_text",
            "target_text",
        ],
    )


    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )


    if args.lr is None:
        lr = 1e-4
    else:
        lr = args.lr


    def compute_metrics(pred):

        predictions = pred.predictions

        if isinstance(
            predictions,
            tuple
        ):
            predictions = predictions[0]

        predictions = np.where(
            predictions != -100,
            predictions,
            tokenizer.pad_token_id,
        )

        pred_text = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        labels = pred.label_ids.copy()

        labels[
            labels == -100
        ] = tokenizer.pad_token_id

        ref_text = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        return {
            "cer":
                cer(
                    ref_text,
                    pred_text,
                ),

            "wer":
                wer(
                    ref_text,
                    pred_text,
                ),
        }


    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=
            args.batch,

        per_device_eval_batch_size=
            args.batch,

        gradient_accumulation_steps=
            args.accum,

        learning_rate=lr,

        warmup_ratio=0.1,

        logging_steps=50,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="cer",

        greater_is_better=False,

        fp16=(
            torch.cuda.is_available()
            and not args.fp32
        ),

        report_to="none",

        seed=args.seed,

        max_steps=args.max_steps,

        predict_with_generate=True,

        generation_max_length=256,

        optim="adamw_torch",
    )


    trainer = Seq2SeqTrainer(
        model=model,

        args=training_args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )


# =========================================================
# Save run config
# =========================================================

run_config = {

    "kind":
        args.kind,

    "target":
        args.target,

    "model":
        model_name,

    "data":
        str(data_dir),

    "train_manifest":
        "train.jsonl",

    "validation_manifest":
        "validation.jsonl",

    "train_manifest_fingerprint":
        train_fingerprint,

    "validation_manifest_fingerprint":
        val_fingerprint,

    "epochs":
        args.epochs,

    "batch":
        args.batch,

    "accum":
        args.accum,

    "learning_rate":
        lr,

    "seed":
        args.seed,

    "train_examples":
        len(train_rows),

    "validation_examples":
        len(val_rows),

    "max_steps":
        args.max_steps,

    "source_lock":
        data_lock,
}


write_json(
    out_dir / "run_config.json",
    run_config,
)


# =========================================================
# Train
# =========================================================

start = time.time()

print()
print("Starting training...")
print()


trainer.train(
    resume_from_checkpoint=
        args.resume
        if args.resume
        else None
)


runtime = time.time() - start


# =========================================================
# Final evaluation
# =========================================================

metrics = trainer.evaluate()

metrics[
    "training_runtime_seconds"
] = runtime


# =========================================================
# Save final model
# =========================================================

final_dir = (
    out_dir /
    "final"
)

final_dir.mkdir(
    parents=True,
    exist_ok=True,
)


trainer.save_model(
    str(final_dir)
)


if args.kind == "wav2vec2":

    processor.save_pretrained(
        str(final_dir)
    )

else:

    tokenizer.save_pretrained(
        str(final_dir)
    )


write_json(
    final_dir / "metrics.json",
    metrics,
)


print()
print("=" * 70)
print("TRAINING COMPLETE")
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
    f"Final model: {final_dir}"
)