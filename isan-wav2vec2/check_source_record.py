
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset


# ==================================================
# 1. Configuration
# ==================================================

DATASET_ID = "typhoon-ai/thai-dialect-isan-dataset"
REVISION = "dfc7e495b3069940880d184f0acfcc6b3cd4f364"

TARGET_ID = "689dc5729edc5727e820cdc6"
TARGET_NAME = "opentyphoon;is;m_042;gen;0102.wav"

LOCAL_AUDIO = Path("data/official/audio") / TARGET_NAME


# ==================================================
# 2. Load the original dataset
# ==================================================

print("Loading source dataset...")

dataset = load_dataset(
    DATASET_ID,
    revision=REVISION,
)


# ==================================================
# 3. Find the target record
# ==================================================

source_row = None
source_split_name = None
source_index = None

for split_name, split_data in dataset.items():
    for index, row in enumerate(split_data):
        if str(row.get("id", "")) == TARGET_ID:
            source_row = row
            source_split_name = split_name
            source_index = index
            break

    if source_row is not None:
        break


if source_row is None:
    print("\nERROR: Source record not found.")
    print("Target ID:", TARGET_ID)
    raise SystemExit(1)


# ==================================================
# 4. Print source record information
# ==================================================

print("\n" + "=" * 60)
print("SOURCE RECORD")
print("=" * 60)

print("Split:", source_split_name)
print("Index:", source_index)

for field in [
    "id",
    "name",
    "speaker",
    "question_id",
    "question",
    "raw",
    "thai_spelling",
    "isan_spelling",
]:
    print(f"{field}: {source_row.get(field)}")


# ==================================================
# 5. Check the local audio file
# ==================================================

print("\n" + "=" * 60)
print("LOCAL AUDIO FILE")
print("=" * 60)

if not LOCAL_AUDIO.is_file():
    print("ERROR: Local audio file not found.")
    print("Expected path:", LOCAL_AUDIO.resolve())
    raise SystemExit(1)

print("Path:", LOCAL_AUDIO.resolve())


# ==================================================
# 6. Compare source audio with local audio
# ==================================================

source_audio = source_row.get("audio")

if not isinstance(source_audio, dict):
    print("ERROR: Source record has no usable audio object.")
    raise SystemExit(1)

if "array" not in source_audio or "sampling_rate" not in source_audio:
    print("ERROR: Source audio is missing array or sampling_rate.")
    raise SystemExit(1)

source_sr = int(source_audio["sampling_rate"])
source_array = np.asarray(
    source_audio["array"],
    dtype=np.float32,
)

local_array, local_sr = sf.read(
    str(LOCAL_AUDIO),
    dtype="float32",
    always_2d=False,
)

print("\n" + "=" * 60)
print("AUDIO COMPARISON")
print("=" * 60)

print("Source sample rate:", source_sr)
print("Local sample rate:", local_sr)

print("Source shape:", source_array.shape)
print("Local shape:", local_array.shape)

print("Source duration (seconds):", len(source_array) / source_sr)
print("Local duration (seconds):", len(local_array) / local_sr)


# ==================================================
# 7. Compare decoded audio samples
# ==================================================

if source_sr != local_sr:
    print("\nRESULT: Sample rates differ.")
    print("Direct sample-by-sample comparison is unavailable.")

elif source_array.shape != local_array.shape:
    print("\nRESULT: Audio array shapes differ.")
    print("The audio arrays are not identical in shape.")

else:
    same_samples = np.allclose(
        source_array,
        local_array,
        atol=1e-6,
        rtol=0,
    )

    max_difference = float(
        np.max(np.abs(source_array - local_array))
    )

    print("\nSame decoded samples:", same_samples)
    print("Maximum absolute sample difference:", max_difference)

    if same_samples:
        print("RESULT: Source and local audio samples match.")
    else:
        print("RESULT: Source and local audio samples differ.")


print("\nInspection completed.")


from scipy.signal import resample_poly

# Resample source audio from 48 kHz to 16 kHz
source_16k = resample_poly(source_array, 1, 3).astype(np.float32)

# Compare the resampled source with the local audio.
print("\n" + "=" * 60)
print("RESAMPLED AUDIO COMPARISON")
print("=" * 60)

print("Resampled source shape:", source_16k.shape)
print("Local audio shape:", local_array.shape)

if source_16k.shape == local_array.shape:
    difference = source_16k - local_array

    print(
        "Mean absolute difference:",
        float(np.mean(np.abs(difference))),
    )
    print(
        "Maximum absolute difference:",
        float(np.max(np.abs(difference))),
    )
    print(
        "Correlation:",
        float(np.corrcoef(source_16k, local_array)[0, 1]),
    )
else:
    print("Shapes differ; direct comparison is not possible.")

