# ============================================================
# TRAIN QUICK — EfficientNetB0 + MobileNetV2, NO CBAM
# ~15-25 min on Kaggle T4
#
# Key speedups vs train_fast.py:
#  1. tf.data pipeline (parallel decode + prefetch) instead of ImageDataGenerator
#  2. Mixed float16 precision — 2-3x faster on T4 Tensor Cores
#  3. Batch size 64 — better GPU utilisation
#  4. Phase 1: 8 epochs  Phase 2: 4 epochs (shorter fine-tune)
#
# HOW TO RUN:
#   1. Kaggle → New Notebook → GPU T4 x2 (or P100)
#   2. Add dataset: adithyaadiga1/plantvillage-split-811
#   3. Paste this file, Run All
#   4. Download final_model_float16.tflite from Output
#   5. Drop into Flutter assets/model/ → hot restart
# ============================================================

import os, json, time, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

warnings.filterwarnings("ignore")
tf.random.set_seed(42)

# ── Mixed precision (2-3x speedup on T4 Tensor Cores) ────────────────────────
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("TF :", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("Policy:", tf.keras.mixed_precision.global_policy())

CSV_BASE   = "/kaggle/input/datasets/adithyaadiga1/plantvillage-split-811"
OUTPUT_DIR = "/kaggle/working"
IMG_SIZE   = 224
BATCH      = 64
AUTOTUNE   = tf.data.AUTOTUNE

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

train_df = pd.read_csv(f"{CSV_BASE}/train_split.csv")
val_df   = pd.read_csv(f"{CSV_BASE}/val_split.csv")
test_df  = pd.read_csv(f"{CSV_BASE}/test_split.csv")
print(f"Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}")

all_labels = sorted(train_df['label'].unique())
label2idx  = {l: i for i, l in enumerate(all_labels)}
CLASS_NAMES = all_labels
NUM_CLASSES = len(all_labels)
print(f"Classes: {NUM_CLASSES}")

with open(f"{OUTPUT_DIR}/class_indices.json", "w") as f:
    json.dump(label2idx, f, indent=2)

def decode(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # Same as efficientnet/mobilenetv2 preprocess_input: [0,255] → [-1,1]
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img, label

def make_dataset(df, training=False):
    paths  = df['filepath'].values
    labels = tf.keras.utils.to_categorical(
        [label2idx[l] for l in df['label'].values], NUM_CLASSES
    )
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: (decode(p), l), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_dataset(train_df, training=True)
val_ds   = make_dataset(val_df)
test_ds  = make_dataset(test_df)

# ── Model ─────────────────────────────────────────────────────────────────────

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

eff = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3), name="efficientnetb0")
eff.trainable = False
eff_feat = keras.layers.GlobalAveragePooling2D(name="gap_eff")(
    eff(inp, training=False))

mob = keras.applications.MobileNetV2(
    include_top=False, weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3), name="mobilenetv2")
mob.trainable = False
mob_feat = keras.layers.GlobalAveragePooling2D(name="gap_mob")(
    mob(inp, training=False))

x   = keras.layers.Concatenate(name="concat")([eff_feat, mob_feat])
x   = keras.layers.BatchNormalization(name="bn")(x)
x   = keras.layers.Dense(512, activation="relu", name="dense1")(x)
x   = keras.layers.Dropout(0.4, name="drop1")(x)
x   = keras.layers.Dense(256, activation="relu", name="dense2")(x)
x   = keras.layers.Dropout(0.3, name="drop2")(x)
# dtype=float32 on output keeps softmax numerically stable under mixed precision
out = keras.layers.Dense(
    NUM_CLASSES, activation="softmax", name="classification", dtype="float32")(x)

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
print(f"Trainable params: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")

# ── Phase 1: frozen backbones ─────────────────────────────────────────────────

print("\n── Phase 1: frozen backbones, 8 epochs ──")
start = time.time()

h1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=8,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/best_model.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-6, verbose=1),
    ],
    verbose=1,
)
p1_best = max(h1.history["val_accuracy"])
print(f"\nPhase 1 best val_acc: {p1_best*100:.2f}%  |  {(time.time()-start)/60:.1f} min")

# ── Phase 2: unfreeze top 20 layers each backbone ────────────────────────────

print("\n── Phase 2: unfreeze top 20 layers each, 4 epochs ──")
for layer in eff.layers[-20:]:
    layer.trainable = True
for layer in mob.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
h2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=4,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/best_model.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-7, verbose=1),
    ],
    verbose=1,
)
p2_best = max(h2.history["val_accuracy"])
total_min = (time.time() - start) / 60
print(f"\nPhase 2 best val_acc: {p2_best*100:.2f}%  |  Total: {total_min:.1f} min")

# ── Test evaluation ───────────────────────────────────────────────────────────

print("\n── Test evaluation ──")
best   = keras.models.load_model(f"{OUTPUT_DIR}/best_model.keras")
preds  = best.predict(test_ds, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = [label2idx[l] for l in test_df['label'].values]
test_acc = np.mean(y_pred == np.array(y_true[:len(y_pred)]))
print(f"Test accuracy: {test_acc*100:.2f}%")

# ── TFLite float16 ───────────────────────────────────────────────────────────

print("\n── Converting to TFLite ──")
TFLITE_OUT = f"{OUTPUT_DIR}/final_model_float16.tflite"

# Reset to float32 before conversion — mixed_float16 model converted directly
# to TFLite float16 can hang due to double-quantization confusion.
tf.keras.mixed_precision.set_global_policy('float32')
best = keras.models.load_model(f"{OUTPUT_DIR}/best_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(best)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_bytes = converter.convert()

with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_bytes)
print(f"Saved: {len(tflite_bytes)/1024/1024:.1f} MB")

# ── Sanity check ──────────────────────────────────────────────────────────────

print("\n── Sanity check (must see DIFFERENT predictions per image) ──")
interp = tf.lite.Interpreter(model_path=TFLITE_OUT)
interp.allocate_tensors()
i_d = interp.get_input_details()[0]
o_d = interp.get_output_details()[0]

preds_seen = set()
for i in range(min(8, len(test_df))):
    path = test_df['filepath'].values[i]
    true_label = test_df['label'].values[i]
    img = tf.image.resize(
        tf.cast(tf.image.decode_jpeg(tf.io.read_file(path), channels=3), tf.float32),
        [IMG_SIZE, IMG_SIZE]).numpy() / 127.5 - 1.0
    arr = img[np.newaxis].astype(np.float32)
    interp.set_tensor(i_d["index"], arr)
    interp.invoke()
    p   = interp.get_tensor(o_d["index"])[0]
    top = int(np.argmax(p))
    preds_seen.add(top)
    print(f"  true={true_label:40s}  pred={CLASS_NAMES[top]:40s}  {p[top]:.1%}")

print("\n✓ PASS — different predictions" if len(preds_seen) > 1
      else "\n✗ FAIL — same class for all images, do NOT use this model")

print(f"\nPhase 1 val : {p1_best*100:.2f}%")
print(f"Phase 2 val : {p2_best*100:.2f}%")
print(f"Test acc    : {test_acc*100:.2f}%")
print(f"Total time  : {total_min:.1f} min")
print(f"\n→ Download: {TFLITE_OUT}")
print("→ Replace assets/model/final_model_float16.tflite in Flutter → hot restart")
