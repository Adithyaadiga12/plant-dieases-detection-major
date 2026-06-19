# ============================================================
# TRAIN FAST — EfficientNetB0 + MobileNetV2, NO CBAM
# ~35-45 min on Kaggle T4 GPU
#
# Same dual-backbone hybrid concept, CBAM removed because
# Keras 3 ModelCheckpoint corrupts custom sub-layer weights.
# Both backbones frozen → only the head trains → fast & safe.
#
# HOW TO RUN:
#   1. Kaggle → New Notebook → GPU T4 (or P100)
#   2. Add dataset: adithyaadiga1/plantvillage-split-811
#   3. Paste this entire file, Run All
#   4. Download final_model_float16.tflite from Output
#   5. Drop into Flutter assets/model/ → hot restart
# ============================================================

import os, json, time, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
tf.random.set_seed(42)

print("TF :", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

CSV_BASE   = "/kaggle/input/datasets/adithyaadiga1/plantvillage-split-811"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

train_df = pd.read_csv(f"{CSV_BASE}/train_split.csv")
val_df   = pd.read_csv(f"{CSV_BASE}/val_split.csv")
test_df  = pd.read_csv(f"{CSV_BASE}/test_split.csv")
print(f"Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}")

preprocess = tf.keras.applications.efficientnet.preprocess_input  # (x/127.5)-1.0

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=20, zoom_range=0.15,
    horizontal_flip=True, vertical_flip=True,
    brightness_range=[0.85, 1.15],
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess)

train_data = train_gen.flow_from_dataframe(
    train_df, x_col="filepath", y_col="label",
    target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=True,
)
val_data = val_gen.flow_from_dataframe(
    val_df, x_col="filepath", y_col="label",
    target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=False,
)
test_data = val_gen.flow_from_dataframe(
    test_df, x_col="filepath", y_col="label",
    target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=False,
)

NUM_CLASSES = len(train_data.class_indices)
CLASS_NAMES = [k for k, _ in sorted(train_data.class_indices.items(), key=lambda x: x[1])]
print(f"Classes: {NUM_CLASSES}")

with open(f"{OUTPUT_DIR}/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)

# ── Model: EfficientNetB0 + MobileNetV2, no custom layers ─────────────────────

inp = keras.Input(shape=(224, 224, 3), name="input_image")

# EfficientNetB0 branch
eff = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), name="efficientnetb0"
)
eff.trainable = False
eff_feat = keras.layers.GlobalAveragePooling2D(name="gap_eff")(eff(inp, training=False))

# MobileNetV2 branch
mob = keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), name="mobilenetv2"
)
mob.trainable = False
mob_feat = keras.layers.GlobalAveragePooling2D(name="gap_mob")(mob(inp, training=False))

# Merge + head
x   = keras.layers.Concatenate(name="concat")([eff_feat, mob_feat])
x   = keras.layers.BatchNormalization(name="bn")(x)
x   = keras.layers.Dense(512, activation="relu", name="dense1")(x)
x   = keras.layers.Dropout(0.4, name="drop1")(x)
x   = keras.layers.Dense(256, activation="relu", name="dense2")(x)
x   = keras.layers.Dropout(0.3, name="drop2")(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
print(f"Total params    : {model.count_params():,}")
print(f"Trainable params: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")

# ── Train ─────────────────────────────────────────────────────────────────────

print("\n── Phase 1: frozen backbones, head only, 10 epochs ──")
start = time.time()

h1 = model.fit(
    train_data, validation_data=val_data,
    epochs=10,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/best_model.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-6, verbose=1,
        ),
    ],
    verbose=1,
)
p1_best = max(h1.history["val_accuracy"])
elapsed = (time.time() - start) / 60
print(f"\nPhase 1 best val_acc: {p1_best*100:.2f}%  |  {elapsed:.1f} min")

# ── Phase 2: unfreeze top layers of both backbones ────────────────────────────

print("\n── Phase 2: unfreeze top 40 layers each backbone, 8 epochs ──")
for layer in eff.layers[-40:]:
    layer.trainable = True
for layer in mob.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

h2 = model.fit(
    train_data, validation_data=val_data,
    epochs=8,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/best_model.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-7, verbose=1,
        ),
    ],
    verbose=1,
)
p2_best = max(h2.history["val_accuracy"])
total_min = (time.time() - start) / 60
print(f"\nPhase 2 best val_acc: {p2_best*100:.2f}%  |  Total: {total_min:.1f} min")

# ── Test evaluation ───────────────────────────────────────────────────────────

print("\n── Test evaluation ──")
best = keras.models.load_model(f"{OUTPUT_DIR}/best_model.keras")
preds  = best.predict(test_data, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes
test_acc = np.mean(y_pred == y_true)
print(f"Test accuracy: {test_acc*100:.2f}%")

# ── Convert to TFLite float16 ─────────────────────────────────────────────────

print("\n── Converting to TFLite ──")
TFLITE_OUT = f"{OUTPUT_DIR}/final_model_float16.tflite"

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

test_data.reset()
bx, by = next(iter(test_data))
preds_seen = set()
for i in range(min(8, len(bx))):
    arr = bx[i:i+1].astype(np.float32)
    interp.set_tensor(i_d["index"], arr)
    interp.invoke()
    p   = interp.get_tensor(o_d["index"])[0]
    top = int(np.argmax(p))
    preds_seen.add(top)
    print(f"  true={CLASS_NAMES[np.argmax(by[i])]:40s}  pred={CLASS_NAMES[top]:40s}  {p[top]:.1%}")

if len(preds_seen) > 1:
    print("\n✓ PASS — different predictions for different images")
else:
    print("\n✗ FAIL — still predicting same class, do NOT use this model")

print(f"\nPhase 1 val : {p1_best*100:.2f}%")
print(f"Phase 2 val : {p2_best*100:.2f}%")
print(f"Test acc    : {test_acc*100:.2f}%")
print(f"Total time  : {total_min:.1f} min")
print(f"\n→ Download: {TFLITE_OUT}")
print("→ Replace assets/model/final_model_float16.tflite in Flutter → hot restart")
