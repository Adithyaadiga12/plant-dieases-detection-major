# ============================================================
# TRAIN V2 — EfficientNetB0 ONLY, NO CUSTOM LAYERS
#
# No CBAM, no custom layers → zero serialization risk.
# EfficientNetB0 alone achieves 95%+ on PlantVillage.
# Expected time: ~60-90 min on Kaggle T4 GPU.
#
# HOW TO RUN ON KAGGLE:
#   1. New notebook, GPU accelerator (T4 x2 or P100)
#   2. Add dataset: adithyaadiga1/plantvillage-split-811
#   3. Paste this entire file, run all
#   4. Download final_model_float16.tflite from /kaggle/working/
#   5. Replace assets/model/final_model_float16.tflite in Flutter
# ============================================================

import os, json, time, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

print("TF :", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

CSV_BASE   = "/kaggle/input/datasets/adithyaadiga1/plantvillage-split-811"
OUTPUT_DIR = "/kaggle/working"
IMG_SIZE   = (224, 224)
BATCH      = 32
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

train_df = pd.read_csv(f"{CSV_BASE}/train_split.csv")
val_df   = pd.read_csv(f"{CSV_BASE}/val_split.csv")
test_df  = pd.read_csv(f"{CSV_BASE}/test_split.csv")
print(f"Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}")

preprocess = tf.keras.applications.efficientnet.preprocess_input  # (x/127.5)-1.0

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess)

train_data = train_gen.flow_from_dataframe(
    train_df, x_col="filepath", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=True,
)
val_data = val_gen.flow_from_dataframe(
    val_df, x_col="filepath", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=False,
)
test_data = val_gen.flow_from_dataframe(
    test_df, x_col="filepath", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=False,
)

NUM_CLASSES = len(train_data.class_indices)
CLASS_NAMES = [k for k, _ in sorted(train_data.class_indices.items(), key=lambda x: x[1])]
print(f"Classes: {NUM_CLASSES}")

# Verify order matches Flutter app (alphabetical)
with open(f"{OUTPUT_DIR}/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)
print("class_indices.json saved")

# ── Model (pure Keras, no custom layers) ──────────────────────────────────────

base = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet",
    input_shape=(224, 224, 3),
)
base.trainable = False

inp  = keras.Input(shape=(224, 224, 3), name="input_image")
x    = base(inp, training=False)
x    = keras.layers.GlobalAveragePooling2D()(x)
x    = keras.layers.BatchNormalization()(x)
x    = keras.layers.Dense(512, activation="relu")(x)
x    = keras.layers.Dropout(0.4)(x)
out  = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)
model = keras.Model(inp, out)

print(f"Params: {model.count_params():,}")

# ── Phase 1: frozen backbone ──────────────────────────────────────────────────

print("\n── Phase 1: frozen backbone, LR=1e-3, 15 epochs ──")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

start = time.time()
h1 = model.fit(
    train_data, validation_data=val_data,
    epochs=15,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/best_phase1.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.CSVLogger(f"{OUTPUT_DIR}/log_phase1.csv"),
    ],
)
print(f"Phase 1 best val_acc: {max(h1.history['val_accuracy'])*100:.2f}%")

# ── Phase 2: unfreeze top layers ──────────────────────────────────────────────

print("\n── Phase 2: unfreeze top 100 layers, LR=5e-5, 15 epochs ──")
base.trainable = True
# Freeze early layers (low-level features already good from ImageNet)
for layer in base.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

h2 = model.fit(
    train_data, validation_data=val_data,
    epochs=15,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/best_phase2.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.CSVLogger(f"{OUTPUT_DIR}/log_phase2.csv"),
    ],
)
p2_best = max(h2.history["val_accuracy"])
print(f"Phase 2 best val_acc: {p2_best*100:.2f}%")

# ── Test evaluation ───────────────────────────────────────────────────────────

print("\n── Test set evaluation ──")
best = keras.models.load_model(f"{OUTPUT_DIR}/best_phase2.keras")
preds = best.predict(test_data, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes
test_acc = np.mean(y_pred == y_true)
print(f"Test accuracy: {test_acc*100:.2f}%")

from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
print(report)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n{report}")

# ── Convert to TFLite float16 ─────────────────────────────────────────────────

print("\n── Converting to TFLite float16 ──")
TFLITE_OUT = f"{OUTPUT_DIR}/final_model_float16.tflite"

converter = tf.lite.TFLiteConverter.from_keras_model(best)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_bytes = converter.convert()

with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_bytes)
print(f"TFLite saved: {len(tflite_bytes)/1024/1024:.1f} MB")

# ── TFLite sanity check ───────────────────────────────────────────────────────

interp = tf.lite.Interpreter(model_path=TFLITE_OUT)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"Input : {inp_d['shape']}  {inp_d['dtype'].__name__}")
print(f"Output: {out_d['shape']}  {out_d['dtype'].__name__}")

# Run on first 5 test images and verify different predictions
print("\nVerifying different images give different predictions:")
test_data.reset()
batch_x, batch_y = next(iter(test_data))
for i in range(min(5, len(batch_x))):
    arr = batch_x[i:i+1]
    interp.set_tensor(inp_d["index"], arr.astype(np.float32))
    interp.invoke()
    probs = interp.get_tensor(out_d["index"])[0]
    top1 = np.argmax(probs)
    true1 = np.argmax(batch_y[i])
    print(f"  [{i}] true={CLASS_NAMES[true1]:40s}  pred={CLASS_NAMES[top1]:40s}  conf={probs[top1]:.1%}")

total_time = (time.time() - start) / 60
print(f"\nTotal training time: {total_time:.1f} min")
print(f"\nPhase 1 best: {max(h1.history['val_accuracy'])*100:.2f}%")
print(f"Phase 2 best: {p2_best*100:.2f}%")
print(f"Test acc    : {test_acc*100:.2f}%")
print(f"\nDownload: {TFLITE_OUT}")
print("→ Replace assets/model/final_model_float16.tflite in Flutter and hot restart")
