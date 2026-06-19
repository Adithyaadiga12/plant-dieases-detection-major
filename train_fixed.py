# ============================================================
# FINAL TRAINING — EfficientNetB0 + MobileNetV2 + CBAM (FIXED)
#
# Bug fixed: all three custom layer classes now decorated with
# @keras.saving.register_keras_serializable() AND each build()
# explicitly calls sub-layer .build() so Keras can track and
# save the trained CBAM weights across sessions.
# ============================================================

import os, json, warnings, time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

print("TF  :", tf.__version__)
print("GPU :", tf.config.list_physical_devices("GPU"))

# ============================================================
# 1. CUSTOM LAYERS  — all three classes MUST be registered
# ============================================================

@keras.saving.register_keras_serializable(package="agrovision")
class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        self.gap     = keras.layers.GlobalAveragePooling2D(name="gap")
        self.gmp     = keras.layers.GlobalMaxPooling2D(name="gmp")
        self.reshape = keras.layers.Reshape((1, 1, filters), name="reshape")
        self.fc1     = keras.layers.Dense(
                           filters // self.ratio,
                           activation="relu", use_bias=True, name="fc1")
        self.fc2     = keras.layers.Dense(
                           filters,
                           activation="sigmoid", use_bias=True, name="fc2")
        # Explicitly build each sub-layer so their variables exist
        # BEFORE Keras serializes the model (the training-session bug fix).
        self.gap.build(input_shape)
        self.gmp.build(input_shape)
        self.reshape.build([None, filters])
        self.fc1.build([None, 1, 1, filters])
        self.fc2.build([None, 1, 1, filters // self.ratio])
        super().build(input_shape)

    def call(self, x):
        avg = self.fc2(self.fc1(self.reshape(self.gap(x))))
        mx  = self.fc2(self.fc1(self.reshape(self.gmp(x))))
        return x * (avg + mx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg


@keras.saving.register_keras_serializable(package="agrovision")
class SpatialAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(
            1, 7, padding="same", activation="sigmoid", name="conv")
        h = input_shape[1] if input_shape[1] is not None else 7
        w = input_shape[2] if input_shape[2] is not None else 7
        self.conv.build([None, h, w, 2])
        super().build(input_shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max(x,  axis=-1, keepdims=True)
        return x * self.conv(tf.concat([avg, mx], axis=-1))

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="agrovision")
class CBAM(keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.ca = ChannelAttention(ratio=self.ratio, name="ca")
        self.sa = SpatialAttention(name="sa")
        self.ca.build(input_shape)
        self.sa.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return self.sa(self.ca(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg


CUSTOM_OBJECTS = {
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
    "CBAM"            : CBAM,
}

# ============================================================
# 2. CONFIG
# ============================================================

CSV_BASE   = "/kaggle/input/datasets/adithyaadiga1/plantvillage-split-811"
OUTPUT_DIR = "/kaggle/working"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 3. LOAD CSV SPLITS
# ============================================================

print("\n── Loading CSV splits ──")
train_df = pd.read_csv(f"{CSV_BASE}/train_split.csv")
val_df   = pd.read_csv(f"{CSV_BASE}/val_split.csv")
test_df  = pd.read_csv(f"{CSV_BASE}/test_split.csv")

print(f"Train : {len(train_df):,}")
print(f"Val   : {len(val_df):,}")
print(f"Test  : {len(test_df):,}")

# ============================================================
# 4. DATA GENERATORS
# ============================================================

preprocess = tf.keras.applications.efficientnet.preprocess_input

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess)

train_data = train_gen.flow_from_dataframe(
    train_df, x_col="filepath", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True
)
val_data = val_gen.flow_from_dataframe(
    val_df, x_col="filepath", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)
test_data = val_gen.flow_from_dataframe(
    test_df, x_col="filepath", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

NUM_CLASSES = len(train_data.class_indices)
CLASS_NAMES = list(train_data.class_indices.keys())
print(f"\nClasses : {NUM_CLASSES}")

with open(f"{OUTPUT_DIR}/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)
print("class_indices.json saved")

# ============================================================
# 5. BUILD MODEL
# ============================================================

print("\n── Building model ──")

def build_hybrid_model(num_classes):
    inp = keras.Input(shape=(224, 224, 3), name="input_image")

    eff     = keras.applications.EfficientNetB0(
                  include_top=False, weights="imagenet",
                  input_shape=(224, 224, 3), name="efficientnetb0")
    eff_out = eff(inp)
    eff_att = CBAM(ratio=8, name="cbam")(eff_out)
    eff_feat= keras.layers.GlobalAveragePooling2D(name="gap_eff")(eff_att)

    mob     = keras.applications.MobileNetV2(
                  include_top=False, weights="imagenet",
                  input_shape=(224, 224, 3), name="mobilenetv2")
    mob_out = mob(inp)
    mob_att = CBAM(ratio=8, name="cbam_1")(mob_out)
    mob_feat= keras.layers.GlobalAveragePooling2D(name="gap_mob")(mob_att)

    merged  = keras.layers.Concatenate(name="concat")([eff_feat, mob_feat])
    x       = keras.layers.BatchNormalization(name="bn_head")(merged)
    x       = keras.layers.Dense(512, activation="relu", name="dense")(x)
    x       = keras.layers.Dropout(0.5, name="dropout")(x)
    x       = keras.layers.Dense(256, activation="relu", name="dense_1")(x)
    x       = keras.layers.Dropout(0.4, name="dropout_1")(x)
    out_cls = keras.layers.Dense(
                  num_classes, activation="softmax", name="classification")(x)

    return keras.Model(inputs=inp, outputs=out_cls)


model = build_hybrid_model(NUM_CLASSES)
print(f"Parameters: {model.count_params():,}")

# ============================================================
# 6. CALLBACKS
# ============================================================

def get_callbacks(phase):
    return [
        keras.callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/best_phase{phase}.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(f"{OUTPUT_DIR}/log_phase{phase}.csv"),
    ]

# ============================================================
# 7. PHASE 1 — Frozen backbones
# ============================================================

print("\n" + "="*55)
print("PHASE 1 — Frozen | LR=1e-3 | 10 epochs")
print("="*55)

model.get_layer("efficientnetb0").trainable = False
model.get_layer("mobilenetv2").trainable    = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

start = time.time()
h1 = model.fit(
    train_data, validation_data=val_data,
    epochs=10, callbacks=get_callbacks(1), verbose=1
)
p1_best = max(h1.history["val_accuracy"])
print(f"\nPhase 1 best : {p1_best*100:.2f}%")

# ============================================================
# 8. PHASE 2 — Unfrozen fine-tuning
# ============================================================

print("\n" + "="*55)
print("PHASE 2 — Unfrozen | LR=5e-5 | 8 epochs")
print("="*55)

model.get_layer("efficientnetb0").trainable = True
model.get_layer("mobilenetv2").trainable    = True

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

h2 = model.fit(
    train_data, validation_data=val_data,
    epochs=8, callbacks=get_callbacks(2), verbose=1
)
p2_best = max(h2.history["val_accuracy"])
print(f"\nPhase 2 best : {p2_best*100:.2f}%")

# ============================================================
# 9. TEST SET EVALUATION
# ============================================================

print("\n── Test Set Evaluation ──")

# Load from checkpoint (tests save/load correctness with registered classes)
best_model = keras.models.load_model(f"{OUTPUT_DIR}/best_phase2.keras")
print(f"Loaded best_phase2 — output shape: {best_model.output_shape}")

preds  = best_model.predict(test_data, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes
test_acc = np.mean(y_pred == y_true)
print(f"\nTest Accuracy : {test_acc*100:.2f}%")

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
print(report)

with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n{report}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix — Test Accuracy: {test_acc*100:.2f}%",
          fontsize=14, fontweight="bold")
plt.xticks(rotation=90, fontsize=6); plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
all_acc   = h1.history["accuracy"]     + h2.history["accuracy"]
all_vacc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
all_loss  = h1.history["loss"]         + h2.history["loss"]
all_vloss = h1.history["val_loss"]     + h2.history["val_loss"]
epochs    = range(1, len(all_acc)+1)
axes[0].plot(epochs, all_acc, label="Train")
axes[0].plot(epochs, all_vacc, label="Val", linestyle="--")
axes[0].axvline(x=10.5, color="grey", linestyle=":", alpha=0.7)
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(epochs, all_loss, label="Train")
axes[1].plot(epochs, all_vloss, label="Val", linestyle="--")
axes[1].axvline(x=10.5, color="grey", linestyle=":", alpha=0.7)
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
plt.close()

# ============================================================
# 10. CONVERT TO TFLITE (done here on Kaggle GPU, saves time)
# ============================================================

print("\n── Converting to TFLite float16 ──")
SAVED_MODEL = f"{OUTPUT_DIR}/agrovision_saved_model"
TFLITE_OUT  = f"{OUTPUT_DIR}/final_model_float16.tflite"

tf.saved_model.save(best_model, SAVED_MODEL)
print("SavedModel saved.")

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_bytes = converter.convert()

with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_bytes)
print(f"TFLite saved: {len(tflite_bytes)/1024/1024:.1f} MB")

# Quick TFLite sanity check
interp = tf.lite.Interpreter(model_path=TFLITE_OUT)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"TFLite input : {inp_d['shape']}  {inp_d['dtype'].__name__}")
print(f"TFLite output: {out_d['shape']}  {out_d['dtype'].__name__}")

# Verify TFLite on first test image
import glob
from PIL import Image
imgs = glob.glob("/kaggle/input/**/*.JPG", recursive=True)
if imgs:
    img = Image.open(imgs[0]).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = ((np.array(img, dtype=np.float32) / 127.5) - 1.0)[np.newaxis]
    interp.set_tensor(inp_d['index'], arr)
    interp.invoke()
    probs = interp.get_tensor(out_d['index'])[0]
    top3  = np.argsort(probs)[::-1][:3]
    print(f"\nTFLite sample: {[f'{CLASS_NAMES[i]}({probs[i]:.0%})' for i in top3]}")

# ============================================================
# 11. SAVE FINAL KERAS MODEL + SUMMARY
# ============================================================

best_model.save(f"{OUTPUT_DIR}/final_clean_model_v4.keras")

total_time = time.time() - start
print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
print(f"Phase 1 best val : {p1_best*100:.2f}%")
print(f"Phase 2 best val : {p2_best*100:.2f}%")
print(f"Test accuracy    : {test_acc*100:.2f}%")
print(f"Total time       : {total_time/60:.1f} mins")
print("Output files:")
print("  final_clean_model_v4.keras  <- main model (fixed serialization)")
print("  best_phase1.keras / best_phase2.keras")
print("  final_model_float16.tflite  <- copy this to Flutter assets/model/")
print("  classification_report.txt")
print("  confusion_matrix.png / training_curves.png")
print("  class_indices.json")
print("="*55)
