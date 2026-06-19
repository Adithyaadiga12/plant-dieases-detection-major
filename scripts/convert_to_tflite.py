# ============================================================
# CONVERSION ONLY — paste into a new Kaggle notebook
# Upload best_model.keras as a dataset input first.
#
# Fix: uses TFLiteConverter.from_keras_model() directly,
# which avoids the tf.saved_model.save() _DictWrapper bug.
# No GPU needed. Runs in ~5 minutes.
# ============================================================

import numpy as np
import tensorflow as tf
import keras

print("TF:", tf.__version__)

# ── 1. Re-define the custom layers (required to load .keras) ──

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
        self.fc1     = keras.layers.Dense(filters // self.ratio, activation="relu", use_bias=True, name="fc1")
        self.fc2     = keras.layers.Dense(filters, activation="sigmoid", use_bias=True, name="fc2")
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
        self.conv = keras.layers.Conv2D(1, 7, padding="same", activation="sigmoid", name="conv")
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


# ── 2. Load the downloaded best_model.keras ──
# On Kaggle: upload best_model.keras as a dataset, then set this path.
# The path will be: /kaggle/input/<your-dataset-name>/best_model.keras

MODEL_PATH  = "/kaggle/input/agrovision-best-model/best_model.keras"
OUTPUT_PATH = "/kaggle/working/final_model_float16.tflite"

print(f"\nLoading model from: {MODEL_PATH}")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "CBAM": CBAM,
    }
)
print(f"Loaded. Output shape: {model.output_shape}")

# ── 3. Convert directly — no tf.saved_model.save() needed ──
print("\nConverting to TFLite float16...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_bytes = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_bytes)
print(f"Saved: {len(tflite_bytes)/1024/1024:.1f} MB  →  {OUTPUT_PATH}")

# ── 4. Sanity check ──
interp = tf.lite.Interpreter(model_path=OUTPUT_PATH)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"\nTFLite input : {inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"TFLite output: {out_d['shape']}  dtype={out_d['dtype'].__name__}")

# Run a dummy inference to confirm the model works
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
interp.set_tensor(inp_d["index"], dummy)
interp.invoke()
probs = interp.get_tensor(out_d["index"])[0]
print(f"Dummy inference — max prob: {probs.max():.4f} at class {probs.argmax()}")
print("\nDone. Download final_model_float16.tflite from /kaggle/working/")
