import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
import numpy as np

KERAS_PATH = r'C:\Users\Adithya Adiga\Downloads\best_model (1).keras'
TFLITE_OUT = r'C:\Users\Adithya Adiga\Documents\agrovision_ai\assets\model\final_model_float16.tflite'
NUM_CLASSES = 38
IMG_SIZE    = 224

# ── Step 1: load mixed_float16 model just to extract weights ──────────────────
print("Loading trained weights...")
trained = keras.models.load_model(KERAS_PATH)
weights = trained.get_weights()
print(f"  {len(weights)} weight tensors loaded")
del trained   # free memory

# ── Step 2: rebuild identical architecture in pure float32 ────────────────────
print("Rebuilding model in float32...")
tf.keras.mixed_precision.set_global_policy('float32')

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

eff = keras.applications.EfficientNetB0(
    include_top=False, weights=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3), name="efficientnetb0")
eff_feat = keras.layers.GlobalAveragePooling2D(name="gap_eff")(
    eff(inp, training=False))

mob = keras.applications.MobileNetV2(
    include_top=False, weights=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3), name="mobilenetv2")
mob_feat = keras.layers.GlobalAveragePooling2D(name="gap_mob")(
    mob(inp, training=False))

x   = keras.layers.Concatenate(name="concat")([eff_feat, mob_feat])
x   = keras.layers.BatchNormalization(name="bn")(x)
x   = keras.layers.Dense(512, activation="relu", name="dense1")(x)
x   = keras.layers.Dropout(0.4, name="drop1")(x)
x   = keras.layers.Dense(256, activation="relu", name="dense2")(x)
x   = keras.layers.Dropout(0.3, name="drop2")(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)

model = keras.Model(inp, out)

# ── Step 3: copy weights into float32 model ───────────────────────────────────
print("Copying weights...")
model.set_weights(weights)
print(f"  Input : {model.input_shape}")
print(f"  Output: {model.output_shape}")

# ── Step 4: convert to TFLite float16 ────────────────────────────────────────
print("Converting to TFLite float16...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_bytes = converter.convert()

with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_bytes)
print(f"Saved: {len(tflite_bytes)/1024/1024:.1f} MB")

# ── Step 5: sanity check ──────────────────────────────────────────────────────
print("\nSanity check...")
interp = tf.lite.Interpreter(model_path=TFLITE_OUT)
interp.allocate_tensors()
i_d = interp.get_input_details()[0]
o_d = interp.get_output_details()[0]
print(f"  Input  shape: {i_d['shape']}  dtype: {i_d['dtype'].__name__}")
print(f"  Output shape: {o_d['shape']}  dtype: {o_d['dtype'].__name__}")

dummy = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
interp.set_tensor(i_d['index'], dummy)
interp.invoke()
out = interp.get_tensor(o_d['index'])[0]
print(f"  Output sum  : {out.sum():.4f}  (should be ~1.0)")
print(f"  Top class   : {np.argmax(out)}  confidence: {out.max():.1%}")
print("\nDone. Hot restart Flutter.")
