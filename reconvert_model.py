"""
Reload final_clean_model_v3.keras with the exact CBAM from training code,
test all 38 images, then reconvert to TFLite float16.

Key fix:
  - fc2 uses activation='sigmoid' (not linear + sigmoid after)
  - reshape is applied BEFORE fc1 (not after fc2)
  - sub-layers are explicitly built in parent's build() so weights load
"""
import os, sys, glob, io, zipfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image
import tensorflow as tf
import keras

# ─────────────────────────────────────────────────────────────────────────────
# CBAM — exactly matching the training code (from convert_model.py)
# ─────────────────────────────────────────────────────────────────────────────

@keras.saving.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        # Create sub-layers with attribute names matching the weight file.
        self.gap     = keras.layers.GlobalAveragePooling2D()
        self.gmp     = keras.layers.GlobalMaxPooling2D()
        self.reshape = keras.layers.Reshape((1, 1, channels))
        self.fc1     = keras.layers.Dense(channels // self.ratio,
                                           activation='relu', use_bias=True)
        self.fc2     = keras.layers.Dense(channels,
                                           activation='sigmoid', use_bias=True)
        # Explicitly build each sub-layer so their variables exist
        # BEFORE Keras tries to load weights into them.
        self.gap.build(input_shape)
        self.gmp.build(input_shape)
        self.reshape.build([None, channels])
        self.fc1.build([None, 1, 1, channels])
        self.fc2.build([None, 1, 1, channels // self.ratio])
        super().build(input_shape)

    def call(self, x):
        avg = self.fc2(self.fc1(self.reshape(self.gap(x))))
        mx  = self.fc2(self.fc1(self.reshape(self.gmp(x))))
        return x * (avg + mx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'ratio': self.ratio})
        return cfg


@keras.saving.register_keras_serializable()
class SpatialAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(1, 7, padding='same',
                                         activation='sigmoid', use_bias=True)
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


@keras.saving.register_keras_serializable()
class CBAM(keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.ca = ChannelAttention(ratio=self.ratio)
        self.sa = SpatialAttention()
        self.ca.build(input_shape)
        self.sa.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return self.sa(self.ca(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'ratio': self.ratio})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
KERAS_PATH  = r"C:\Users\Adithya Adiga\Downloads\final_clean_model_v3.keras"
SAVED_MODEL = r"C:\Users\Adithya Adiga\Downloads\agrovision_saved_model_v2"
TFLITE_OUT  = r"C:\Users\Adithya Adiga\Documents\agrovision_ai\assets\model\final_model_float16.tflite"
IMAGE_DIR   = r"C:\Users\Adithya Adiga\Documents\agrovision_ai"

CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy',
]

print(f"TF {tf.__version__} / Keras {keras.__version__}")
print(f"Loading: {KERAS_PATH}")

CUSTOM = {
    'CBAM': CBAM,
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention,
}

model = keras.models.load_model(KERAS_PATH, custom_objects=CUSTOM, compile=False)
print(f"Loaded. Output shape: {model.output_shape}  Params: {model.count_params():,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Test accuracy on the 38 validation images
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(path):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return ((arr / 127.5) - 1.0)[np.newaxis, ...]

def fname_to_expected(fname):
    stem = os.path.splitext(fname)[0]
    for old, new in [
        ('Pepper_bell___', 'Pepper,_bell___'),
        ('Cercospora_leaf_spot_Gray_leaf_spot', 'Cercospora_leaf_spot Gray_leaf_spot'),
        ('Spider_mites_Two-spotted_spider_mite', 'Spider_mites Two-spotted_spider_mite'),
    ]:
        stem = stem.replace(old, new)
    return stem

images = sorted(set(
    f for pat in ['*.JPG','*.jpg','*.jpeg','*.png','*.PNG']
    for f in glob.glob(os.path.join(IMAGE_DIR, pat))
))

correct = total = 0
print(f"{'File':<55} {'Expected':<40} {'Got':<40} {'Conf':>6}  Result")
print("-" * 155)

for path in images:
    fname    = os.path.basename(path)
    expected = fname_to_expected(fname)
    if expected not in CLASS_NAMES:
        print(f"[SKIP] {fname}")
        continue
    exp_idx = CLASS_NAMES.index(expected)
    probs   = model.predict(preprocess(path), verbose=0)[0]
    pred    = int(np.argmax(probs))
    conf    = float(probs[pred])
    ok      = pred == exp_idx
    if ok: correct += 1
    total += 1
    status = "OK" if ok else "WRONG"
    print(f"[{status}] {fname:<52} {expected:<40} {CLASS_NAMES[pred]:<40} {conf:6.1%}")

print()
print("=" * 80)
pct = 100 * correct / max(total, 1)
print(f"KERAS ACCURACY: {correct}/{total} correct  ({pct:.1f}%)")
print()

if pct < 80:
    print("ERROR: Model accuracy still too low. The CBAM loading is still broken.")
    print("Aborting conversion — the TFLite model would be degenerate.")
    sys.exit(1)

print("Accuracy is acceptable. Proceeding with TFLite conversion...\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Export as SavedModel, then convert to TFLite float16
# ─────────────────────────────────────────────────────────────────────────────
print(f"Exporting SavedModel → {SAVED_MODEL}")
tf.saved_model.save(model, SAVED_MODEL)
print("SavedModel saved.\n")

print("Converting to TFLite float16...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_bytes = converter.convert()

os.makedirs(os.path.dirname(TFLITE_OUT), exist_ok=True)
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_bytes)
size_mb = len(tflite_bytes) / 1024 / 1024
print(f"TFLite saved: {TFLITE_OUT}  ({size_mb:.1f} MB)\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Verify TFLite model on the same 38 images
# ─────────────────────────────────────────────────────────────────────────────
print("Verifying TFLite model on 38 test images...")
interp = tf.lite.Interpreter(model_path=TFLITE_OUT)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
print(f"TFLite Input : {inp_d['shape']}  {inp_d['dtype'].__name__}")
print(f"TFLite Output: {out_d['shape']}  {out_d['dtype'].__name__}\n")

correct_t = total_t = 0
print(f"{'File':<55} {'Expected':<40} {'Got':<40} {'Conf':>6}  Result")
print("-" * 155)

for path in images:
    fname    = os.path.basename(path)
    expected = fname_to_expected(fname)
    if expected not in CLASS_NAMES:
        continue
    exp_idx = CLASS_NAMES.index(expected)
    inp_arr = preprocess(path)
    interp.set_tensor(inp_d['index'], inp_arr)
    interp.invoke()
    probs   = interp.get_tensor(out_d['index'])[0]
    pred    = int(np.argmax(probs))
    conf    = float(probs[pred])
    ok      = pred == exp_idx
    if ok: correct_t += 1
    total_t += 1
    print(f"[{'OK' if ok else 'WRONG'}] {fname:<52} {expected:<40} {CLASS_NAMES[pred]:<40} {conf:6.1%}")

print()
print("=" * 80)
pct_t = 100 * correct_t / max(total_t, 1)
print(f"TFLITE ACCURACY: {correct_t}/{total_t} correct  ({pct_t:.1f}%)")
