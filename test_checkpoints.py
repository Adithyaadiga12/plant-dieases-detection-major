"""Test best_phase1.keras and best_phase2.keras to find which checkpoint works."""
import os, sys, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap     = keras.layers.GlobalAveragePooling2D()
        self.gmp     = keras.layers.GlobalMaxPooling2D()
        self.reshape = keras.layers.Reshape((1, 1, channels))
        self.fc1     = keras.layers.Dense(channels // self.ratio, activation='relu', use_bias=True)
        self.fc2     = keras.layers.Dense(channels, activation='sigmoid', use_bias=True)
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
        cfg = super().get_config(); cfg.update({'ratio': self.ratio}); return cfg

@keras.saving.register_keras_serializable()
class SpatialAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=True)
        h = input_shape[1] if input_shape[1] is not None else 7
        w = input_shape[2] if input_shape[2] is not None else 7
        self.conv.build([None, h, w, 2])
        super().build(input_shape)
    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max(x, axis=-1, keepdims=True)
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
        cfg = super().get_config(); cfg.update({'ratio': self.ratio}); return cfg

CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]
IMAGE_DIR = r"C:\Users\Adithya Adiga\Documents\agrovision_ai"
CUSTOM = {'CBAM': CBAM, 'ChannelAttention': ChannelAttention, 'SpatialAttention': SpatialAttention}

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

def test_model(model_path):
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(model_path)}")
    print('='*70)
    try:
        model = keras.models.load_model(model_path, custom_objects=CUSTOM, compile=False)
        print(f"Loaded. Output: {model.output_shape}  Params: {model.count_params():,}")
    except Exception as e:
        print(f"FAILED to load: {e}")
        return

    correct = total = 0
    results = {}
    for path in images:
        fname    = os.path.basename(path)
        expected = fname_to_expected(fname)
        if expected not in CLASS_NAMES: continue
        probs  = model.predict(preprocess(path), verbose=0)[0]
        pred   = int(np.argmax(probs))
        conf   = float(probs[pred])
        ok     = CLASS_NAMES[pred] == expected
        if ok: correct += 1
        total += 1
        results[fname] = (expected, CLASS_NAMES[pred], conf, ok)

    # Show first 5 and last 5
    items = list(results.items())
    for fname, (exp, got, conf, ok) in items[:5] + items[-5:]:
        print(f"  [{'OK' if ok else 'NO'}] {fname:<50} -> {got} ({conf:.0%})")
    pct = 100 * correct / max(total, 1)
    print(f"\n  RESULT: {correct}/{total} ({pct:.1f}%)")
    return pct, model

candidates = [
    r"C:\Users\Adithya Adiga\Downloads\best_phase2.keras",
    r"C:\Users\Adithya Adiga\Downloads\best_phase1.keras",
    r"C:\Users\Adithya Adiga\Downloads\best_clean_model.keras",
    r"C:\Users\Adithya Adiga\Downloads\final_clean_model.keras",
    r"C:\Users\Adithya Adiga\Downloads\final_clean_model_v3.keras",
]

best_pct = 0
best_model = None
best_path = None
for path in candidates:
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    result = test_model(path)
    if result and result[0] > best_pct:
        best_pct, best_model = result
        best_path = path

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_path} with {best_pct:.1f}% accuracy")
