"""
Test the original Keras model (CBAM custom layer required).
If this works → issue is in TFLite conversion.
If this also fails → training is broken.
"""
import os, sys, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image
import tensorflow as tf
import keras  # standalone Keras 3

# ── CBAM sub-layers — must match training code structure exactly ─────────────
# Weight hierarchy in model.weights.h5:
#   cbam/ca/{gap,gmp,fc1,fc2,reshape}/vars/...
#   cbam/sa/conv/vars/...
# fc1/fc2 and sa/conv all have biases (use_bias=True).

@keras.saving.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap     = keras.layers.GlobalAveragePooling2D(name='gap')
        self.gmp     = keras.layers.GlobalMaxPooling2D(name='gmp')
        self.reshape = keras.layers.Reshape((1, 1, channels), name='reshape')
        self.fc1     = keras.layers.Dense(channels // self.ratio,
                                           activation='relu', name='fc1')
        self.fc2     = keras.layers.Dense(channels, name='fc2')
        super().build(input_shape)

    def call(self, x):
        avg = self.fc2(self.fc1(self.gap(x)))
        mx  = self.fc2(self.fc1(self.gmp(x)))
        return x * tf.sigmoid(self.reshape(avg + mx))

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
                                         activation='sigmoid', name='conv')
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
        self.ca = ChannelAttention(self.ratio, name='ca')
        self.sa = SpatialAttention(name='sa')
        super().build(input_shape)

    def call(self, x):
        return self.sa(self.ca(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'ratio': self.ratio})
        return cfg


CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

KERAS_PATH = r"C:\Users\Adithya Adiga\Downloads\final_clean_model_v3.keras"
IMAGE_DIR  = r"C:\Users\Adithya Adiga\Documents\agrovision_ai"

print(f"TensorFlow: {tf.__version__}")
print(f"Loading: {KERAS_PATH}")
print("(May take 30-60 seconds...)\n")

try:
    model = keras.models.load_model(
        KERAS_PATH,
        custom_objects={'CBAM': CBAM},
        compile=False,
    )
    print("Model loaded successfully.\n")
except Exception as e:
    print(f"Load failed: {e}")
    sys.exit(1)

print(f"Input shape : {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Num classes : {model.output_shape[-1]}\n")

def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr[np.newaxis, ...]

def fname_to_expected(fname):
    stem = os.path.splitext(fname)[0]
    replacements = {
        'Pepper_bell___': 'Pepper,_bell___',
        'Cercospora_leaf_spot_Gray_leaf_spot': 'Cercospora_leaf_spot Gray_leaf_spot',
        'Spider_mites_Two-spotted_spider_mite': 'Spider_mites Two-spotted_spider_mite',
    }
    for old, new in replacements.items():
        stem = stem.replace(old, new)
    return stem

patterns = ['*.JPG', '*.jpg', '*.jpeg', '*.png', '*.PNG']
images = []
for pat in patterns:
    images.extend(glob.glob(os.path.join(IMAGE_DIR, pat)))
images = sorted(set(images))

correct = 0
total   = 0
failures = []

print(f"{'File':<55} {'Expected':<42} {'Got':<42} {'Conf':>6}  Result")
print("-" * 160)

for path in images:
    fname    = os.path.basename(path)
    expected = fname_to_expected(fname)

    if expected not in CLASS_NAMES:
        print(f"[SKIP] {fname:<52} '{expected}' not in class list")
        continue

    exp_idx  = CLASS_NAMES.index(expected)
    inp      = preprocess(path)
    probs    = model.predict(inp, verbose=0)[0]

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    ok = pred_idx == exp_idx

    if ok:
        correct += 1
        status = "OK"
    else:
        status = "WRONG"
        top3 = np.argsort(probs)[::-1][:3]
        failures.append({
            'file': fname, 'expected': expected,
            'got': CLASS_NAMES[pred_idx], 'conf': confidence,
            'top3': [(CLASS_NAMES[i], float(probs[i])) for i in top3],
        })

    total += 1
    print(f"[{status}] {fname:<52} {expected:<42} {CLASS_NAMES[pred_idx]:<42} {confidence:6.1%}")

print()
print("=" * 80)
print(f"KERAS RESULT: {correct}/{total} correct  ({100*correct/max(total,1):.1f}%)")
print()

if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  {f['file']}")
        print(f"    Expected : {f['expected']}")
        print(f"    Got      : {f['got']}  ({f['conf']:.1%})")
        print(f"    Top-3    : {f['top3']}")
        print()
