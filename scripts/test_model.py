import os, sys, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image
import tensorflow as tf

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

MODEL_PATH = r"C:\Users\Adithya Adiga\Documents\agrovision_ai\assets\model\final_model_float16.tflite"
IMAGE_DIR  = r"C:\Users\Adithya Adiga\Documents\agrovision_ai"

# Load TFLite interpreter
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp_d  = interp.get_input_details()[0]
out_d  = interp.get_output_details()[0]
print(f"Model  : {MODEL_PATH}")
print(f"Input  : {inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"Output : {out_d['shape']}  dtype={out_d['dtype'].__name__}")
print(f"Classes: {len(CLASS_NAMES)}")
print()

def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # EfficientNet preprocess_input: (x / 127.5) - 1.0  ->  [-1, 1]
    arr = (arr / 127.5) - 1.0
    return arr[np.newaxis, ...]   # shape [1, 224, 224, 3]

def fname_to_expected(fname):
    stem = os.path.splitext(fname)[0]
    # Image filenames use underscores where class names use spaces or commas
    replacements = {
        'Pepper_bell___': 'Pepper,_bell___',
        'Cercospora_leaf_spot_Gray_leaf_spot': 'Cercospora_leaf_spot Gray_leaf_spot',
        'Spider_mites_Two-spotted_spider_mite': 'Spider_mites Two-spotted_spider_mite',
    }
    for old, new in replacements.items():
        stem = stem.replace(old, new)
    return stem

# Collect test images
patterns = ['*.JPG', '*.jpg', '*.jpeg', '*.png', '*.PNG']
images = []
for pat in patterns:
    images.extend(glob.glob(os.path.join(IMAGE_DIR, pat)))
images = sorted(set(images))

correct = 0
total   = 0
failures = []

print(f"{'File':<55} {'Expected':<40} {'Got':<40} {'Conf':>6}  Result")
print("-" * 155)

for path in images:
    fname    = os.path.basename(path)
    expected = fname_to_expected(fname)

    if expected not in CLASS_NAMES:
        print(f"{'[SKIP] ' + fname:<55} '{expected}' not in class list")
        continue

    exp_idx = CLASS_NAMES.index(expected)
    inp_arr = preprocess(path)

    interp.set_tensor(inp_d['index'], inp_arr)
    interp.invoke()
    probs   = interp.get_tensor(out_d['index'])[0]

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    correct_p  = pred_idx == exp_idx

    if correct_p:
        correct += 1
        status = "OK"
    else:
        status = "WRONG"
        top3 = np.argsort(probs)[::-1][:3]
        failures.append({
            'file': fname,
            'expected': expected,
            'got': CLASS_NAMES[pred_idx],
            'conf': confidence,
            'top3': [(CLASS_NAMES[i], float(probs[i])) for i in top3],
        })

    total += 1
    got_name = CLASS_NAMES[pred_idx]
    print(f"[{status}] {fname:<52} {expected:<40} {got_name:<40} {confidence:6.1%}")

print()
print("=" * 80)
print(f"RESULT: {correct}/{total} correct  ({100*correct/total:.1f}%)")
print()

if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  {f['file']}")
        print(f"    Expected : {f['expected']}")
        print(f"    Got      : {f['got']}  ({f['conf']:.1%})")
        print(f"    Top-3    : {f['top3']}")
        print()
