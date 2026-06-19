"""Check if model output varies with different inputs (is model input-sensitive?)"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = keras.layers.GlobalAveragePooling2D(name='gap')
        self.gmp = keras.layers.GlobalMaxPooling2D(name='gmp')
        self.reshape = keras.layers.Reshape((1, 1, channels), name='reshape')
        self.fc1 = keras.layers.Dense(channels // self.ratio, activation='relu', name='fc1')
        self.fc2 = keras.layers.Dense(channels, name='fc2')
        super().build(input_shape)
    def call(self, x):
        avg = self.fc2(self.fc1(self.gap(x)))
        mx  = self.fc2(self.fc1(self.gmp(x)))
        return x * tf.sigmoid(self.reshape(avg + mx))
    def get_config(self):
        cfg = super().get_config(); cfg.update({'ratio': self.ratio}); return cfg

@keras.saving.register_keras_serializable()
class SpatialAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid', name='conv')
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
        cfg = super().get_config(); cfg.update({'ratio': self.ratio}); return cfg

print("Loading model...")
model = keras.models.load_model(
    r"C:\Users\Adithya Adiga\Downloads\final_clean_model_v3.keras",
    custom_objects={'CBAM': CBAM, 'ChannelAttention': ChannelAttention,
                    'SpatialAttention': SpatialAttention},
    compile=False,
)
print("Loaded.\n")

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
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy',
]

tests = {
    'zeros [-1 range]':    np.full([1,224,224,3], -1.0, dtype=np.float32),
    'ones  [+1 range]':    np.full([1,224,224,3],  1.0, dtype=np.float32),
    'mid   [0.0 range]':   np.full([1,224,224,3],  0.0, dtype=np.float32),
    'random_1':            (np.random.randn(1,224,224,3)*0.5).astype(np.float32),
    'random_2':            (np.random.randn(1,224,224,3)*0.5).astype(np.float32),
}

all_probs = []
for name, inp in tests.items():
    probs = model.predict(inp, verbose=0)[0]
    all_probs.append(probs)
    idx  = int(np.argmax(probs))
    top3 = list(np.argsort(probs)[::-1][:3])
    print(f"{name:20}: [{CLASS_NAMES[idx]}]  {probs[idx]*100:.1f}%")
    print(f"{'':22}  top3: " + " | ".join(f"{CLASS_NAMES[i]}({probs[i]*100:.1f}%)" for i in top3))

print()
# Measure how much outputs vary across inputs
arr = np.stack(all_probs)
print(f"Output std across all test inputs (per class, mean): {arr.std(axis=0).mean():.6f}")
print(f"Max variation in any class across inputs:            {arr.std(axis=0).max():.6f}")
print()
print("If std ~ 0, the model is completely input-insensitive (degenerate).")
print("If std > 0.01, the model sees input differences but still predicts wrong class.")
