"""Check backbone sensitivity and output bias."""
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
        self.gap = keras.layers.GlobalAveragePooling2D()
        self.gmp = keras.layers.GlobalMaxPooling2D()
        self.reshape = keras.layers.Reshape((1, 1, channels))
        self.fc1 = keras.layers.Dense(channels // self.ratio, activation='relu', use_bias=True)
        self.fc2 = keras.layers.Dense(channels, activation='sigmoid', use_bias=True)
        self.gap.build(input_shape); self.gmp.build(input_shape)
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
        self.ca.build(input_shape); self.sa.build(input_shape)
        super().build(input_shape)
    def call(self, x):
        return self.sa(self.ca(x))
    def get_config(self):
        cfg = super().get_config(); cfg.update({'ratio': self.ratio}); return cfg

KERAS_PATH = r"C:\Users\Adithya Adiga\Downloads\final_clean_model_v3.keras"
CUSTOM = {'CBAM': CBAM, 'ChannelAttention': ChannelAttention, 'SpatialAttention': SpatialAttention}

model = keras.models.load_model(KERAS_PATH, custom_objects=CUSTOM, compile=False)
print(f"Layers: {[l.name for l in model.layers]}\n")

# Get backbone
backbone_eff = model.get_layer('efficientnetb0')
print(f"EfficientNetB0 weights: {len(backbone_eff.get_weights())} tensors")

# Check backbone output sensitivity
inp1 = np.full([1, 224, 224, 3], -1.0, dtype=np.float32)
inp2 = np.full([1, 224, 224, 3],  1.0, dtype=np.float32)
inp3 = np.random.randn(1, 224, 224, 3).astype(np.float32)

out1 = backbone_eff(inp1).numpy()
out2 = backbone_eff(inp2).numpy()
out3 = backbone_eff(inp3).numpy()

print(f"Backbone output shape: {out1.shape}")
print(f"all-neg:  mean={out1.mean():.4f}  std={out1.std():.4f}")
print(f"all-pos:  mean={out2.mean():.4f}  std={out2.std():.4f}")
print(f"random:   mean={out3.mean():.4f}  std={out3.std():.4f}")
diff12 = np.abs(out1 - out2).mean()
diff13 = np.abs(out1 - out3).mean()
print(f"Diff (neg vs pos): {diff12:.6f}")
print(f"Diff (neg vs rand): {diff13:.6f}")

if diff12 < 1e-4:
    print("=> BACKBONE IS DEAD (all-neg = all-pos output) - backbone weights broken!")
else:
    print("=> Backbone IS sensitive - backbone works")

# Check output layer bias
print("\n--- Output layer classification bias ---")
CLASS_NAMES = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']

try:
    out_layer = model.get_layer('classification')
    bias = out_layer.bias.numpy()
    sorted_bias = sorted(enumerate(bias), key=lambda x: x[1], reverse=True)
    print("Top-5 by bias:")
    for idx, val in sorted_bias[:5]:
        print(f"  [{idx:2d}] {CLASS_NAMES[idx]}: {val:.4f}")
    print("Bottom-5 by bias:")
    for idx, val in sorted_bias[-5:]:
        print(f"  [{idx:2d}] {CLASS_NAMES[idx]}: {val:.4f}")
except Exception as e:
    print(f"Could not get classification layer: {e}")

# If backbone works but final model still predicts Tomato Late Blight,
# test what happens if we feed zeros directly to the dense layers
print("\n--- Testing dense head with zeros ---")
dense_in = np.zeros([1, 2560], dtype=np.float32)  # 1280+1280 concatenated backbone features
try:
    bn = model.get_layer('bn_head')
    d1 = model.get_layer('dense')
    d2 = model.get_layer('dense_1')
    clf = model.get_layer('classification')

    x = bn(dense_in)
    x = d1(x)
    x = d2(x)
    logits = clf(x).numpy()[0]
    probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    idx = np.argmax(probs)
    print(f"Dense head (zeros in) -> {CLASS_NAMES[idx]} ({probs[idx]:.1%})")
    top3 = np.argsort(probs)[::-1][:3]
    for i in top3:
        print(f"  {CLASS_NAMES[i]}: {probs[i]:.1%}")
except Exception as e:
    print(f"Dense head test failed: {e}")
