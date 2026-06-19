"""
Check if the EfficientNetB0 backbone extracts different features for different images.
If backbone outputs are identical → backbone weights failed to load.
"""
import os, io, zipfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import h5py
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
print(f"Full model: {model.input_shape} -> {model.output_shape}")
print(f"Layer names: {[l.name for l in model.layers]}\n")

# Find EfficientNetB0 backbone (the first 'functional' layer)
backbone_eff = None
backbone_mob = None
for layer in model.layers:
    if layer.name == 'functional' or layer.name == 'efficientnetb0':
        backbone_eff = layer
    elif layer.name == 'functional_1' or layer.name == 'mobilenetv2':
        backbone_mob = layer

print(f"EfficientNetB0 backbone: {backbone_eff}")
print(f"MobileNetV2 backbone:    {backbone_mob}\n")

# Check backbone weight statistics
if backbone_eff is not None:
    all_w = backbone_eff.get_weights()
    print(f"EfficientNetB0 weight tensors: {len(all_w)}")
    if all_w:
        w0 = all_w[0]
        print(f"First weight shape: {w0.shape}  mean={w0.mean():.4f}  std={w0.std():.4f}")
        print(f"All-zero? {np.allclose(w0, 0)}")

    # Check if backbone weights from file match
    with zipfile.ZipFile(KERAS_PATH) as zf:
        wb = zf.read('model.weights.h5')
    with h5py.File(io.BytesIO(wb), 'r') as f:
        if 'functional' in f['layers']:
            fn = f['layers']['functional']['layers']
            # Get first conv weight from file
            for lname in fn.keys():
                grp = fn[lname]
                if 'vars' in grp and list(grp['vars'].keys()):
                    vk = list(grp['vars'].keys())[0]
                    v = grp['vars'][vk][:]
                    if len(v.shape) == 4 and v.shape[3] == 32:  # first conv
                        print(f"\nFile conv weight: {v.shape}  mean={v.mean():.4f}  std={v.std():.4f}")
                        if np.allclose(all_w[0], v, atol=1e-3):
                            print("BACKBONE WEIGHTS MATCH - backbone loaded correctly")
                        else:
                            print("BACKBONE WEIGHTS MISMATCH - backbone NOT loaded correctly!")
                        break

# Create intermediate model to check backbone output sensitivity
if backbone_eff is not None:
    print("\n--- Testing backbone output sensitivity ---")
    inp1 = np.full([1, 224, 224, 3], -1.0, dtype=np.float32)
    inp2 = np.full([1, 224, 224, 3],  1.0, dtype=np.float32)
    inp3 = np.random.randn(1, 224, 224, 3).astype(np.float32)

    out1 = backbone_eff(inp1).numpy()
    out2 = backbone_eff(inp2).numpy()
    out3 = backbone_eff(inp3).numpy()

    print(f"Backbone output shape: {out1.shape}")
    print(f"all-neg input:  mean={out1.mean():.4f}  std={out1.std():.4f}  max={out1.max():.4f}")
    print(f"all-pos input:  mean={out2.mean():.4f}  std={out2.std():.4f}  max={out2.max():.4f}")
    print(f"random input:   mean={out3.mean():.4f}  std={out3.std():.4f}  max={out3.max():.4f}")

    diff12 = np.abs(out1 - out2).mean()
    diff13 = np.abs(out1 - out3).mean()
    print(f"\nMean abs diff (all-neg vs all-pos):  {diff12:.6f}")
    print(f"Mean abs diff (all-neg vs random):   {diff13:.6f}")
    if diff12 < 1e-4 and diff13 < 1e-4:
        print("BACKBONE IS INSENSITIVE TO INPUT - backbone weights are zero/broken!")
    else:
        print("Backbone IS sensitive to input - backbone works OK")
        print("The problem is in the head (Dense layers or output layer bias)")

# Check output layer bias directly
print("\n--- Output layer (dense_2) bias ---")
for layer in model.layers:
    if layer.name == 'dense_2':
        bias = layer.bias.numpy()
        sorted_bias = sorted(enumerate(bias), key=lambda x: x[1], reverse=True)
        CLASS_NAMES = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
        print("Top-5 classes by output bias:")
        for idx, val in sorted_bias[:5]:
            print(f"  {CLASS_NAMES[idx]}: bias={val:.4f}")
        print("Bottom-3 classes by output bias:")
        for idx, val in sorted_bias[-3:]:
            print(f"  {CLASS_NAMES[idx]}: bias={val:.4f}")
        break
