"""Check if CBAM weights are actually loaded or staying at random init values."""
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

KERAS_PATH = r"C:\Users\Adithya Adiga\Downloads\final_clean_model_v3.keras"
model = keras.models.load_model(KERAS_PATH,
    custom_objects={'CBAM': CBAM, 'ChannelAttention': ChannelAttention, 'SpatialAttention': SpatialAttention},
    compile=False)

# Get CBAM layer and check its weights
cbam_layer = None
for layer in model.layers:
    if 'cbam' in layer.name:
        cbam_layer = layer
        break

if cbam_layer is None:
    print("No CBAM layer found in top-level layers. Searching nested...")
    for layer in model.layers:
        for sublayer in getattr(layer, 'layers', []):
            if 'cbam' in sublayer.name:
                cbam_layer = sublayer
                break

print(f"CBAM layer: {cbam_layer}")
print(f"CBAM.ca attributes: {[attr for attr in dir(cbam_layer) if not attr.startswith('_') and hasattr(cbam_layer, attr)][:20]}")

# Read EXPECTED weights from the .keras file
with zipfile.ZipFile(KERAS_PATH) as zf:
    wb = zf.read('model.weights.h5')
with h5py.File(io.BytesIO(wb), 'r') as f:
    expected_fc1_kernel = f['layers/cbam/ca/fc1/vars/0'][:]
    expected_fc2_kernel = f['layers/cbam/ca/fc2/vars/0'][:]
    expected_conv_kernel = f['layers/cbam/sa/conv/vars/0'][:]

print(f"\nExpected cbam.ca.fc1.kernel from file: shape={expected_fc1_kernel.shape}  mean={expected_fc1_kernel.mean():.4f}  std={expected_fc1_kernel.std():.4f}")
print(f"Expected cbam.ca.fc2.kernel from file: shape={expected_fc2_kernel.shape}  mean={expected_fc2_kernel.mean():.4f}  std={expected_fc2_kernel.std():.4f}")
print(f"Expected cbam.sa.conv.kernel from file: shape={expected_conv_kernel.shape}  mean={expected_conv_kernel.mean():.4f}  std={expected_conv_kernel.std():.4f}")

# Get ACTUAL loaded weights
try:
    actual_fc1_kernel = cbam_layer.ca.fc1.kernel.numpy()
    print(f"\nActual  cbam.ca.fc1.kernel loaded:    shape={actual_fc1_kernel.shape}  mean={actual_fc1_kernel.mean():.4f}  std={actual_fc1_kernel.std():.4f}")
    match_fc1 = np.allclose(expected_fc1_kernel, actual_fc1_kernel, atol=1e-3)
    print(f"fc1 kernel MATCHES expected: {match_fc1}")

    actual_fc2_kernel = cbam_layer.ca.fc2.kernel.numpy()
    print(f"\nActual  cbam.ca.fc2.kernel loaded:    shape={actual_fc2_kernel.shape}  mean={actual_fc2_kernel.mean():.4f}  std={actual_fc2_kernel.std():.4f}")
    match_fc2 = np.allclose(expected_fc2_kernel, actual_fc2_kernel, atol=1e-3)
    print(f"fc2 kernel MATCHES expected: {match_fc2}")

    actual_conv_kernel = cbam_layer.sa.conv.kernel.numpy()
    print(f"\nActual  cbam.sa.conv.kernel loaded:   shape={actual_conv_kernel.shape}  mean={actual_conv_kernel.mean():.4f}  std={actual_conv_kernel.std():.4f}")
    match_conv = np.allclose(expected_conv_kernel, actual_conv_kernel, atol=1e-3)
    print(f"conv kernel MATCHES expected: {match_conv}")

    print(f"\nSummary: fc1={match_fc1}, fc2={match_fc2}, conv={match_conv}")
    if all([match_fc1, match_fc2, match_conv]):
        print("ALL CBAM WEIGHTS LOADED CORRECTLY - problem is elsewhere (bad training?)")
    else:
        print("CBAM WEIGHTS NOT LOADED - this is the bug!")
except Exception as e:
    print(f"\nError accessing CBAM sub-layer weights: {e}")
    print("This means the CBAM sub-layers aren't properly accessible")
