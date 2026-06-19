# AgroVision AI

A modern, offline-first Flutter application for plant disease detection using TensorFlow Lite. Detects 38 plant disease classes entirely on-device — no internet required.

---

## ⚠️ Important: Real-World Limitations

**This app is trained on PlantVillage — a controlled laboratory dataset.** The images in that dataset were taken under consistent studio lighting, against plain backgrounds, with leaves filling the entire frame. Real-world photos differ significantly.

### What this means in practice

| Condition | Lab (PlantVillage) | Real World |
|---|---|---|
| Background | Plain / uniform | Soil, grass, other plants |
| Lighting | Controlled, even | Sunlight, shadows, mixed |
| Leaf coverage | Full frame | Partial, angled, distant |
| Image quality | High resolution, sharp | Variable |

**The model may hallucinate (output a wrong disease label with high confidence) when given real-world photos that differ from lab conditions.** This is a known limitation of models trained solely on PlantVillage data.

### For best results

- Place the leaf **flat and close** to the camera — it should fill most of the frame
- Use **even, diffused lighting** — avoid harsh direct sunlight or deep shadows
- Shoot against a **plain background** if possible
- Keep the camera **steady** — blur causes wrong predictions
- Scan **one leaf at a time**

The app has a quality gate that rejects blurry, too-dark, overexposed, and non-leaf images, and a confidence threshold that suppresses low-certainty predictions. However, these cannot fully compensate for the domain gap between lab and field conditions.

---

## ✨ Features

- **38-class plant disease classification** — covers Apple, Tomato, Potato, Corn, Grape, and more
- **Fully offline** — TFLite model runs entirely on-device
- **Image quality gate** — rejects blurry, dark, overexposed, and non-leaf photos before inference
- **Confidence thresholding** — suppresses uncertain predictions
- **Scan history** — Hive-backed local history of all scans
- **Disease library** — browse all 38 classes with symptoms, causes, and treatment info
- **Text-to-speech** — reads out results in Kannada, Hindi, or English
- **Material 3 UI** — dark/light theme, smooth animations

---

## 🧪 Test Images

A curated zip of test images is included in this repo for evaluating the app:

**`test_images_10per_class.zip`** — 10 images per class sampled from PlantVillage.

> These are **lab-condition images** and represent the best-case scenario for the model. Use them to verify the app is working correctly before testing with real-world photos.

To use:
1. Extract the zip
2. Transfer images to your phone
3. Open AgroVision AI → Scan → Gallery → select an image

---

## 🚀 Getting Started

### Prerequisites
- Flutter SDK 3.x
- Android device or emulator (API 21+)

### Run from source

```bash
git clone https://github.com/Adithyaadiga12/plant-dieases-detection-major.git
cd plant-dieases-detection-major
flutter pub get
flutter run
```

### Build APK

```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

---

## 🏗️ Architecture

| Layer | Technology |
|---|---|
| UI | Flutter, Material 3 |
| ML inference | TensorFlow Lite (`tflite_flutter`) |
| Model | EfficientNetB0 + MobileNetV2 + CBAM attention |
| Local storage | Hive |
| Image handling | `image_picker`, `image` |
| TTS | `flutter_tts` |

**Model input:** `[1, 224, 224, 3]` — EfficientNet preprocessing `(pixel / 127.5) - 1.0`  
**Model output:** `[1, 38]` softmax probabilities  
**Confidence threshold:** 0.75 (below this → "Not Recognized")

---

## 📋 Supported Classes

38 PlantVillage classes across: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
