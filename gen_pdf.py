from fpdf import FPDF

W = 186  # A4 210 - 12 left - 12 right

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(130, 130, 130)
        self.cell(W, 6, "AgroVision AI  -  Final Exam Preparation Guide", align="L")
        self.ln(6)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(W, 5, f"Page {self.page_no()}", align="C")

    def h1(self, text):
        self.ln(5)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(21, 101, 192)
        self.cell(W, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(21, 101, 192)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.l_margin + W, self.get_y())
        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)
        self.set_text_color(26, 26, 26)
        self.ln(4)

    def h2(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(74, 20, 140)
        self.cell(W, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(26, 26, 26)
        self.ln(1)

    def para(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(26, 26, 26)
        self.write(5.5, text)
        self.ln(6)

    def b(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(26, 26, 26)
        self.set_x(self.l_margin + 4)
        self.write(5.5, "- " + text)
        self.ln(6)

    def tbl(self, rows, c1w=62):
        c2w = W - c1w
        for i, (c1, c2, hdr) in enumerate(rows):
            self.set_font("Helvetica", "B" if hdr else "", 9.5)
            if hdr:
                self.set_fill_color(21, 101, 192)
                self.set_text_color(255, 255, 255)
            elif i % 2 == 0:
                self.set_fill_color(245, 245, 245)
                self.set_text_color(26, 26, 26)
            else:
                self.set_fill_color(255, 255, 255)
                self.set_text_color(26, 26, 26)
            self.cell(c1w, 7, f"  {c1}", border=1, fill=True)
            self.cell(c2w, 7, f"  {c2}", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(26, 26, 26)
        self.ln(3)

    def qa(self, q, a):
        # Estimate total height needed; add page if won't fit
        needed = 30  # rough: 1 line Q + 3 lines A + spacing
        if self.get_y() + needed > (self.h - self.b_margin):
            self.add_page()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 70, 127)
        self.write(6, q)
        self.ln(6)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(26, 26, 26)
        self.write(5.5, a)
        self.ln(9)

    def slide(self, num, title, bullets):
        # Estimate height; add page if won't fit
        needed = 10 + len(bullets) * 6
        if self.get_y() + needed > (self.h - self.b_margin):
            self.add_page()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(13, 71, 161)
        self.write(6, f"Slide {num}: {title}")
        self.ln(7)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(26, 26, 26)
        for bul in bullets:
            self.set_x(self.l_margin + 5)
            self.write(5.5, "- " + bul)
            self.ln(6)
        self.ln(2)


# ── Build PDF ──────────────────────────────────────────────────────────────────
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=16)
pdf.set_margins(12, 14, 12)
pdf.add_page()

# ── COVER ──────────────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(27, 94, 32)
pdf.cell(W, 12, "AgroVision AI", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 13)
pdf.set_text_color(21, 101, 192)
pdf.cell(W, 8, "Offline Plant Disease Detection Using Deep Learning on Mobile", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(90, 90, 90)
pdf.cell(W, 7, "Final Exam Preparation Guide", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(50, 50, 50)
pdf.cell(W, 6, "Stack:  Flutter  |  TFLite  |  EfficientNetB0 + MobileNetV2 + CBAM  |  PlantVillage  |  Python/Keras", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)

# ── 1. ABSTRACT ───────────────────────────────────────────────────────────────
pdf.h1("1. Abstract")
pdf.para(
    "Plant diseases cause significant crop losses globally, yet early and accurate diagnosis remains "
    "inaccessible to smallholder farmers due to expensive expert consultation and unreliable internet "
    "connectivity in rural areas."
)
pdf.para(
    "This project presents AgroVision AI, an offline-first Android application that performs real-time "
    "plant disease classification entirely on-device using a custom dual-backbone deep learning model. "
    "The model combines EfficientNetB0 and MobileNetV2 feature extractors, enhanced with a Convolutional "
    "Block Attention Module (CBAM), trained on the PlantVillage dataset (87,848 images, 38 disease "
    "classes, 14 crop species). Test accuracy is approximately 98.3%. The 14.8 MB TFLite Float16 model "
    "runs offline on any Android 5.0+ device. The Flutter app supports 5 languages (EN, KN, HI, TA, TE), "
    "voice output via TTS, image quality validation, scan history, and an offline disease treatment library."
)
pdf.set_font("Helvetica", "BI", 10)
pdf.set_text_color(0, 100, 0)
pdf.write(5.5,
    'Key sentence: "AgroVision AI brings expert-level plant disease diagnosis offline, directly to a '
    "farmer's smartphone, with 98.3% accuracy and no internet dependency.\""
)
pdf.set_text_color(26, 26, 26)
pdf.ln(6)

# ── 2. METHODOLOGY ────────────────────────────────────────────────────────────
pdf.h1("2. Methodology")

pdf.h2("2.1  Dataset")
pdf.b("PlantVillage dataset - 87,848 RGB leaf images (Kaggle, open-source)")
pdf.b("38 classes: diseases + healthy, covering 14 crops (Tomato, Pepper, Grape, Apple, Corn, Potato...)")
pdf.b("Split: 80% Train (70,278) / 10% Validation (8,785) / 10% Test (8,785)")
pdf.b("Preprocessing: resize 224x224, normalize pixel = (px / 127.5) - 1.0  -> range [-1, 1]")
pdf.b("Augmentation (train only): random H/V flip, rotation +/-25 deg, zoom +/-20%, brightness +/-20%")
pdf.ln(2)

pdf.h2("2.2  Model Architecture - Dual Backbone + CBAM")
pdf.para("Hybrid dual-backbone CNN with attention applied to each backbone output:")
pdf.tbl([
    ("Component", "Detail", True),
    ("Input", "224 x 224 x 3 RGB image", False),
    ("EfficientNetB0", "ImageNet pretrained, 5.3M params, MBConv blocks -> 1,280-d via GAP", False),
    ("CBAM #1", "Channel + Spatial Attention on EfficientNet output", False),
    ("MobileNetV2", "ImageNet pretrained, 3.4M params, Inverted Residuals -> 1,280-d via GAP", False),
    ("CBAM #2", "Channel + Spatial Attention on MobileNet output", False),
    ("Concatenate", "1,280 + 1,280 = 2,560-d joint feature vector", False),
    ("Dense(512) + BN", "BatchNorm + ReLU + Dropout(0.5)", False),
    ("Dense(256)", "ReLU + Dropout(0.4)", False),
    ("Dense(38) Softmax", "Output: probability over 38 disease classes", False),
])

pdf.h2("2.3  CBAM - Convolutional Block Attention Module")
pdf.b("Channel Attention: GAP + GMP -> shared FC layers -> sigmoid -> weights which feature channels matter most")
pdf.b("Spatial Attention: avg + max across channels -> 7x7 Conv -> sigmoid mask -> highlights WHERE disease is on leaf")
pdf.b("Applied sequentially: Channel Attention first, then Spatial Attention")
pdf.b("Why: disease lesions are localized - attention suppresses background, forces focus on lesion region")
pdf.ln(2)

pdf.h2("2.4  Training Strategy - Two-Phase Transfer Learning")
pdf.tbl([
    ("Phase", "Epochs | LR | Frozen Layers | Goal", True),
    ("Phase 1 - Head only", "8 epochs | lr=1e-3 | Both backbones frozen | Train head", False),
    ("Phase 2 - Fine-tune", "4 epochs | lr=5e-5 | Top-20 backbone layers unfrozen | Adapt", False),
])
pdf.b("Optimizer: Adam with Mixed Float16 precision (faster GPU training)")
pdf.b("Loss function: Categorical Cross-Entropy")
pdf.b("Callbacks: ModelCheckpoint (save best val_accuracy), EarlyStopping, ReduceLROnPlateau")
pdf.b("Platform: Kaggle NVIDIA T4 GPU | approx. 2 hours total training time")
pdf.ln(2)

pdf.h2("2.5  TFLite Conversion")
pdf.b("Best .keras checkpoint reloaded in pure Float32")
pdf.b("Converted with TFLite Float16 quantization (tf.lite.Optimize.DEFAULT)")
pdf.b("Output: final_model_float16.tflite  -  14.8 MB  |  Input [1,224,224,3]  |  Output [1,38]")
pdf.ln(2)

pdf.h2("2.6  Flutter Application Architecture")
pdf.b("Presentation Layer: SplashScreen, Home, Scan, Result, History, Library, Settings (5-tab nav bar)")
pdf.b("Service Layer: TFLiteService, ImageQualityService, TtsService, HistoryService, DatabaseService, LocaleService")
pdf.b("Data Layer: Hive (scan history, offline), SQLite (treatment DB), TFLite model asset (on-device)")
pdf.b("100% offline - zero network calls at any time")
pdf.ln(2)

pdf.h2("2.7  On-Device Inference Pipeline (6 Steps)")
pdf.b("Step 1: User captures image via camera or selects from gallery")
pdf.b("Step 2 - Quality Gate: file <=15 MB, dimensions >=100px, brightness avg 20-245, Laplacian variance >8 (blur), HSV green ratio >2% (leaf check)")
pdf.b("Step 3 - Preprocess: bake EXIF orientation, bicubic resize 224x224, normalize (px/127.5)-1.0 -> Float32List of 150,528 values")
pdf.b("Step 4 - Inference: Float32List -> TFLite interpreter.invoke() -> output [1,38] softmax probabilities - all on-device")
pdf.b("Step 5 - Post-process: argmax -> top class, confidence >=40% threshold, severity: <75%=Mild, 75-90%=Moderate, >=90%=Severe")
pdf.b("Step 6 - Result: disease name, confidence %, severity badge, symptoms/remedy/prevention, TTS readout, auto-save to history")
pdf.ln(2)

# ── 3. RESULTS ────────────────────────────────────────────────────────────────
pdf.h1("3. Results Obtained")

pdf.h2("3.1  Model Performance")
pdf.tbl([
    ("Metric", "Value", True),
    ("Test Set Accuracy", "approx. 98.3%", False),
    ("Model Size (Float16 TFLite)", "14.8 MB", False),
    ("Number of Classes", "38 (diseases + healthy)", False),
    ("Crop Species Covered", "14 species", False),
    ("Total Dataset", "87,848 images", False),
    ("Training Images", "70,278", False),
    ("Training Platform", "Kaggle NVIDIA T4 GPU", False),
    ("Inference", "100% on-device - no internet required", False),
])

pdf.h2("3.2  Application Features Delivered")
pdf.tbl([
    ("Feature", "Implementation", True),
    ("Offline inference", "TFLite model bundled in APK, zero network calls", False),
    ("Image quality gate", "5 checks: size, blur, brightness, leaf detect, dimensions", False),
    ("Confidence threshold", "Below 40% -> 'Not Recognized' (prevents false diagnoses)", False),
    ("Severity rating", "Mild (<75%) / Moderate (75-90%) / Severe (>=90%)", False),
    ("Voice output TTS", "EN, Kannada, Hindi, Tamil, Telugu", False),
    ("Scan history", "All scans stored locally in Hive NoSQL (offline)", False),
    ("Disease library", "38 diseases: symptoms, causes, remedy, prevention (SQLite)", False),
    ("Multilingual UI", "Full app UI in 5 languages", False),
    ("Material 3 UI", "Dark/light theme, animations, Hero transitions", False),
])

pdf.h2("3.3  Key Design Decisions")
pdf.b("Dual backbone: EfficientNetB0 (fine textures/spots) + MobileNetV2 (leaf structure) give complementary features, boosting accuracy vs. single backbone.")
pdf.b("CBAM: Lesions are localized. Attention forces focus on disease region, suppressing irrelevant background.")
pdf.b("Float16 quantization: Halves model size (29MB -> 14.8MB) with <0.5% accuracy loss, enabling small APK.")
pdf.b("Offline-first: Farmers in rural India have unreliable internet. App must work with zero connectivity.")
pdf.b("Image quality gate: Bad input produces meaningless predictions - gating prevents harmful misdiagnoses.")
pdf.ln(2)

# ── 4. PPT OUTLINE ────────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("4. PPT Slide Outline  (12 Slides, 10-15 Minutes)")
pdf.para("Aim for 1 to 1.5 minutes per slide. Reference DIAGRAMS.md for Figures 1-5.")

pdf.slide(1, "Title Slide", [
    "AgroVision AI - Offline Plant Disease Detection Using Deep Learning on Mobile",
    "Your name, date, institution name",
    "Add app logo or a diseased leaf photo as visual",
])
pdf.slide(2, "Problem Statement & Motivation", [
    "Plant diseases cause 20-40% annual crop loss worldwide (FAO)",
    "Expert diagnosis is expensive, slow, and unavailable in rural areas",
    "Farmers in rural India have no reliable internet for cloud AI tools",
    "No existing affordable, offline, accurate diagnostic tool -> AgroVision AI solves this",
])
pdf.slide(3, "Project Objectives", [
    "Build a deep learning model with >95% accuracy on 38 plant disease classes",
    "Deploy as a lightweight TFLite model on Android (no GPU, no server required)",
    "Develop a fully offline Flutter app with multilingual support (5 languages)",
    "Provide actionable disease treatment information directly to farmers",
])
pdf.slide(4, "System Architecture", [
    "Show Fig. 1 from DIAGRAMS.md (Overall System Architecture)",
    "4 layers: Presentation (Flutter screens), Service (logic), Data (Hive/SQLite/model), ML Engine",
    "Key message: Everything runs on the phone. No server, no internet required.",
])
pdf.slide(5, "Dataset", [
    "PlantVillage - 87,848 images | 38 classes | 14 crop species",
    "Split: 80% Train (70,278) / 10% Val / 10% Test (8,785 each)",
    "Preprocessing: resize 224x224, normalize (px/127.5)-1.0 -> range [-1, 1]",
    "Augmentation (train only): flip, rotation, zoom, brightness variations",
    "Tip: add a 3x3 grid of sample leaf images showing different disease classes",
])
pdf.slide(6, "Model Architecture - Dual Backbone + CBAM", [
    "Show Fig. 2 from DIAGRAMS.md (CNN Model Architecture)",
    "EfficientNetB0 (5.3M params) + MobileNetV2 (3.4M params) - both ImageNet pretrained",
    "CBAM attention applied to each backbone output independently",
    "Concatenate: 1,280 + 1,280 = 2,560-d joint feature vector",
    "Head: BatchNorm -> Dense(512, ReLU) -> Dense(256, ReLU) -> Dense(38, Softmax)",
])
pdf.slide(7, "CBAM - Attention Mechanism", [
    "Channel Attention: GAP + GMP -> FC layers -> sigmoid -> weights which channels matter",
    "Spatial Attention: avg+max across channels -> 7x7 Conv -> sigmoid mask -> WHERE is disease",
    "Applied sequentially: Channel first, then Spatial",
    "Why: disease lesions are localized - attention removes background, focuses on lesion",
])
pdf.slide(8, "Training Pipeline", [
    "Show Fig. 3 from DIAGRAMS.md (Training Pipeline)",
    "Phase 1 (8 epochs, lr=1e-3): backbones frozen, train classification head only",
    "Phase 2 (4 epochs, lr=5e-5): top-20 layers per backbone unfrozen for fine-tuning",
    "Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau",
    "Kaggle T4 GPU | Mixed Float16 | ~2 hours | converted to TFLite Float16 -> 14.8 MB",
])
pdf.slide(9, "On-Device Inference Pipeline", [
    "Show Fig. 4 from DIAGRAMS.md (Mobile App Inference Pipeline)",
    "Steps 1-2: Image input -> 5-check quality gate (reject blurry/dark/non-leaf images)",
    "Step 3: Preprocess - resize 224x224, normalize -> Float32List of 150,528 values",
    "Step 4: TFLite interpreter.invoke() runs forward pass entirely on-device",
    "Steps 5-6: Argmax + threshold + severity -> Result screen + TTS + auto-save history",
])
pdf.slide(10, "App Screenshots / Demo", [
    "Scan screen: camera + gallery buttons, pulse animation, quality tips",
    "Result screen: disease name, confidence bar, severity badge (Mild/Moderate/Severe)",
    "Disease library: searchable grid of 38 disease cards with treatment bottom sheet",
    "History screen: past scan cards with filter chips (All / Healthy / Disease)",
    "Settings: language selector (EN, KN, HI, TA, TE) and TTS toggle",
])
pdf.slide(11, "Results & Performance", [
    "Test Accuracy: approx. 98.3% on 8,785-image held-out test set",
    "Model Size: 14.8 MB Float16 TFLite - fits inside the app bundle",
    "38 disease classes across 14 crop species",
    "5 language support with full TTS voice output",
    "0% internet dependency - works in airplane mode",
])
pdf.slide(12, "Conclusion & Future Work", [
    "Built offline plant disease detector achieving ~98.3% accuracy",
    "Runs on any Android 5.0+ device with no internet and no cost",
    "Multilingual TTS makes it accessible to non-literate farmers",
    "Future: expand to Indian crop diseases, INT8 quantization, lesion segmentation, iOS port",
])

# ── 5. EXAM Q&A ───────────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("5. Likely Exam Questions & Answers")

pdf.qa(
    "Q1: Why did you use two backbones instead of one?",
    "Two backbones extract complementary features. EfficientNetB0 focuses on fine-grained textures "
    "(disease spots, color change) while MobileNetV2 captures structural patterns (leaf morphology). "
    "Concatenating both gives a richer 2,560-d representation than either alone, improving accuracy."
)
pdf.qa(
    "Q2: What is CBAM and why use it?",
    "CBAM (Convolutional Block Attention Module) applies channel-wise then spatial attention. Channel "
    "attention learns which feature maps matter most; spatial attention highlights where on the leaf "
    "the disease is. Since disease occupies a small localized region, attention suppresses background "
    "and improves detection focus."
)
pdf.qa(
    "Q3: Why TFLite and not a cloud server?",
    "Target users are farmers in rural India with no reliable internet. Cloud inference would make "
    "the app unusable in the field. TFLite enables fully offline, low-latency inference on any "
    "Android 5.0+ device with zero network dependency."
)
pdf.qa(
    "Q4: What is Float16 quantization? Does it hurt accuracy?",
    "Float16 stores weights in 16-bit floating point instead of 32-bit, halving model size from "
    "~29 MB to 14.8 MB. Accuracy loss is negligible (typically <0.5%). It also speeds up inference "
    "on hardware with native Float16 support."
)
pdf.qa(
    "Q5: Why do you have a confidence threshold?",
    "A misdiagnosis is harmful - a farmer might apply the wrong pesticide. If the model confidence "
    "is below 40%, we show 'Not Recognized' instead of a potentially wrong answer, prompting the "
    "user to retake the photo or consult an expert."
)
pdf.qa(
    "Q6: What is transfer learning and why use it?",
    "Transfer learning reuses a model pretrained on ImageNet (1.2M images, 1000 classes). The "
    "backbone already detects edges, textures, and shapes. We fine-tune it for plant disease "
    "classification with far less data and compute than training from scratch."
)
pdf.qa(
    "Q7: What is two-phase training?",
    "Phase 1 freezes the backbone and trains only the classification head - fast convergence, "
    "avoids destroying pretrained weights. Phase 2 unfreezes the top backbone layers and fine-tunes "
    "them at a very low lr (5e-5) to adapt to plant leaves. More stable than unfreezing all at once."
)
pdf.qa(
    "Q8: What does the image quality service do?",
    "5 checks run before inference: file <=15 MB, dimensions >=100px, brightness avg 20-245 "
    "(not dark/overexposed), Laplacian variance >8 (not blurry), HSV green ratio >2% (actually a "
    "leaf). Any failure prompts the user to retake the photo."
)
pdf.qa(
    "Q9: How does the app show disease info offline?",
    "All 38 disease treatments (symptoms, causes, remedy, prevention) are stored in an SQLite "
    "database bundled inside the APK at install time. No internet call is ever made."
)
pdf.qa(
    "Q10: What is the PlantVillage dataset?",
    "PlantVillage is an open-source dataset of 87,848 labeled leaf images covering 38 disease "
    "categories across 14 crop species. It is the standard benchmark for plant disease "
    "classification research."
)

# ── 6. TECH STACK ─────────────────────────────────────────────────────────────
pdf.h1("6. Tech Stack Quick Reference")
pdf.tbl([
    ("Component", "Technology / Detail", True),
    ("Mobile Framework", "Flutter 3.41 / Dart 3.11 | Android arm64", False),
    ("ML Runtime", "tflite_flutter ^0.12.1 | On-device, fully offline", False),
    ("Model", "EfficientNetB0 + MobileNetV2 + CBAM | 38-class Softmax", False),
    ("Model File", "Float16 TFLite | 14.8 MB | Input [1,224,224,3] | Output [1,38]", False),
    ("Dataset", "PlantVillage | 87,848 images | 38 classes | 14 crops", False),
    ("Training", "Keras 3 / TensorFlow | Kaggle T4 GPU | Mixed Float16", False),
    ("Scan History DB", "Hive (NoSQL, offline)", False),
    ("Disease Treatment DB", "SQLite | 38 diseases, symptoms + remedy + prevention", False),
    ("Voice Output TTS", "flutter_tts ^4.1.0 | EN, Kannada, Hindi, Tamil, Telugu", False),
    ("UI Design", "Material 3 | Dark/light theme | Hero animations", False),
    ("Min Android", "API 21 (Android 5.0+)", False),
])

pdf.ln(4)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(150, 150, 150)
pdf.cell(W, 5, "AgroVision AI  |  Exam Prep  |  Good luck!", align="C")

out = "C:/Users/Adithya Adiga/Documents/agrovision_ai/AgroVision_Exam_Prep.pdf"
pdf.output(out)
print(f"Done: {out}")
