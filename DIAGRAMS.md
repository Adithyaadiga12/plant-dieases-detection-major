# AgroVision AI — Architecture Diagrams

> **How to export high-quality images:**
> 1. Go to **https://mermaid.live**
> 2. Paste any diagram block below (including the ` ```mermaid ` fences)
> 3. Click **Export → SVG** (vector, infinite resolution — best for paper/report)
>    or **Export → PNG** at 4× scale for slides
>
> Alternatively, open this file in **VS Code** with the _Mermaid Preview_ extension.

---

## Fig. 1 — Overall System Architecture

*Use in: paper (Section III — System Design), report cover architecture section, PPT slide 4–5*

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '15px', 'fontFamily': 'Segoe UI, Arial'}}}%%
flowchart TB
    USR(["👤  Farmer / Agronomist\n(End User)"])

    subgraph APP["AgroVision AI — Flutter Android Application"]
        direction TB

        subgraph PL["①  Presentation Layer — Flutter Screens"]
            direction LR
            Sc1["Home\nScreen"]
            Sc2["Scan\nScreen"]
            Sc3["Result\nScreen"]
            Sc4["History\nScreen"]
            Sc5["Disease\nLibrary"]
            Sc6["Settings\nScreen"]
        end

        subgraph SL["②  Service Layer — Business Logic"]
            direction LR
            Sv1["TFLite\nService"]
            Sv2["Image Quality\nService"]
            Sv3["TTS\nService"]
            Sv4["History\nService"]
            Sv5["Database\nService"]
            Sv6["Locale Service\nEN · KN · HI · TA · TE"]
        end

        subgraph DL["③  Data Layer — On-Device Storage"]
            direction LR
            Da1[("Hive\nScan History")]
            Da2[("SQLite\n38 Disease Treatments")]
            Da3[/"TFLite Model\nFloat16 · 14.8 MB"/]
        end

        subgraph ML["④  ML Engine — On-Device TFLite Runtime"]
            direction LR
            Ml1["EfficientNetB0\nFeature Extractor"]
            Ml2["MobileNetV2\nFeature Extractor"]
            Ml3["38-Class\nSoftmax Classifier"]
        end
    end

    USR        -->|"Camera / Gallery photo"| PL
    PL         -->|"UI events → service calls"| SL
    SL         -->|"persist / query"| DL
    Da3        -->|"model weights loaded once"| ML
    ML         -->|"disease class + confidence"| SL
    SL         -->|"display results"| PL
    PL         -->|"TTS voice output"| USR
```

---

## Fig. 2 — Dual-Backbone CNN Model Architecture

*Use in: paper (Section IV — Proposed Model), report model section, PPT slide 7–8*
*This is the most important diagram for the research paper.*

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '15px', 'fontFamily': 'Segoe UI, Arial'}}}%%
flowchart TB
    IN["Input Image\n224 × 224 × 3  (RGB)"]

    IN --> EFF
    IN --> MOB

    subgraph EFF["EfficientNetB0 Backbone  (5.3 M params)"]
        direction TB
        E1["Stem Conv  3×3\n+ MBConv Blocks  ×16\nCompound Scaled Feature Maps"]
        E2["Global Average Pooling 2D\n→ 1,280-d Feature Vector"]
        E1 --> E2
    end

    subgraph MOB["MobileNetV2 Backbone  (3.4 M params)"]
        direction TB
        M1["Stem Conv  3×3\n+ Inverted Residual Blocks  ×17\nDepthwise Separable Convolutions"]
        M2["Global Average Pooling 2D\n→ 1,280-d Feature Vector"]
        M1 --> M2
    end

    E2 --> CAT
    M2 --> CAT

    CAT["Concatenate\n1,280 + 1,280 = 2,560-d Joint Feature Vector"]

    subgraph HEAD["Classification Head  (trainable from epoch 1)"]
        direction TB
        BN["Batch Normalization"]
        D1["Dense  512  ·  ReLU\nDropout  0.4"]
        D2["Dense  256  ·  ReLU\nDropout  0.3"]
        OUT["Dense  38  ·  Softmax\n38 Plant Disease Classes"]
        BN --> D1 --> D2 --> OUT
    end

    CAT --> HEAD

    OUT --> PRED["Predicted Disease Class\n+ Confidence Score  (0–100 %)"]

    classDef backbone fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef head     fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef io       fill:#FFF3E0,stroke:#E65100,color:#BF360C,rx:8

    class EFF,MOB backbone
    class HEAD head
    class IN,PRED io
```

---

## Fig. 3 — Model Training Pipeline

*Use in: paper (Section V — Methodology), report training section, PPT slide 9–10*

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '14px', 'fontFamily': 'Segoe UI, Arial'}}}%%
flowchart LR
    DS[("PlantVillage Dataset\n87,848 Images\n38 Disease Classes\n14 Crop Species")]

    DS --> SPL

    subgraph SPL["Dataset Split  80 / 10 / 10"]
        direction TB
        TR["Train\n70,278 images"]
        VA["Validation\n8,785 images"]
        TE["Test\n8,785 images"]
    end

    SPL --> PIPE

    subgraph PIPE["tf.data GPU Pipeline  (parallel, prefetched)"]
        direction TB
        P1["Decode JPEG\nResize → 224 × 224"]
        P2["Normalize\npixel ÷ 127.5 − 1.0  →  [−1, 1]"]
        P3["Augmentation  (train only)\nRandom H/V flip  ·  Brightness ±0.15"]
        P4["Batch 64  +  Prefetch AUTOTUNE"]
        P1 --> P2 --> P3 --> P4
    end

    PIPE --> TRAIN

    subgraph TRAIN["Two-Phase Transfer Learning  (Mixed Float16)"]
        direction TB
        PH1["Phase 1 — 8 Epochs\nBackbones frozen\nAdam  lr = 1 × 10⁻³\nHead-only training"]
        PH2["Phase 2 — 4 Epochs\nTop-20 layers unfrozen per backbone\nAdam  lr = 5 × 10⁻⁵\nFine-tuning"]
        CB["Callbacks\nModelCheckpoint  ·  EarlyStopping\nReduceLROnPlateau"]
        PH1 --> PH2
        CB -. monitors .-> PH1 & PH2
    end

    TRAIN --> EVAL["Test Set Evaluation\nAccuracy  ≈ 98.3 %"]

    EVAL --> CONV

    subgraph CONV["TFLite Conversion  (local, CPU)"]
        direction TB
        C1["Reload best_model.keras\nin pure Float32\n(avoids double-quantization bug)"]
        C2["TFLite Float16 Quantization\ntf.lite.Optimize.DEFAULT\n+ target_spec Float16"]
        C3["final_model_float16.tflite\n14.8 MB  ·  input [1,224,224,3]  ·  output [1,38]"]
        C1 --> C2 --> C3
    end

    classDef phase fill:#E8EAF6,stroke:#3949AB,color:#1A237E
    classDef conv  fill:#FCE4EC,stroke:#C62828,color:#B71C1C
    classDef io    fill:#FFF8E1,stroke:#F57F17,color:#E65100

    class TRAIN phase
    class CONV conv
    class DS,EVAL io
```

---

## Fig. 4 — Mobile App Inference Pipeline (End-to-End)

*Use in: paper (Section VI — Implementation), report inference section, PPT slide 11–12*

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '14px', 'fontFamily': 'Segoe UI, Arial'}}}%%
flowchart TD
    SRC(["📷  Camera\nor  🖼️  Gallery"])

    SRC --> IMG["Raw Image File\nJPEG / PNG"]

    IMG --> QC

    subgraph QC["Image Quality Gate  (ImageQualityService)"]
        direction TB
        Q1["File Size  ≤ 15 MB"]
        Q2["Dimensions  ≥ 100 × 100 px"]
        Q3["Brightness  20 < avg pixel < 245"]
        Q4["Blur  ·  Laplacian Variance > 8"]
        Q5["Leaf Detection  ·  HSV pixel ratio > 2 %"]
        Q1 --> Q2 --> Q3 --> Q4 --> Q5
    end

    QC -->|"❌  Any check fails"| ERR(["Quality Error Dialog\n(user prompted to retake)"])
    QC -->|"✅  All checks pass"| PRE

    subgraph PRE["Image Preprocessing  (Dart / image package)"]
        direction TB
        PR1["Bake EXIF Orientation"]
        PR2["Bicubic Resize  →  224 × 224"]
        PR3["Per-channel Normalize\nr  g  b  →  (px ÷ 127.5) − 1.0\nFloat32List  [150,528 values]"]
        PR1 --> PR2 --> PR3
    end

    PRE --> INF

    subgraph INF["On-Device TFLite Inference  (tflite_flutter 0.12)"]
        direction TB
        I1["setTo(Float32List)\nInput Tensor  [1, 224, 224, 3]"]
        I2["invoke()\nForward Pass  ·  Float16 ops"]
        I3["copyTo(List 1×38)\nOutput Tensor  [1, 38]"]
        I1 --> I2 --> I3
    end

    INF --> POST

    subgraph POST["Post-Processing"]
        direction TB
        PS1["Argmax  →  Top-1 Class Index"]
        PS2["Confidence Threshold  > 40 %"]
        PS3a["Low-Confidence Result\n'Not Recognized'"]
        PS3b["Severity Mapping\n< 75 % → Mild\n75–90 % → Moderate\n≥ 90 % → Severe"]
        PS1 --> PS2
        PS2 -->|"< 40 %"| PS3a
        PS2 -->|"≥ 40 %"| PS3b
    end

    PS3b --> RES(["Result Screen\nDisease name  ·  Confidence  ·  Severity\nSymptoms · Remedy · Prevention\nTTS Voice Output  ·  Saved to History"])

    classDef gate  fill:#FFF3E0,stroke:#E65100,color:#BF360C
    classDef pre   fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef inf   fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef post  fill:#EDE7F6,stroke:#4527A0,color:#311B92

    class QC gate
    class PRE pre
    class INF inf
    class POST post
```

---

## Fig. 5 — Application Screen Navigation Flow

*Use in: report UX section, PPT slide 6, final project report*

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '14px', 'fontFamily': 'Segoe UI, Arial'}}}%%
flowchart TD
    SPS["Splash Screen\nModel pre-warm  ·  Services init"]

    SPS --> SHELL

    subgraph SHELL["Main Shell — Bottom Navigation Bar  (5 tabs)"]
        direction LR
        T1["🏠  Home"] ~~~ T2["🔬  Scan"] ~~~ T3["📋  History"] ~~~ T4["📚  Library"] ~~~ T5["⚙️  Settings"]
    end

    subgraph HOME["Home Screen"]
        H1["Model accuracy stats\nRecent scan card\nQuick tips"]
    end

    subgraph SCAN["Scan Screen"]
        S1["Camera button"] & S2["Gallery button"]
        S3["Pulse animation  ·  Quality tips"]
    end

    subgraph QUAL["Quality Check"]
        QA["Pass  ✅"] & QB["Fail  ❌"]
    end

    subgraph RESULT["Result Screen  (2 tabs)"]
        R1["Overview Tab\nDisease name  ·  Confidence bar\nSeverity badge  ·  TTS FAB"]
        R2["Details Tab\nSymptoms  ·  Causes\nPrevention  ·  Remedy"]
    end

    subgraph HIST["History Screen"]
        HF["Filter chips\nAll  ·  Healthy  ·  Disease"]
    end

    subgraph LIB["Disease Library"]
        LB["Search bar\nGrid of 38 disease cards\nDetail bottom sheet"]
    end

    subgraph SETT["Settings Screen"]
        ST["Language selector\nEN · KN · HI · TA · TE\nTTS on/off  ·  About"]
    end

    T1 --- HOME
    T2 --- SCAN
    T3 --- HIST
    T4 --- LIB
    T5 --- SETT

    S1 & S2 --> QUAL
    QB -->|"Error dialog  →  dismiss"| SCAN
    QA -->|"Inference"| RESULT
    RESULT -->|"Auto-saved"| HIST
    HOME -->|"Scan shortcut"| SCAN

    classDef screen fill:#E3F2FD,stroke:#1565C0,color:#0D47A1,rx:8
    classDef action fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef warn   fill:#FFF3E0,stroke:#E65100,color:#BF360C

    class HOME,SCAN,RESULT,HIST,LIB,SETT screen
    class QUAL action
```

---

## Quick Reference — What to use where

| Diagram | Paper Section | Report Chapter | PPT Slide |
|---|---|---|---|
| Fig. 1 — System Architecture | III. System Design | Architecture Overview | 4–5 |
| Fig. 2 — CNN Model Architecture | IV. Proposed Model | Model Design | 7–8 |
| Fig. 3 — Training Pipeline | V. Methodology | Training & Conversion | 9–10 |
| Fig. 4 — Inference Pipeline | VI. Implementation | App Implementation | 11–12 |
| Fig. 5 — App Navigation | VII. User Interface | UX & Screens | 6 |

## Tech Stack Summary (for paper table)

| Component | Technology | Detail |
|---|---|---|
| Mobile Framework | Flutter 3.41 / Dart 3.11 | Android arm64 |
| ML Runtime | TFLite Flutter 0.12 | On-device, offline |
| Model | EfficientNetB0 + MobileNetV2 | Dual backbone, 38-class |
| Model Size | 14.8 MB (Float16) | Fits in app bundle |
| Dataset | PlantVillage | 87,848 images, 38 classes |
| Test Accuracy | ≈ 98.3 % | 8,785-image test split |
| Local DB | Hive + SQLite | History + treatment data |
| Languages | EN, KN, HI, TA, TE | Full UI + TTS |
| Min Android | API 21 (Android 5.0) | — |
