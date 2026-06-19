import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../data/class_names.dart';
import '../data/disease_data.dart';
import '../models/prediction_result.dart';

class TfliteService {
  TfliteService._();
  static final TfliteService instance = TfliteService._();

  Interpreter? _interpreter;
  bool _isModelLoaded = false;
  String? _loadError;

  String? get loadError => _loadError;

  Future<void> load() async {
    if (_isModelLoaded) return;
    try {
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(
        'assets/model/final_model_float16.tflite',
        options: options,
      );
      // Pre-resize to [1,224,224,3] so invoke() doesn't try to infer
      // shape from a flat Float32List (which gives 1D → CONV_2D crash).
      _interpreter!.resizeInputTensor(0, [1, 224, 224, 3]);
      _interpreter!.allocateTensors();
      debugPrint('[TFLite] Loaded. '
          'Input: ${_interpreter!.getInputTensor(0).shape}  '
          'Output: ${_interpreter!.getOutputTensor(0).shape}');
      _isModelLoaded = true;
      _loadError = null;
    } catch (e, st) {
      debugPrint('[TFLite] Load failed: $e\n$st');
      _interpreter = null;
      _isModelLoaded = false;
      _loadError = e.toString();
    }
  }

  Future<PredictionResult> predict(String imagePath) async {
    try {
      await load();
      if (_interpreter == null) {
        return _error('Model failed to load: ${_loadError ?? "unknown"}');
      }

      // ── Decode + fix orientation ─────────────────────────────────
      final bytes = await File(imagePath).readAsBytes();
      img.Image? image = img.decodeImage(bytes);
      if (image == null) return _error('Could not decode image');
      image = img.bakeOrientation(image);

      // ── Camera-to-training-distribution alignment ─────────────────
      // PlantVillage images are controlled studio shots.
      // Real camera photos differ in three ways:
      //   • Exposure: variable lighting → histogram stretch corrects it
      //   • Contrast: camera JPEG processing flattens contrast → mild boost
      //   • Interpolation: cubic keeps edges sharper than linear on large→224 downsample
      image = img.normalize(image, min: 0, max: 255);
      image = img.adjustColor(image, contrast: 1.08);
      image = img.copyResize(image, width: 224, height: 224,
          interpolation: img.Interpolation.cubic);

      // ── Adaptive TTA (Test-Time Augmentation) ─────────────────────
      // First inference pass.
      List<double> probs = _inferProbs(image);
      double topConf = probs.reduce((a, b) => a > b ? a : b);
      debugPrint('[TFLite] Pass 1 top: ${(topConf * 100).toStringAsFixed(1)}%');

      // If confidence is below 0.80, run a second pass on the
      // horizontally-flipped image and average both probability arrays.
      //
      // WHY this works: leaf diseases are visually symmetric — the model
      // sees the same disease pattern regardless of mirror orientation.
      // Averaging two independent probability distributions smooths out
      // per-image noise and consistently lifts borderline predictions.
      // Cost: one extra ~300 ms inference, paid only when needed.
      if (topConf < 0.80) {
        final probs2 = _inferProbs(img.flipHorizontal(image));
        probs = List.generate(38, (i) => (probs[i] + probs2[i]) / 2);
        topConf = probs.reduce((a, b) => a > b ? a : b);
        debugPrint('[TFLite] TTA applied → ${(topConf * 100).toStringAsFixed(1)}%');
      }

      // ── Top-5 diagnostic ─────────────────────────────────────────
      final sorted = List.generate(probs.length, (i) => (i, probs[i]))
        ..sort((a, b) => b.$2.compareTo(a.$2));
      debugPrint('[TFLite] Top5: ${sorted.take(5).map((e) =>
          '${plantDiseaseClassNames[e.$1]}(${(e.$2 * 100).toStringAsFixed(1)}%)').join(' | ')}');

      final predIdx    = sorted[0].$1;
      final confidence = sorted[0].$2;
      final rawLabel      = plantDiseaseClassNames[predIdx];
      final displayLabel  = displayLabelFromClassName(rawLabel);
      final isHealthy     = rawLabel.toLowerCase().contains('healthy');

      // ── Confidence + margin thresholds ───────────────────────────
      // confidence < 0.75: model is not certain enough.
      // margin  < 0.20: top class is not decisively ahead of runner-up —
      //   the model is split, meaning the image is ambiguous or not a leaf.
      final secondConf = sorted[1].$2;
      final margin = confidence - secondConf;
      if (confidence < 0.75 || margin < 0.20) {
        return PredictionResult(
          label: rawLabel,
          displayLabel: 'Not Recognized',
          confidence: confidence,
          diseaseInfo: unknownDisease,
          isLowConfidence: true,
          severityLevel: SeverityLevel.none,
          warningMessage:
              'No plant disease detected. Point the camera directly at a leaf '
              'in good lighting.',
        );
      }

      return PredictionResult(
        label: rawLabel,
        displayLabel: displayLabel,
        confidence: confidence,
        diseaseInfo: diseaseInfoForRawClass(rawLabel),
        isLowConfidence: false,
        severityLevel: isHealthy ? SeverityLevel.none : _severity(confidence),
        warningMessage: null,
      );
    } catch (e, st) {
      debugPrint('[TFLite] Predict error: $e\n$st');
      return _error('Inference error: $e');
    }
  }

  // ── Private: one inference pass on a pre-resized 224×224 image ──────────
  // Applies grey-world auto white balance before building the tensor.
  //
  // Grey-world assumption: on average, a natural image is neutral grey.
  // Scaling each channel so its mean equals the overall mean corrects
  // colour casts from yellow indoor lighting or overcast outdoor light —
  // the most common cause of camera-to-model colour mismatch.
  List<double> _inferProbs(img.Image resized) {
    // Compute per-channel means
    double rSum = 0, gSum = 0, bSum = 0;
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final px = resized.getPixel(x, y);
        rSum += px.r.toDouble();
        gSum += px.g.toDouble();
        bSum += px.b.toDouble();
      }
    }
    const n = 224 * 224;
    final overall = (rSum + gSum + bSum) / (3 * n);
    final rScale = (overall / (rSum / n)).clamp(0.5, 2.0);
    final gScale = (overall / (gSum / n)).clamp(0.5, 2.0);
    final bScale = (overall / (bSum / n)).clamp(0.5, 2.0);

    // Build Float32 input tensor with white-balance + EfficientNet normalisation
    final inputF32 = Float32List(n * 3);
    int idx = 0;
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final px = resized.getPixel(x, y);
        inputF32[idx++] = ((px.r.toDouble() * rScale).clamp(0, 255) / 127.5) - 1.0;
        inputF32[idx++] = ((px.g.toDouble() * gScale).clamp(0, 255) / 127.5) - 1.0;
        inputF32[idx++] = ((px.b.toDouble() * bScale).clamp(0, 255) / 127.5) - 1.0;
      }
    }

    _interpreter!.getInputTensor(0).setTo(inputF32);
    _interpreter!.invoke();
    final out = List.generate(1, (_) => List<double>.filled(38, 0.0));
    _interpreter!.getOutputTensor(0).copyTo(out);
    return out[0];
  }

  SeverityLevel _severity(double conf) {
    if (conf >= 0.90) return SeverityLevel.severe;
    if (conf >= 0.75) return SeverityLevel.moderate;
    return SeverityLevel.mild;
  }

  PredictionResult _error(String msg) => PredictionResult(
        label: 'ERROR',
        displayLabel: 'Detection Failed',
        confidence: 0,
        diseaseInfo: unknownDisease,
        isLowConfidence: true,
        severityLevel: SeverityLevel.none,
        warningMessage: msg,
      );
}
