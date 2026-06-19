# AgroVision Model Contract

The app is designed so a fine-tuned model can be swapped with minimal code changes.

## Current Expected Files

- `assets/model/final_model_float16.tflite`
- `assets/model/labels.txt`

## Current Expected Model

- Input: `[1, 224, 224, 3]`
- Input type: `float32`
- Preprocessing: raw RGB float values in `0..255`
- Output: one softmax classification tensor
- Classes: `38`

## Plug-And-Play Checklist

When replacing the model:

1. Keep the file name as `final_model_float16.tflite`, or update `currentModelConfig.modelAssetPath`.
2. Keep `labels.txt` in the exact same order as the model output indices.
3. If input size changes, update `currentModelConfig.inputSize`.
4. If preprocessing changes, update `currentModelConfig.preprocessingMode`.
5. If class count changes, update `currentModelConfig.expectedClassCount`.
6. Run `flutter analyze` and test one known image.

The app validates model input shape and output class count during model loading.
