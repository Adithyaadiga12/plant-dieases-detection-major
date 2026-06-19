import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

enum ImageQualityIssue {
  none,
  tooLarge,
  unsupportedFormat,
  tooSmall,
  tooDark,
  overExposed,
  blurry,
  noLeaf,
}

extension ImageQualityIssueExt on ImageQualityIssue {
  String get message => switch (this) {
        ImageQualityIssue.none => '',
        ImageQualityIssue.tooLarge =>
          'Image is too large (>15 MB). Please choose a smaller photo.',
        ImageQualityIssue.unsupportedFormat =>
          'Cannot read this image format. Try a JPEG or PNG photo.',
        ImageQualityIssue.tooSmall =>
          'Image is too small. Use a photo at least 100×100 pixels.',
        ImageQualityIssue.tooDark =>
          'Image is too dark. Move to better lighting and try again.',
        ImageQualityIssue.overExposed =>
          'Image is overexposed. Avoid direct sunlight on the lens.',
        ImageQualityIssue.blurry =>
          'Image is too blurry. Hold the camera steady and tap to focus.',
        ImageQualityIssue.noLeaf =>
          'No plant detected. Point the camera directly at a leaf.',
      };
}

class ImageQualityService {
  static const _maxFileSizeBytes = 15 * 1024 * 1024; // 15 MB
  static const _minDimension = 100;

  static const _darkThreshold = 20.0;
  static const _brightThreshold = 245.0;
  static const _blurThreshold = 7.0;

  // At least 13% of pixels must look like leaf tissue.
  // A real leaf close-up clears 30%+; random photos rarely reach this.
  static const _leafPixelThreshold = 0.13;

  // Solid green objects score <0.04; real leaves score >0.06 due to
  // veins, disease spots, and lighting variation.
  static const _leafTextureThreshold = 0.052;

  // Leaf-coloured pixels must appear in at least 2 of 4 image quadrants.
  // A green bottle in one corner won't reach both halves.
  static const _minLeafQuadrants = 2;

  Future<ImageQualityIssue> check(String imagePath) async {
    final file = File(imagePath);
    final fileSize = await file.length();
    if (fileSize > _maxFileSizeBytes) return ImageQualityIssue.tooLarge;

    final bytes = await file.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) return ImageQualityIssue.unsupportedFormat;

    if (image.width < _minDimension || image.height < _minDimension) {
      return ImageQualityIssue.tooSmall;
    }

    // Downsample once for all pixel-level checks
    final small = img.copyResize(image, width: 64, height: 64);

    final brightness = _avgBrightness(small);
    if (brightness < _darkThreshold) return ImageQualityIssue.tooDark;
    if (brightness > _brightThreshold) return ImageQualityIssue.overExposed;

    if (_laplacianVariance(small) < _blurThreshold) {
      return ImageQualityIssue.blurry;
    }

    if (_leafPixelRatio(small) < _leafPixelThreshold) {
      return ImageQualityIssue.noLeaf;
    }

    if (_leafBrightnessStdDev(small) < _leafTextureThreshold) {
      return ImageQualityIssue.noLeaf;
    }

    if (_leafQuadrantCount(small) < _minLeafQuadrants) {
      return ImageQualityIssue.noLeaf;
    }

    return ImageQualityIssue.none;
  }

  double _avgBrightness(img.Image image) {
    double total = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final px = image.getPixel(x, y);
        total += (px.r.toDouble() + px.g.toDouble() + px.b.toDouble()) / 3;
      }
    }
    return total / (image.width * image.height);
  }

  double _laplacianVariance(img.Image image) {
    final values = <double>[];
    for (int y = 1; y < image.height - 1; y++) {
      for (int x = 1; x < image.width - 1; x++) {
        final c = _gray(image.getPixel(x, y));
        final t = _gray(image.getPixel(x, y - 1));
        final b = _gray(image.getPixel(x, y + 1));
        final l = _gray(image.getPixel(x - 1, y));
        final r = _gray(image.getPixel(x + 1, y));
        values.add((t + b + l + r) - 4 * c);
      }
    }
    if (values.isEmpty) return 0;
    final mean = values.reduce((a, b) => a + b) / values.length;
    return values
            .map((v) => (v - mean) * (v - mean))
            .reduce((a, b) => a + b) /
        values.length;
  }

  // HSV leaf colour test — shared by ratio, texture, and quadrant checks.
  //
  // Healthy leaf green : hue 60–165°, sat ≥ 0.18, val ≥ 0.10
  // Diseased tissue    : hue 25–70°,  sat ≥ 0.25, val ≥ 0.12
  // Skin sits at hue 0–25° and falls outside both ranges entirely.
  bool _isLeafPixel(img.Pixel px) {
    final r = px.r.toDouble() / 255.0;
    final g = px.g.toDouble() / 255.0;
    final b = px.b.toDouble() / 255.0;

    final maxC = r > g ? (r > b ? r : b) : (g > b ? g : b);
    final minC = r < g ? (r < b ? r : b) : (g < b ? g : b);
    final delta = maxC - minC;
    final value = maxC;
    final saturation = maxC > 0 ? delta / maxC : 0.0;

    double hue = 0;
    if (delta > 0) {
      if (maxC == r) {
        hue = 60 * (((g - b) / delta) % 6);
      } else if (maxC == g) {
        hue = 60 * ((b - r) / delta + 2);
      } else {
        hue = 60 * ((r - g) / delta + 4);
      }
      if (hue < 0) hue += 360;
    }

    final isLeafGreen =
        hue >= 60 && hue <= 165 && saturation >= 0.18 && value >= 0.10;
    final isDiseasedTissue =
        hue >= 25 && hue <= 70 && saturation >= 0.25 && value >= 0.12;

    return isLeafGreen || isDiseasedTissue;
  }

  double _leafPixelRatio(img.Image image) {
    int leafCount = 0;
    final total = image.width * image.height;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        if (_isLeafPixel(image.getPixel(x, y))) leafCount++;
      }
    }
    return leafCount / total;
  }

  double _leafBrightnessStdDev(img.Image image) {
    final values = <double>[];
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final px = image.getPixel(x, y);
        if (!_isLeafPixel(px)) continue;
        final r = px.r.toDouble() / 255.0;
        final g = px.g.toDouble() / 255.0;
        final b = px.b.toDouble() / 255.0;
        final maxC = r > g ? (r > b ? r : b) : (g > b ? g : b);
        values.add(maxC); // V channel
      }
    }
    if (values.length < 20) return 0.0;
    final mean = values.reduce((a, b) => a + b) / values.length;
    final variance = values
            .map((v) => (v - mean) * (v - mean))
            .reduce((a, b) => a + b) /
        values.length;
    return math.sqrt(variance);
  }

  // Counts how many of the 4 image quadrants contain >4% leaf-coloured pixels.
  // Rejects images where green appears only in a small corner region.
  int _leafQuadrantCount(img.Image image) {
    final halfW = image.width ~/ 2;
    final halfH = image.height ~/ 2;
    final bounds = [
      (0, 0, halfW, halfH),
      (halfW, 0, image.width, halfH),
      (0, halfH, halfW, image.height),
      (halfW, halfH, image.width, image.height),
    ];
    int filled = 0;
    for (final (x0, y0, x1, y1) in bounds) {
      int leafCount = 0;
      final total = (x1 - x0) * (y1 - y0);
      for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
          if (_isLeafPixel(image.getPixel(x, y))) leafCount++;
        }
      }
      if (leafCount / total > 0.04) filled++;
    }
    return filled;
  }

  double _gray(img.Pixel p) =>
      0.299 * p.r.toDouble() +
      0.587 * p.g.toDouble() +
      0.114 * p.b.toDouble();
}
