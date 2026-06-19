class ScanHistory {
  const ScanHistory({
    required this.imagePath,
    required this.prediction,
    required this.confidence,
    required this.scannedAt,
    required this.isHealthy,
  });

  final String imagePath;
  final String prediction;
  final double confidence;
  final DateTime scannedAt;
  final bool isHealthy;

  Map<String, dynamic> toJson() => {
        'imagePath': imagePath,
        'prediction': prediction,
        'confidence': confidence,
        'scannedAt': scannedAt.toIso8601String(),
        'isHealthy': isHealthy,
      };

  factory ScanHistory.fromJson(Map<dynamic, dynamic> json) {
    try {
      return ScanHistory(
        imagePath: json['imagePath'] as String? ?? '',
        prediction: json['prediction'] as String? ?? 'Unknown',
        confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
        scannedAt: json['scannedAt'] != null
            ? DateTime.tryParse(json['scannedAt'] as String) ?? DateTime.now()
            : DateTime.now(),
        isHealthy: json['isHealthy'] as bool? ?? false,
      );
    } catch (_) {
      return ScanHistory(
        imagePath: '',
        prediction: 'Unknown',
        confidence: 0.0,
        scannedAt: DateTime.now(),
        isHealthy: false,
      );
    }
  }
}
