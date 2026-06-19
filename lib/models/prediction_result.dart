import 'disease_info.dart';

enum SeverityLevel { none, mild, moderate, severe }

extension SeverityLevelExt on SeverityLevel {
  String get label => switch (this) {
        SeverityLevel.none => 'Healthy',
        SeverityLevel.mild => 'Mild',
        SeverityLevel.moderate => 'Moderate',
        SeverityLevel.severe => 'Severe',
      };
}

class PredictionResult {
  const PredictionResult({
    required this.label,
    required this.displayLabel,
    required this.confidence,
    required this.diseaseInfo,
    required this.isLowConfidence,
    required this.severityLevel,
    this.warningMessage,
  });

  final String label;
  final String displayLabel;
  final double confidence;
  final DiseaseInfo diseaseInfo;
  final bool isLowConfidence;
  final SeverityLevel severityLevel;
  final String? warningMessage;

  bool get isHealthy => diseaseInfo.isHealthy;
}
