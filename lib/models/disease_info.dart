class DiseaseInfo {
  const DiseaseInfo({
    required this.name,
    required this.description,
    required this.symptoms,
    required this.causes,
    required this.prevention,
    required this.remedy,
    required this.isHealthy,
  });

  final String name;
  final String description;
  final String symptoms;
  final String causes;
  final String prevention;
  final String remedy;
  final bool isHealthy;
}
