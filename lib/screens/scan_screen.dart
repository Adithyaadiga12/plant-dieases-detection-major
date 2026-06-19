import 'package:flutter/material.dart';

import '../models/scan_history.dart';
import '../screens/result_screen.dart';
import '../services/history_service.dart';
import '../services/image_quality_service.dart';
import '../services/image_service.dart';
import '../services/locale_service.dart';
import '../services/permission_service.dart';
import '../services/tflite_service.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with SingleTickerProviderStateMixin {
  final _imageService = ImageService();
  final _qualityService = ImageQualityService();
  bool _isProcessing = false;
  String _processingStep = '';
  late final AnimationController _pulseCtrl;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  Future<void> _scan(bool fromCamera) async {
    if (_isProcessing) return;

    // ── Permission check ────────────────────────────────────────────
    final permResult = fromCamera
        ? await PermissionService.instance.checkCamera()
        : await PermissionService.instance.checkGallery();

    if (!mounted) return;
    if (permResult == PermissionResult.permanentlyDenied) {
      PermissionService.instance
          .showSettingsDialog(context, fromCamera ? 'Camera' : 'Gallery');
      return;
    }
    if (permResult == PermissionResult.denied) {
      PermissionService.instance
          .showDeniedSnackbar(context, fromCamera ? 'Camera' : 'Gallery');
      return;
    }
    if (permResult != PermissionResult.granted) return;

    // ── Pick image ──────────────────────────────────────────────────
    final file = fromCamera
        ? await _imageService.captureFromCamera()
        : await _imageService.pickFromGallery();
    if (file == null || !mounted) return;

    setState(() {
      _isProcessing = true;
      _processingStep = LocaleService.instance.s('checkQuality');
    });

    try {
      // ── Quality gate ────────────────────────────────────────────
      final issue = await _qualityService.check(file.path);
      if (!mounted) return;

      if (issue != ImageQualityIssue.none) {
        _showQualityError(issue.message);
        return;
      }

      // ── Run inference ───────────────────────────────────────────
      setState(() => _processingStep = LocaleService.instance.s('runningModel'));
      final result = await TfliteService.instance.predict(file.path);
      if (!mounted) return;

      await HistoryService.instance.add(ScanHistory(
        imagePath: file.path,
        prediction: result.displayLabel,
        confidence: result.confidence,
        scannedAt: DateTime.now(),
        isHealthy: result.isHealthy,
      ));

      if (!mounted) return;
      await Navigator.pushNamed(
        context,
        ResultScreen.routeName,
        arguments: ResultScreenArgs(imagePath: file.path, result: result),
      );
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  void _showQualityError(String message) {
    if (!mounted) return;
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.image_search_rounded, size: 40),
        title: Text(LocaleService.instance.s('qualityTitle')),
        content: Text(message),
        actions: [
          FilledButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(LocaleService.instance.s('gotIt')),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: Text(LocaleService.instance.s('scanTitle'),
            style: const TextStyle(fontWeight: FontWeight.w800)),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              // ── Scanner area ───────────────────────────────────────
              Expanded(
                child: AnimatedBuilder(
                  animation: _pulseCtrl,
                  builder: (_, child) => Container(
                    decoration: BoxDecoration(
                      color: cs.surfaceContainerLow,
                      borderRadius: BorderRadius.circular(28),
                      border: Border.all(
                        color: cs.primary
                            .withValues(alpha: 0.2 + 0.2 * _pulseCtrl.value),
                        width: 2,
                      ),
                    ),
                    child: child,
                  ),
                  child: _isProcessing
                      ? _AnalyzingWidget(cs: cs, step: _processingStep)
                      : _ScanPlaceholder(cs: cs, pulseCtrl: _pulseCtrl),
                ),
              ),
              const SizedBox(height: 28),

              if (!_isProcessing) ...[
                // ── Camera button ──────────────────────────────────
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: () => _scan(true),
                    icon: const Icon(Icons.camera_alt_rounded),
                    label: Text(LocaleService.instance.s('btnCamera')),
                  ),
                ),
                const SizedBox(height: 12),
                // ── Gallery button ─────────────────────────────────
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: () => _scan(false),
                    icon: const Icon(Icons.photo_library_rounded),
                    label: Text(LocaleService.instance.s('btnGallery')),
                  ),
                ),
              ],
              const SizedBox(height: 16),

              // ── Quality tips ───────────────────────────────────────
              if (!_isProcessing) ...[
                _QualityTips(cs: cs),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ── Sub-widgets ──────────────────────────────────────────────────────────────

class _QualityTips extends StatelessWidget {
  const _QualityTips({required this.cs});
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        children: [
          _tip(context, Icons.wb_sunny_outlined, LocaleService.instance.s('tipLight')),
          _tip(context, Icons.center_focus_strong_outlined, LocaleService.instance.s('tipFocus')),
          _tip(context, Icons.no_photography_outlined, LocaleService.instance.s('tipNoSelfie')),
        ],
      ),
    );
  }

  Widget _tip(BuildContext context, IconData icon, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          Icon(icon, size: 15, color: cs.primary),
          const SizedBox(width: 8),
          Text(
            text,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: cs.onSurfaceVariant,
                ),
          ),
        ],
      ),
    );
  }
}

class _ScanPlaceholder extends StatelessWidget {
  const _ScanPlaceholder({required this.cs, required this.pulseCtrl});
  final ColorScheme cs;
  final Animation<double> pulseCtrl;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          AnimatedBuilder(
            animation: pulseCtrl,
            builder: (_, child) => Transform.scale(
              scale: 1.0 + 0.06 * pulseCtrl.value,
              child: child,
            ),
            child: Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(
                color: cs.primaryContainer,
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.document_scanner_outlined,
                  size: 48, color: cs.primary),
            ),
          ),
          const SizedBox(height: 20),
          Text(
            LocaleService.instance.s('readyToScan'),
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w800,
                ),
          ),
          const SizedBox(height: 6),
          Text(
            LocaleService.instance.s('tapToStart'),
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: cs.onSurfaceVariant,
                ),
          ),
          const SizedBox(height: 24),
          _CornerGuide(cs: cs),
        ],
      ),
    );
  }
}

class _CornerGuide extends StatelessWidget {
  const _CornerGuide({required this.cs});
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 120,
      height: 120,
      child: CustomPaint(painter: _GuideLinePainter(color: cs.primary)),
    );
  }
}

class _GuideLinePainter extends CustomPainter {
  _GuideLinePainter({required this.color});
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    const c = 24.0;
    final corners = [
      Offset(0, 0),
      Offset(size.width, 0),
      Offset(0, size.height),
      Offset(size.width, size.height),
    ];
    final dirs = [
      [Offset(c, 0), Offset(0, c)],
      [Offset(-c, 0), Offset(0, c)],
      [Offset(c, 0), Offset(0, -c)],
      [Offset(-c, 0), Offset(0, -c)],
    ];
    for (int i = 0; i < 4; i++) {
      canvas.drawLine(corners[i], corners[i] + dirs[i][0], p);
      canvas.drawLine(corners[i], corners[i] + dirs[i][1], p);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter o) => false;
}

class _AnalyzingWidget extends StatelessWidget {
  const _AnalyzingWidget({required this.cs, required this.step});
  final ColorScheme cs;
  final String step;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 64,
            height: 64,
            child: CircularProgressIndicator(
              strokeWidth: 4,
              color: cs.primary,
            ),
          ),
          const SizedBox(height: 20),
          Text(
            LocaleService.instance.s('analyzing'),
            style: Theme.of(context)
                .textTheme
                .titleMedium
                ?.copyWith(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(
            step,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: cs.onSurfaceVariant,
                ),
          ),
        ],
      ),
    );
  }
}
