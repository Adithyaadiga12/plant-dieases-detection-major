import 'dart:io';

import 'package:flutter/material.dart';

import '../models/prediction_result.dart';
import '../services/locale_service.dart';
import '../services/tts_service.dart';

class ResultScreenArgs {
  const ResultScreenArgs({required this.imagePath, required this.result});
  final String imagePath;
  final PredictionResult result;
}

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.args});
  static const routeName = '/result';

  final ResultScreenArgs args;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  @override
  void dispose() {
    TtsService.instance.stop();
    super.dispose();
  }

  Future<void> _toggleTts() =>
      TtsService.instance.announceResult(widget.args.result);

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final r = widget.args.result;

    final (statusColor, statusIcon) = switch (r.severityLevel) {
      SeverityLevel.none when r.isHealthy =>
        (Colors.green, Icons.check_circle_rounded),
      SeverityLevel.mild => (Colors.lightGreen, Icons.warning_rounded),
      SeverityLevel.moderate => (Colors.orange, Icons.warning_rounded),
      SeverityLevel.severe => (Colors.red, Icons.dangerous_rounded),
      _ => (Colors.grey, Icons.help_outline_rounded),
    };

    return Scaffold(
      floatingActionButton: LocaleService.instance.ttsEnabled
          ? ValueListenableBuilder<bool>(
              valueListenable: TtsService.instance.speakingState,
              builder: (_, speaking, _) => FloatingActionButton(
                onPressed: _toggleTts,
                tooltip: speaking
                    ? LocaleService.instance.s('stopSpeaking')
                    : LocaleService.instance.s('speakResult'),
                child: Icon(
                  speaking ? Icons.stop_rounded : Icons.volume_up_rounded,
                ),
              ),
            )
          : null,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 280,
            pinned: true,
            flexibleSpace: FlexibleSpaceBar(
              background: Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(
                    File(widget.args.imagePath),
                    fit: BoxFit.cover,
                    errorBuilder: (_, _, _) => Container(
                      color: cs.surfaceContainerHighest,
                      child: Icon(Icons.broken_image_outlined,
                          size: 64, color: cs.onSurfaceVariant),
                    ),
                  ),
                  DecoratedBox(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.transparent,
                          Colors.black.withValues(alpha: 0.4),
                        ],
                        stops: const [0.5, 1.0],
                      ),
                    ),
                  ),
                ],
              ),
            ),
            foregroundColor: Colors.white,
            backgroundColor: cs.primary,
            title: Text(LocaleService.instance.s('resultTitle'),
                style: const TextStyle(fontWeight: FontWeight.w800)),
          ),
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                // ── Status card ──────────────────────────────────────
                _StatusCard(
                    r: r,
                    cs: cs,
                    statusColor: statusColor,
                    statusIcon: statusIcon),
                const SizedBox(height: 16),

                // ── Warning banner ───────────────────────────────────
                if (r.warningMessage != null) ...[
                  _WarningBanner(message: r.warningMessage!, cs: cs),
                  const SizedBox(height: 16),
                ],

                // ── Disease info tabs ────────────────────────────────
                if (!r.isLowConfidence && r.label != 'ERROR')
                  _DiseaseInfoTabs(r: r, cs: cs),

                const SizedBox(height: 32),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Sub-widgets ──────────────────────────────────────────────────────────────

class _StatusCard extends StatelessWidget {
  const _StatusCard({
    required this.r,
    required this.cs,
    required this.statusColor,
    required this.statusIcon,
  });
  final PredictionResult r;
  final ColorScheme cs;
  final Color statusColor;
  final IconData statusIcon;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Text(
                    r.displayLabel,
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w900,
                          height: 1.3,
                        ),
                  ),
                ),
                const SizedBox(width: 8),
                _SeverityChip(
                    color: statusColor,
                    icon: statusIcon,
                    label: r.severityLevel.label),
              ],
            ),
            const SizedBox(height: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(LocaleService.instance.s('confidenceLabel'),
                        style: Theme.of(context)
                            .textTheme
                            .labelMedium
                            ?.copyWith(color: cs.onSurfaceVariant)),
                    Text(
                      '${(r.confidence * 100).toStringAsFixed(1)}%',
                      style: Theme.of(context).textTheme.labelLarge?.copyWith(
                            fontWeight: FontWeight.w800,
                            color: statusColor,
                          ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0, end: r.confidence),
                  duration: const Duration(milliseconds: 900),
                  curve: Curves.easeOutCubic,
                  builder: (_, v, _) => ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: v,
                      minHeight: 10,
                      color: statusColor,
                      backgroundColor: statusColor.withValues(alpha: 0.15),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SeverityChip extends StatelessWidget {
  const _SeverityChip(
      {required this.color, required this.icon, required this.label});
  final Color color;
  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 5),
          Text(
            label,
            style: TextStyle(
                color: color, fontWeight: FontWeight.w800, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class _WarningBanner extends StatelessWidget {
  const _WarningBanner({required this.message, required this.cs});
  final String message;
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.errorContainer,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline_rounded,
              color: cs.onErrorContainer, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: TextStyle(
                  color: cs.onErrorContainer, fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }
}

class _DiseaseInfoTabs extends StatelessWidget {
  const _DiseaseInfoTabs({required this.r, required this.cs});
  final PredictionResult r;
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    final info = r.diseaseInfo;
    return DefaultTabController(
      length: 4,
      child: Column(
        children: [
          TabBar(
            isScrollable: true,
            tabAlignment: TabAlignment.start,
            tabs: [
              Tab(text: LocaleService.instance.s('tabOverview')),
              Tab(text: LocaleService.instance.s('tabSymptoms')),
              Tab(text: LocaleService.instance.s('tabCauses')),
              Tab(text: LocaleService.instance.s('tabTreatment')),
            ],
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 220,
            child: TabBarView(
              children: [
                _InfoTab(
                    icon: Icons.description_outlined,
                    text: info.description),
                _InfoTab(
                    icon: Icons.coronavirus_outlined, text: info.symptoms),
                _InfoTab(icon: Icons.science_outlined, text: info.causes),
                _InfoTab(
                    icon: Icons.medication_outlined,
                    text: '${info.prevention}\n\n${info.remedy}'),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _InfoTab extends StatelessWidget {
  const _InfoTab({required this.icon, required this.text});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, size: 20, color: cs.primary),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  text,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        height: 1.6,
                        color: cs.onSurfaceVariant,
                      ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
