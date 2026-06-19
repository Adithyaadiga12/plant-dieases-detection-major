import 'dart:io';

import 'package:flutter/material.dart';

import '../services/history_service.dart';
import '../services/locale_service.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final history = HistoryService.instance.getAll();
    final totalScans = history.length;
    final healthyCount = history.where((h) => h.isHealthy).length;
    final diseasedCount = totalScans - healthyCount;

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 220,
            pinned: true,
            stretch: true,
            flexibleSpace: FlexibleSpaceBar(
              stretchModes: const [StretchMode.zoomBackground],
              background: _HeroBanner(cs: cs),
            ),
            title: Row(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.asset(
                    'assets/images/logo.jpeg',
                    width: 28,
                    height: 28,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  'AgroVision AI',
                  style: TextStyle(
                    color: cs.onPrimary,
                    fontWeight: FontWeight.w800,
                    fontSize: 18,
                  ),
                ),
              ],
            ),
            backgroundColor: cs.primary,
            foregroundColor: cs.onPrimary,
          ),
          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                // ── Stats row ──
                Row(
                  children: [
                    _StatCard(
                      label: LocaleService.instance.s('statTotal'),
                      value: '$totalScans',
                      icon: Icons.document_scanner_rounded,
                      color: cs.primary,
                    ),
                    const SizedBox(width: 12),
                    _StatCard(
                      label: LocaleService.instance.s('statHealthy'),
                      value: '$healthyCount',
                      icon: Icons.check_circle_outline_rounded,
                      color: Colors.green,
                    ),
                    const SizedBox(width: 12),
                    _StatCard(
                      label: LocaleService.instance.s('statDiseased'),
                      value: '$diseasedCount',
                      icon: Icons.warning_amber_rounded,
                      color: Colors.orange,
                    ),
                  ],
                ),
                const SizedBox(height: 24),

                // ── Tip card ──
                _TipCard(cs: cs),
                const SizedBox(height: 24),

                // ── Recent scan ──
                if (history.isNotEmpty) ...[
                  Text(
                    LocaleService.instance.s('recentScan'),
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                  ),
                  const SizedBox(height: 12),
                  _RecentScanCard(history: history.first, cs: cs),
                  const SizedBox(height: 24),
                ],

                // ── About ──
                _InfoCard(cs: cs),
                const SizedBox(height: 32),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

class _HeroBanner extends StatelessWidget {
  const _HeroBanner({required this.cs});
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [cs.primary, cs.tertiary],
        ),
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 60, 24, 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Text(
                LocaleService.instance.s('homeHero'),
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      color: cs.onPrimary,
                      fontWeight: FontWeight.w900,
                      height: 1.2,
                    ),
              ),
              const SizedBox(height: 6),
              Text(
                LocaleService.instance.s('homeHeroSub'),
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: cs.onPrimary.withValues(alpha: 0.8),
                    ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  const _StatCard({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });
  final String label, value;
  final IconData icon;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
          child: Column(
            children: [
              Icon(icon, color: color, size: 22),
              const SizedBox(height: 6),
              Text(
                value,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w900,
                      color: color,
                    ),
              ),
              const SizedBox(height: 2),
              Text(
                label,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _TipCard extends StatelessWidget {
  const _TipCard({required this.cs});
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.primaryContainer,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Icon(Icons.lightbulb_outline_rounded,
              color: cs.onPrimaryContainer, size: 26),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  LocaleService.instance.s('tipTitle'),
                  style: Theme.of(context).textTheme.labelLarge?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: cs.onPrimaryContainer,
                      ),
                ),
                const SizedBox(height: 2),
                Text(
                  LocaleService.instance.s('tipBody'),
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: cs.onPrimaryContainer.withValues(alpha: 0.8),
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _RecentScanCard extends StatelessWidget {
  const _RecentScanCard({required this.history, required this.cs});
  final dynamic history;
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    final isHealthy = history.isHealthy as bool;
    final color = isHealthy ? Colors.green : Colors.orange;

    return Card(
      child: ListTile(
        contentPadding: const EdgeInsets.all(12),
        leading: ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: history.imagePath != null
              ? Image.file(
                  File(history.imagePath as String),
                  width: 56,
                  height: 56,
                  fit: BoxFit.cover,
                  errorBuilder: (_, _, _) => Container(
                    width: 56,
                    height: 56,
                    color: cs.surfaceContainerHighest,
                    child: Icon(Icons.image_outlined,
                        color: cs.onSurfaceVariant),
                  ))
              : Container(
                  width: 56,
                  height: 56,
                  color: cs.surfaceContainerHighest,
                  child:
                      Icon(Icons.image_outlined, color: cs.onSurfaceVariant),
                ),
        ),
        title: Text(
          history.prediction as String,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(fontWeight: FontWeight.w700),
        ),
        subtitle: Text(
          '${((history.confidence as double) * 100).toStringAsFixed(1)}% ${LocaleService.instance.s('confidence')}',
          style: TextStyle(color: cs.onSurfaceVariant),
        ),
        trailing: Chip(
          label: Text(
            isHealthy
                ? LocaleService.instance.s('healthy')
                : LocaleService.instance.s('diseased'),
            style: TextStyle(
                color: color, fontSize: 12, fontWeight: FontWeight.w700),
          ),
          backgroundColor: color.withValues(alpha: 0.12),
          side: BorderSide.none,
          padding: EdgeInsets.zero,
          visualDensity: VisualDensity.compact,
        ),
      ),
    );
  }
}

class _InfoCard extends StatelessWidget {
  const _InfoCard({required this.cs});
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Icon(Icons.info_outline_rounded,
                color: cs.onSurfaceVariant, size: 22),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Powered by EfficientNetB0 + MobileNetV2 with CBAM attention. Runs fully offline on your device.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: cs.onSurfaceVariant,
                      height: 1.5,
                    ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
