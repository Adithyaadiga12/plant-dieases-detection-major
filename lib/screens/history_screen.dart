import 'dart:io';

import 'package:flutter/material.dart';

import '../models/scan_history.dart';
import '../services/history_service.dart';
import '../services/locale_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  int _filter = 0; // 0=All, 1=Healthy, 2=Diseased

  List<ScanHistory> _filtered(List<ScanHistory> all) => switch (_filter) {
        1 => all.where((h) => h.isHealthy).toList(),
        2 => all.where((h) => !h.isHealthy).toList(),
        _ => all,
      };

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final all = HistoryService.instance.getAll();
    final items = _filtered(all);

    return Scaffold(
      appBar: AppBar(
        title: Text(LocaleService.instance.s('historyTitle'),
            style: const TextStyle(fontWeight: FontWeight.w800)),
        actions: [
          if (all.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_sweep_outlined),
              tooltip: LocaleService.instance.s('clearAll'),
              onPressed: () => _confirmClear(context),
            ),
        ],
      ),
      body: Column(
        children: [
          // ── Filter chips ──
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
            child: Row(
              children: [
                _FilterChip(label: LocaleService.instance.s('filterAll'), selected: _filter == 0, onTap: () => setState(() => _filter = 0)),
                const SizedBox(width: 8),
                _FilterChip(label: LocaleService.instance.s('filterHealthy'), selected: _filter == 1, onTap: () => setState(() => _filter = 1), color: Colors.green),
                const SizedBox(width: 8),
                _FilterChip(label: LocaleService.instance.s('filterDiseased'), selected: _filter == 2, onTap: () => setState(() => _filter = 2), color: Colors.orange),
              ],
            ),
          ),
          const Divider(height: 1),
          // ── List ──
          Expanded(
            child: items.isEmpty
                ? _EmptyState(cs: cs, isFiltered: _filter != 0)
                : ListView.separated(
                    padding: const EdgeInsets.all(16),
                    itemCount: items.length,
                    separatorBuilder: (_, _) => const SizedBox(height: 10),
                    itemBuilder: (_, i) => _HistoryCard(
                      history: items[i],
                      cs: cs,
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Future<void> _confirmClear(BuildContext context) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: Text(LocaleService.instance.s('clearHistory')),
        content: Text(LocaleService.instance.s('clearConfirm')),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text(LocaleService.instance.s('cancel'))),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: Text(LocaleService.instance.s('clear'))),
        ],
      ),
    );
    if (ok == true) {
      await HistoryService.instance.clear();
      if (mounted) setState(() {});
    }
  }
}

class _FilterChip extends StatelessWidget {
  const _FilterChip({
    required this.label,
    required this.selected,
    required this.onTap,
    this.color,
  });
  final String label;
  final bool selected;
  final VoidCallback onTap;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final c = color ?? cs.primary;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        decoration: BoxDecoration(
          color: selected ? c.withValues(alpha: 0.12) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: selected ? c : cs.outlineVariant),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: selected ? c : cs.onSurfaceVariant,
            fontWeight: selected ? FontWeight.w700 : FontWeight.normal,
            fontSize: 13,
          ),
        ),
      ),
    );
  }
}

class _HistoryCard extends StatelessWidget {
  const _HistoryCard({required this.history, required this.cs});
  final ScanHistory history;
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    final isHealthy = history.isHealthy;
    final statusColor = isHealthy ? Colors.green : Colors.orange;

    return Card(
      child: ListTile(
        contentPadding: const EdgeInsets.all(12),
        leading: ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: SizedBox(
            width: 60,
            height: 60,
            child: _ImageOrPlaceholder(path: history.imagePath, cs: cs),
          ),
        ),
        title: Text(
          history.prediction,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 4),
            Text(
              '${(history.confidence * 100).toStringAsFixed(1)}% ${LocaleService.instance.s('confidence')}',
              style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
            ),
            Text(
              _formatDate(history.scannedAt),
              style: TextStyle(color: cs.onSurfaceVariant, fontSize: 11),
            ),
          ],
        ),
        trailing: Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            color: statusColor,
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }

  String _formatDate(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${dt.day}/${dt.month}/${dt.year}';
  }
}

class _ImageOrPlaceholder extends StatelessWidget {
  const _ImageOrPlaceholder({required this.path, required this.cs});
  final String path;
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    final file = File(path);
    if (file.existsSync()) {
      return Image.file(file, fit: BoxFit.cover);
    }
    return Container(
      color: cs.surfaceContainerHighest,
      child: Icon(Icons.image_not_supported_outlined,
          color: cs.onSurfaceVariant, size: 28),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.cs, required this.isFiltered});
  final ColorScheme cs;
  final bool isFiltered;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(isFiltered ? Icons.filter_list_off_rounded : Icons.history_rounded,
              size: 64, color: cs.onSurfaceVariant.withValues(alpha: 0.4)),
          const SizedBox(height: 16),
          Text(
            isFiltered
                ? LocaleService.instance.s('noMatch')
                : LocaleService.instance.s('noScansYet'),
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: cs.onSurfaceVariant,
                  fontWeight: FontWeight.w700,
                ),
          ),
          const SizedBox(height: 6),
          Text(
            isFiltered
                ? LocaleService.instance.s('tryFilter')
                : LocaleService.instance.s('scanFirst'),
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: cs.onSurfaceVariant.withValues(alpha: 0.7),
                ),
          ),
        ],
      ),
    );
  }
}
