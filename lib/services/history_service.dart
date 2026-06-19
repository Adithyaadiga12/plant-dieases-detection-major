import 'package:hive/hive.dart';

import '../models/scan_history.dart';

class HistoryService {
  HistoryService._();
  static final HistoryService instance = HistoryService._();

  static const _boxName = 'scan_history';
  static const _maxEntries = 50;
  Box<Map>? _box;

  Future<void> init() async {
    _box ??= await Hive.openBox<Map>(_boxName);
  }

  List<ScanHistory> getAll() {
    final values = _box?.values.toList() ?? [];
    return values
        .map((v) {
          try {
            return ScanHistory.fromJson(v);
          } catch (_) {
            return null;
          }
        })
        .whereType<ScanHistory>()
        .toList()
      ..sort((a, b) => b.scannedAt.compareTo(a.scannedAt));
  }

  Future<void> add(ScanHistory history) async {
    if (_box == null) return;
    await _box!.add(history.toJson());
    // Trim oldest entries beyond the cap
    while (_box!.length > _maxEntries) {
      await _box!.deleteAt(0);
    }
  }

  Future<void> delete(int boxIndex) async {
    await _box?.deleteAt(boxIndex);
  }

  Future<void> clear() async {
    await _box?.clear();
  }
}
