import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../data/disease_data.dart';
import '../models/disease_info.dart';

class DatabaseService {
  DatabaseService._();
  static final DatabaseService instance = DatabaseService._();

  Database? _db;

  Future<void> init() async {
    _db = await openDatabase(
      p.join(await getDatabasesPath(), 'agrovision.db'),
      version: 1,
      onCreate: _onCreate,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE treatments (
        class_name  TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        description  TEXT NOT NULL,
        symptoms     TEXT NOT NULL,
        causes       TEXT NOT NULL,
        prevention   TEXT NOT NULL,
        remedy       TEXT NOT NULL,
        is_healthy   INTEGER NOT NULL DEFAULT 0
      )
    ''');

    final batch = db.batch();
    for (final entry in allDiseaseEntries) {
      batch.insert('treatments', {
        'class_name':   entry.key,
        'display_name': entry.value.name,
        'description':  entry.value.description,
        'symptoms':     entry.value.symptoms,
        'causes':       entry.value.causes,
        'prevention':   entry.value.prevention,
        'remedy':       entry.value.remedy,
        'is_healthy':   entry.value.isHealthy ? 1 : 0,
      }, conflictAlgorithm: ConflictAlgorithm.ignore);
    }
    await batch.commit(noResult: true);
  }

  Future<DiseaseInfo> getTreatment(String className) async {
    if (_db == null) return unknownDisease;
    final rows = await _db!.query(
      'treatments',
      where: 'class_name = ?',
      whereArgs: [className],
    );
    if (rows.isEmpty) return unknownDisease;
    final r = rows.first;
    return DiseaseInfo(
      name:        r['display_name'] as String,
      description: r['description']  as String,
      symptoms:    r['symptoms']     as String,
      causes:      r['causes']       as String,
      prevention:  r['prevention']   as String,
      remedy:      r['remedy']       as String,
      isHealthy:   (r['is_healthy']  as int) == 1,
    );
  }

  Future<List<MapEntry<String, DiseaseInfo>>> getAllTreatments() async {
    if (_db == null) return allDiseaseEntries;
    final rows = await _db!.query('treatments', orderBy: 'class_name ASC');
    return rows.map((r) => MapEntry(
      r['class_name'] as String,
      DiseaseInfo(
        name:        r['display_name'] as String,
        description: r['description']  as String,
        symptoms:    r['symptoms']     as String,
        causes:      r['causes']       as String,
        prevention:  r['prevention']   as String,
        remedy:      r['remedy']       as String,
        isHealthy:   (r['is_healthy']  as int) == 1,
      ),
    )).toList();
  }
}
