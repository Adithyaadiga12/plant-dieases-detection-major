import 'package:flutter/material.dart';
import 'package:hive_flutter/hive_flutter.dart';

import 'app.dart';
import 'services/database_service.dart';
import 'services/history_service.dart';
import 'services/locale_service.dart';
import 'services/tflite_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Hive.initFlutter();
  await LocaleService.instance.init();
  await HistoryService.instance.init();
  await DatabaseService.instance.init();
  TfliteService.instance.load();
  runApp(const AgroVisionApp());
}
