import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';

import '../models/prediction_result.dart';
import 'locale_service.dart';

class TtsService {
  TtsService._();
  static final TtsService instance = TtsService._();

  final FlutterTts _tts = FlutterTts();
  bool _initialized = false;

  // Reactive speaking state — listen to this instead of polling isSpeaking.
  final ValueNotifier<bool> speakingState = ValueNotifier(false);

  bool get isSpeaking => speakingState.value;

  Future<void> init() async {
    if (_initialized) return;
    await _tts.setVolume(1.0);
    await _tts.setSpeechRate(0.75); // 0.42 was uncomfortably slow
    await _tts.setPitch(1.0);
    _tts.setCompletionHandler(() => speakingState.value = false);
    _tts.setCancelHandler(() => speakingState.value = false);
    _tts.setErrorHandler((_) => speakingState.value = false);
    _initialized = true;
  }

  Future<void> announceResult(PredictionResult result) async {
    if (!LocaleService.instance.ttsEnabled) return;
    await init();
    if (speakingState.value) {
      await stop();
      return;
    }
    final locale = LocaleService.instance;
    final text = result.isHealthy
        ? locale.ttsHealthyText()
        : locale.ttsDiseaseText(result.displayLabel);
    await _speak(locale.ttsLocaleCode, text);
  }

  Future<void> _speak(String localeCode, String text) async {
    final available = await _isLanguageAvailable(localeCode);
    await _tts.setLanguage(available ? localeCode : 'en-US');
    speakingState.value = true;
    await _tts.speak(text);
  }

  Future<bool> _isLanguageAvailable(String code) async {
    try {
      final langs = await _tts.getLanguages;
      final prefix = code.split('-').first.toLowerCase();
      return (langs as List?)
              ?.any((l) => l.toString().toLowerCase().startsWith(prefix)) ??
          false;
    } catch (_) {
      return false;
    }
  }

  Future<void> stop() async {
    speakingState.value = false;
    await _tts.stop();
  }
}
