import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../l10n/app_strings.dart';

class LocaleService extends ChangeNotifier {
  LocaleService._();
  static final LocaleService instance = LocaleService._();

  static const _key = 'app_locale';
  static const _ttsKey = 'tts_enabled';

  String _code = 'en';
  bool _ttsEnabled = true;

  String get code => _code;
  bool get ttsEnabled => _ttsEnabled;
  Locale get locale => Locale(_code);

  static const languages = [
    ('en', 'English', 'English'),
    ('kn', 'ಕನ್ನಡ', 'Kannada'),
    ('hi', 'हिंदी', 'Hindi'),
    ('ta', 'தமிழ்', 'Tamil'),
    ('te', 'తెలుగు', 'Telugu'),
  ];

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _code = prefs.getString(_key) ?? 'en';
    _ttsEnabled = prefs.getBool(_ttsKey) ?? true;
  }

  Future<void> setLocale(String code) async {
    if (_code == code) return;
    _code = code;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, code);
    notifyListeners();
  }

  Future<void> setTtsEnabled(bool value) async {
    _ttsEnabled = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_ttsKey, value);
    notifyListeners();
  }

  String s(String key) =>
      appStrings[_code]?[key] ?? appStrings['en']?[key] ?? key;

  String get ttsLocaleCode => switch (_code) {
        'kn' => 'kn-IN',
        'hi' => 'hi-IN',
        'ta' => 'ta-IN',
        'te' => 'te-IN',
        _ => 'en-US',
      };

  String ttsHealthyText() => switch (_code) {
        'kn' => 'ಸಸ್ಯ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಯಾವುದೇ ರೋಗ ಕಂಡುಬಂದಿಲ್ಲ.',
        'hi' => 'पौधा स्वस्थ है। कोई बीमारी नहीं मिली।',
        'ta' => 'தாவரம் ஆரோக்கியமாக உள்ளது. எந்த நோயும் இல்லை.',
        'te' => 'మొక్క ఆరోగ్యంగా ఉంది. ఎటువంటి వ్యాధి లేదు.',
        _ => 'Plant is healthy. No disease detected.',
      };

  String ttsDiseaseText(String label) => switch (_code) {
        'kn' => 'ರೋಗ ಪತ್ತೆಯಾಗಿದೆ: $label. ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.',
        'hi' => 'रोग पाया गया: $label. कृषि विशेषज्ञ से परामर्श लें।',
        'ta' => 'நோய் கண்டறியப்பட்டது: $label. விவசாய நிபுணரை அணுகவும்.',
        'te' => 'వ్యాధి గుర్తించబడింది: $label. వ్యవసాయ నిపుణుడిని సంప్రదించండి.',
        _ => 'Disease detected: $label. Please consult an agricultural expert.',
      };
}

extension L10n on BuildContext {
  String s(String key) => LocaleService.instance.s(key);
}
