import 'package:flutter/material.dart';

import '../services/locale_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String _selectedCode = LocaleService.instance.code;
  bool _ttsEnabled = LocaleService.instance.ttsEnabled;

  Future<void> _setLocale(String code) async {
    setState(() => _selectedCode = code);
    await LocaleService.instance.setLocale(code);
  }

  Future<void> _setTts(bool value) async {
    setState(() => _ttsEnabled = value);
    await LocaleService.instance.setTtsEnabled(value);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final l = LocaleService.instance;

    return Scaffold(
      appBar: AppBar(
        title: Text(l.s('settingsTitle'),
            style: const TextStyle(fontWeight: FontWeight.w800)),
      ),
      body: ListView(
        children: [
          // ── Language ────────────────────────────────────────────────────────
          _SectionHeader(label: l.s('sectionLanguage'), cs: cs),
          ...LocaleService.languages.map((lang) {
            final (code, native, english) = lang;
            final selected = _selectedCode == code;
            return RadioListTile<String>(
              value: code,
              groupValue: _selectedCode,
              onChanged: (v) => _setLocale(v!),
              title: Text(
                native,
                style: TextStyle(
                  fontWeight: selected ? FontWeight.w700 : FontWeight.normal,
                ),
              ),
              subtitle: code != 'en'
                  ? Text(english,
                      style: TextStyle(
                          color: cs.onSurfaceVariant, fontSize: 12))
                  : null,
              activeColor: cs.primary,
              secondary: selected
                  ? Icon(Icons.check_circle_rounded,
                      color: cs.primary, size: 20)
                  : const SizedBox(width: 20),
            );
          }),

          const Divider(height: 1),

          // ── TTS ─────────────────────────────────────────────────────────────
          _SectionHeader(label: l.s('sectionTts'), cs: cs),
          SwitchListTile(
            value: _ttsEnabled,
            onChanged: _setTts,
            activeThumbColor: cs.primary,
            title: Text(l.s('ttsEnabled')),
            secondary: Icon(
              _ttsEnabled ? Icons.volume_up_rounded : Icons.volume_off_rounded,
              color: cs.primary,
            ),
          ),
          if (_ttsEnabled)
            Padding(
              padding: const EdgeInsets.fromLTRB(72, 0, 16, 12),
              child: Text(
                _ttsLanguageNote(l.code),
                style: TextStyle(
                    color: cs.onSurfaceVariant,
                    fontSize: 12,
                    height: 1.4),
              ),
            ),

          const Divider(height: 1),

          // ── About ───────────────────────────────────────────────────────────
          _SectionHeader(label: l.s('sectionAbout'), cs: cs),
          ListTile(
            leading: Icon(Icons.psychology_outlined, color: cs.primary),
            title: Text(l.s('aboutModel'),
                style: const TextStyle(fontWeight: FontWeight.w600)),
            subtitle: Text(l.s('aboutModelDesc'),
                style:
                    TextStyle(color: cs.onSurfaceVariant, height: 1.4)),
          ),
          ListTile(
            leading: Icon(Icons.info_outline_rounded,
                color: cs.onSurfaceVariant),
            title: Text(l.s('aboutVersion'),
                style: TextStyle(color: cs.onSurfaceVariant)),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  String _ttsLanguageNote(String code) {
    final lang = LocaleService.languages
        .firstWhere((l) => l.$1 == code, orElse: () => ('en', 'English', 'English'));
    return 'Voice will speak in ${lang.$2} (${lang.$3}). '
        'If unavailable on your device, English is used as fallback.';
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.label, required this.cs});
  final String label;
  final ColorScheme cs;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 8),
      child: Text(
        label,
        style: TextStyle(
          color: cs.primary,
          fontWeight: FontWeight.w800,
          fontSize: 13,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
