import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class TutorialOverlay extends StatefulWidget {
  const TutorialOverlay({super.key, required this.child});
  final Widget child;

  @override
  State<TutorialOverlay> createState() => _TutorialOverlayState();
}

class _TutorialOverlayState extends State<TutorialOverlay>
    with SingleTickerProviderStateMixin {
  static const _prefKey = 'tutorial_shown_v1';

  bool _show = false;
  int _page = 0;
  late final AnimationController _ctrl;
  late final Animation<double> _fade;

  static const _pages = [
    _TutorialPage(
      icon: Icons.document_scanner_rounded,
      title: 'Scan a Leaf',
      body:
          'Tap the Scan tab and capture a single leaf photo with your camera, '
          'or pick one from your gallery.',
    ),
    _TutorialPage(
      icon: Icons.wb_sunny_outlined,
      title: 'Good Lighting',
      body:
          'Use natural light and fill the frame with one leaf. '
          'Avoid shadows, blurry shots, and direct sunlight.',
    ),
    _TutorialPage(
      icon: Icons.analytics_outlined,
      title: 'Instant Results',
      body:
          'The on-device AI identifies 38 plant diseases in seconds — '
          'no internet required. Tap the speaker icon to hear the result.',
    ),
    _TutorialPage(
      icon: Icons.local_florist_outlined,
      title: 'Disease Library',
      body:
          'Browse all 38 detectable diseases in the Library tab to learn '
          'symptoms, causes, and treatments.',
    ),
  ];

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 350));
    _fade = CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut);
    _checkFirstLaunch();
  }

  Future<void> _checkFirstLaunch() async {
    final prefs = await SharedPreferences.getInstance();
    final shown = prefs.getBool(_prefKey) ?? false;
    if (!shown && mounted) {
      setState(() => _show = true);
      _ctrl.forward();
    }
  }

  Future<void> _dismiss() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefKey, true);
    await _ctrl.reverse();
    if (mounted) setState(() => _show = false);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_show) return widget.child;

    return Stack(
      children: [
        widget.child,
        FadeTransition(
          opacity: _fade,
          child: _TutorialSheet(
            pages: _pages,
            page: _page,
            onNext: () {
              if (_page < _pages.length - 1) {
                setState(() => _page++);
              } else {
                _dismiss();
              }
            },
            onSkip: _dismiss,
          ),
        ),
      ],
    );
  }
}

class _TutorialSheet extends StatelessWidget {
  const _TutorialSheet({
    required this.pages,
    required this.page,
    required this.onNext,
    required this.onSkip,
  });
  final List<_TutorialPage> pages;
  final int page;
  final VoidCallback onNext;
  final VoidCallback onSkip;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final current = pages[page];
    final isLast = page == pages.length - 1;

    return Container(
      color: Colors.black54,
      alignment: Alignment.bottomCenter,
      child: Container(
        margin: const EdgeInsets.all(16),
        padding: const EdgeInsets.fromLTRB(24, 32, 24, 24),
        decoration: BoxDecoration(
          color: cs.surface,
          borderRadius: BorderRadius.circular(28),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: cs.primaryContainer,
                shape: BoxShape.circle,
              ),
              child: Icon(current.icon, size: 40, color: cs.primary),
            ),
            const SizedBox(height: 20),
            Text(
              current.title,
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(fontWeight: FontWeight.w900),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 10),
            Text(
              current.body,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: cs.onSurfaceVariant,
                    height: 1.6,
                  ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            // Page dots
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                pages.length,
                (i) => AnimatedContainer(
                  duration: const Duration(milliseconds: 250),
                  width: i == page ? 20 : 8,
                  height: 8,
                  margin: const EdgeInsets.symmetric(horizontal: 3),
                  decoration: BoxDecoration(
                    color: i == page
                        ? cs.primary
                        : cs.primary.withValues(alpha: 0.25),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                if (isLast)
                  Expanded(
                    child: FilledButton(
                      onPressed: onNext,
                      child: const Text('Get Started'),
                    ),
                  )
                else ...[
                  TextButton(
                    onPressed: onSkip,
                    child: const Text('Skip'),
                  ),
                  const Spacer(),
                  FilledButton(
                    style: FilledButton.styleFrom(
                      minimumSize: const Size(120, 48),
                    ),
                    onPressed: onNext,
                    child: const Text('Next'),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _TutorialPage {
  const _TutorialPage({
    required this.icon,
    required this.title,
    required this.body,
  });
  final IconData icon;
  final String title;
  final String body;
}
