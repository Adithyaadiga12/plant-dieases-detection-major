import 'package:flutter/material.dart';

import '../services/tflite_service.dart';
import 'main_shell.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
  static const routeName = '/';

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _fade;
  late final Animation<double> _scale;
  String _status = 'Loading model…';
  bool _modelError = false;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1200));
    _fade = CurvedAnimation(parent: _ctrl, curve: Curves.easeIn);
    _scale = Tween(begin: 0.7, end: 1.0)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.elasticOut));
    _ctrl.forward();
    _initialize();
  }

  Future<void> _initialize() async {
    // Warm up the model on splash so first scan is instant
    await TfliteService.instance.load();

    if (!mounted) return;

    if (TfliteService.instance.loadError != null) {
      setState(() {
        _modelError = true;
        _status = 'Model failed to load';
      });
      return;
    }

    setState(() => _status = 'Ready');
    await Future.delayed(const Duration(milliseconds: 600));
    if (mounted) Navigator.pushReplacementNamed(context, MainShell.routeName);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: cs.primary,
      body: Center(
        child: FadeTransition(
          opacity: _fade,
          child: ScaleTransition(
            scale: _scale,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: Image.asset(
                    'assets/images/logo.jpeg',
                    width: 104,
                    height: 104,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(height: 28),
                Text(
                  'AgroVision AI',
                  style:
                      Theme.of(context).textTheme.headlineMedium?.copyWith(
                            color: cs.onPrimary,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 0.5,
                          ),
                ),
                const SizedBox(height: 6),
                Text(
                  'Plant Disease Detection',
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: cs.onPrimary.withValues(alpha: 0.8),
                        fontWeight: FontWeight.w500,
                      ),
                ),
                const SizedBox(height: 52),
                if (_modelError) ...[
                  Icon(Icons.error_outline_rounded,
                      color: cs.onPrimary, size: 32),
                  const SizedBox(height: 10),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 40),
                    child: Text(
                      TfliteService.instance.loadError ?? 'Unknown error',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                          color: cs.onPrimary.withValues(alpha: 0.85), fontSize: 13),
                    ),
                  ),
                  const SizedBox(height: 16),
                  OutlinedButton(
                    onPressed: () {
                      setState(() {
                        _modelError = false;
                        _status = 'Retrying…';
                      });
                      _initialize();
                    },
                    style: OutlinedButton.styleFrom(
                        foregroundColor: cs.onPrimary,
                        side: BorderSide(color: cs.onPrimary.withValues(alpha: 0.6))),
                    child: const Text('Retry'),
                  ),
                ] else ...[
                  SizedBox(
                    width: 28,
                    height: 28,
                    child: CircularProgressIndicator(
                      color: cs.onPrimary.withValues(alpha: 0.7),
                      strokeWidth: 2.5,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _status,
                    style: TextStyle(
                        color: cs.onPrimary.withValues(alpha: 0.6), fontSize: 13),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
