import 'package:flutter/material.dart';

import '../services/locale_service.dart';
import '../widgets/tutorial_overlay.dart';
import 'disease_library_screen.dart';
import 'history_screen.dart';
import 'home_screen.dart';
import 'scan_screen.dart';
import 'settings_screen.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key});
  static const routeName = '/home';

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _index = 0;

  static const _screens = [
    HomeScreen(),
    ScanScreen(),
    HistoryScreen(),
    DiseaseLibraryScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    final l = LocaleService.instance;
    return TutorialOverlay(
      child: Scaffold(
        body: IndexedStack(index: _index, children: _screens),
        bottomNavigationBar: NavigationBar(
          selectedIndex: _index,
          onDestinationSelected: (i) => setState(() => _index = i),
          destinations: [
            NavigationDestination(
              icon: const Icon(Icons.home_outlined),
              selectedIcon: const Icon(Icons.home_rounded),
              label: l.s('navHome'),
            ),
            NavigationDestination(
              icon: const Icon(Icons.document_scanner_outlined),
              selectedIcon: const Icon(Icons.document_scanner_rounded),
              label: l.s('navScan'),
            ),
            NavigationDestination(
              icon: const Icon(Icons.history_outlined),
              selectedIcon: const Icon(Icons.history_rounded),
              label: l.s('navHistory'),
            ),
            NavigationDestination(
              icon: const Icon(Icons.local_florist_outlined),
              selectedIcon: const Icon(Icons.local_florist_rounded),
              label: l.s('navLibrary'),
            ),
            NavigationDestination(
              icon: const Icon(Icons.settings_outlined),
              selectedIcon: const Icon(Icons.settings_rounded),
              label: l.s('navSettings'),
            ),
          ],
        ),
      ),
    );
  }
}
