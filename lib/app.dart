import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'screens/main_shell.dart';
import 'screens/result_screen.dart';
import 'screens/splash_screen.dart';
import 'services/locale_service.dart';
import 'theme/app_theme.dart';

class AgroVisionApp extends StatelessWidget {
  const AgroVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: LocaleService.instance,
      builder: (context, _) => MaterialApp(
        title: 'AgroVision AI',
        debugShowCheckedModeBanner: false,
        themeMode: ThemeMode.system,
        theme: lightTheme(),
        darkTheme: darkTheme(),
        locale: LocaleService.instance.locale,
        localizationsDelegates: const [
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        supportedLocales:
            LocaleService.languages.map((l) => Locale(l.$1)).toList(),
        initialRoute: SplashScreen.routeName,
        routes: {
          SplashScreen.routeName: (_) => const SplashScreen(),
          MainShell.routeName: (_) => const MainShell(),
        },
        onGenerateRoute: (settings) {
          if (settings.name == ResultScreen.routeName) {
            final args = settings.arguments as ResultScreenArgs;
            return MaterialPageRoute(
              builder: (_) => ResultScreen(args: args),
            );
          }
          return null;
        },
      ),
    );
  }
}
