import 'package:agrovision_ai/app.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('app launches', (WidgetTester tester) async {
    await tester.pumpWidget(const AgroVisionApp());
    expect(find.byType(AgroVisionApp), findsOneWidget);
  });
}
