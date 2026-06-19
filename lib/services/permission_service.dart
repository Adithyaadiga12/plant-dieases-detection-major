import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

enum PermissionResult { granted, denied, permanentlyDenied }

class PermissionService {
  static final PermissionService instance = PermissionService._();
  PermissionService._();

  Future<PermissionResult> checkCamera() => _check(Permission.camera);

  Future<PermissionResult> checkGallery() => _check(Permission.photos);

  Future<PermissionResult> _check(Permission perm) async {
    final status = await perm.status;
    if (status.isGranted) return PermissionResult.granted;
    if (status.isPermanentlyDenied) return PermissionResult.permanentlyDenied;

    final result = await perm.request();
    if (result.isGranted) return PermissionResult.granted;
    if (result.isPermanentlyDenied) return PermissionResult.permanentlyDenied;
    return PermissionResult.denied;
  }

  void showSettingsDialog(BuildContext context, String label) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.lock_outline_rounded, size: 36),
        title: Text('$label Access Blocked'),
        content: Text(
          '$label permission was permanently denied. '
          'Please enable it in App Settings to continue.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.pop(ctx);
              openAppSettings();
            },
            child: const Text('Open Settings'),
          ),
        ],
      ),
    );
  }

  void showDeniedSnackbar(BuildContext context, String label) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('$label permission denied.'),
        action: SnackBarAction(
          label: 'Settings',
          onPressed: openAppSettings,
        ),
      ),
    );
  }
}
