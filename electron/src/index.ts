import type { CapacitorElectronConfig } from '@capacitor-community/electron';
import { getCapacitorElectronConfig, setupElectronDeepLinking } from '@capacitor-community/electron';
import type { MenuItemConstructorOptions } from 'electron';
import { app, MenuItem, session, ipcMain } from 'electron';
import electronIsDev from 'electron-is-dev';
import unhandled from 'electron-unhandled';
import { autoUpdater } from 'electron-updater';
import { join } from 'path';

import { ElectronCapacitorApp, setupContentSecurityPolicy, setupReloadWatcher } from './setup';
import { detectArcFacePreprocess, initNativeFace, runArcFace } from './face/native-face';

// Graceful handling of unhandled errors.
unhandled();

// Define our menu templates (these are optional)
const trayMenuTemplate: (MenuItemConstructorOptions | MenuItem)[] = [new MenuItem({ label: 'Quit App', role: 'quit' })];
const appMenuBarMenuTemplate: (MenuItemConstructorOptions | MenuItem)[] = [
  { role: process.platform === 'darwin' ? 'appMenu' : 'fileMenu' },
  { role: 'editMenu' },
  { role: 'viewMenu' },
];

// Get Config options from capacitor.config
const capacitorFileConfig: CapacitorElectronConfig = getCapacitorElectronConfig();

// Initialize our app. You can pass menu templates into the app here.
// const myCapacitorApp = new ElectronCapacitorApp(capacitorFileConfig);
const myCapacitorApp = new ElectronCapacitorApp(capacitorFileConfig, trayMenuTemplate, appMenuBarMenuTemplate);

// If deeplinking is enabled then we will set it up here.
if (capacitorFileConfig.electron?.deepLinkingEnabled) {
  setupElectronDeepLinking(myCapacitorApp, {
    customProtocol: capacitorFileConfig.electron.deepLinkingCustomProtocol ?? 'mycapacitorapp',
  });
}

// If we are in Dev mode, use the file watcher components.
if (electronIsDev) {
  setupReloadWatcher(myCapacitorApp);
}

ipcMain.handle('face-init', async () => {
  await initNativeFace();
});

ipcMain.handle('face-arcface', async (_event, payload) => runArcFace(payload));
ipcMain.handle('face-arcface-preprocess', async () => detectArcFacePreprocess());

// Run Application
(async () => {
  // Wait for electron app to be ready.
  await app.whenReady();

  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
    const permissionName = String(permission);
    const allowed =
      permissionName === 'media' ||
      permissionName === 'mediaVideoCapture' ||
      permissionName === 'mediaAudioCapture';
    callback(allowed);
  });
  session.defaultSession.setPermissionCheckHandler((_webContents, permission) => {
    const permissionName = String(permission);
    return (
      permissionName === 'media' ||
      permissionName === 'mediaVideoCapture' ||
      permissionName === 'mediaAudioCapture'
    );
  });
  // Security - Set Content-Security-Policy based on whether or not we are in dev mode.
  // Custom CSP setup to include local-resource:
  const customScheme = myCapacitorApp.getCustomURLScheme();
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          electronIsDev
            ? `default-src ${customScheme}://* 'unsafe-inline' 'unsafe-eval' blob: data: local-resource: http://localhost:3001 ws://localhost:3001 https://static.byganxing.com; script-src ${customScheme}://* 'unsafe-inline' 'unsafe-eval' blob: data: local-resource: http://localhost:3001; connect-src ${customScheme}://* 'unsafe-inline' blob: data: local-resource: http://localhost:3001 ws://localhost:3001 https://static.byganxing.com; img-src ${customScheme}://* 'unsafe-inline' blob: data: local-resource: http://localhost:3001 https://static.byganxing.com; worker-src ${customScheme}://* 'unsafe-inline' 'unsafe-eval' blob: data: local-resource: http://localhost:3001;`
            : `default-src ${customScheme}://* 'unsafe-inline' 'unsafe-eval' blob: data: local-resource: https://static.byganxing.com; script-src ${customScheme}://* 'unsafe-inline' 'unsafe-eval' blob: data: local-resource:; connect-src ${customScheme}://* 'unsafe-inline' blob: data: local-resource: https://static.byganxing.com; img-src ${customScheme}://* 'unsafe-inline' blob: data: local-resource: https://static.byganxing.com; worker-src ${customScheme}://* 'unsafe-inline' 'unsafe-eval' blob: data: local-resource:;`,
        ],
        'Cross-Origin-Opener-Policy': ['same-origin'],
        'Cross-Origin-Embedder-Policy': ['require-corp'],
        'Access-Control-Allow-Origin': ['*'],
      },
    });
  });

  // Initialize our app, build windows, and load content.
  await myCapacitorApp.init();
  // Check for updates if we are in a packaged app.
  autoUpdater.checkForUpdatesAndNotify();
})();

// Handle when all of our windows are close (platforms have their own expectations).
app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// When the dock icon is clicked.
app.on('activate', async function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (myCapacitorApp.getMainWindow().isDestroyed()) {
    await myCapacitorApp.init();
  }
});

// Place all ipc or other electron api calls and custom functionality under this line
