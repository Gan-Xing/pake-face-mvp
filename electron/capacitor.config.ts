import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.pake.faceattendance',
  appName: 'Face Attendance',
  webDir: 'out',
  server: {
    androidScheme: 'https',
    iosScheme: 'https'
  }
};

export default config;
