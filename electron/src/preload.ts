import { contextBridge, ipcRenderer } from 'electron';

import './rt/electron-rt';

contextBridge.exposeInMainWorld('faceNative', {
      init: () => ipcRenderer.invoke('face-init'),
      runArcFace: (input: Float32Array) => ipcRenderer.invoke('face-arcface', { input }),
      getArcFacePreprocess: () => ipcRenderer.invoke('face-arcface-preprocess'),
    });
  
    console.log('User Preload!');
