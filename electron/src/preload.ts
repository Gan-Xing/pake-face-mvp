import { contextBridge, ipcRenderer } from 'electron';

import './rt/electron-rt';

contextBridge.exposeInMainWorld('faceNative', {
      init: () => ipcRenderer.invoke('face-init'),
      runArcFace: (input: Float32Array) => ipcRenderer.invoke('face-arcface', { input }),
      getArcFacePreprocess: () => ipcRenderer.invoke('face-arcface-preprocess'),
      detectFace: (payload: {
        input: Float32Array;
        inputWidth: number;
        inputHeight: number;
        scaleX: number;
        scaleY: number;
      }) => ipcRenderer.invoke('face-detect', payload),
    });
  
    console.log('User Preload!');
