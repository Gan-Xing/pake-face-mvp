import type { FaceNativeAPI } from "../lib/face/native-types";

declare global {
  interface Window {
    faceNative?: FaceNativeAPI;
  }
}

export {};
