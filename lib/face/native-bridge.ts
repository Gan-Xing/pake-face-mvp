import type {
  FaceNativeAPI,
  NativeDetectPayload,
  NativeFaceDetection,
  NativeRawTensor,
} from "./native-types";

export const getFaceNative = (): FaceNativeAPI | null => {
  if (typeof window === "undefined") return null;
  return window.faceNative ?? null;
};

export const isFaceNativeAvailable = (): boolean => Boolean(getFaceNative());

export const isFaceNativeDetectionAvailable = (): boolean =>
  Boolean(getFaceNative()?.detectFace);

export const requireFaceNative = (): FaceNativeAPI => {
  const api = getFaceNative();
  if (!api) {
    throw new Error("未检测到本地推理环境");
  }
  return api;
};

export const initFaceNative = async (): Promise<FaceNativeAPI> => {
  const api = requireFaceNative();
  await api.init();
  return api;
};

export const runArcFaceNative = async (input: Float32Array): Promise<NativeRawTensor> => {
  const api = requireFaceNative();
  return api.runArcFace(input);
};

export const detectFaceNative = async (
  payload: NativeDetectPayload
): Promise<NativeFaceDetection | null> => {
  const api = requireFaceNative();
  if (!api.detectFace) {
    throw new Error("未检测到本地人脸检测能力");
  }
  return api.detectFace(payload);
};

export const getArcFacePreprocessNative = async (): Promise<string | null> => {
  const api = requireFaceNative();
  if (!api.getArcFacePreprocess) return null;
  return api.getArcFacePreprocess();
};
