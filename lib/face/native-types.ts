export type NativeRawTensor = {
  data: ArrayBuffer;
  dims: number[];
};

export type NativeFaceBox = {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
};

export type NativeFaceDetection = {
  score: number;
  box: NativeFaceBox;
  landmarks?: number[][];
};

export type NativeDetectPayload = {
  input: Float32Array;
  inputWidth: number;
  inputHeight: number;
  scaleX: number;
  scaleY: number;
};

export type FaceNativeAPI = {
  init: () => Promise<void>;
  runArcFace: (input: Float32Array) => Promise<NativeRawTensor>;
  getArcFacePreprocess?: () => Promise<string | null>;
  detectFace?: (payload: NativeDetectPayload) => Promise<NativeFaceDetection | null>;
};
