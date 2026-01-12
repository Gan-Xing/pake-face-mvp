export type NativeRawTensor = {
  data: ArrayBuffer;
  dims: number[];
};

export type FaceNativeAPI = {
  init: () => Promise<void>;
  runArcFace: (input: Float32Array) => Promise<NativeRawTensor>;
  getArcFacePreprocess?: () => Promise<string | null>;
};
