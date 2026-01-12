import { ARC_FACE_INPUT_SIZE } from "../../lib/face/utils";

export const FACE_DETECTION_MODEL_URL = "/mediapipe/face_detection";
export const FACE_MESH_MODEL_URL = "/mediapipe/face_mesh";
export const DEFAULT_THRESHOLD = 0.95;

export const buildInputFromCanvas = (
  aligned: HTMLCanvasElement
): Float32Array => {
  const ctx = aligned.getContext("2d");
  if (!ctx) throw new Error("无法读取人脸图像");
  const imageData = ctx.getImageData(0, 0, ARC_FACE_INPUT_SIZE, ARC_FACE_INPUT_SIZE);
  const { data } = imageData;
  const planeSize = ARC_FACE_INPUT_SIZE * ARC_FACE_INPUT_SIZE;
  const input = new Float32Array(planeSize * 3);

  // Preprocess: RGB, 0..255 (Raw)
  for (let i = 0; i < planeSize; i += 1) {
    const base = i * 4;
    const r = data[base];
    const g = data[base + 1];
    const b = data[base + 2];
    
    // RGB 0..255
    input[i] = r;
    input[planeSize + i] = g;
    input[planeSize * 2 + i] = b;
  }
  return input;
};
