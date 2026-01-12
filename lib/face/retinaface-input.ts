export type RetinaFaceInput = {
  input: Float32Array;
  inputWidth: number;
  inputHeight: number;
  scaleX: number;
  scaleY: number;
};

const DEFAULT_INPUT_WIDTH = 640;
const DEFAULT_INPUT_HEIGHT = 608;

export const buildRetinaFaceInput = (
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  inputWidth = DEFAULT_INPUT_WIDTH,
  inputHeight = DEFAULT_INPUT_HEIGHT
): RetinaFaceInput => {
  const canvas = document.createElement("canvas");
  canvas.width = inputWidth;
  canvas.height = inputHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("无法读取图像数据");
  }
  ctx.drawImage(source, 0, 0, inputWidth, inputHeight);
  const { data } = ctx.getImageData(0, 0, inputWidth, inputHeight);
  const planeSize = inputWidth * inputHeight;
  const input = new Float32Array(planeSize * 3);
  const mean = [104, 117, 123];

  for (let i = 0; i < planeSize; i += 1) {
    const base = i * 4;
    const r = data[base];
    const g = data[base + 1];
    const b = data[base + 2];
    const outBase = i * 3;
    input[outBase] = b - mean[0];
    input[outBase + 1] = g - mean[1];
    input[outBase + 2] = r - mean[2];
  }

  return {
    input,
    inputWidth,
    inputHeight,
    scaleX: sourceWidth / inputWidth,
    scaleY: sourceHeight / inputHeight,
  };
};
