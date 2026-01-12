import * as ort from 'onnxruntime-node';
import { app } from 'electron';
import { join } from 'path';
import { existsSync } from 'fs';

type RetinaFacePayload = {
  input: ArrayBuffer | Float32Array | number[];
  inputWidth: number;
  inputHeight: number;
  scaleX: number;
  scaleY: number;
};

export type RetinaFaceDetection = {
  score: number;
  box: { xMin: number; yMin: number; width: number; height: number };
  landmarks: number[][];
};

const VARIANCES: [number, number] = [0.1, 0.2];
const CONF_THRESHOLD = 0.3;
const NMS_THRESHOLD = 0.4;
const TOP_K = 1000;
const KEEP_TOP_K = 20;

const getModelPath = () => {
  if (app.isPackaged) {
    return join(process.resourcesPath, 'models', 'retinaface', 'retinaface_mbn025.onnx');
  }
  return join(app.getAppPath(), 'app', 'models', 'retinaface', 'retinaface_mbn025.onnx');
};

const RETINAFACE_MODEL_PATH = getModelPath();

let retinafaceSession: ort.InferenceSession | null = null;
let cachedPriors:
  | { width: number; height: number; priors: Float32Array }
  | null = null;
let debugLogged = false;

const toFloat32Array = (input: ArrayBuffer | Float32Array | number[]): Float32Array => {
  if (input instanceof Float32Array) return input;
  if (ArrayBuffer.isView(input)) {
    return new Float32Array(input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength));
  }
  if (input instanceof ArrayBuffer) return new Float32Array(input);
  return Float32Array.from(input);
};

const generatePriors = (inputWidth: number, inputHeight: number): Float32Array => {
  if (cachedPriors && cachedPriors.width === inputWidth && cachedPriors.height === inputHeight) {
    return cachedPriors.priors;
  }
  const minSizes = [
    [16, 32],
    [64, 128],
    [256, 512],
  ];
  const steps = [8, 16, 32];
  const featureMapSizes = steps.map((step) => [
    Math.ceil(inputHeight / step),
    Math.ceil(inputWidth / step),
  ]);
  const priors: number[] = [];

  for (let k = 0; k < featureMapSizes.length; k += 1) {
    const [featureH, featureW] = featureMapSizes[k];
    const step = steps[k];
    for (let i = 0; i < featureH; i += 1) {
      for (let j = 0; j < featureW; j += 1) {
        for (let m = 0; m < minSizes[k].length; m += 1) {
          const minSize = minSizes[k][m];
          const sKx = minSize / inputWidth;
          const sKy = minSize / inputHeight;
          const cx = (j + 0.5) * step / inputWidth;
          const cy = (i + 0.5) * step / inputHeight;
          priors.push(cx, cy, sKx, sKy);
        }
      }
    }
  }

  const priorsArray = Float32Array.from(priors);
  cachedPriors = { width: inputWidth, height: inputHeight, priors: priorsArray };
  return priorsArray;
};

const decodeBoxes = (
  loc: Float32Array,
  priors: Float32Array
): Float32Array => {
  const numPriors = priors.length / 4;
  const boxes = new Float32Array(numPriors * 4);
  for (let i = 0; i < numPriors; i += 1) {
    const priorIdx = i * 4;
    const locIdx = i * 4;
    const cx = priors[priorIdx] + loc[locIdx] * VARIANCES[0] * priors[priorIdx + 2];
    const cy = priors[priorIdx + 1] + loc[locIdx + 1] * VARIANCES[0] * priors[priorIdx + 3];
    const w = priors[priorIdx + 2] * Math.exp(loc[locIdx + 2] * VARIANCES[1]);
    const h = priors[priorIdx + 3] * Math.exp(loc[locIdx + 3] * VARIANCES[1]);
    boxes[locIdx] = cx - w / 2;
    boxes[locIdx + 1] = cy - h / 2;
    boxes[locIdx + 2] = cx + w / 2;
    boxes[locIdx + 3] = cy + h / 2;
  }
  return boxes;
};

const decodeLandmarks = (
  landms: Float32Array,
  priors: Float32Array
): Float32Array => {
  const numPriors = priors.length / 4;
  const decoded = new Float32Array(numPriors * 10);
  for (let i = 0; i < numPriors; i += 1) {
    const priorIdx = i * 4;
    const landmIdx = i * 10;
    for (let k = 0; k < 5; k += 1) {
      const x = priors[priorIdx] + landms[landmIdx + k * 2] * VARIANCES[0] * priors[priorIdx + 2];
      const y =
        priors[priorIdx + 1] +
        landms[landmIdx + k * 2 + 1] * VARIANCES[0] * priors[priorIdx + 3];
      decoded[landmIdx + k * 2] = x;
      decoded[landmIdx + k * 2 + 1] = y;
    }
  }
  return decoded;
};

const iou = (a: number[], b: number[]): number => {
  const interXMin = Math.max(a[0], b[0]);
  const interYMin = Math.max(a[1], b[1]);
  const interXMax = Math.min(a[2], b[2]);
  const interYMax = Math.min(a[3], b[3]);
  const interW = Math.max(0, interXMax - interXMin);
  const interH = Math.max(0, interYMax - interYMin);
  const interArea = interW * interH;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - interArea;
  return union <= 0 ? 0 : interArea / union;
};

const nms = (boxes: number[][], scores: number[]): number[] => {
  const order = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K)
    .map((entry) => entry.idx);
  const keep: number[] = [];
  while (order.length > 0) {
    const current = order.shift();
    if (current === undefined) break;
    keep.push(current);
    const rest: number[] = [];
    for (let i = 0; i < order.length; i += 1) {
      const idx = order[i];
      if (iou(boxes[current], boxes[idx]) <= NMS_THRESHOLD) {
        rest.push(idx);
      }
    }
    order.splice(0, order.length, ...rest);
    if (keep.length >= KEEP_TOP_K) break;
  }
  return keep;
};

const pickOutputs = (
  outputMap: Record<string, ort.Tensor>,
  numPriors: number
): {
  loc: Float32Array;
  conf: Float32Array;
  landms: Float32Array;
} => {
  const outputs = Object.values(outputMap).map((tensor) => tensor.data as Float32Array);
  if (outputs.length < 3) {
    throw new Error('RetinaFace outputs missing');
  }

  const expectedLoc = numPriors * 4;
  const expectedConf = numPriors * 2;
  const expectedLandms = numPriors * 10;

  let loc: Float32Array | undefined;
  let conf: Float32Array | undefined;
  let landms: Float32Array | undefined;

  outputs.forEach((data) => {
    if (data.length === expectedLoc) loc = data;
    if (data.length === expectedConf) conf = data;
    if (data.length === expectedLandms) landms = data;
  });

  if (!loc || !conf || !landms) {
    const fallback = outputs.slice(0, 3);
    loc = loc ?? fallback[0];
    conf = conf ?? fallback[1];
    landms = landms ?? fallback[2];
  }

  if (!loc || !conf || !landms) {
    throw new Error('RetinaFace outputs missing');
  }

  if (!debugLogged) {
    debugLogged = true;
    console.log('[RetinaFace] Output lengths:', {
      loc: loc.length,
      conf: conf.length,
      landms: landms.length,
      expectedLoc: numPriors * 4,
      expectedConf: numPriors * 2,
      expectedLandms: numPriors * 10,
    });
  }

  return { loc, conf, landms };
};

export const initRetinaFace = async (): Promise<void> => {
  if (!retinafaceSession) {
    if (!existsSync(RETINAFACE_MODEL_PATH)) {
      throw new Error(`RetinaFace model not found at: ${RETINAFACE_MODEL_PATH}`);
    }
    retinafaceSession = await ort.InferenceSession.create(RETINAFACE_MODEL_PATH, {
      executionProviders: ['cpu'],
    });
  }
};

export const runRetinaFace = async (
  payload: RetinaFacePayload
): Promise<RetinaFaceDetection | null> => {
  await initRetinaFace();
  if (!retinafaceSession) {
    throw new Error('RetinaFace session not available');
  }
  const input = toFloat32Array(payload.input);
  const inputWidth = payload.inputWidth;
  const inputHeight = payload.inputHeight;
  const tensor = new ort.Tensor('float32', input, [1, inputHeight, inputWidth, 3]);
  const outputMap = await retinafaceSession.run({ [retinafaceSession.inputNames[0]]: tensor });

  const priors = generatePriors(inputWidth, inputHeight);
  const numPriors = priors.length / 4;
  const { loc, conf, landms } = pickOutputs(outputMap, numPriors);
  if (loc.length !== numPriors * 4 || landms.length !== numPriors * 10) {
    throw new Error('RetinaFace output size mismatch');
  }

  const boxes = decodeBoxes(loc, priors);
  const landmarks = decodeLandmarks(landms, priors);

  const scores: number[] = [];
  const boxList: number[][] = [];
  const landmarkList: number[][][] = [];
  const confStride = Math.max(1, Math.floor(conf.length / numPriors));
  let maxScore = -Infinity;
  let maxIndex = -1;

  for (let i = 0; i < numPriors; i += 1) {
    const score = confStride > 1 ? conf[i * confStride + 1] : conf[i];
    if (score > maxScore) {
      maxScore = score;
      maxIndex = i;
    }
    if (score < CONF_THRESHOLD) continue;
    const boxIdx = i * 4;
    const xMin = boxes[boxIdx];
    const yMin = boxes[boxIdx + 1];
    const xMax = boxes[boxIdx + 2];
    const yMax = boxes[boxIdx + 3];
    boxList.push([xMin, yMin, xMax, yMax]);
    scores.push(score);

    const landmIdx = i * 10;
    const points: number[][] = [];
    for (let k = 0; k < 5; k += 1) {
      points.push([landmarks[landmIdx + k * 2], landmarks[landmIdx + k * 2 + 1]]);
    }
    landmarkList.push(points);
  }

  if (scores.length === 0) {
    if (!debugLogged) {
      debugLogged = true;
      console.log('[RetinaFace] No detections. Max score:', maxScore);
    }
    if (maxIndex < 0 || maxScore < 0.1) {
      return null;
    }
    const boxIdx = maxIndex * 4;
    const xMin = boxes[boxIdx];
    const yMin = boxes[boxIdx + 1];
    const xMax = boxes[boxIdx + 2];
    const yMax = boxes[boxIdx + 3];
    const landmIdx = maxIndex * 10;
    const bestLandmarks: number[][] = [];
    for (let k = 0; k < 5; k += 1) {
      bestLandmarks.push([landmarks[landmIdx + k * 2], landmarks[landmIdx + k * 2 + 1]]);
    }
    return {
      score: maxScore,
      box: {
        xMin: Math.max(0, xMin * inputWidth * payload.scaleX),
        yMin: Math.max(0, yMin * inputHeight * payload.scaleY),
        width: Math.max(1, (xMax - xMin) * inputWidth * payload.scaleX),
        height: Math.max(1, (yMax - yMin) * inputHeight * payload.scaleY),
      },
      landmarks: bestLandmarks.map(([x, y]) => [
        x * inputWidth * payload.scaleX,
        y * inputHeight * payload.scaleY,
      ]),
    };
  }
  const keep = nms(boxList, scores);
  if (keep.length === 0) return null;

  const bestIdx = keep[0];
  const bestBox = boxList[bestIdx];
  const bestScore = scores[bestIdx];
  const bestLandmarks = landmarkList[bestIdx];

  const xMin = Math.max(0, bestBox[0] * inputWidth * payload.scaleX);
  const yMin = Math.max(0, bestBox[1] * inputHeight * payload.scaleY);
  const xMax = Math.max(0, bestBox[2] * inputWidth * payload.scaleX);
  const yMax = Math.max(0, bestBox[3] * inputHeight * payload.scaleY);

  return {
    score: bestScore,
    box: {
      xMin,
      yMin,
      width: Math.max(1, xMax - xMin),
      height: Math.max(1, yMax - yMin),
    },
    landmarks: bestLandmarks.map(([x, y]) => [
      x * inputWidth * payload.scaleX,
      y * inputHeight * payload.scaleY,
    ]),
  };
};
