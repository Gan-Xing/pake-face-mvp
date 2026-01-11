import * as ort from 'onnxruntime-node';
import { app } from 'electron';
import { join } from 'path';
import { readFile } from 'fs/promises';

type RawTensor = {
  data: ArrayBuffer;
  dims: number[];
};

type ArcFacePayload = {
  input: ArrayBuffer | Float32Array | number[];
};

const ARC_FACE_MODEL_PATH = join(
  app.getAppPath(),
  'app',
  'models',
  'arcface',
  'arcfaceresnet100-8.onnx'
);

let arcfaceSession: ort.InferenceSession | null = null;

const toFloat32Array = (input: ArrayBuffer | Float32Array | number[]): Float32Array => {
  if (input instanceof Float32Array) return input;
  if (ArrayBuffer.isView(input)) {
    return new Float32Array(input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength));
  }
  if (input instanceof ArrayBuffer) return new Float32Array(input);
  return Float32Array.from(input);
};

const serializeTensor = (tensor: ort.Tensor): RawTensor => {
  const data = tensor.data as Float32Array;
  const copied = data.slice().buffer;
  return { data: copied, dims: [...tensor.dims] };
};

export const initNativeFace = async (): Promise<void> => {
  if (!arcfaceSession) {
    arcfaceSession = await ort.InferenceSession.create(ARC_FACE_MODEL_PATH, {
      executionProviders: ['cpu'],
    });
  }
};

export const runArcFace = async (payload: ArcFacePayload): Promise<RawTensor> => {
  await initNativeFace();
  if (!arcfaceSession) {
    throw new Error('ArcFace session not available');
  }
  const input = toFloat32Array(payload.input);
  const tensor = new ort.Tensor('float32', input, [1, 3, 112, 112]);
  const outputMap = await arcfaceSession.run({ [arcfaceSession.inputNames[0]]: tensor });
  const outputName = arcfaceSession.outputNames[0];
  const output = outputMap[outputName];
  if (!output) {
    throw new Error('ArcFace output missing');
  }
  return serializeTensor(output);
};

const extractAsciiStrings = (buffer: Buffer, minLength = 3): string[] => {
  const results: string[] = [];
  let current = '';
  for (let i = 0; i < buffer.length; i += 1) {
    const byte = buffer[i];
    if (byte >= 32 && byte <= 126) {
      current += String.fromCharCode(byte);
    } else if (current.length >= minLength) {
      results.push(current);
      current = '';
    } else {
      current = '';
    }
  }
  if (current.length >= minLength) {
    results.push(current);
  }
  return results;
};

export const detectArcFacePreprocess = async (): Promise<'rgb_0_255' | 'rgb_-1_1' | null> => {
  try {
    const data = await readFile(ARC_FACE_MODEL_PATH);
    const strings = extractAsciiStrings(data);
    const hasSub = strings.some((value) => value === 'Sub');
    const hasMul = strings.some((value) => value === 'Mul');
    if (hasSub && hasMul) {
      return 'rgb_0_255';
    }
    return 'rgb_-1_1';
  } catch {
    return null;
  }
};
