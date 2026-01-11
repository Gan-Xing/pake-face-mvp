export const ARC_FACE_INPUT_SIZE = 112;

export const ARC_FACE_LANDMARK_TEMPLATE: Array<[number, number]> = [
  [38.2946, 51.6963], // left eye
  [73.5318, 51.5014], // right eye
  [56.0252, 71.7366], // nose
  [41.5493, 92.3655], // left mouth
  [70.7299, 92.2041], // right mouth
];

export const ARC_FACE_EYE_TEMPLATE = {
  right: [38.2946, 51.6963],
  left: [73.5318, 51.5014],
};

export type FaceBox = {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
};

export const clampBox = (box: FaceBox, width: number, height: number): FaceBox => {
  const xMin = Math.max(0, Math.min(width - 1, box.xMin));
  const yMin = Math.max(0, Math.min(height - 1, box.yMin));
  const xMax = Math.max(xMin + 1, Math.min(width, box.xMin + box.width));
  const yMax = Math.max(yMin + 1, Math.min(height, box.yMin + box.height));
  return {
    xMin,
    yMin,
    width: Math.max(1, xMax - xMin),
    height: Math.max(1, yMax - yMin),
  };
};

export const estimateSimilarityTransform = (
  src: Array<[number, number]>,
  dst: Array<[number, number]>
): Array<[number, number, number]> | null => {
  if (src.length !== dst.length || src.length === 0) return null;
  const count = src.length;
  let srcMeanX = 0;
  let srcMeanY = 0;
  let dstMeanX = 0;
  let dstMeanY = 0;

  for (let i = 0; i < count; i += 1) {
    srcMeanX += src[i][0];
    srcMeanY += src[i][1];
    dstMeanX += dst[i][0];
    dstMeanY += dst[i][1];
  }

  srcMeanX /= count;
  srcMeanY /= count;
  dstMeanX /= count;
  dstMeanY /= count;

  let varSrc = 0;
  let a = 0;
  let b = 0;

  for (let i = 0; i < count; i += 1) {
    const sx = src[i][0] - srcMeanX;
    const sy = src[i][1] - srcMeanY;
    const dx = dst[i][0] - dstMeanX;
    const dy = dst[i][1] - dstMeanY;
    varSrc += sx * sx + sy * sy;
    a += sx * dx + sy * dy;
    b += sx * dy - sy * dx;
  }

  if (varSrc === 0) return null;
  const norm = Math.hypot(a, b);
  if (norm === 0) return null;

  const cos = a / norm;
  const sin = b / norm;
  const scale = norm / varSrc;

  const m00 = scale * cos;
  const m01 = -scale * sin;
  const m10 = scale * sin;
  const m11 = scale * cos;
  const t0 = dstMeanX - m00 * srcMeanX - m01 * srcMeanY;
  const t1 = dstMeanY - m10 * srcMeanX - m11 * srcMeanY;

  return [
    [m00, m01, t0],
    [m10, m11, t1],
  ];
};

export const alignFaceToTemplate = (
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  box: FaceBox,
  landmarks?: number[][]
): HTMLCanvasElement => {
  const canvas = document.createElement("canvas");
  canvas.width = ARC_FACE_INPUT_SIZE;
  canvas.height = ARC_FACE_INPUT_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  if (!landmarks || landmarks.length < 2) {
    ctx.drawImage(
      source,
      Math.max(0, box.xMin),
      Math.max(0, box.yMin),
      Math.max(1, Math.min(sourceWidth - box.xMin, box.width)),
      Math.max(1, Math.min(sourceHeight - box.yMin, box.height)),
      0,
      0,
      ARC_FACE_INPUT_SIZE,
      ARC_FACE_INPUT_SIZE
    );
    return canvas;
  }

  if (landmarks.length >= 5) {
    const srcPoints = landmarks.slice(0, 5) as Array<[number, number]>;
    const dstPoints = ARC_FACE_LANDMARK_TEMPLATE;
    const transform = estimateSimilarityTransform(srcPoints, dstPoints);
    if (transform) {
      ctx.setTransform(
        transform[0][0],
        transform[1][0],
        transform[0][1],
        transform[1][1],
        transform[0][2],
        transform[1][2]
      );
      ctx.drawImage(source, 0, 0, sourceWidth, sourceHeight);
      return canvas;
    }
  }

  const rightEye = landmarks[0];
  const leftEye = landmarks[1];
  const dx = leftEye[0] - rightEye[0];
  const dy = leftEye[1] - rightEye[1];
  const angle = Math.atan2(dy, dx);
  const distance = Math.hypot(dx, dy) || 1;
  const desiredDistance = ARC_FACE_EYE_TEMPLATE.left[0] - ARC_FACE_EYE_TEMPLATE.right[0];
  const scale = desiredDistance / distance;

  ctx.translate(ARC_FACE_EYE_TEMPLATE.right[0], ARC_FACE_EYE_TEMPLATE.right[1]);
  ctx.rotate(-angle);
  ctx.scale(scale, scale);
  ctx.translate(-rightEye[0], -rightEye[1]);
  ctx.drawImage(source, 0, 0);
  return canvas;
};

export const l2Normalize = (vector: Float32Array): number[] => {
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) {
    sum += vector[i] * vector[i];
  }
  const norm = Math.sqrt(sum) || 1;
  return Array.from(vector, (value) => value / norm);
};

export const l2NormalizeArray = (vector: number[]): number[] => {
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) {
    sum += vector[i] * vector[i];
  }
  const norm = Math.sqrt(sum) || 1;
  return vector.map((value) => value / norm);
};

export const FACE_MESH_ARCFACE_INDEX = {
  rightEye: [33, 133, 159, 145],
  leftEye: [263, 362, 386, 374],
  nose: 1,
  mouth: [61, 291],
};

export const FACE_MESH_BLINK_INDEX = {
  // p1, p2, p3, p4, p5, p6
  right: [33, 160, 158, 133, 153, 144],
  left: [362, 385, 387, 263, 373, 380],
};

export const calculateEyeAspectRatio = (landmarks: MeshLandmark[]): number | null => {
  if (!landmarks || landmarks.length === 0) return null;

  const dist = (p1: MeshLandmark, p2: MeshLandmark) => 
    Math.hypot(p1.x - p2.x, p1.y - p2.y);

  const getEAR = (indices: number[]) => {
    const p1 = landmarks[indices[0]];
    const p2 = landmarks[indices[1]];
    const p3 = landmarks[indices[2]];
    const p4 = landmarks[indices[3]];
    const p5 = landmarks[indices[4]];
    const p6 = landmarks[indices[5]];
    
    if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return 0;

    const num = dist(p2, p6) + dist(p3, p5);
    const den = 2 * dist(p1, p4);
    return den === 0 ? 0 : num / den;
  };

  const earRight = getEAR(FACE_MESH_BLINK_INDEX.right);
  const earLeft = getEAR(FACE_MESH_BLINK_INDEX.left);

  return (earRight + earLeft) / 2;
};

// Quality Constants
export const QUALITY_MIN_SCORE = 0.7;
export const QUALITY_MIN_WIDTH_RATIO = 0.15;
export const QUALITY_MAX_WIDTH_RATIO = 0.9;
export const QUALITY_MAX_CENTER_OFFSET = 0.45;
export const QUALITY_MIN_BOX_SIZE = 50;

export type MeshLandmark = { x: number; y: number; z?: number };

export type DetectedFace = {
  score: number;
  box: FaceBox;
  landmarks?: number[][];
  eyeAspectRatio?: number;
};

export const getQualityHint = (
  detection: DetectedFace,
  videoWidth: number,
  videoHeight: number
): string | null => {
  const { box, score } = detection;
  
  // 1. Check Score (Clarity/Confidence)
  if (score < QUALITY_MIN_SCORE) {
    return "人脸细节模糊，请正对光线或擦拭镜头";
  }

  // 2. Check Absolute Size (Pixels)
  if (box.width < QUALITY_MIN_BOX_SIZE || box.height < QUALITY_MIN_BOX_SIZE) {
    return "检测到的人脸太小";
  }

  const widthRatio = box.width / videoWidth;
  
  // 3. Check Relative Size (Distance)
  if (widthRatio < QUALITY_MIN_WIDTH_RATIO) {
    return "距离太远，请靠近摄像头";
  }
  if (widthRatio > QUALITY_MAX_WIDTH_RATIO) {
    return "距离太近，请稍微后退";
  }

  // 4. Check Centering
  const centerX = box.xMin + box.width / 2;
  const centerY = box.yMin + box.height / 2;
  const offsetX = Math.abs(centerX / videoWidth - 0.5);
  const offsetY = Math.abs(centerY / videoHeight - 0.5);
  
  if (offsetX > QUALITY_MAX_CENTER_OFFSET || offsetY > QUALITY_MAX_CENTER_OFFSET) {
    return "请将人脸移至屏幕中央区域";
  }

  return null;
};

export const getArcFaceLandmarksFromMesh = (
  landmarks: MeshLandmark[],
  width: number,
  height: number
): number[][] | undefined => {
  const averagePoints = (indices: number[]): [number, number] | null => {
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    indices.forEach((idx) => {
      const point = landmarks[idx];
      if (!point) return;
      sumX += point.x;
      sumY += point.y;
      count += 1;
    });
    if (count === 0) return null;
    return [sumX / count, sumY / count];
  };

  const rightEye = averagePoints(FACE_MESH_ARCFACE_INDEX.rightEye);
  const leftEye = averagePoints(FACE_MESH_ARCFACE_INDEX.leftEye);
  const nose = landmarks[FACE_MESH_ARCFACE_INDEX.nose];
  const mouthPoints = FACE_MESH_ARCFACE_INDEX.mouth
    .map((idx) => landmarks[idx])
    .filter((point): point is MeshLandmark => Boolean(point));

  if (!leftEye || !rightEye || !nose || mouthPoints.length < 2) return undefined;

  const eyePoints: Array<[number, number]> = [
    [leftEye[0] * width, leftEye[1] * height],
    [rightEye[0] * width, rightEye[1] * height],
  ];
  const sortedEyes = eyePoints.sort((a, b) => a[0] - b[0]);

  const mouthCandidates: Array<[number, number]> = mouthPoints.map((point) => [
    point.x * width,
    point.y * height,
  ]);
  const sortedMouth = mouthCandidates.sort((a, b) => a[0] - b[0]);

  return [
    sortedEyes[0], // Left Eye
    sortedEyes[1], // Right Eye
    [nose.x * width, nose.y * height], // Nose
    sortedMouth[0], // Left Mouth
    sortedMouth[1], // Right Mouth
  ];
};
