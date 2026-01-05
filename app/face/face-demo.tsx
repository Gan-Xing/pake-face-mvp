"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as blazeface from "@tensorflow-models/blazeface";
import * as mobilenet from "@tensorflow-models/mobilenet";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";
import { loadFaceStore, saveEmbedding, listEmbeddings } from "../../lib/face/storage";
import { cosineSimilarity } from "../../lib/face/similarity";

const EMBEDDING_MODEL_URL = "/models/mobilenet/model.json";
const FACE_DETECTOR_MODEL_URL = "/models/blazeface/model.json";
const EMBEDDING_SIZE = 1024;
const SIMILARITY_THRESHOLD = 0.45;

type StoredFace = {
  name: string;
  embedding: number[];
  createdAt: number;
};

type FaceBox = {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
};

export default function FaceDemo() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<blazeface.BlazeFaceModel | null>(null);
  const embedderRef = useRef<mobilenet.MobileNet | null>(null);

  const [status, setStatus] = useState("等待启动摄像头…");
  const [name, setName] = useState("");
  const [faces, setFaces] = useState<StoredFace[]>([]);
  const [ready, setReady] = useState(false);
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");

  const updateStoredFaces = useCallback(async () => {
    const list = await listEmbeddings();
    setFaces(list);
  }, []);

  useEffect(() => {
    void updateStoredFaces();
  }, [updateStoredFaces]);

  const refreshDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter((device) => device.kind === "videoinput");
      setVideoDevices(inputs);
      if (!selectedDeviceId && inputs[0]) {
        setSelectedDeviceId(inputs[0].deviceId);
      }
    } catch {
      setStatus("无法读取摄像头设备列表");
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    void refreshDevices();
  }, [refreshDevices]);

  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    setStatus("启动摄像头…");
    const constraints: MediaStreamConstraints = {
      video: selectedDeviceId
        ? { deviceId: { exact: selectedDeviceId }, width: 640, height: 480 }
        : { facingMode: "user", width: 640, height: 480 },
      audio: false,
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    await refreshDevices();
    setStatus("摄像头已启动");
  }, [refreshDevices, selectedDeviceId]);

  const initModels = useCallback(async () => {
    setStatus("初始化模型中…");
    await tf.setBackend("webgl");
    await tf.ready();

    detectorRef.current = await blazeface.load({
      maxFaces: 1,
      modelUrl: FACE_DETECTOR_MODEL_URL,
    });

    try {
      embedderRef.current = await mobilenet.load({
        version: 2,
        alpha: 1.0,
        modelUrl: EMBEDDING_MODEL_URL,
      });
    } catch {
      embedderRef.current = await mobilenet.load({
        version: 2,
        alpha: 1.0,
      });
    }
    setReady(true);
    setStatus("模型就绪，可以注册 / 打卡");
  }, []);

  useEffect(() => {
    void initModels();
    loadFaceStore();
  }, [initModels]);

  const drawBox = useCallback((box?: FaceBox) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!box) return;
    ctx.strokeStyle = "rgba(47, 111, 237, 0.9)";
    ctx.lineWidth = 3;
    ctx.strokeRect(box.xMin, box.yMin, box.width, box.height);
  }, []);

  const getEmbeddingFromVideo = useCallback(async (): Promise<number[]> => {
    const detector = detectorRef.current;
    const embedder = embedderRef.current;
    const video = videoRef.current;
    if (!detector || !embedder || !video) {
      throw new Error("模型未准备好");
    }

    const facesDetected = await detector.estimateFaces(video, false);
    const face = facesDetected[0];
    if (!face) {
      drawBox();
      throw new Error("未检测到人脸");
    }
    const [xMin, yMin] = face.topLeft as [number, number];
    const [xMax, yMax] = face.bottomRight as [number, number];
    const box = { xMin, yMin, width: xMax - xMin, height: yMax - yMin };
    drawBox(box);

    const input = tf.tidy(() => {
      const frame = tf.browser.fromPixels(video).toFloat();
      const [height, width] = frame.shape.slice(0, 2);

      const y1 = Math.max(box.yMin, 0);
      const x1 = Math.max(box.xMin, 0);
      const y2 = Math.min(box.yMin + box.height, height);
      const x2 = Math.min(box.xMin + box.width, width);

      const boxes = [[y1 / height, x1 / width, y2 / height, x2 / width]];
      const batched = frame.expandDims(0);
      const crop = tf.image.cropAndResize(batched, boxes, [0], [224, 224]);
      return crop.squeeze();
    });

    const output = embedder.infer(input, true) as tf.Tensor;
    const data = await output.data();
    tf.dispose([input, output]);

    if (data.length < EMBEDDING_SIZE) {
      throw new Error("Embedding 维度不正确");
    }
    return Array.from(data).slice(0, EMBEDDING_SIZE);
  }, [drawBox]);

  const handleRegister = useCallback(async () => {
    if (!ready) return;
    if (!name.trim()) {
      setStatus("请输入名字再注册");
      return;
    }
    setStatus("捕获人脸并生成 embedding…");
    try {
      const embedding = await getEmbeddingFromVideo();
      await saveEmbedding({ name: name.trim(), embedding });
      await updateStoredFaces();
      setStatus(`已注册：${name.trim()}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "注册失败");
    }
  }, [getEmbeddingFromVideo, name, ready, updateStoredFaces]);

  const handleCheckIn = useCallback(async () => {
    if (!ready) return;
    setStatus("识别中…");
    try {
      const embedding = await getEmbeddingFromVideo();
      const stored = await listEmbeddings();
      let best: { name: string; score: number } | null = null;

      stored.forEach((item) => {
        const score = cosineSimilarity(embedding, item.embedding);
        if (!best || score > best.score) {
          best = { name: item.name, score };
        }
      });

      if (!best || best.score < SIMILARITY_THRESHOLD) {
        setStatus("未匹配到已注册的人脸");
        return;
      }

      setStatus(`识别到：${best.name} (相似度 ${best.score.toFixed(3)})`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "识别失败");
    }
  }, [getEmbeddingFromVideo, ready]);

  return (
    <div className="grid">
      <div className="video-wrap">
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} />
      </div>
      <div className="controls">
        <div className="row">
          <button onClick={startCamera}>启动摄像头</button>
          <button className="secondary" onClick={handleCheckIn} disabled={!ready}>
            Check in
          </button>
          <button className="ghost" onClick={handleRegister} disabled={!ready}>
            Register
          </button>
        </div>
        <div className="row">
          <label>
            摄像头：
            <select
              value={selectedDeviceId}
              onChange={(event) => setSelectedDeviceId(event.target.value)}
            >
              {videoDevices.length === 0 ? (
                <option value="">未检测到摄像头</option>
              ) : (
                videoDevices.map((device, index) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Camera ${index + 1}`}
                  </option>
                ))
              )}
            </select>
          </label>
        </div>
        <input
          placeholder="输入名字，例如 Alice"
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
        <div className="status">{status}</div>
        <div className="list">
          <strong>已注册</strong>
          {faces.length === 0 ? (
            <span className="subtitle">暂无数据</span>
          ) : (
            faces.map((item) => (
              <span key={item.name} className="tag">
                {item.name}
              </span>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
