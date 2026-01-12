"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { saveEmbedding } from "../../../lib/face/storage";
import {
  alignFaceToTemplate,
  clampBox,
  l2Normalize,
  l2NormalizeArray,
  ARC_FACE_INPUT_SIZE,
  FaceBox,
  getArcFaceLandmarksFromMesh,
  getQualityHint,
  DetectedFace,
} from "../../../lib/face/utils";
import styles from "./register.module.css";

// Constants
const FACE_DETECTION_MODEL_URL = "/mediapipe/face_detection";
const FACE_MESH_MODEL_URL = "/mediapipe/face_mesh";

// Helper
const buildInputFromCanvas = (aligned: HTMLCanvasElement): Float32Array => {
  const ctx = aligned.getContext("2d");
  if (!ctx) throw new Error("无法读取人脸图像");
  const imageData = ctx.getImageData(0, 0, ARC_FACE_INPUT_SIZE, ARC_FACE_INPUT_SIZE);
  const { data } = imageData;
  const planeSize = ARC_FACE_INPUT_SIZE * ARC_FACE_INPUT_SIZE;
  const input = new Float32Array(planeSize * 3);

  for (let i = 0; i < planeSize; i += 1) {
    const base = i * 4;
    const r = data[base];
    const g = data[base + 1];
    const b = data[base + 2];
    input[i] = r;
    input[planeSize + i] = g;
    input[planeSize * 2 + i] = b;
  }
  return input;
};

type TempPhoto = {
  id: string;
  url: string;
  embedding: number[];
  timestamp: number;
};



function RegisterPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const detectionRef = useRef<any | null>(null);
  const meshRef = useRef<any | null>(null);
  const detectionResolveRef = useRef<((results: any) => void) | null>(null);

  const [status, setStatus] = useState("初始化中...");
  const [isReady, setReady] = useState(false);
  const [regName, setRegName] = useState("");
  const [qualityHint, setQualityHint] = useState<string | null>(null);
  
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  
  const [tempPhotos, setTempPhotos] = useState<TempPhoto[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // Init
  const stopCamera = () => {
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  useEffect(() => {
    let mounted = true;
    
    navigator.mediaDevices.enumerateDevices().then(devs => {
        if (mounted) {
            const videoInputs = devs.filter(d => d.kind === 'videoinput');
            setDevices(videoInputs);
            if (videoInputs.length > 0) setSelectedDeviceId(videoInputs[0].deviceId);
        }
    });

    const init = async () => {
      try {
        const { FaceDetection } = require("@mediapipe/face_detection");
        const { FaceMesh } = require("@mediapipe/face_mesh");

        console.log("Loading FaceDetection...");
        const detector = new FaceDetection({
          locateFile: (file: string) => `${FACE_DETECTION_MODEL_URL}/${file}`,
        });
        detector.setOptions({ model: "short", minDetectionConfidence: 0.5 });
        detector.onResults((results: any) => {
            if (detectionResolveRef.current) {
                detectionResolveRef.current(results);
                detectionResolveRef.current = null;
            }
        });
        detectionRef.current = detector;

        console.log("Loading FaceMesh...");
        const mesh = new FaceMesh({
          locateFile: (file: string) => `${FACE_MESH_MODEL_URL}/${file}`,
        });
        mesh.setOptions({ maxNumFaces: 1, minDetectionConfidence: 0.5 });
        meshRef.current = mesh;

        if (window.faceNative) {
          console.log("Init FaceNative...");
          await window.faceNative.init();
        }

        if (mounted) {
          setReady(true);
          setStatus("请启动摄像头");
          startCamera(); 
        }
      } catch (err) {
        console.error(err);
        if (mounted) setStatus("初始化失败");
      }
    };
    init();
    return () => {
      mounted = false;
      stopCamera();
    };
  }, []);

  const detectFace = async (video: HTMLVideoElement) => {
      if (!detectionRef.current) return null;
      // Safety check for video
      if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0 || video.paused) {
          return null;
      }
      return new Promise<any>((resolve) => {
          detectionResolveRef.current = resolve;
          detectionRef.current.send({ image: video });
      });
  };

  const drawResults = (results: any) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results && results.detections && results.detections.length > 0) {
      const det = results.detections[0];
      const w = canvas.width;
      const h = canvas.height;
      
      let rectX, rectY, rectW, rectH;

      if (det.boundingBox) {
          const { xCenter, yCenter, width, height } = det.boundingBox;
          const xMin = xCenter - width / 2;
          const yMin = yCenter - height / 2;
          rectX = xMin * w;
          rectY = yMin * h;
          rectW = width * w;
          rectH = height * h;
      } else if (det.locationData && det.locationData.relativeBoundingBox) {
          const box = det.locationData.relativeBoundingBox;
          rectX = box.xMin * w;
          rectY = box.yMin * h;
          rectW = box.width * w;
          rectH = box.height * h;
      } else {
          return;
      }
      // Try to find score in various places
      let score = 0;
      if (det.score && det.score.length > 0) score = det.score[0];
      else if (det.confidence && det.confidence.length > 0) score = det.confidence[0];
      else if (det.V && det.V.length > 0 && typeof det.V[0].ga === 'number') score = det.V[0].ga;
      
      const face: DetectedFace = {
        score: score,
        box: { xMin: rectX, yMin: rectY, width: rectW, height: rectH }
      };

      const hint = getQualityHint(face, w, h);
      
      // Debug metrics
      const widthRatio = face.box.width / w;
      const centerX = face.box.xMin + face.box.width / 2;
      const centerY = face.box.yMin + face.box.height / 2;
      const offsetX = Math.abs(centerX / w - 0.5);
      const offsetY = Math.abs(centerY / h - 0.5);
      
      if (hint) {
          setQualityHint(`${hint} [S:${face.score.toFixed(2)} W:${widthRatio.toFixed(2)} X:${offsetX.toFixed(2)} Y:${offsetY.toFixed(2)}]`);
      } else {
          setQualityHint(null);
      }

      ctx.strokeStyle = hint ? "#fbbf24" : "rgba(47, 111, 237, 0.9)";
      ctx.lineWidth = 3;
      ctx.strokeRect(face.box.xMin, face.box.yMin, face.box.width, face.box.height);
    } else {
      setQualityHint("未检测到人脸");
    }
  };

  const startCamera = async () => {
    if (!videoRef.current) return;
    try {
      stopCamera();
      const constraints = {
        video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("请正对摄像头，尝试不同角度拍摄");
      
      const loop = async () => {
        if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) return;
        const results = await detectFace(videoRef.current);
        drawResults(results);
        rafIdRef.current = requestAnimationFrame(loop);
      };
      loop();
    } catch (err) {
      setStatus("摄像头启动失败");
    }
  };

  useEffect(() => {
      if (isReady) {
          startCamera();
      }
  }, [selectedDeviceId, isReady]);

  const extractEmbedding = async (): Promise<{ emb: number[], photo: string } | null> => {
    if (!videoRef.current || !window.faceNative || !meshRef.current) {
        alert("模型或环境未就绪");
        return null;
    }
    
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(videoRef.current, 0, 0);

    const photoUrl = canvas.toDataURL("image/jpeg", 0.85);

    return new Promise((resolve) => {
        meshRef.current!.onResults(async (results: any) => {
            if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                resolve(null);
                return;
            }
            const landmarks = results.multiFaceLandmarks[0];
            const w = canvas.width;
            const h = canvas.height;
            
            let minX = 1, minY = 1, maxX = 0, maxY = 0;
            landmarks.forEach((p: any) => {
                if(p.x < minX) minX = p.x;
                if(p.y < minY) minY = p.y;
                if(p.x > maxX) maxX = p.x;
                if(p.y > maxY) maxY = p.y;
            });
            const box: FaceBox = {
                xMin: minX * w,
                yMin: minY * h,
                width: (maxX - minX) * w,
                height: (maxY - minY) * h
            };

            const fivePoints = getArcFaceLandmarksFromMesh(landmarks, w, h);
            let alignedCanvas: HTMLCanvasElement;
            if (fivePoints) {
               alignedCanvas = alignFaceToTemplate(canvas, w, h, box, fivePoints);
            } else {
               const safeBox = clampBox(box, w, h);
               alignedCanvas = document.createElement('canvas');
               alignedCanvas.width = ARC_FACE_INPUT_SIZE;
               alignedCanvas.height = ARC_FACE_INPUT_SIZE;
               const aCtx = alignedCanvas.getContext('2d');
               aCtx?.drawImage(canvas, safeBox.xMin, safeBox.yMin, safeBox.width, safeBox.height, 0, 0, ARC_FACE_INPUT_SIZE, ARC_FACE_INPUT_SIZE);
            }

            try {
                const input = buildInputFromCanvas(alignedCanvas);
                const output = await window.faceNative!.runArcFace(input);
                const floatData = new Float32Array(output.data);
                resolve({ emb: l2Normalize(floatData), photo: photoUrl });
            } catch(e) {
                console.error(e);
                resolve(null);
            }
        });
        meshRef.current!.send({ image: canvas });
    });
  };

  const handleCapture = async () => {
    if (qualityHint) {
        if (qualityHint === "未检测到人脸") {
            alert("未检测到人脸，无法抓拍");
            return;
        }
        // Allow user to bypass quality check
        const confirmCapture = confirm(`当前拍摄质量提示：${qualityHint}\n\n是否强制抓拍？(质量差可能影响识别准确率)`);
        if (!confirmCapture) return;
    }
    
    setStatus("正在处理...");
    try {
        const result = await extractEmbedding();
        if (!result) {
          setStatus("未检测到人脸，请重试");
          return;
        }
        
        const newPhoto: TempPhoto = {
          id: Date.now().toString() + Math.random(),
          url: result.photo,
          embedding: result.emb,
          timestamp: Date.now()
        };

        setTempPhotos(prev => [newPhoto, ...prev]);
        setSelectedIds(prev => (prev.length < 10 ? [...prev, newPhoto.id] : prev));
        setStatus("已抓拍，请继续");
    } catch (e) {
        console.error(e);
        setStatus("抓拍出错");
    }
  };

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    if (!meshRef.current || !window.faceNative) {
        alert("模型尚未加载完成，请稍候");
        return;
    }

    setStatus("正在处理上传...");
    let successCount = 0;
    let failCount = 0;
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        try {
            const url = await new Promise<string>((resolve) => {
                const reader = new FileReader();
                reader.onload = () => resolve(String(reader.result));
                reader.readAsDataURL(file);
            });

            const img = new Image();
            img.src = url;
            await new Promise((r) => (img.onload = r));

            await new Promise<void>((resolve) => {
                const onResults = async (results: any) => {
                    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                        console.warn(`Upload skipped: No face in ${file.name}`);
                        failCount++;
                        resolve();
                        return;
                    }
                    const landmarks = results.multiFaceLandmarks[0];
                    const w = img.width;
                    const h = img.height;
                    
                    let minX = 1, minY = 1, maxX = 0, maxY = 0;
                    landmarks.forEach((p: any) => {
                        if(p.x < minX) minX = p.x;
                        if(p.y < minY) minY = p.y;
                        if(p.x > maxX) maxX = p.x;
                        if(p.y > maxY) maxY = p.y;
                    });
                    const box: FaceBox = {
                        xMin: minX * w, yMin: minY * h, width: (maxX - minX) * w, height: (maxY - minY) * h
                    };

                    const fivePoints = getArcFaceLandmarksFromMesh(landmarks, w, h);
                    const canvas = document.createElement('canvas');
                    canvas.width = w;
                    canvas.height = h;
                    const ctx = canvas.getContext('2d');
                    ctx?.drawImage(img, 0, 0);

                    let alignedCanvas: HTMLCanvasElement;
                    if (fivePoints) {
                       alignedCanvas = alignFaceToTemplate(canvas, w, h, box, fivePoints);
                    } else {
                       const safeBox = clampBox(box, w, h);
                       alignedCanvas = document.createElement('canvas');
                       alignedCanvas.width = ARC_FACE_INPUT_SIZE;
                       alignedCanvas.height = ARC_FACE_INPUT_SIZE;
                       const aCtx = alignedCanvas.getContext('2d');
                       aCtx?.drawImage(canvas, safeBox.xMin, safeBox.yMin, safeBox.width, safeBox.height, 0, 0, ARC_FACE_INPUT_SIZE, ARC_FACE_INPUT_SIZE);
                    }

                    const input = buildInputFromCanvas(alignedCanvas);
                    const output = await window.faceNative!.runArcFace(input);
                    const floatData = new Float32Array(output.data);
                    
                    const newPhoto: TempPhoto = {
                      id: Date.now().toString() + Math.random(),
                      url: url,
                      embedding: l2Normalize(floatData),
                      timestamp: Date.now()
                    };
                    
                    setTempPhotos(prev => [newPhoto, ...prev]);
                    setSelectedIds(prev => (prev.length < 10 ? [...prev, newPhoto.id] : prev));
                    successCount++;
                    resolve();
                };

                meshRef.current!.onResults(onResults);
                meshRef.current!.send({ image: img });
            });
        } catch (e) {
            console.error("Upload failed", e);
            failCount++;
        }
    }
    
    setStatus(`上传完成：成功 ${successCount} 张，失败 ${failCount} 张`);
    if (failCount > 0) {
        alert(`${failCount} 张照片因未检测到人脸而被跳过，请尝试更清晰的正脸照片。`);
    }
  };

  const toggleSelect = (id: string) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(prev => prev.filter(x => x !== id));
    } else {
      if (selectedIds.length >= 10) {
        alert("最多选择 10 张");
        return;
      }
      setSelectedIds(prev => [...prev, id]);
    }
  };

  const handleDownload = (photo: TempPhoto) => {
    const a = document.createElement("a");
    a.href = photo.url;
    a.download = `face-${regName || 'capture'}-${photo.timestamp}.jpg`;
    a.click();
  };

  const handleRegister = async () => {
    if (!regName.trim()) {
      alert("请输入姓名");
      return;
    }
    if (selectedIds.length < 3) {
      alert("请至少选择 3 张照片以保证识别准确率");
      return;
    }

    setStatus("正在注册...");
    
    const selectedPhotos = tempPhotos.filter(p => selectedIds.includes(p.id));
    const size = selectedPhotos[0].embedding.length;
    const sum = new Array(size).fill(0);
    
    selectedPhotos.forEach(p => {
        for(let i=0; i<size; i++) sum[i] += p.embedding[i];
    });
    
    const averageEmb = l2NormalizeArray(sum.map(v => v / selectedPhotos.length));
    const avatarUrl = selectedPhotos[0].url;

    try {
      await saveEmbedding({ 
          name: regName.trim(), 
          embedding: averageEmb, 
          photoDataUrl: avatarUrl 
      });
      setStatus(`注册成功: ${regName}`);
      setTimeout(() => router.push("/demo2"), 1500);
    } catch (e) {
      setStatus("保存失败: " + String(e));
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>多角度注册</h1>
        <p className={styles.subtitle}>{status}</p>
        <div style={{ height: 24 }}>
           {qualityHint && <div className={styles.alert}>⚠️ {qualityHint}</div>}
        </div>
      </div>

      <div className={styles.grid}>
        <div className={styles.leftColumn}>
            <div className={styles.videoWrapper}>
                <video ref={videoRef} playsInline muted className={styles.video} />
                <canvas ref={canvasRef} className={styles.canvas} />
            </div>
            
            {devices.length > 0 && (
                <select 
                  className={styles.select}
                  value={selectedDeviceId} 
                  onChange={e => setSelectedDeviceId(e.target.value)}
                >
                  {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.slice(0,5)}`}</option>)}
                </select>
            )}
            
            <div className={styles.inputRow}>
               <input 
                 className={styles.input}
                 type="text" 
                 placeholder="请输入姓名" 
                 value={regName}
                 onChange={e => setRegName(e.target.value)}
               />
            </div>

            <div className={styles.actionButtons}>
                <label className={styles.fileLabel}>
                   📂 上传照片
                   <input type="file" multiple accept="image/*" style={{ display: 'none' }} onChange={(e) => handleUpload(e.target.files)} />
                </label>
                <button onClick={handleCapture} className={`${styles.btn} ${styles.btnPrimary}`}>
                   📸 抓拍一张
                </button>
            </div>
            
            <div className={styles.tips}>
               💡 技巧：请分别拍摄 <strong>正脸、微侧、抬头、低头</strong> 等不同角度，有助于提高识别通过率。建议上传或抓拍至少 3-5 张清晰照片。
            </div>

            <div className={styles.footer}>
               <button 
                 onClick={handleRegister} 
                 className={`${styles.btn} ${styles.btnPrimary}`}
                 disabled={selectedIds.length < 3}
               >
                 确认注册 ({selectedIds.length}/10)
               </button>
               <button onClick={() => router.back()} className={`${styles.btn} ${styles.btnOutline}`}>
                 返回
               </button>
            </div>
        </div>

        <div className={styles.rightColumn}>
            <h4 className={styles.sectionTitle}>
                照片池 
                <span style={{ fontSize: '0.9rem', color: '#6b7280', fontWeight: 'normal' }}>
                    ({selectedIds.length} / {tempPhotos.length})
                </span>
            </h4>
            
            {tempPhotos.length > 0 ? (
                <div className={styles.photoGrid}>
                {tempPhotos.map(p => {
                    const isSelected = selectedIds.includes(p.id);
                    return (
                    <div 
                        key={p.id} 
                        className={`${styles.photoCard} ${isSelected ? styles.photoCardSelected : ''}`}
                        onClick={() => toggleSelect(p.id)}
                    >
                        <img src={p.url} className={styles.photoImg} />
                        {isSelected && <div className={styles.selectedBadge}>✓</div>}
                        <div 
                            className={styles.downloadBtn}
                            onClick={(e) => { e.stopPropagation(); handleDownload(p); }}
                        >
                            下载原图
                        </div>
                    </div>
                    );
                })}
                </div>
            ) : (
                <div className={styles.emptyState}>
                    暂无照片，请点击左侧拍照或上传
                </div>
            )}
        </div>
      </div>
    </div>
  );
}

export default dynamic(() => Promise.resolve(RegisterPage), { ssr: false });
