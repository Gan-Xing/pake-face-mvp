import { useEffect, useRef, RefObject } from 'react';
import {
  alignFaceToTemplate,
  clampBox,
  l2Normalize,
  ARC_FACE_INPUT_SIZE,
  FaceBox,
  getArcFaceLandmarksFromMesh,
  calculateEyeAspectRatio,
} from "../../../lib/face/utils";
import { FACE_DETECTION_MODEL_URL, FACE_MESH_MODEL_URL, buildInputFromCanvas } from "../utils";

interface UseFaceDetectionProps {
    videoRef: RefObject<HTMLVideoElement>;
    canvasRef: RefObject<HTMLCanvasElement>;
    addLog: (msg: string) => void;
    setStatus: (msg: string) => void;
    setReady: (ready: boolean) => void;
}

export function useFaceDetection({ videoRef, canvasRef, addLog, setStatus, setReady }: UseFaceDetectionProps) {
    const detectionRef = useRef<any | null>(null);
    const meshRef = useRef<any | null>(null);
    const detectionResolveRef = useRef<((results: any) => void) | null>(null);

    useEffect(() => {
        let mounted = true;
        const init = async () => {
            // Require inside useEffect to avoid SSR issues
            const { FaceDetection } = require("@mediapipe/face_detection");
            const { FaceMesh } = require("@mediapipe/face_mesh");

            addLog("[Demo2] Loading models...");
            try {
                // Keep FaceDetection for consistency if needed, but we switch to FaceMesh for main loop
                const detector = new FaceDetection({
                    locateFile: (file: string) => `${FACE_DETECTION_MODEL_URL}/${file}`,
                });
                detector.setOptions({ model: "short", minDetectionConfidence: 0.5 });
                detectionRef.current = detector;

                addLog("[Demo2] Loading FaceMesh model...");
                const mesh = new FaceMesh({
                    locateFile: (file: string) => `${FACE_MESH_MODEL_URL}/${file}`,
                });
                mesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
                mesh.onResults((results: any) => {
                    if (detectionResolveRef.current) {
                        detectionResolveRef.current(results);
                    }
                });
                meshRef.current = mesh;

                if (window.faceNative) {
                    addLog("[Demo2] Init FaceNative...");
                    await window.faceNative.init();
                }
                
                if(mounted) {
                     setReady(true);
                     setStatus("就绪");
                     addLog("[Demo2] Ready!");
                }
            } catch (err: any) {
                addLog(`[Demo2] Init Error: ${err.message}`);
                if(mounted) setStatus("初始化失败");
            }
        };
        init();
        return () => { mounted = false; };
    }, [addLog, setReady, setStatus]);

    const detectFace = async () => {
        if (!meshRef.current || !videoRef.current) return null;
        
        const video = videoRef.current;
        if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0 || video.paused) {
            return null;
        }

        return new Promise<any>((resolve) => {
            // Overwrite the resolver each frame
            detectionResolveRef.current = (results: any) => {
                if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                    resolve(null);
                    return;
                }
                const landmarks = results.multiFaceLandmarks[0];
                const w = video.videoWidth;
                const h = video.videoHeight;

                // Calc Box
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

                const ear = calculateEyeAspectRatio(landmarks);

                resolve({ box, landmarks, ear, score: 0.95 });
            };

            try {
                meshRef.current.send({ image: video });
            } catch (e) {
                console.error("Detection Send Error:", e);
                resolve(null);
            }
        });
    };

    const getEmbedding = async (): Promise<{ embedding: number[], photo: string } | null> => {
        if (!videoRef.current || !window.faceNative || !meshRef.current) return null;
        
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        ctx.drawImage(videoRef.current, 0, 0);
        
        const photo = canvas.toDataURL("image/jpeg", 0.8);

        return new Promise((resolve) => {
            if (!meshRef.current) {
                 resolve(null);
                 return;
            }
            // ... existing resolver logic ...
            const prevResolver = detectionResolveRef.current;
            detectionResolveRef.current = async (results: any) => {
                if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                    resolve(null);
                    return;
                }
                const landmarks = results.multiFaceLandmarks[0];
                const w = canvas.width;
                const h = canvas.height;
                
                // Box
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

                // Align
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

                const input = buildInputFromCanvas(alignedCanvas);
                const output = await window.faceNative!.runArcFace(input);
                const floatData = new Float32Array(output.data);
                resolve({ embedding: l2Normalize(floatData), photo });
            };
            
            meshRef.current.send({ image: canvas });
        });
    };

    const drawResults = (results: any) => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return false;

        const ctx = canvas.getContext("2d");
        if (!ctx) return false;

        if (video.videoWidth > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (results && results.box) {
          const { xMin, yMin, width, height } = results.box;

          ctx.save();
          ctx.beginPath();
          ctx.strokeStyle = "rgba(47, 111, 237, 0.9)";
          ctx.lineWidth = 5;
          ctx.rect(xMin, yMin, width, height);
          ctx.stroke();
          ctx.restore();
          
          return true;
        }
        return false;
    };

    return { detectFace, getEmbedding, drawResults };
}
