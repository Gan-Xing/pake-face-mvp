"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { loadFaceStore, listEmbeddings, deleteEmbedding, renameEmbedding } from "../../lib/face/storage";
import { cosineSimilarity } from "../../lib/face/similarity";
import { DEFAULT_THRESHOLD } from "./utils";
import { UserList } from "./components/UserList";
import { ControlPanel } from "./components/ControlPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { AttendanceLog, LogEntry } from "./components/AttendanceLog";
import { RenameModal } from "./components/RenameModal";
import { ImageModal } from "./components/ImageModal";
import { useCamera } from "./hooks/useCamera";
import { useFaceDetection } from "./hooks/useFaceDetection";
import { useLiveness } from "./hooks/useLiveness";
import styles from "./demo2.module.css";

function FaceDemoV2() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState("初始化中...");
  const [isReady, setReady] = useState(false);
  const [faces, setFaces] = useState<any[]>([]);
  const [checkResult, setCheckResult] = useState<string>("");
  const [topMatches, setTopMatches] = useState<Array<{ name: string; score: number; photo?: string }>>([]);
  const [attendanceLogs, setAttendanceLogs] = useState<LogEntry[]>([]);
  const lastCheckInMapRef = useRef<Map<string, number>>(new Map());
  
  // Debug Logs
  const [logs, setLogs] = useState<string[]>([]);
  const addLog = useCallback((msg: string) => {
    console.log(msg);
    setLogs(prev => [`[${new Date().toISOString().slice(11, 23)}] ${msg}`, ...prev].slice(0, 50));
  }, []);

  // Liveness State
  const [isLivenessEnabled, setLivenessEnabled] = useState(true);
  const { isAlive, updateLiveness, checkLiveness, resetLiveness } = useLiveness(isLivenessEnabled);

  // Modal States
  const [renamingUser, setRenamingUser] = useState<any>(null);
  const [viewImage, setViewImage] = useState<string | null>(null);

  // Hooks
  const { devices, selectedDeviceId, setSelectedDeviceId, startCamera, isStreamActive } = useCamera({ videoRef, addLog, setStatus });
  const { detectFace, getEmbedding, drawResults, supportsLiveness } = useFaceDetection({ videoRef, canvasRef, addLog, setStatus, setReady });

  useEffect(() => {
    if (!supportsLiveness) {
      setLivenessEnabled(false);
    }
  }, [supportsLiveness]);
  
  // Auto Mode State
  const [autoMode, setAutoMode] = useState(false);
  const lastCheckTimeRef = useRef<number>(0);
  
  // Calibration State
  const [similarityThreshold, setSimilarityThreshold] = useState(DEFAULT_THRESHOLD);
  const [isCalibrating, setCalibrating] = useState(false);
  const [calibrationTarget, setCalibrationTarget] = useState("");

  // Init Data
  useEffect(() => {
    const saved = localStorage.getItem("demo2_threshold");
    if (saved) setSimilarityThreshold(parseFloat(saved));

    const loadFaces = async () => {
      try {
        await loadFaceStore();
        const list = await listEmbeddings();
        addLog(`[Demo2] Loaded faces: ${list.length}`);
        setFaces(list);
      } catch (err: any) {
        addLog(`[Demo2] Load Faces Error: ${err.message}`);
      }
    };
    loadFaces();
  }, [addLog]);

  const handleRenameConfirm = async (newName: string) => {
      if (!renamingUser) return;
      const oldName = renamingUser.name;
      try {
          await renameEmbedding(oldName, newName);
          const list = await listEmbeddings();
          setFaces(list);
          addLog(`Renamed ${oldName} to ${newName}`);
          setRenamingUser(null);
      } catch (e: any) {
          alert("改名失败: " + e.message);
      }
  };

  const handleCheck = useCallback(async () => {
    // Liveness Check
    if (isLivenessEnabled && !checkLiveness()) {
        setStatus("⚠️ 请眨眼证明是真人");
        return;
    }

    setStatus("正在识别...");
    setCheckResult("");
    setTopMatches([]);
    
    if (faces.length === 0) {
        setStatus("无注册人脸");
        addLog("Check failed: No registered faces loaded");
        return;
    }

    const emb = await getEmbedding();
    if (!emb) {
      setStatus("未检测到人脸");
      return;
    }

    const scored = faces.map(face => ({
      ...face,
      score: cosineSimilarity(emb.embedding, face.embedding)
    }));

    scored.sort((a, b) => b.score - a.score);

    const top2 = scored.slice(0, 2).map(f => ({
      name: f.name, 
      score: f.score,
      photo: f.photoDataUrl 
    }));
    setTopMatches(top2);
    const bestMatch = top2[0];

    if (bestMatch && bestMatch.score > similarityThreshold) {
      setCheckResult(`识别成功: ${bestMatch.name} (${bestMatch.score.toFixed(2)})`);
      addLog(`Identify Success: ${bestMatch.name} (${bestMatch.score.toFixed(2)})`);
      
      // Reset Liveness immediately after successful check
      if (isLivenessEnabled) {
          resetLiveness();
          setStatus(`打卡成功：${bestMatch.name}。请下一位眨眼`);
      }

      // Log Attendance (with cooldown)
      const now = Date.now();
      const lastTime = lastCheckInMapRef.current.get(bestMatch.name) || 0;
      // 30 seconds cooldown per person to avoid spam
      if (now - lastTime > 30000) {
          lastCheckInMapRef.current.set(bestMatch.name, now);
          const newLog: LogEntry = {
              id: now.toString() + Math.random(),
              name: bestMatch.name,
              score: bestMatch.score,
              time: now,
              photo: emb.photo
          };
          setAttendanceLogs(prev => [newLog, ...prev]);
      }

    } else {
      const bestScore = bestMatch ? bestMatch.score.toFixed(2) : "0.00";
      setCheckResult(`未知人员 (最高相似度: ${bestScore})`);
      addLog(`Identify Failed: Unknown (Max: ${bestScore})`);
    }
    setStatus(`阈值: ${similarityThreshold.toFixed(2)}`);
  }, [faces, similarityThreshold, addLog, getEmbedding, isLivenessEnabled, checkLiveness, resetLiveness]);

  // Main Detection Loop
  useEffect(() => {
    if (!isReady) return;
    let cancelled = false;

    const tick = async () => {
      if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) return;
      
      try {
        const results = await detectFace();
        if (cancelled) return;
        
        // Update Liveness
        updateLiveness(results?.ear ?? null);
        
        const hasFace = drawResults(results);
        
        if (autoMode && hasFace) {
            const now = Date.now();
            if (now - lastCheckTimeRef.current > 3000) { 
                // Only trigger if alive (if enabled)
                if (isLivenessEnabled && !checkLiveness()) {
                    setStatus("请眨眼...");
                    return;
                }

                lastCheckTimeRef.current = now;
                addLog("Auto-check triggered");
                handleCheck().catch(console.error);
            }
        }
      } catch (e: any) {
         addLog(`Detection loop error: ${e.message}`);
      }
    };

    const interval = setInterval(() => {
        void tick();
    }, 100); // 10 FPS (100ms) - Balanced for power/performance

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [isReady, autoMode, detectFace, drawResults, handleCheck, addLog, updateLiveness, isLivenessEnabled, checkLiveness]);

  const handleCalibration = async () => {
     if (!calibrationTarget) {
      alert("请先选择当前在摄像头前的人是谁");
      return;
    }
    
    setStatus("开始校准，请保持姿势...");
    setCalibrating(true);
    addLog(`Starting calibration for ${calibrationTarget}`);
    
    try {
      const samples: number[][] = [];
      for (let i = 0; i < 5; i++) {
        setStatus(`采集样本 ${i + 1}/5...`);
        addLog(`Calibration sample ${i + 1}/5`);
        const emb = await getEmbedding();
        if (emb) samples.push(emb.embedding);
        await new Promise(r => setTimeout(r, 300));
      }

      if (samples.length < 3) {
        alert("采集有效样本过少，校准失败");
        setCalibrating(false);
        addLog("Calibration failed: too few samples");
        return;
      }

      const targetFace = faces.find(f => f.name === calibrationTarget);
      if (!targetFace) {
        alert("找不到目标用户数据");
        setCalibrating(false);
        return;
      }

      let minSelfScore = 1.0;
      for (const sample of samples) {
        const score = cosineSimilarity(sample, targetFace.embedding);
        if (score < minSelfScore) minSelfScore = score;
      }

      let maxOtherScore = 0.0;
      const otherFaces = faces.filter(f => f.name !== calibrationTarget);
      
      if (otherFaces.length > 0) {
        for (const sample of samples) {
          for (const other of otherFaces) {
            const score = cosineSimilarity(sample, other.embedding);
            if (score > maxOtherScore) maxOtherScore = score;
          }
        }
      } else {
        maxOtherScore = minSelfScore - 0.2; 
      }

      const gap = minSelfScore - maxOtherScore;
      let recommended = 0.6;

      if (gap > 0.3) {
        recommended = minSelfScore - 0.15;
      } else {
        recommended = (maxOtherScore + minSelfScore) / 2;
      }
      
      if (recommended > 0.95) recommended = 0.95;
      if (recommended < 0.35) recommended = 0.35;

      setSimilarityThreshold(recommended);
      localStorage.setItem("demo2_threshold", recommended.toFixed(3));
      
      const msg = `校准完成！\n本人最低相似度: ${minSelfScore.toFixed(2)}\n他人最高相似度: ${maxOtherScore.toFixed(2)}\n\n新阈值已设为: ${recommended.toFixed(3)}`;
      alert(msg);
      addLog(msg.replace(/\n/g, ' '));
      setCalibrating(false);
      setCalibrationTarget(""); 
      
    } catch (e: any) {
      alert("校准出错");
      addLog(`Calibration error: ${e.message}`);
      setCalibrating(false);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`确定要删除用户 "${name}" 吗？`)) return;
    try {
      await deleteEmbedding(name);
      const list = await listEmbeddings();
      setFaces(list);
      addLog(`Deleted user: ${name}`);
    } catch (e: any) {
      alert("删除失败");
      addLog(`Delete error: ${e.message}`);
    }
  };

  const handleDeleteLog = (id: string) => {
      setAttendanceLogs(prev => prev.filter(log => log.id !== id));
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
           <h1 className={styles.title}>Face Demo V2</h1>
           <button className={`${styles.button} ${styles.btnSecondary}`} onClick={() => router.push('/demo2/register')}> 
             + 注册新用户
           </button>
      </div>

      <div className={`${styles.row} ${styles.rowStart}`}> 
          <div className={styles.leftColumn}>
             <div className={styles.videoWrapper}>
                {!isStreamActive && (
                    <div className={styles.placeholder}>
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                            <circle cx="12" cy="13" r="4"></circle>
                        </svg>
                        <span>点击 "启动摄像头" 开始</span>
                    </div>
                )}
                <video ref={videoRef} playsInline muted className={styles.video} />
                <canvas ref={canvasRef} className={styles.canvas} />
             </div>
             
             <ControlPanel 
                devices={devices}
                selectedDeviceId={selectedDeviceId}
                setSelectedDeviceId={setSelectedDeviceId}
                startCamera={startCamera}
                isReady={isReady}
                autoMode={autoMode}
                setAutoMode={setAutoMode}
                handleCheck={handleCheck}
                checkResult={checkResult}
                topMatches={topMatches}
                isLivenessEnabled={isLivenessEnabled}
                setLivenessEnabled={setLivenessEnabled}
                isAlive={isAlive}
                supportsLiveness={supportsLiveness}
             />
          </div>

          <div className={styles.rightColumn}>
              <UserList 
                faces={faces} 
                onDelete={handleDelete} 
                onEdit={setRenamingUser} 
              />
              
              <SettingsPanel 
                 similarityThreshold={similarityThreshold}
                 faces={faces}
                 calibrationTarget={calibrationTarget}
                 setCalibrationTarget={setCalibrationTarget}
                 handleCalibration={handleCalibration}
                 isCalibrating={isCalibrating}
                 logs={logs}
              />
          </div>
      </div>
      
      <AttendanceLog 
         logs={attendanceLogs} 
         onClear={() => setAttendanceLogs([])} 
         onViewImage={setViewImage} 
         onDeleteLog={handleDeleteLog}
      />

      <RenameModal 
        isOpen={!!renamingUser}
        initialName={renamingUser?.name || ''}
        onClose={() => setRenamingUser(null)}
        onConfirm={handleRenameConfirm}
      />
      
      <ImageModal 
        isOpen={!!viewImage}
        src={viewImage || ''}
        onClose={() => setViewImage(null)}
      />
    </div>
  );
}

// Export dynamic component (No SSR)
export default dynamic(() => Promise.resolve(FaceDemoV2), { ssr: false });
