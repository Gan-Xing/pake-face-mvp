import { useState, useEffect, useCallback, RefObject, useRef } from 'react';

interface UseCameraProps {
    videoRef: RefObject<HTMLVideoElement>;
    addLog: (msg: string) => void;
    setStatus: (status: string) => void;
}

export function useCamera({ videoRef, addLog, setStatus }: UseCameraProps) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isStreamActive, setStreamActive] = useState(false);
  
  // Track if we are currently mounting/unmounting to avoid unsafe state updates
  const isMountedRef = useRef(true);
  const stopStream = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const oldStream = videoRef.current.srcObject as MediaStream;
      oldStream.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
  }, [videoRef]);

  useEffect(() => {
    isMountedRef.current = true;
    navigator.mediaDevices.enumerateDevices().then(devs => {
        if (isMountedRef.current) {
            const videoInputs = devs.filter(d => d.kind === 'videoinput');
            addLog(`Found ${videoInputs.length} video devices`);
            setDevices(videoInputs);
            if (videoInputs.length > 0) setSelectedDeviceId(videoInputs[0].deviceId);
        }
    });
    return () => {
      isMountedRef.current = false;
      stopStream();
    };
  }, [addLog, stopStream]);

  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    
    // 1. Cleanup old stream
    stopStream();

    addLog(`Starting camera...`);
    
    try {
      const constraints = {
        video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
        audio: false,
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (!isMountedRef.current || !videoRef.current) {
          // Component unmounted during async request
          stream.getTracks().forEach(t => t.stop());
          return;
      }

      // 2. Assign stream
      videoRef.current.srcObject = stream;
      
      // 3. CRITICAL: Wait for browser to digest the new source before playing
      // This prevents "The play() request was interrupted by a new load request"
      await new Promise(r => setTimeout(r, 150));

      try {
         await videoRef.current.play();
         
         // 4. Update State ONLY AFTER successful play
         // Updating state earlier causes re-renders that might interrupt play()
         if (isMountedRef.current) {
             setStreamActive(true);
             addLog(`Camera started. ${videoRef.current.videoWidth}x${videoRef.current.videoHeight}`);
             setStatus("摄像头已启动");
         }
      } catch (playErr: any) {
         // 5. Handle AbortError gracefully
         if (playErr.name === 'AbortError') {
             // In 99% of cases, the video is actually playing or ready to play despite this error
             console.warn("Play interrupted but stream is active");
             if (isMountedRef.current) setStreamActive(true);
         } else {
             throw playErr;
         }
      }

    } catch (err: any) {
      addLog(`Camera error: ${err.message}`);
      if (isMountedRef.current) {
          setStatus("摄像头启动失败");
          setStreamActive(false);
      }
    }
  }, [selectedDeviceId, videoRef, addLog, setStatus]);

  // NO AUTO-START USE-EFFECT. Manual control only.

  return { devices, selectedDeviceId, setSelectedDeviceId, startCamera, isStreamActive };
}
