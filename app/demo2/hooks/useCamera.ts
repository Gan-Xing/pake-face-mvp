import { useState, useEffect, useCallback, RefObject } from 'react';

interface UseCameraProps {
    videoRef: RefObject<HTMLVideoElement>;
    addLog: (msg: string) => void;
    setStatus: (status: string) => void;
}

export function useCamera({ videoRef, addLog, setStatus }: UseCameraProps) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isStreamActive, setStreamActive] = useState(false);

  useEffect(() => {
    let mounted = true;
    navigator.mediaDevices.enumerateDevices().then(devs => {
        if (mounted) {
            const videoInputs = devs.filter(d => d.kind === 'videoinput');
            addLog(`Found ${videoInputs.length} video devices`);
            setDevices(videoInputs);
            if (videoInputs.length > 0) setSelectedDeviceId(videoInputs[0].deviceId);
        }
    });
    return () => { mounted = false; };
  }, [addLog]);

  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    addLog(`Starting camera with deviceId: ${selectedDeviceId}`);
    try {
      const constraints = {
        video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStreamActive(true);
      addLog("Camera started. Video dimensions: " + videoRef.current.videoWidth + "x" + videoRef.current.videoHeight);
      setStatus("摄像头已启动");
    } catch (err: any) {
      addLog(`Camera error: ${err.message}`);
      setStatus("摄像头启动失败");
      setStreamActive(false);
    }
  }, [selectedDeviceId, videoRef, addLog, setStatus]);

  return { devices, selectedDeviceId, setSelectedDeviceId, startCamera, isStreamActive };
}
