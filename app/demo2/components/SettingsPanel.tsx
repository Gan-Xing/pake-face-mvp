import React from 'react';
import styles from '../demo2.module.css';
import { LogPanel } from './LogPanel';

interface SettingsPanelProps {
    similarityThreshold: number;
    faces: any[];
    calibrationTarget: string;
    setCalibrationTarget: (target: string) => void;
    handleCalibration: () => void;
    isCalibrating: boolean;
    logs: string[];
}

export function SettingsPanel({
    similarityThreshold, faces, calibrationTarget, setCalibrationTarget,
    handleCalibration, isCalibrating, logs
}: SettingsPanelProps) {
    return (
        <div className={styles.controlsSection}>
            <div className={styles.card}>
                <h4 className={styles.cardTitle}>校准与设置</h4>
                <div style={{ marginBottom: "12px", fontSize: "0.9rem", color: "#666" }}>
                    当前阈值: <strong>{similarityThreshold.toFixed(3)}</strong>
                </div>
                {!isCalibrating ? (
                    <div className={styles.controlGroup}>
                        <select
                            className={styles.select}
                            value={calibrationTarget}
                            onChange={e => setCalibrationTarget(e.target.value)}
                        >
                            <option value="">选择校准对象...</option>
                            {faces.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
                        </select>
                        <button
                            className={`${styles.button} ${styles.btnSecondary}`}
                            onClick={handleCalibration}
                            disabled={!calibrationTarget}
                        >
                            🎯 自动校准
                        </button>
                    </div>
                ) : (
                    <div style={{ color: "orange", fontWeight: "bold", padding: "10px", background: "#fff7ed", borderRadius: "8px" }}>⚡️ 正在校准中，请保持姿势...</div>
                )}
            </div>

            <LogPanel logs={logs} />
        </div>
    );
}
