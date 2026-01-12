import React from 'react';
import styles from '../demo2.module.css';

interface ControlPanelProps {
    devices: MediaDeviceInfo[];
    selectedDeviceId: string;
    setSelectedDeviceId: (id: string) => void;
    startCamera: () => void;
    isReady: boolean;
    autoMode: boolean;
    setAutoMode: (mode: boolean) => void;
    handleCheck: () => void;
    checkResult: string;
    topMatches: any[];
    isLivenessEnabled: boolean;
    setLivenessEnabled: (val: boolean) => void;
    isAlive: boolean;
    supportsLiveness: boolean;
}

export function ControlPanel({
    devices, selectedDeviceId, setSelectedDeviceId, startCamera, isReady,
    autoMode, setAutoMode, handleCheck, checkResult, topMatches,
    isLivenessEnabled, setLivenessEnabled, isAlive, supportsLiveness
}: ControlPanelProps) {
    return (
        <div className={styles.card}>
            <h4 className={styles.cardTitle}>操作控制</h4>
            <div className={styles.controlGroup}>
                {devices.length > 0 && (
                    <select
                        className={styles.select}
                        value={selectedDeviceId}
                        onChange={e => setSelectedDeviceId(e.target.value)}
                    >
                        {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.slice(0, 5)}`}</option>)}
                    </select>
                )}
                <button className={`${styles.button} ${styles.btnSecondary}`} onClick={startCamera} disabled={!isReady}>
                    📸 启动
                </button>
                <button
                    className={`${styles.button} ${autoMode ? styles.btnActive : styles.btnGhost}`}
                    onClick={() => setAutoMode(!autoMode)}
                >
                    {autoMode ? "🛑 停止" : "🤖 自动"}
                </button>
                <button onClick={handleCheck} className={`${styles.button} ${styles.btnPrimary}`}>
                    🔍 单次
                </button>
            </div>

            <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 10, padding: '10px', background: '#f9fafb', borderRadius: '8px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.9rem', cursor: 'pointer', userSelect: 'none' }}>
                    <input 
                        type="checkbox" 
                        checked={isLivenessEnabled} 
                        onChange={e => setLivenessEnabled(e.target.checked)} 
                        disabled={!supportsLiveness}
                        style={{ width: 16, height: 16, cursor: supportsLiveness ? 'pointer' : 'not-allowed' }}
                    />
                    活体检测 (眨眼){supportsLiveness ? "" : " - 当前检测不支持"}
                </label>
                {isLivenessEnabled && (
                    <span className={styles.tag} style={{ 
                        background: isAlive ? '#d1fae5' : '#fee2e2', 
                        color: isAlive ? '#047857' : '#b91c1c',
                        fontSize: '0.8rem',
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontWeight: 600
                    }}>
                        {isAlive ? "通过 ✓" : "请眨眼 👀"}
                    </span>
                )}
            </div>

            {checkResult && (
                <div className={styles.resultBox}>
                    {checkResult}
                </div>
            )}

            {topMatches.length > 0 && (
                <div style={{ marginTop: "16px" }}>
                    {topMatches.map((m, i) => (
                        <div key={m.name} className={styles.topMatch}>
                            <span className={`${styles.rank} ${i === 0 ? styles.rankFirst : ''}`}>#{i + 1}</span>
                            {m.photo && <img src={m.photo} className={styles.avatar} style={{width: 30, height: 30}} alt={m.name} />}
                            <span>{m.name}</span>
                            <span className={styles.score}>{m.score.toFixed(3)}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
