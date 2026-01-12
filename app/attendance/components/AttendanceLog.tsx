import React from 'react';
import styles from '../attendance.module.css';

export interface LogEntry {
  id: string;
  name: string;
  score: number;
  time: number;
  photo?: string;
}

interface AttendanceLogProps {
  logs: LogEntry[];
  onClear: () => void;
  onViewImage: (src: string) => void;
  onDeleteLog: (id: string) => void;
}

export function AttendanceLog({ logs, onClear, onViewImage, onDeleteLog }: AttendanceLogProps) {
  if (logs.length === 0) return null;

  const handleDownload = (log: LogEntry) => {
      if (!log.photo) return;
      const a = document.createElement('a');
      a.href = log.photo;
      a.download = `attendance-${log.name}-${log.time}.jpg`;
      a.click();
  };

  return (
    <div className={styles.card} style={{ marginTop: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h4 className={styles.cardTitle} style={{ margin: 0 }}>📋 考勤记录 ({logs.length})</h4>
        <button onClick={onClear} className={styles.btnDestructive} style={{ padding: '6px 12px' }}>
          清空记录
        </button>
      </div>
      
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #f3f4f6', textAlign: 'left', color: '#6b7280' }}>
              <th style={{ padding: '10px' }}>时间</th>
              <th style={{ padding: '10px' }}>抓拍</th>
              <th style={{ padding: '10px' }}>人员</th>
              <th style={{ padding: '10px' }}>相似度</th>
              <th style={{ padding: '10px' }}>状态</th>
              <th style={{ padding: '10px' }}>操作</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => (
              <tr key={log.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '10px', whiteSpace: 'nowrap', color: '#6b7280' }}>
                  {new Date(log.time).toLocaleTimeString()}
                </td>
                <td style={{ padding: '10px' }}>
                    {log.photo ? (
                      <img 
                        src={log.photo} 
                        style={{ width: 50, height: 50, borderRadius: '8px', objectFit: 'cover', cursor: 'zoom-in', border: '1px solid #ddd' }} 
                        onClick={() => onViewImage(log.photo!)}
                        title="点击查看大图"
                      />
                    ) : <span style={{color:'#ccc'}}>无</span>}
                </td>
                <td style={{ padding: '10px', fontWeight: 500 }}>
                  {log.name}
                </td>
                <td style={{ padding: '10px', fontFamily: 'monospace' }}>
                  {log.score.toFixed(3)}
                </td>
                <td style={{ padding: '10px' }}>
                  <span style={{ 
                    background: '#d1fae5', 
                    color: '#047857', 
                    padding: '2px 8px', 
                    borderRadius: '4px',
                    fontSize: '0.8rem'
                  }}>
                    打卡成功
                  </span>
                </td>
                <td style={{ padding: '10px' }}>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button 
                            onClick={() => handleDownload(log)}
                            className={styles.btnGhost}
                            style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                            disabled={!log.photo}
                        >
                            下载
                        </button>
                        <button 
                            onClick={() => onDeleteLog(log.id)}
                            className={styles.btnDestructive}
                            style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                        >
                            删除
                        </button>
                    </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
