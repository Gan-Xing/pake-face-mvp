import React from 'react';
import styles from '../demo2.module.css';

interface LogPanelProps {
  logs: string[];
}

export function LogPanel({ logs }: LogPanelProps) {
  return (
    <div className={styles.logPanel}>
        <div className={styles.logHeader}>System Logs</div>
        {logs.map((log, i) => (
            <div key={i}>{log}</div>
        ))}
    </div>
  );
}
