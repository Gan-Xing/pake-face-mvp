import React, { useState, useEffect } from 'react';
import styles from '../demo2.module.css';

interface RenameModalProps {
  isOpen: boolean;
  initialName: string;
  onClose: () => void;
  onConfirm: (newName: string) => void;
}

export function RenameModal({ isOpen, initialName, onClose, onConfirm }: RenameModalProps) {
  const [name, setName] = useState(initialName);

  useEffect(() => {
    setName(initialName);
  }, [initialName, isOpen]);

  if (!isOpen) return null;

  return (
    <div style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.5)',
        display: 'grid',
        placeItems: 'center',
        zIndex: 100
    }}>
      <div className={styles.card} style={{ width: '300px', padding: '24px' }}>
        <h4 className={styles.cardTitle}>修改名称</h4>
        <input 
            type="text" 
            value={name}
            onChange={(e) => setName(e.target.value)}
            className={styles.select} // Reuse input style
            style={{ width: '100%', marginBottom: '20px' }}
            autoFocus
        />
        <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
            <button onClick={onClose} className={styles.btnGhost}>取消</button>
            <button 
                onClick={() => onConfirm(name)} 
                className={styles.btnPrimary}
                disabled={!name.trim() || name === initialName}
            >
                确认
            </button>
        </div>
      </div>
    </div>
  );
}
