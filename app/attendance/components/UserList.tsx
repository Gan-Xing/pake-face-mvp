import React from 'react';
import styles from '../attendance.module.css';

interface UserListProps {
  faces: any[];
  onDelete: (name: string) => void;
  onEdit: (user: any) => void;
}

export function UserList({ faces, onDelete, onEdit }: UserListProps) {
  return (
    <div className={`${styles.card} ${styles.userListCard}`}>
      <h4 className={styles.cardTitle}>已注册用户 ({faces.length})</h4>
      <ul className={styles.userList}>
        {faces.map(f => (
          <li key={f.name} className={styles.userItem}>
            <div className={styles.userInfo}>
               {f.photoDataUrl ? (
                   <img src={f.photoDataUrl} className={styles.avatar} alt={f.name} />
               ) : (
                   <div className={styles.avatar} /> 
               )}
               <span className={styles.userName}>{f.name}</span>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  onClick={() => onEdit(f)} 
                  className={styles.btnGhost}
                  style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                >
                  改名
                </button>
                <button 
                  onClick={() => onDelete(f.name)} 
                  className={styles.btnDestructive}
                >
                  删除
                </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
