import React from 'react';
import styles from '../demo2.module.css';

interface ImageModalProps {
  isOpen: boolean;
  src: string;
  onClose: () => void;
}

export function ImageModal({ isOpen, src, onClose }: ImageModalProps) {
  if (!isOpen || !src) return null;

  return (
    <div 
        style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.8)',
            display: 'grid',
            placeItems: 'center',
            zIndex: 200,
            cursor: 'zoom-out'
        }}
        onClick={onClose}
    >
      <img 
        src={src} 
        style={{ 
            maxWidth: '90vw', 
            maxHeight: '90vh', 
            borderRadius: '12px', 
            boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
            border: '2px solid white'
        }} 
        onClick={(e) => e.stopPropagation()} // Prevent close when clicking image
      />
      <button 
        onClick={onClose}
        style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '40px',
            height: '40px',
            fontSize: '24px',
            cursor: 'pointer',
            display: 'grid',
            placeItems: 'center'
        }}
      >
        ×
      </button>
    </div>
  );
}
