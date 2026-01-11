import { useState, useCallback, useRef } from 'react';

// EAR threshold for blink
const BLINK_EAR_THRESHOLD = 0.22; 
// Time window to consider a blink valid
const BLINK_VALID_MS = 3000;

export function useLiveness(enabled: boolean) {
    const [isAlive, setIsAlive] = useState(false);
    const [blinkCount, setBlinkCount] = useState(0);
    const lastBlinkTimeRef = useRef<number>(0);
    const wasClosedRef = useRef<boolean>(false);

    const updateLiveness = useCallback((ear: number | null) => {
        if (!enabled || ear === null) {
             // Reset if tracking lost? No, keep state.
             return;
        }

        // Simple state machine
        if (ear < BLINK_EAR_THRESHOLD) {
            wasClosedRef.current = true;
        } else {
            if (wasClosedRef.current) {
                // Closed -> Open = Blink
                const now = Date.now();
                lastBlinkTimeRef.current = now;
                setBlinkCount(c => c + 1);
                setIsAlive(true);
                wasClosedRef.current = false;
            }
        }
    }, [enabled]);

    const checkLiveness = useCallback(() => {
        if (!enabled) return true;
        const now = Date.now();
        const isValid = (now - lastBlinkTimeRef.current) < BLINK_VALID_MS;
        if (!isValid && isAlive) setIsAlive(false); // Auto expire UI
        return isValid;
    }, [enabled, isAlive]);

    const resetLiveness = useCallback(() => {
        setIsAlive(false);
        lastBlinkTimeRef.current = 0;
        wasClosedRef.current = false;
    }, []);

    return { 
        isAlive, 
        blinkCount, 
        updateLiveness, 
        checkLiveness,
        resetLiveness
    };
}
