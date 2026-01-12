const DB_NAME = "face-attendance-db";
const STORE_NAME = "faces";

export type FaceRecord = {
  name: string;
  embedding: number[];
  createdAt: number;
  photoDataUrl?: string;
};

export function loadFaceStore(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "name" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveEmbedding(record: {
  name: string;
  embedding: number[];
  photoDataUrl?: string;
}) {
  const db = await loadFaceStore();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put({
      ...record,
      createdAt: Date.now(),
    });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function listEmbeddings(): Promise<FaceRecord[]> {
  const db = await loadFaceStore();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).getAll();
    request.onsuccess = () => resolve(request.result as FaceRecord[]);
    request.onerror = () => reject(request.error);
  });
}

export async function deleteEmbedding(name: string) {
  const db = await loadFaceStore();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(name);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function renameEmbedding(oldName: string, newName: string) {
  const db = await loadFaceStore();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const getReq = store.get(oldName);
    getReq.onsuccess = () => {
      const record = getReq.result as FaceRecord | undefined;
      if (!record) {
        return;
      }
      store.delete(oldName);
      store.put({ ...record, name: newName, createdAt: Date.now() });
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
