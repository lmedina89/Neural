const DB_NAME = 'micromind-lab';
const STORE = 'checkpoints';
const VERSION = 1;

function openDb() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in globalThis)) return reject(new Error('IndexedDB unavailable'));
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB open failed'));
  });
}

export async function saveCheckpoint(snapshot, name = 'latest') {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put({ id: name, savedAt: Date.now(), snapshot });
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { const e = tx.error; db.close(); reject(e); };
  });
}

export async function saveCheckpointRecord(record) {
  if (!record?.id || !record?.snapshot) throw new Error('Invalid checkpoint record');
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(record);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { const e = tx.error; db.close(); reject(e); };
  });
}

export async function loadCheckpointRecord(name = 'latest') {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(name);
    req.onsuccess = () => { db.close(); resolve(req.result || null); };
    req.onerror = () => { const e = req.error; db.close(); reject(e); };
  });
}

export async function loadCheckpoint(name = 'latest') {
  const record = await loadCheckpointRecord(name);
  return record?.snapshot || null;
}

export async function loadCheckpointRecords(names = ['latest', 'autosave']) {
  const records = await Promise.all(names.map(loadCheckpointRecord));
  return records.filter(Boolean);
}

export async function loadNewestCheckpoint(names = ['latest', 'autosave']) {
  const records = await loadCheckpointRecords(names);
  if (!records.length) return null;
  records.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
  return records[0];
}

export const HALL_OF_FAME_SLOT = 'hall-of-fame';

export async function saveHallOfFame(archive) {
  return saveCheckpoint(archive, HALL_OF_FAME_SLOT);
}

export async function loadHallOfFameRecord() {
  return loadCheckpointRecord(HALL_OF_FAME_SLOT);
}
