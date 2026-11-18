import { useState, useEffect } from 'react';

// Pure helpers take the storage object so they can be tested without a browser.
export function loadSaved(key, fallback, storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem(key);
    return raw === null || raw === undefined ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function save(key, value, storage = globalThis.localStorage) {
  try {
    storage?.setItem(key, JSON.stringify(value));
  } catch { /* storage unavailable: choice just won't persist */ }
}

// useState that remembers its value in localStorage until the user changes it.
export function usePersistedState(key, fallback) {
  const [value, setValue] = useState(() => loadSaved(key, fallback));
  useEffect(() => save(key, value), [key, value]);
  return [value, setValue];
}
