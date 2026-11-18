import test from 'node:test';
import assert from 'node:assert/strict';
import { loadSaved, save } from '../src/persistedState.js';

const mem = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, v) }; };

test('returns the fallback when nothing is saved', () => {
  assert.deepEqual(loadSaved('seats', []), []);
});
test('round-trips a saved choice', () => {
  const s = mem();
  save('seats', [{ provider: 'openrouter', model: 'a/b' }], s);
  assert.deepEqual(loadSaved('seats', [], s), [{ provider: 'openrouter', model: 'a/b' }]);
});
test('corrupt or unavailable storage falls back instead of throwing', () => {
  const bad = { getItem: () => '{not json', setItem: () => { throw new Error('full'); } };
  assert.equal(loadSaved('k', 'dflt', bad), 'dflt');
  assert.doesNotThrow(() => save('k', 1, bad));
  assert.equal(loadSaved('k', 'dflt', undefined) !== undefined, true);
});
