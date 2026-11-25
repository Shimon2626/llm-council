import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_RUN_SETTINGS, STORAGE_KEY, loadRunSettings, saveRunSettings, buildRunPayload,
  estimateInputTokens, seatsCapTokens, attachmentCutOff,
} from '../src/runSettings.js';

const mem = (init = {}) => {
  const d = { ...init };
  return { getItem: (k) => (k in d ? d[k] : null), setItem: (k, v) => { d[k] = v; } };
};

test('defaults match the brief', () => {
  assert.deepEqual(DEFAULT_RUN_SETTINGS, {
    stage1_max_tokens: '40000', retry_max_tokens_ceiling: '64000', stage1_reasoning_effort: 'low',
    chairman_reasoning_effort: 'medium', stage2_max_tokens: '12000', chairman_max_tokens: '40000', max_office_extract_chars: '400000',
  });
});

test('load falls back to defaults on missing, bad JSON, or bad reasoning', () => {
  assert.deepEqual(loadRunSettings(mem()), DEFAULT_RUN_SETTINGS);
  assert.deepEqual(loadRunSettings(mem({ [STORAGE_KEY]: '{oops' })), DEFAULT_RUN_SETTINGS);
  const bad = loadRunSettings(mem({ [STORAGE_KEY]: JSON.stringify({ stage1_reasoning_effort: 'extreme' }) }));
  assert.equal(bad.stage1_reasoning_effort, DEFAULT_RUN_SETTINGS.stage1_reasoning_effort);
});

test('save then load round-trips', () => {
  const s = mem();
  saveRunSettings(s, { ...DEFAULT_RUN_SETTINGS, stage1_max_tokens: '50000' });
  assert.equal(loadRunSettings(s).stage1_max_tokens, '50000');
});

test('payload converts digit strings to ints and drops junk', () => {
  const p = buildRunPayload({ ...DEFAULT_RUN_SETTINGS, stage2_max_tokens: 'abc' });
  assert.equal(p.stage1_max_tokens, 40000);
  assert.equal(p.stage1_reasoning_effort, 'low');
  assert.ok(!('stage2_max_tokens' in p));
  assert.equal(p.chairman_reasoning_effort, 'medium');
  assert.ok(!('chairman_reasoning_effort' in buildRunPayload({ ...DEFAULT_RUN_SETTINGS, chairman_reasoning_effort: 'extreme' })));
  const bad = loadRunSettings(mem({ [STORAGE_KEY]: JSON.stringify({ chairman_reasoning_effort: 'extreme' }) }));
  assert.equal(bad.chairman_reasoning_effort, 'medium');
});

test('estimates and cut-off helper', () => {
  assert.equal(estimateInputTokens(3600, 0), 1000);
  assert.equal(seatsCapTokens(5, '40000'), 200000);
  assert.equal(seatsCapTokens(5, 'x'), null);
  assert.equal(attachmentCutOff(400000, 400000), true);
  assert.equal(attachmentCutOff(399999, 400000), false);
});
