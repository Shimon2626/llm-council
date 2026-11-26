import test from 'node:test';
import assert from 'node:assert/strict';
import { estimateRows, councilReady } from '../src/runSettings.js';

const est = {
  seats: [
    { model: 'a/seat1', input_cost_usd: 0.0001, max_output_cost_usd: 0.002 },
    { model: 'b/seat2', input_cost_usd: 0.0003, max_output_cost_usd: 0.004 },
  ],
  chairman: { model: 'c/chair', input_cost_usd: 0.0005, max_output_cost_usd: 0.018 },
};

test('estimateRows lists each seat then the chairman with formatted dollars', () => {
  assert.deepEqual(estimateRows(est), [
    { role: 'seat', model: 'a/seat1', input: '$0.0001', output: '$0.0020' },
    { role: 'seat', model: 'b/seat2', input: '$0.0003', output: '$0.0040' },
    { role: 'chairman', model: 'c/chair', input: '$0.0005', output: '$0.0180' },
  ]);
});

test('estimateRows is empty for a missing or old-shape estimate', () => {
  assert.deepEqual(estimateRows(null), []);
  assert.deepEqual(estimateRows({ estimated_cost_usd: 1 }), []);
});

test('councilReady: seats and chairman both needed', () => {
  assert.equal(councilReady([{ model: 'a' }], { model: 'c' }), true);
  assert.equal(councilReady([{ model: 'a' }], null, { seats: [], chairman: null }), false);
  assert.equal(councilReady([], { model: 'c' }, { seats: [], chairman: 'c' }), false);
});

test('councilReady: empty selection falls back to the server default', () => {
  assert.equal(councilReady([], null, { seats: ['a'], chairman: 'c' }), true);
  assert.equal(councilReady([], null, { seats: [], chairman: 'c' }), false);
  assert.equal(councilReady([], null, undefined), true);
});
