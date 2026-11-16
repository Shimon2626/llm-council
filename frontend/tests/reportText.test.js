import test from 'node:test';
import assert from 'node:assert/strict';
import { previewAttachments, attachmentPreviewBlocks, typedPrompt, attachmentTable, healthLines, seatTable } from '../src/reportText.js';

const q = `\n\n**Attached file: Plan.docx**\n\`\`\`\n${'x'.repeat(1000)}\n\`\`\``;

test('preview is labelled with the full length and the number of characters shown', () => {
  const out = previewAttachments(q, 600);
  assert.match(out, /full length 1,000 chars/);
  assert.match(out, /preview of the first 600 characters only/);
  assert.ok(!out.includes('x'.repeat(601)));
});

test('a short file is shown whole and the label says so', () => {
  const out = previewAttachments('**Attached file: a.txt**\n```\nhello\n```', 600);
  assert.match(out, /full length 5 chars; preview of the first 5 characters/);
  assert.ok(!out.includes('…'));
});

test('typed prompt is the full text before the first attachment', () => {
  assert.equal(typedPrompt('Check the plan.\n\n**Attached file: a**\n```\nz\n```'), 'Check the plan.');
  assert.equal(typedPrompt(q), '');
});

test('attachment table carries full length and hash', () => {
  const t = attachmentTable({ attachments: [{ name: 'Plan.docx', text_chars: 9072, text_sha256: 'ab12' }] }).join('\n');
  assert.match(t, /\| Plan\.docx \| 9072 \| ab12 \|/);
});

test('health lists failures with reason, replacement, duplicates and non-compliance', () => {
  const s1 = [
    { model: 'a', status: 'ok' },
    { model: 'b', status: 'error', error_detail: 'HTTP 410 retired', replaced_by: 'r1' },
    { model: 'r1', status: 'ok', replaces: 'b (error)' },
    { model: 'c:free', status: 'duplicate', error_detail: 'same base model as another seat (c)' },
    { model: 'd', status: 'ok', compliance: { ok: false, issues: ['line 1 is not a verdict'], reasked: true } },
  ];
  const t = healthLines(s1).join('\n');
  assert.match(t, /4 seats asked: 3 complete, 0 cut off, 1 returned nothing/);
  assert.match(t, /FAILED: b: HTTP 410 retired\. Replaced by r1/);
  assert.match(t, /REMOVED \(duplicate model\): c:free/);
  assert.match(t, /RESERVE: r1/);
  assert.match(t, /NON-COMPLIANT: d: line 1 is not a verdict \(re-asked once\)/);
});

test('seat table shows finish_reason, tokens and duration, null shown as null', () => {
  const t = seatTable([{ seat: 's', model: 'm', status: 'error', finish_reason: null, completion_tokens: 0, duration_s: null }]).join('\n');
  assert.match(t, /\| s \| m \| error \| null \| \? \| 0 \| 0 \| \? \| \? \|/);
});

test('attachmentPreviewBlocks returns only the labelled previews', () => {
  const out = attachmentPreviewBlocks(`typed words${q}`, 600);
  assert.ok(!out.includes('typed words'));
  assert.match(out, /full length 1,000 chars/);
});
