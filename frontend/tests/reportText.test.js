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

import { reportFileName, stampNow } from '../src/reportText.js';

const att = (...names) => names.map((n) => `**Attached file: ${n}**\n\`\`\`\nx\n\`\`\``).join('\n\n');
const NOW = new Date(2026, 10, 16, 14, 5);

test('file name is LLM-Council- plus the uploaded plan file name, as asked', () => {
  const q = att('Table_Optimization_Plan_20261116_1351_v1_0.docx', 'Council_Prompt_Table_Optimization_20261116_1351_v1_0.docx');
  assert.equal(reportFileName(q, 'whatever', NOW), 'LLM-Council-Table_Optimization_Plan_20261116_1351_v1_0.md');
});

test('the council prompt file is skipped even when it is attached first', () => {
  const q = att('Council_Prompt_X_20261116_1351_v1_0.docx', 'Plan_20261116_1351_v1_0.docx');
  assert.equal(reportFileName(q, '', NOW), 'LLM-Council-Plan_20261116_1351_v1_0.md');
});

test('only a prompt file attached: use it rather than nothing', () => {
  assert.equal(reportFileName(att('Council_Prompt_A_20261116_1351_v1_0.docx'), '', NOW), 'LLM-Council-Council_Prompt_A_20261116_1351_v1_0.md');
});

test('an uploaded name without a date gets the current date and time appended', () => {
  assert.equal(reportFileName(att('notes.txt'), '', NOW), `LLM-Council-notes_${stampNow(NOW)}.md`);
  assert.equal(stampNow(NOW), '20261116_1405');
});

test('no attachment: topic is used, made file-safe, with date and time', () => {
  assert.equal(reportFileName('just typed', 'Pricing: Q4 / EU plan?', NOW), 'LLM-Council-Pricing_Q4_EU_plan_20261116_1405.md');
});

test('no attachment and no usable topic falls back to "report"', () => {
  assert.equal(reportFileName('x', 'New Conversation', NOW), 'LLM-Council-report_20261116_1405.md');
  assert.equal(reportFileName('x', undefined, NOW), 'LLM-Council-report_20261116_1405.md');
});
