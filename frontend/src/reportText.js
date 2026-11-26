// Pure text builders for the markdown report (no React), so they can be tested
// with `node --test frontend/tests`.

const ATTACHED_BLOCK = /\*\*Attached file: ([^\n]*?)( \[PROMPT FILE\])?\*\*\n```\n([\s\S]*?)\n```/g;

export const typedPrompt = (query) => (query || '').split('\n\n**Attached file:')[0].trim();

// A preview is only a preview: the label always carries the full length, and
// says how many characters are shown.
const previewBlock = (name, body, n) => {
  const shown = Math.min(n, body.length);
  return (
    `**Attached file: ${name}** (full length ${body.length.toLocaleString()} chars; ` +
    `preview of the first ${shown.toLocaleString()} characters only, not the full text sent to the seats)\n` +
    `\`\`\`\n${body.slice(0, n)}${body.length > n ? '\n…' : ''}\n\`\`\``
  );
};

export const previewAttachments = (query, n = 600) =>
  (query || '').replace(ATTACHED_BLOCK, (_, name, mark, body) => previewBlock(name + (mark || ''), body, n));

// Only the attachment previews, without the typed prompt.
export const attachmentPreviewBlocks = (query, n = 600) =>
  [...(query || '').matchAll(ATTACHED_BLOCK)].map((m) => previewBlock(m[1] + (m[2] || ''), m[3], n)).join('\n\n');

const cell = (v) => String(v ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');

export const attachmentTable = (runRecord) => {
  const rows = runRecord?.attachments || [];
  if (rows.length === 0) return ['_no attachments_', ''];
  const cutText = (a) => (a.cut_off === true ? 'CUT OFF yes' : a.cut_off === false ? 'CUT OFF no' : 'CUT OFF not recorded');
  return [
    '| File | Full extracted length (chars) | SHA-256 of full extracted text |',
    '|---|---|---|',
    ...rows.map((a) => `| ${cell(a.name)}${a.prompt_file ? ' (PROMPT FILE)' : ''} | ${a.text_chars} | ${a.text_sha256} |`),
    '',
    'Extracted chars per attachment:',
    ...rows.map((a) => `- ${a.name}${a.prompt_file ? ' [PROMPT FILE]' : ''}, ${a.text_chars} chars, cap ${a.cap ?? 'not recorded'}, ${cutText(a)}${a.cut_off_source === 'extractor' ? ' (extractor flag)' : a.cut_off != null ? ' (length heuristic)' : ''}`),
    '',
  ];
};

const SETTING_LABELS = [
  ['stage1_max_tokens', 'Stage 1 max tokens per seat'],
  ['retry_max_tokens_ceiling', 'Retry ceiling (tokens)'],
  ['stage1_reasoning_effort', 'Stage 1 reasoning effort'],
  ['chairman_reasoning_effort', 'Chairman reasoning effort'],
  ['stage2_max_tokens', 'Stage 2 max tokens per reviewer'],
  ['chairman_max_tokens', 'Chairman max tokens'],
  ['max_office_extract_chars', 'Max extracted chars per attachment'],
];

export const settingsTable = (runRecord) => {
  const st = runRecord?.settings;
  if (!st) return ['_not recorded for this conversation_', ''];
  const show = (k, v) => (k.endsWith('_reasoning_effort') ? (v === '' || v == null ? 'default (none sent)' : v) : v);
  const rows = [
    '| Setting | Value used |',
    '|---|---|',
    ...SETTING_LABELS.map(([k, label]) => `| ${label} | ${cell(show(k, st[k]))} |`),
  ];
  if (st.clamped && st.clamped.length) rows.push('', `Clamped to limits: ${st.clamped.join('; ')}`);
  rows.push('');
  return rows;
};

const statusOf = (r) => r.status || 'ok';

export const healthLines = (s1, s2 = [], s3 = null) => {
  const asked = s1.filter((r) => statusOf(r) !== 'duplicate');
  const ok = asked.filter((r) => statusOf(r) === 'ok').length;
  const cut = asked.filter((r) => statusOf(r) === 'truncated').length;
  const none = asked.filter((r) => ['error', 'empty'].includes(statusOf(r))).length;
  const lines = [`${asked.length} seats asked: ${ok} complete, ${cut} cut off, ${none} returned nothing. Only complete answers count as full reviews.`];
  s1.forEach((r) => {
    if (['error', 'empty'].includes(statusOf(r))) {
      const swap = r.replaced_by ? ` Replaced by ${r.replaced_by}.` : ' No reserve seat was left to replace it.';
      lines.push(`- FAILED: ${r.model}: ${r.error_detail || statusOf(r)}.${swap}`);
    } else if (statusOf(r) === 'duplicate') {
      lines.push(`- REMOVED (duplicate model): ${r.model}: ${r.error_detail}.`);
    }
    if (r.replaces) lines.push(`- RESERVE: ${r.model} was seated to replace ${r.replaces}.`);
    if (r.compliance && r.compliance.ok === false) {
      lines.push(`- NON-COMPLIANT: ${r.model}: ${r.compliance.issues.join('; ')}${r.compliance.reasked ? ' (re-asked once)' : ''}.`);
    }
  });
  const cut2 = (s2 || []).filter((r) => r.truncated || r.finish_reason === 'length');
  if (cut2.length) {
    lines.push(`- STAGE 2 CUT OFF: ${cut2.length} of ${s2.length} reviewer rankings hit the token cap (finish_reason=length): ${cut2.map((r) => r.model).join(', ')}. Those rankings may be incomplete.`);
  }
  if (s3 && (s3.status === 'truncated' || s3.finish_reason === 'length')) {
    lines.push(`- STAGE 3 CUT OFF: ${s3.model}. Chairman output was cut off; the merged table is incomplete.`);
  }
  return lines;
};

export const reasoningShare = (r) => {
  const c = Number(r?.completion_tokens) || 0;
  if (c <= 0) return 'n/a';
  return `${(((Number(r?.reasoning_tokens) || 0) / c) * 100).toFixed(1)}%`;
};

export const seatTable = (s1) => [
  '| Seat | Answered by | Status | finish_reason | Prompt tok | Completion tok | Reasoning tok | Reasoning share | Seconds | Attempts |',
  '|---|---|---|---|---|---|---|---|---|---|',
  ...s1.map((r) =>
    `| ${cell(r.seat || r.model)} | ${cell(r.model)} | ${statusOf(r)} | ${r.finish_reason ?? 'null'} | ` +
    `${r.prompt_tokens ?? '?'} | ${r.completion_tokens ?? 0} | ${r.reasoning_tokens ?? 0} | ${reasoningShare(r)} | ${r.duration_s ?? '?'} | ${r.attempts ?? '?'} |`),
  '',
];

// ---- Report file name ------------------------------------------------------
// LLM-Council-<name of the uploaded file, no extension>.md. With several files,
// the first one that is not the council prompt/instructions is used (the prompt
// file only says how to review; the subject is the other one). With no file,
// the conversation topic is used. A name with no date-and-time in it gets the
// current YYYYMMDD_HHMM appended so two runs never share a name.

const HAS_STAMP = /\d{8}[_-]\d{4}/;
const PROMPT_LIKE = /^(council[_ -]?prompt|prompt)(?![A-Za-z])/i;

const pad = (n) => String(n).padStart(2, '0');
export const stampNow = (d = new Date()) =>
  `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`;

const stripExt = (name) => name.replace(/\.[A-Za-z0-9]{1,5}$/, '');

// Letters, digits, underscore, hyphen, dot. Anything else becomes "_".
const safe = (s) =>
  (s || '').normalize('NFKD').replace(/[^\w.-]+/g, '_').replace(/_+/g, '_').replace(/^[_.-]+|[_.-]+$/g, '');

export const attachmentNames = (query) => [...(query || '').matchAll(ATTACHED_BLOCK)].map((m) => m[1]);

export const reportFileName = (query, topic, now = new Date()) => {
  const names = attachmentNames(query);
  const subject = names.find((n) => !PROMPT_LIKE.test(n)) ?? names[0];
  let base = safe(subject ? stripExt(subject) : topic);
  if (!base || base.toLowerCase() === 'new_conversation') base = 'report';
  if (!HAS_STAMP.test(base)) base = `${base}_${stampNow(now)}`;
  return `LLM-Council-${base}.md`;
};
