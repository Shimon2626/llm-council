// Pure text builders for the markdown report (no React), so they can be tested
// with `node --test frontend/tests`.

const ATTACHED_BLOCK = /\*\*Attached file: ([^\n]*)\*\*\n```\n([\s\S]*?)\n```/g;

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
  (query || '').replace(ATTACHED_BLOCK, (_, name, body) => previewBlock(name, body, n));

// Only the attachment previews, without the typed prompt.
export const attachmentPreviewBlocks = (query, n = 600) =>
  [...(query || '').matchAll(ATTACHED_BLOCK)].map((m) => previewBlock(m[1], m[2], n)).join('\n\n');

const cell = (v) => String(v ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');

export const attachmentTable = (runRecord) => {
  const rows = runRecord?.attachments || [];
  if (rows.length === 0) return ['_no attachments_', ''];
  return [
    '| File | Full extracted length (chars) | SHA-256 of full extracted text |',
    '|---|---|---|',
    ...rows.map((a) => `| ${cell(a.name)} | ${a.text_chars} | ${a.text_sha256} |`),
    '',
  ];
};

const statusOf = (r) => r.status || 'ok';

export const healthLines = (s1) => {
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
  return lines;
};

export const seatTable = (s1) => [
  '| Seat | Answered by | Status | finish_reason | Prompt tok | Completion tok | Reasoning tok | Seconds | Attempts |',
  '|---|---|---|---|---|---|---|---|---|',
  ...s1.map((r) =>
    `| ${cell(r.seat || r.model)} | ${cell(r.model)} | ${statusOf(r)} | ${r.finish_reason ?? 'null'} | ` +
    `${r.prompt_tokens ?? '?'} | ${r.completion_tokens ?? 0} | ${r.reasoning_tokens ?? 0} | ${r.duration_s ?? '?'} | ${r.attempts ?? '?'} |`),
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
