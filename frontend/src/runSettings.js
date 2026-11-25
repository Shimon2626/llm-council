// Per-run token and reasoning limits: defaults, storage, payload, estimates.
// Pure helpers (no React) so they can be tested with node --test.

export const STORAGE_KEY = 'llm-council.runSettings.v1';
export const CHARS_PER_TOKEN = 3.6;

export const DEFAULT_RUN_SETTINGS = {
  stage1_max_tokens: '40000',
  retry_max_tokens_ceiling: '64000',
  stage1_reasoning_effort: 'low',
  chairman_reasoning_effort: 'medium',
  stage2_max_tokens: '12000',
  chairman_max_tokens: '40000',
  max_office_extract_chars: '400000',
};

export const REASONING_OPTIONS = ['default', 'low', 'medium', 'high'];

export const REASONING_KEYS = ['stage1_reasoning_effort', 'chairman_reasoning_effort'];

export const SETTING_FIELDS = [
  { key: 'stage1_max_tokens', label: 'Stage 1 max tokens', help: 'Cap per seat answer. Includes hidden reasoning, so a low cap can cut the visible answer short.' },
  { key: 'retry_max_tokens_ceiling', label: 'Retry ceiling', help: 'A cut-off answer is retried once with a doubled cap, never above this number.' },
  { key: 'stage1_reasoning_effort', label: 'Stage 1 reasoning', help: '"default" sends no reasoning setting. Lower effort leaves more of the cap for the answer.' },
  { key: 'chairman_reasoning_effort', label: 'Chairman reasoning', help: 'Stage 3 reasoning effort for the chairman. "default" sends no reasoning setting. Lower effort leaves more of the chairman cap for the merged answer.' },
  { key: 'stage2_max_tokens', label: 'Stage 2 max tokens', help: 'Cap per peer-ranking response.' },
  { key: 'chairman_max_tokens', label: 'Chairman max tokens', help: 'Cap for the final merged answer. If it is hit, the merged table is incomplete.' },
  { key: 'max_office_extract_chars', label: 'Attachment extract chars', help: 'Characters kept from each xlsx/pptx attachment. Anything longer is cut off.' },
];

export function loadRunSettings(storage) {
  try {
    const raw = storage && storage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_RUN_SETTINGS };
    const saved = JSON.parse(raw);
    const out = { ...DEFAULT_RUN_SETTINGS };
    for (const k of Object.keys(out)) {
      if (typeof saved[k] === 'string') out[k] = saved[k];
    }
    for (const k of REASONING_KEYS) {
      if (!REASONING_OPTIONS.includes(out[k])) out[k] = DEFAULT_RUN_SETTINGS[k];
    }
    return out;
  } catch {
    return { ...DEFAULT_RUN_SETTINGS };
  }
}

export function saveRunSettings(storage, settings) {
  try {
    if (storage) storage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // storage blocked: the panel still works for this page load
  }
}

function toInt(value) {
  const s = String(value ?? '').trim();
  return /^\d+$/.test(s) ? parseInt(s, 10) : null;
}

// Request body fields. Blank or non-numeric number fields are omitted so the
// server falls back to its own default.
export function buildRunPayload(settings) {
  const out = {};
  for (const key of ['stage1_max_tokens', 'retry_max_tokens_ceiling', 'stage2_max_tokens', 'chairman_max_tokens', 'max_office_extract_chars']) {
    const n = toInt(settings[key]);
    if (n !== null) out[key] = n;
  }
  for (const k of REASONING_KEYS) {
    if (REASONING_OPTIONS.includes(settings[k])) out[k] = settings[k];
  }
  return out;
}

export function estimateInputTokens(extractedChars, promptChars) {
  return Math.ceil(((extractedChars || 0) + (promptChars || 0)) / CHARS_PER_TOKEN);
}

export function seatsCapTokens(seatCount, stage1MaxTokens) {
  const n = toInt(stage1MaxTokens);
  return n === null ? null : seatCount * n;
}

// The extractor's boolean `truncated` flag is authoritative. Only when it is
// missing does reaching the cap act as the fallback signal.
export function attachmentCutOff(textChars, cap, truncated) {
  if (typeof truncated === 'boolean') return truncated;
  const c = toInt(cap);
  return c !== null && textChars >= c;
}
