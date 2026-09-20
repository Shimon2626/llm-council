import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import { api } from '../api';
import './ChatInterface.css';

// A "seat" is a single-provider pick, e.g. { provider: 'nvidia', model: 'meta/llama-3.1-70b-instruct' }.
// Serialized to the backend spec shape { nvidia: '...' } or { openrouter: '...' } — no cross-provider
// fallback, since picking a provider explicitly means you want exactly that one.
const seatKey = (seat) => `${seat.provider}:${seat.model}`;

export default function ChatInterface({
  conversation,
  onSendMessage,
  onStopMessage,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [availableModels, setAvailableModels] = useState({ nvidia: [], openrouter: [] });
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [selectedSeats, setSelectedSeats] = useState([]); // [] means "use backend default council"
  const [selectedChairman, setSelectedChairman] = useState(null); // null means "use backend default chairman"
  // Strings, not numbers: lets the field be genuinely empty (falls back to the
  // backend's own default) instead of coercing to 0. No provider accepts a
  // literal "unlimited" — this is a per-request override of the finite cap.
  const [councilMaxTokens, setCouncilMaxTokens] = useState('');
  const [chairmanMaxTokens, setChairmanMaxTokens] = useState('');
  // Full live OpenRouter catalog (id strings only), fetched once — used only
  // to validate a free-typed custom slug, not rendered as a checkbox list.
  const [openRouterCatalog, setOpenRouterCatalog] = useState(null);
  const [customSeatInput, setCustomSeatInput] = useState('');
  const [customSeatError, setCustomSeatError] = useState('');
  const [customChairmanInput, setCustomChairmanInput] = useState('');
  const [customChairmanError, setCustomChairmanError] = useState('');
  const [costEstimate, setCostEstimate] = useState(null);
  const [estimating, setEstimating] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  useEffect(() => {
    api.getAvailableModels().then(setAvailableModels).catch((e) => console.error('Failed to load model list:', e));
    api.getOpenRouterCatalog()
      .then((r) => setOpenRouterCatalog(r.models))
      .catch((e) => console.error('Failed to load OpenRouter catalog (custom model entry will be unavailable):', e));
  }, []);

  // Any change to the query, attachments, or model selection invalidates a
  // previous cost estimate — it no longer reflects what would actually run.
  useEffect(() => {
    setCostEstimate(null);
  }, [input, attachments, selectedSeats, selectedChairman]);

  const toggleSeat = (provider, model) => {
    setSelectedSeats((prev) => {
      const key = seatKey({ provider, model });
      const exists = prev.some((s) => seatKey(s) === key);
      return exists ? prev.filter((s) => seatKey(s) !== key) : [...prev, { provider, model }];
    });
  };

  // Free-typed OpenRouter slugs (e.g. moonshotai/kimi-k3) aren't in the
  // curated checkbox list, so they're validated against the live catalog
  // (fetched once on mount) before being added as a seat — same trust bar
  // AVAILABLE_MODELS entries get, just checked at request time instead of
  // by a human editing config.py in advance.
  const addCustomSeat = async () => {
    const model = customSeatInput.trim();
    if (!model) return;
    if (openRouterCatalog === null) {
      setCustomSeatError('Still loading the OpenRouter catalog — try again in a moment.');
      return;
    }
    if (!openRouterCatalog.includes(model)) {
      setCustomSeatError(`"${model}" not found on OpenRouter. Check the exact slug (e.g. moonshotai/kimi-k3).`);
      return;
    }
    setCustomSeatError('');
    setCustomSeatInput('');
    // Persist it server-side so it's a normal checkbox option from now on,
    // not just for this browser tab — a failure here still lets the seat
    // work for the current run, it just won't have stuck for next time.
    try {
      const updated = await api.addCustomModel('openrouter', model);
      setAvailableModels(updated);
    } catch (e) {
      console.error('Failed to permanently save custom model (added for this session only):', e);
    }
    setSelectedSeats((prev) =>
      prev.some((s) => seatKey(s) === seatKey({ provider: 'openrouter', model }))
        ? prev
        : [...prev, { provider: 'openrouter', model }]
    );
  };

  const addCustomChairman = async () => {
    const model = customChairmanInput.trim();
    if (!model) return;
    if (openRouterCatalog === null) {
      setCustomChairmanError('Still loading the OpenRouter catalog — try again in a moment.');
      return;
    }
    if (!openRouterCatalog.includes(model)) {
      setCustomChairmanError(`"${model}" not found on OpenRouter. Check the exact slug (e.g. moonshotai/kimi-k3).`);
      return;
    }
    setCustomChairmanError('');
    setCustomChairmanInput('');
    try {
      const updated = await api.addCustomModel('openrouter', model);
      setAvailableModels(updated);
    } catch (e) {
      console.error('Failed to permanently save custom model (added for this session only):', e);
    }
    setSelectedChairman({ provider: 'openrouter', model });
  };

  const councilModelsPayload = () =>
    selectedSeats.length === 0 ? undefined : selectedSeats.map((s) => ({ [s.provider]: s.model }));

  const chairmanModelPayload = () =>
    selectedChairman ? { [selectedChairman.provider]: selectedChairman.model } : undefined;

  // '' or non-positive → undefined, so the backend falls back to its own default
  // rather than receiving max_tokens: 0 or a NaN.
  const toTokenOverride = (value) => {
    const n = parseInt(value, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  };

  const handleEstimateCost = async () => {
    setEstimating(true);
    setCostEstimate(null);
    try {
      const content = buildMessageContent();
      const result = await api.estimateCost(content, councilModelsPayload(), chairmanModelPayload());
      setCostEstimate(result);
    } catch (e) {
      console.error('Cost estimate failed:', e);
      setCostEstimate({ error: true });
    } finally {
      setEstimating(false);
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    files.forEach((file) => {
      const lowerName = file.name.toLowerCase();
      if (lowerName.endsWith('.zip')) {
        uploadZipFile(file);
        return;
      }
      if (lowerName.endsWith('.docx')) {
        uploadDocxFile(file);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        setAttachments((prev) => [...prev, { name: file.name, content: reader.result }]);
      };
      reader.readAsText(file);
    });
    e.target.value = ''; // allow re-selecting the same file
  };

  // Zip is binary — extracted server-side (api.extractZip) rather than
  // read as text like other attachments. Shows an "extracting…" chip
  // while the request is in flight; if the chip gets removed mid-upload,
  // the result is discarded instead of resurrecting a removed attachment.
  const uploadZipFile = async (file) => {
    setAttachments((prev) => [...prev, { name: file.name, content: null, uploading: true }]);
    try {
      const result = await api.extractZip(file);
      const notes = [
        result.files_skipped_binary.length > 0
          ? `${result.files_skipped_binary.length} binary file(s) skipped`
          : null,
        result.files_skipped_oversized.length > 0
          ? `${result.files_skipped_oversized.length} oversized file(s) skipped`
          : null,
        result.truncated ? 'content truncated to fit' : null,
      ].filter(Boolean).join(', ');
      const content =
        `[Extracted ${result.files_extracted} file(s) from ${file.name}` +
        `${notes ? ` — ${notes}` : ''}]\n\n${result.extracted_text}`;
      setAttachments((prev) =>
        prev.some((a) => a.name === file.name)
          ? prev.map((a) => (a.name === file.name ? { name: file.name, content } : a))
          : prev
      );
    } catch (error) {
      console.error('Failed to extract zip:', error);
      setAttachments((prev) => prev.filter((a) => a.name !== file.name));
      alert(`Could not extract ${file.name}: ${error.message}`);
    }
  };

  // Same reasoning as uploadZipFile above: .docx is a binary zip container,
  // not text, so it's extracted server-side (api.extractDocx) instead of
  // going through the generic FileReader.readAsText path.
  const uploadDocxFile = async (file) => {
    setAttachments((prev) => [...prev, { name: file.name, content: null, uploading: true }]);
    try {
      const result = await api.extractDocx(file);
      const content =
        `[Extracted text from ${file.name}${result.truncated ? ' — content truncated to fit' : ''}]\n\n` +
        result.extracted_text;
      setAttachments((prev) =>
        prev.some((a) => a.name === file.name)
          ? prev.map((a) => (a.name === file.name ? { name: file.name, content } : a))
          : prev
      );
    } catch (error) {
      console.error('Failed to extract docx:', error);
      setAttachments((prev) => prev.filter((a) => a.name !== file.name));
      alert(`Could not extract ${file.name}: ${error.message}`);
    }
  };

  const removeAttachment = (name) => {
    setAttachments((prev) => prev.filter((a) => a.name !== name));
  };

  const attachmentsUploading = attachments.some((a) => a.uploading);

  const buildMessageContent = () => {
    const ready = attachments.filter((a) => a.content !== null);
    if (ready.length === 0) return input;
    const attachmentBlocks = ready
      .map((a) => `**Attached file: ${a.name}**\n\`\`\`\n${a.content}\n\`\`\``)
      .join('\n\n');
    return `${input}\n\n${attachmentBlocks}`;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((input.trim() || attachments.length > 0) && !isLoading && !attachmentsUploading) {
      onSendMessage(
        buildMessageContent(),
        councilModelsPayload(),
        chairmanModelPayload(),
        toTokenOverride(councilMaxTokens),
        toTokenOverride(chairmanMaxTokens)
      );
      setInput('');
      setAttachments([]);
      setCostEstimate(null);
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const buildMarkdownReport = (msg, userQuery) => {
    const lines = [`# LLM Council Report`, ``, `**Date:** ${new Date().toISOString()}`, ``];

    if (userQuery) {
      lines.push(`## Query`, ``, userQuery, ``);
    }

    lines.push(`## Stage 1 — Individual Responses`, ``);
    (msg.stage1 || []).forEach((r) => {
      lines.push(`### ${r.model}`, ``, r.response, ``);
    });

    lines.push(`## Stage 2 — Peer Rankings`, ``);
    (msg.stage2 || []).forEach((r) => {
      lines.push(`### ${r.model}`, ``, r.ranking, ``);
    });

    const aggregate = msg.metadata?.aggregate_rankings;
    if (aggregate?.length) {
      lines.push(`### Aggregate Rankings`, ``, `| Model | Average Rank | Rankings Count |`, `|---|---|---|`);
      aggregate.forEach((a) => {
        lines.push(`| ${a.model} | ${a.average_rank} | ${a.rankings_count} |`);
      });
      lines.push(``);
    }

    lines.push(`## Stage 3 — Chairman Final Verdict (${msg.stage3?.model || 'unknown'})`, ``, msg.stage3?.response || '', ``);

    return lines.join('\n');
  };

  const downloadReport = (msg, userQuery) => {
    const markdown = buildMarkdownReport(msg, userQuery);
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llm-council-report-${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // The topbar (settings gear) must render regardless of whether a
  // conversation is selected — it was previously stuck below an early
  // `if (!conversation) return ...`, which made it disappear entirely on
  // the "Welcome" screen and on any already-answered (single-turn) chat.
  const topbar = (
    <div className="chat-topbar">
      <button
        type="button"
        className="model-picker-toggle"
        onClick={() => setShowModelPicker((v) => !v)}
      >
        ⚙ Settings — {selectedSeats.length > 0 ? `${selectedSeats.length} council seat(s)` : 'default council'}
        {selectedChairman ? ` · Chairman: ${selectedChairman.model}` : ''}
        {(councilMaxTokens || chairmanMaxTokens) ? ' · custom max tokens' : ''}
      </button>

      {showModelPicker && (
        <div className="model-picker-panel">
          {['nvidia', 'openrouter'].map((provider) => {
            // Custom-added seats (typed via the field below, only possible for
            // OpenRouter) aren't in the curated AVAILABLE_MODELS list — merge
            // them in so they render as a checked, toggleable entry too,
            // instead of existing invisibly in selectedSeats.
            const curated = availableModels[provider] || [];
            const custom = selectedSeats
              .filter((s) => s.provider === provider && !curated.includes(s.model))
              .map((s) => s.model);
            const models = [...curated, ...custom];
            return (
              <div key={provider} className="model-picker-group">
                <div className="model-picker-group-label">{provider === 'nvidia' ? 'NVIDIA' : 'OpenRouter'}</div>
                {models.map((model) => {
                  const checked = selectedSeats.some((s) => s.provider === provider && s.model === model);
                  return (
                    <label key={model} className="model-picker-option">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleSeat(provider, model)}
                      />
                      {model}
                    </label>
                  );
                })}

                {provider === 'openrouter' && (
                  <div className="model-picker-custom-row">
                    <input
                      type="text"
                      className="model-picker-custom-input"
                      placeholder="Custom slug, e.g. moonshotai/kimi-k3"
                      value={customSeatInput}
                      onChange={(e) => { setCustomSeatInput(e.target.value); setCustomSeatError(''); }}
                      onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addCustomSeat(); } }}
                    />
                    <button type="button" onClick={addCustomSeat} disabled={!customSeatInput.trim()}>
                      Add
                    </button>
                    {customSeatError && <div className="model-picker-custom-error">{customSeatError}</div>}
                  </div>
                )}
              </div>
            );
          })}

          <div className="model-picker-group">
            <div className="model-picker-group-label">Chairman</div>
            <label className="model-picker-option">
              <input
                type="radio"
                name="chairman"
                checked={!selectedChairman}
                onChange={() => setSelectedChairman(null)}
              />
              Default
            </label>
            {['nvidia', 'openrouter'].map((provider) => {
              const curated = availableModels[provider] || [];
              // Surface a custom-picked chairman (typed below) as its own
              // selected radio, same reasoning as the custom seat merge above.
              const custom =
                selectedChairman?.provider === provider && !curated.includes(selectedChairman.model)
                  ? [selectedChairman.model]
                  : [];
              return [...curated, ...custom].map((model) => (
                <label key={`${provider}:${model}`} className="model-picker-option">
                  <input
                    type="radio"
                    name="chairman"
                    checked={selectedChairman?.provider === provider && selectedChairman?.model === model}
                    onChange={() => setSelectedChairman({ provider, model })}
                  />
                  {provider === 'nvidia' ? 'NVIDIA' : 'OpenRouter'}: {model}
                </label>
              ));
            })}

            <div className="model-picker-custom-row">
              <input
                type="text"
                className="model-picker-custom-input"
                placeholder="Custom chairman slug, e.g. moonshotai/kimi-k3"
                value={customChairmanInput}
                onChange={(e) => { setCustomChairmanInput(e.target.value); setCustomChairmanError(''); }}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addCustomChairman(); } }}
              />
              <button type="button" onClick={addCustomChairman} disabled={!customChairmanInput.trim()}>
                Add
              </button>
              {customChairmanError && <div className="model-picker-custom-error">{customChairmanError}</div>}
            </div>
          </div>

          <div className="model-picker-group">
            <div className="model-picker-group-label">Max tokens (blank = server default)</div>
            <label className="model-picker-option model-picker-token-input">
              Council seats
              <input
                type="number"
                min="1"
                placeholder="8000"
                value={councilMaxTokens}
                onChange={(e) => setCouncilMaxTokens(e.target.value)}
              />
            </label>
            <label className="model-picker-option model-picker-token-input">
              Chairman
              <input
                type="number"
                min="1"
                placeholder="16000"
                value={chairmanMaxTokens}
                onChange={(e) => setChairmanMaxTokens(e.target.value)}
              />
            </label>
            <div className="model-picker-token-note">
              No provider accepts a literal "unlimited" value — this overrides the
              finite per-request cap. Higher values cost more per query and can
              400 if above a model's real max-output window.
            </div>
          </div>
        </div>
      )}
    </div>
  );

  if (!conversation) {
    return (
      <div className="chat-interface">
        {topbar}
        <div className="empty-state">
          <h2>Welcome to LLM Council</h2>
          <p>Create a new conversation to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-interface">
      {topbar}

      <div className="messages-container">
        {conversation.messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Ask a question to consult the LLM Council</p>
          </div>
        ) : (
          conversation.messages.map((msg, index) => (
            <div key={index} className="message-group">
              {msg.role === 'user' ? (
                <div className="user-message">
                  <div className="message-label">You</div>
                  <div className="message-content">
                    <div className="markdown-content">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="assistant-message">
                  <div className="message-label">LLM Council</div>

                  {msg.error && (
                    <div className="stage-loading" style={{ color: '#c0392b' }}>
                      <span>⚠ {msg.error}</span>
                    </div>
                  )}

                  {/* Stage 1 */}
                  {msg.loading?.stage1 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 1: Collecting individual responses...</span>
                    </div>
                  )}
                  {msg.stage1 && <Stage1 responses={msg.stage1} />}

                  {/* Stage 2 */}
                  {msg.loading?.stage2 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 2: Peer rankings...</span>
                    </div>
                  )}
                  {msg.stage2 && (
                    <Stage2
                      rankings={msg.stage2}
                      labelToModel={msg.metadata?.label_to_model}
                      aggregateRankings={msg.metadata?.aggregate_rankings}
                    />
                  )}

                  {/* Stage 3 */}
                  {msg.loading?.stage3 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 3: Final synthesis...</span>
                    </div>
                  )}
                  {msg.stage3 && <Stage3 finalResponse={msg.stage3} />}

                  {msg.stage3 && (
                    <button
                      type="button"
                      className="download-report-button"
                      onClick={() => downloadReport(msg, conversation.messages[index - 1]?.content)}
                    >
                      ⬇ Download report (.md)
                    </button>
                  )}

                  {/* Real (not estimated) cost — computed from actual token usage
                      the instant Stage 3 finishes, not the pre-send worst-case
                      guess from the Estimate button. The "calculating" gap only
                      applies to the message currently streaming in — older,
                      already-loaded messages either have actual_cost persisted
                      or predate this feature entirely (nothing to wait for). */}
                  {msg.stage3 && !msg.actual_cost && isLoading && index === conversation.messages.length - 1 && (
                    <div className="actual-cost-banner actual-cost-pending">Calculating actual cost…</div>
                  )}
                  {msg.actual_cost && (
                    <div className="actual-cost-banner">
                      Actual cost: ${msg.actual_cost.actual_cost_usd.toFixed(4)}
                      {msg.actual_cost.actual_cost_usd === 0 && ' (all seats served free / NVIDIA)'}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Consulting the council...</span>
            <button
              type="button"
              className="stop-button"
              onClick={onStopMessage}
              title="Cancel this run — stops the backend from making further model calls too"
            >
              Stop
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {conversation.messages.length === 0 && (
        <form className="input-form-wrapper" onSubmit={handleSubmit}>
          {attachments.length > 0 && (
            <div className="attachment-chips">
              {attachments.map((a) => (
                <span key={a.name} className={`attachment-chip${a.uploading ? ' attachment-chip-uploading' : ''}`}>
                  {a.name}
                  {a.uploading && <span className="attachment-chip-status"> — extracting…</span>}
                  <button
                    type="button"
                    className="attachment-chip-remove"
                    onClick={() => removeAttachment(a.name)}
                    disabled={isLoading}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="input-form">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              multiple
              style={{ display: 'none' }}
            />
            <button
              type="button"
              className="attach-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              title="Attach file(s)"
            >
              📎
            </button>
            <textarea
              className="message-input"
              placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={3}
            />
            <button
              type="button"
              className="estimate-button"
              onClick={handleEstimateCost}
              disabled={(!input.trim() && attachments.length === 0) || isLoading || estimating || attachmentsUploading}
              title="Estimate worst-case cost before sending"
            >
              {estimating ? '…' : '💲 Estimate'}
            </button>
            <button
              type="submit"
              className="send-button"
              disabled={(!input.trim() && attachments.length === 0) || isLoading || attachmentsUploading}
            >
              Send
            </button>
          </div>

          {costEstimate && (
            <div className="cost-estimate-row">
              {costEstimate.error ? (
                <span className="cost-estimate-error">Could not estimate cost.</span>
              ) : (
                <>
                  <span className="cost-estimate-total">
                    Estimated max cost: ${costEstimate.estimated_cost_usd.toFixed(4)}
                  </span>
                  <span className="cost-estimate-note">{costEstimate.note}</span>
                </>
              )}
            </div>
          )}
        </form>
      )}
    </div>
  );
}
