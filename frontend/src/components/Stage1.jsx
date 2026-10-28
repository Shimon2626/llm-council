import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage1.css';

const STATUS_LABEL = { empty: 'EMPTY', error: 'FAILED', truncated: 'CUT OFF' };

export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  return (
    <div className="stage stage1">
      <h3 className="stage-title">Stage 1: Individual Responses</h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {resp.model.split('/')[1] || resp.model}
            {STATUS_LABEL[resp.status] && <span className="tab-status"> ⚠ {STATUS_LABEL[resp.status]}</span>}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">{responses[activeTab].model}</div>
        {STATUS_LABEL[responses[activeTab].status] && (
          <div className="response-warning">
            ⚠ {STATUS_LABEL[responses[activeTab].status]}: finish_reason={String(responses[activeTab].finish_reason)},
            {' '}{responses[activeTab].completion_tokens} completion tokens
            ({responses[activeTab].reasoning_tokens || 0} reasoning). This seat is not a complete review.
          </div>
        )}
        <div className="response-text markdown-content">
          <ReactMarkdown>{responses[activeTab].response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
