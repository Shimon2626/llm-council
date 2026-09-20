import { useState, useEffect } from 'react';
import { api } from '../api';
import './SetupModal.css';

/**
 * "Setup" panel — lets you view whether each provider's API key is
 * configured and paste in a new one, instead of hand-editing .env.
 * Keys are write-only from this UI: the backend never sends a saved key
 * back, only a configured/not-configured boolean (config.key_status()).
 */
export default function SetupModal({ onClose }) {
  const [status, setStatus] = useState(null);
  const [nvidiaKey, setNvidiaKey] = useState('');
  const [openrouterKey, setOpenrouterKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      const s = await api.getConfigStatus();
      setStatus(s);
    } catch (error) {
      console.error('Failed to load config status:', error);
      setMessage({ type: 'error', text: 'Could not reach the backend to check key status.' });
    }
  };

  const handleSave = async () => {
    if (!nvidiaKey && !openrouterKey) {
      setMessage({ type: 'error', text: 'Enter at least one key to save.' });
      return;
    }
    setSaving(true);
    setMessage(null);
    try {
      const newStatus = await api.saveApiKeys(nvidiaKey, openrouterKey);
      setStatus(newStatus);
      setNvidiaKey('');
      setOpenrouterKey('');
      setMessage({ type: 'success', text: 'Saved — takes effect immediately, no restart needed.' });
    } catch (error) {
      console.error('Failed to save API keys:', error);
      setMessage({ type: 'error', text: 'Failed to save keys — check the backend is running.' });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="setup-modal-overlay" onClick={onClose}>
      <div className="setup-modal" onClick={(e) => e.stopPropagation()}>
        <div className="setup-modal-header">
          <h2>Setup</h2>
          <button className="setup-modal-close" onClick={onClose}>×</button>
        </div>

        <p className="setup-modal-intro">
          Keys are stored in <code>.env</code> on this machine and read by the
          backend only — they're never sent back to this screen once saved.
          You only need one provider to run the council; both gives automatic
          fallback if a model errors out on its primary provider.
        </p>

        <div className="setup-key-row">
          <label htmlFor="nvidia-key">
            NVIDIA_API_KEY{' '}
            <span className={`setup-status-badge ${status?.nvidia ? 'configured' : 'not-configured'}`}>
              {status === null ? '…' : status.nvidia ? 'configured' : 'not set'}
            </span>
          </label>
          <input
            id="nvidia-key"
            type="password"
            placeholder={status?.nvidia ? '•••••••••••••• (leave blank to keep)' : 'nvapi-...'}
            value={nvidiaKey}
            onChange={(e) => setNvidiaKey(e.target.value)}
          />
          <a href="https://build.nvidia.com/" target="_blank" rel="noreferrer">
            Get a free NVIDIA key →
          </a>
        </div>

        <div className="setup-key-row">
          <label htmlFor="openrouter-key">
            OPENROUTER_API_KEY{' '}
            <span className={`setup-status-badge ${status?.openrouter ? 'configured' : 'not-configured'}`}>
              {status === null ? '…' : status.openrouter ? 'configured' : 'not set'}
            </span>
          </label>
          <input
            id="openrouter-key"
            type="password"
            placeholder={status?.openrouter ? '•••••••••••••• (leave blank to keep)' : 'sk-or-...'}
            value={openrouterKey}
            onChange={(e) => setOpenrouterKey(e.target.value)}
          />
          <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer">
            Get an OpenRouter key →
          </a>
        </div>

        {message && (
          <div className={`setup-message ${message.type}`}>{message.text}</div>
        )}

        <div className="setup-modal-actions">
          <button className="setup-cancel-btn" onClick={onClose}>
            Close
          </button>
          <button className="setup-save-btn" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
