/**
 * API client for the LLM Council backend.
 */

const API_BASE = 'http://localhost:8001';

export const api = {
  /**
   * List selectable models per provider (nvidia / openrouter).
   */
  async getAvailableModels() {
    const response = await fetch(`${API_BASE}/api/models`);
    if (!response.ok) {
      throw new Error('Failed to list available models');
    }
    return response.json();
  },

  /**
   * Permanently add a custom-typed model slug (from the model picker's free-
   * typed "Add" field) so it shows up as a normal option in every future
   * session instead of only in this browser tab's in-memory state. Returns
   * the updated { nvidia: [...], openrouter: [...] } list.
   */
  async addCustomModel(provider, model) {
    const response = await fetch(`${API_BASE}/api/models/custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, model }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Failed to save custom model');
    }
    return response.json();
  },

  /**
   * Full live OpenRouter model catalog (cached server-side), used to
   * validate a free-typed custom model slug before it's added as a seat.
   */
  async getOpenRouterCatalog() {
    const response = await fetch(`${API_BASE}/api/openrouter/models`);
    if (!response.ok) {
      throw new Error('Failed to fetch OpenRouter catalog');
    }
    return response.json();
  },

  /**
   * Estimate the worst-case USD cost of a council run before sending it.
   */
  async estimateCost(content, councilModels, chairmanModel) {
    const response = await fetch(`${API_BASE}/api/estimate-cost`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content,
        council_models: councilModels,
        chairman_model: chairmanModel,
      }),
    });
    if (!response.ok) {
      throw new Error('Failed to estimate cost');
    }
    return response.json();
  },

  /**
   * Whether each provider's API key is currently configured (booleans
   * only — the Setup modal never receives the actual key value back).
   */
  async getConfigStatus() {
    const response = await fetch(`${API_BASE}/api/config/status`);
    if (!response.ok) {
      throw new Error('Failed to fetch config status');
    }
    return response.json();
  },

  /**
   * Save one or both provider API keys from the Setup modal. Either can
   * be omitted — an empty field never clears an already-configured key.
   */
  async saveApiKeys(nvidiaApiKey, openrouterApiKey) {
    const response = await fetch(`${API_BASE}/api/config/keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nvidia_api_key: nvidiaApiKey || null,
        openrouter_api_key: openrouterApiKey || null,
      }),
    });
    if (!response.ok) {
      throw new Error('Failed to save API keys');
    }
    return response.json();
  },

  /**
   * Extract readable text from a .zip file's contents server-side (a zip
   * is binary — reading it with FileReader.readAsText, like the other
   * attachment types below, would produce garbage, not usable content).
   */
  async extractZip(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/api/extract-zip`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Failed to extract zip file');
    }
    return response.json();
  },

  /**
   * Extract readable text from a .docx file's contents server-side (like
   * .zip above, a .docx is binary — FileReader.readAsText would return the
   * raw zip bytes, not the document's actual text).
   */
  async extractDocx(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/api/extract-docx`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Failed to extract docx file');
    }
    return response.json();
  },

  /**
   * List all conversations.
   */
  async listConversations() {
    const response = await fetch(`${API_BASE}/api/conversations`);
    if (!response.ok) {
      throw new Error('Failed to list conversations');
    }
    return response.json();
  },

  /**
   * Create a new conversation.
   */
  async createConversation() {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      throw new Error('Failed to create conversation');
    }
    return response.json();
  },

  /**
   * Get a specific conversation.
   */
  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`
    );
    if (!response.ok) {
      throw new Error('Failed to get conversation');
    }
    return response.json();
  },

  /**
   * Send a message in a conversation.
   */
  async sendMessage(conversationId, content) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    return response.json();
  },

  /**
   * Send a message and receive streaming updates.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback function for each event: (eventType, data) => void
   * @returns {Promise<void>}
   */
  async sendMessageStream(conversationId, content, onEvent, councilModels, chairmanModel, councilMaxTokens, chairmanMaxTokens, signal) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          council_models: councilModels,
          chairman_model: chairmanModel,
          council_max_tokens: councilMaxTokens,
          chairman_max_tokens: chairmanMaxTokens,
        }),
        signal,
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }
  },
};
