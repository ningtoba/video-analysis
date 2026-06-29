/**
 * Video Analysis Platform — Application Shell
 *
 * Alpine.js application for tab switching, global state, toast notifications,
 * and WebSocket-based job progress tracking.
 */
document.addEventListener('alpine:init', () => {
  Alpine.data('appState', () => ({
    activeTab: 'analysis',
    videoId: '',
    videoPath: '',
    busy: false,
    version: '',

    init() {
      // Restore last active tab from sessionStorage
      const saved = sessionStorage.getItem('va_active_tab');
      if (saved && this.tabExists(saved)) {
        this.activeTab = saved;
      }

      // Listen for global events
      window.addEventListener('va:switchTab', (e) => {
        this.switchTab(e.detail);
      });

      // Load version from footer or meta
      const verEl = document.querySelector('[data-version]');
      if (verEl) this.version = verEl.dataset.version;
    },

    tabExists(name) {
      return ['analysis', 'import', 'batch', 'search', 'library',
              'camera', 'monitor', 'comparison', 'kg', 'events', 'settings'].includes(name);
    },

    switchTab(name) {
      if (!this.tabExists(name)) return;
      this.activeTab = name;
      sessionStorage.setItem('va_active_tab', name);
      // Load tab content lazily on first visit via HTMX
      const container = document.querySelector(`[data-tab="${name}"]`);
      if (container && container.dataset.loaded === 'false') {
        htmx.trigger(container, 'loadTab');
      }
    },

    // Toast notifications
    notify(message, type = 'info') {
      window.dispatchEvent(new CustomEvent('new-toast', {
        detail: { id: Date.now(), message, type }
      }));
    },

    // Set processing state (disables buttons that check :disabled="busy")
    setBusy(v) {
      this.busy = v;
    }
  }));
});

/**
 * WebSocket Manager — handles job progress and real-time updates.
 *
 * Usage:
 *   const ws = connectJobWS(jobId, {
 *     onProgress: (data) => { ... },
 *     onComplete: (data) => { ... },
 *     onError: (data) => { ... },
 *   });
 */
window.connectJobWS = function(jobId, callbacks) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/jobs/${jobId}`;
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log(`[WS] Connected to job ${jobId}`);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.status === 'completed' && callbacks.onComplete) {
        callbacks.onComplete(data);
        ws.close();
      } else if (data.status === 'failed' && callbacks.onError) {
        callbacks.onError(data);
        ws.close();
      } else if (callbacks.onProgress) {
        callbacks.onProgress(data);
      }
    } catch (e) {
      console.error('[WS] Parse error:', e);
    }
  };

  ws.onerror = (err) => {
    console.error('[WS] Error:', err);
    if (callbacks.onError) callbacks.onError({ error: 'WebSocket connection failed' });
  };

  ws.onclose = () => {
    console.log(`[WS] Disconnected from job ${jobId}`);
  };

  return ws;
};

/**
 * SSE Chat Streaming client.
 *
 * Usage:
 *   const stream = chatStream('/api/chat/stream', { video_id: 'xxx', query: '...' }, {
 *     onToken: (token) => appendToChat(token),
 *     onDone: () => finalize(),
 *     onError: (err) => showError(err),
 *   });
 */
window.chatStream = function(url, body, callbacks) {
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(async (response) => {
    if (!response.ok) {
      const err = await response.json();
      if (callbacks.onError) callbacks.onError(err.detail || 'Chat error');
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            if (callbacks.onDone) callbacks.onDone();
            return;
          }
          try {
            const parsed = JSON.parse(data);
            if (parsed.token && callbacks.onToken) {
              callbacks.onToken(parsed.token);
            } else if (parsed.error && callbacks.onError) {
              callbacks.onError(parsed.error);
            }
          } catch {
            // Plain text token (non-JSON SSE)
            if (callbacks.onToken) callbacks.onToken(data);
          }
        }
      }
    }
    if (callbacks.onDone) callbacks.onDone();
  }).catch((err) => {
    if (callbacks.onError) callbacks.onError(err.message);
  });
};

/**
 * Format seconds to HH:MM:SS.mmm timestamp.
 */
window.formatTimestamp = function(seconds) {
  if (seconds == null) return '--:--:--';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = (seconds % 60).toFixed(1);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(4, '0')}`;
};

/**
 * Simple markdown-to-HTML (handles **bold**, *italic*, `code`, newlines → <br>).
 */
window.simpleMarkdown = function(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
};
