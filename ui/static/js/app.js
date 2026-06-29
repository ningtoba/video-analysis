/** Video Analysis — App Shell */
document.addEventListener('alpine:init', () => {
  Alpine.data('appState', () => ({
    activeTab: 'analyze',
    videoId: '',
    videoPath: '',
    busy: false,
    init() {
      const saved = sessionStorage.getItem('va_tab');
      if (saved && ['analyze','library','camera','settings'].includes(saved)) this.activeTab = saved;
      this.version = document.querySelector('[data-version]')?.dataset?.version || '';
    },
    switchTab(name) {
      if (!['analyze','library','camera','settings'].includes(name)) return;
      this.activeTab = name;
      sessionStorage.setItem('va_tab', name);
    },
    notify(msg, type = 'info') {
      window.dispatchEvent(new CustomEvent('new-toast', { detail: { id: Date.now(), message: msg, type } }));
    },
    onVideoReady(detail) {
      this.videoId = detail.videoId;
      this.videoPath = detail.videoPath || '';
      this.busy = false;
      this.switchTab('analyze');
    }
  }));
  window.addEventListener('va:videoReady', (e) => {
    const app = document.querySelector('[x-data]').__x.$data;
    app.onVideoReady(e.detail);
  });
});

/** WebSocket job progress */
window.connectJobWS = function(jobId, callbacks) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/jobs/${jobId}`);
  ws.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.status === 'completed') { callbacks.onComplete?.(d); ws.close(); }
      else if (d.status === 'failed') { callbacks.onError?.(d); ws.close(); }
      else callbacks.onProgress?.(d);
    } catch(ex) { console.error('[WS]', ex); }
  };
  ws.onerror = () => callbacks.onError?.({ error: 'Connection failed' });
  return ws;
};

/** Simple markdown */
window.md = (t) => t ? t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\*(.+?)\*/g,'<em>$1</em>').replace(/`(.+?)`/g,'<code>$1</code>').replace(/\n/g,'<br>') : '';

/** Format seconds */
window.fmtTs = (s) => { if (s == null) return '--:--'; const h=Math.floor(s/3600), m=Math.floor((s%3600)/60); return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${(s%60).toFixed(0).padStart(2,'0')}`; };
