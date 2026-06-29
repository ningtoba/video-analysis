/**
 * Timeline Hover Preview — Sprite-sheet based thumbnail preview for
 * HTML5 video players.  Ported from the Gradio UI's custom JavaScript
 * (ui/app.py lines 1376–1609).
 *
 * Usage:
 *   <video id="main-video-player" ...></video>
 *   setupTimelinePreview(videoEl, videoId, spritesUrl);
 *
 * Creates a fixed-position preview div that shows the corresponding
 * sprite tile when the user hovers near the bottom of the video.
 */
(function () {
  'use strict';

  let _previewEl = null;
  let _videoObserver = null;
  let _spritesCache = {};

  /**
   * Create the preview DOM element (once, reused for all videos).
   */
  function _ensurePreview() {
    if (_previewEl) return _previewEl;
    _previewEl = document.createElement('div');
    _previewEl.id = 'timeline-preview';
    _previewEl.style.cssText = [
      'position: fixed;',
      'pointer-events: none;',
      'z-index: 9999;',
      'display: none;',
      'width: 160px;',
      'height: 90px;',
      'border: 2px solid var(--primary);',
      'border-radius: var(--radius-sm);',
      'overflow: hidden;',
      'background: var(--bg);',
      'box-shadow: var(--shadow-lg);',
      'background-size: cover;',
      'background-position: center;',
      'image-rendering: auto;',
    ].join('');
    document.body.appendChild(_previewEl);
    return _previewEl;
  }

  /**
   * Load sprite metadata for a video.
   */
  async function _loadSprites(videoId) {
    if (_spritesCache[videoId]) return _spritesCache[videoId];

    // Try multiple paths
    const urls = [
      `/data/thumbnails/${videoId}_sprite.json`,
      `/file=data/thumbnails/${videoId}_sprite.json`,
    ];

    for (const url of urls) {
      try {
        const resp = await fetch(url);
        if (resp.ok) {
          const data = await resp.json();
          _spritesCache[videoId] = data;
          return data;
        }
      } catch (e) {
        // try next URL
      }
    }
    return null;
  }

  /**
   * Find the sprite tile URL for a given timestamp.
   */
  function _findTile(sprites, seconds) {
    if (!sprites || !sprites.tiles) return null;
    const tiles = sprites.tiles;
    // Binary search for the tile covering this timestamp
    let lo = 0, hi = tiles.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const tile = tiles[mid];
      if (seconds >= tile.start && seconds <= tile.end) {
        return tile;
      }
      if (seconds < tile.start) {
        hi = mid - 1;
      } else {
        lo = mid + 1;
      }
    }
    return null;
  }

  /**
   * Main entry point: attach timeline preview to a video element.
   */
  window.setupTimelinePreview = function (videoEl, videoId, spritesMeta) {
    const preview = _ensurePreview();
    let sprites = spritesMeta || null;
    let loadedVideoId = null;

    videoEl.addEventListener('mousemove', async function (e) {
      const rect = videoEl.getBoundingClientRect();
      const relY = e.clientY - rect.top;
      const relX = e.clientX - rect.left;

      // Only activate in the bottom 15% of the video (timeline area)
      const timelineZone = rect.height * 0.85;
      if (relY < timelineZone || !videoEl.duration) {
        preview.style.display = 'none';
        return;
      }

      const fraction = relX / rect.width;
      const timestamp = fraction * videoEl.duration;

      // Load sprites on first hover
      if (videoId && videoId !== loadedVideoId) {
        sprites = await _loadSprites(videoId);
        loadedVideoId = videoId;
      }

      const tile = _findTile(sprites, timestamp);
      if (tile && tile.url) {
        preview.style.display = 'block';
        preview.style.backgroundImage = `url(${tile.url})`;
        preview.style.backgroundPosition = `-${tile.x || 0}px -${tile.y || 0}px`;
        preview.style.left = `${e.clientX - 80}px`;
        preview.style.top = `${rect.top - 100}px`;
      } else {
        preview.style.display = 'none';
      }
    });

    videoEl.addEventListener('mouseleave', function () {
      preview.style.display = 'none';
    });
  };

  /**
   * Auto-detect video elements and attach preview.
   * Surveys every 2s for new video elements (handles lazy tab rendering).
   */
  function _autoSetup() {
    const videos = document.querySelectorAll('video[id]');
    videos.forEach(function (video) {
      if (video.dataset.timelineReady) return;
      video.dataset.timelineReady = '1';

      // Try to extract videoId from src
      const src = video.src || video.querySelector('source')?.src || '';
      const match = src.match(/\/data\/videos\/([^/]+)\.mp4/);
      const videoId = match ? match[1] : null;

      setupTimelinePreview(video, videoId);
    });
  }

  // Survey on load and periodically
  document.addEventListener('DOMContentLoaded', _autoSetup);
  setInterval(_autoSetup, 2000);

  // Also watch for Alpine.js tab switches
  document.addEventListener('alpine:initialized', function () {
    setTimeout(_autoSetup, 500);
  });
})();
