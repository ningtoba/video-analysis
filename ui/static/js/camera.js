/**
 * Camera Tab — Webcam capture using getUserMedia API.
 * Ported from the Gradio Camera tab (ui/camera.py).
 */

(function () {
  'use strict';

  let _stream = null;
  let _autoInterval = null;

  window.startCamera = async function (videoEl, deviceId) {
    try {
      if (_stream) {
        _stream.getTracks().forEach(t => t.stop());
      }

      const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
        audio: false,
      };

      _stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = _stream;
      await videoEl.play();
      return { ok: true };
    } catch (err) {
      console.error('Camera error:', err);
      return { ok: false, error: err.message };
    }
  };

  window.stopCamera = function () {
    if (_stream) {
      _stream.getTracks().forEach(t => t.stop());
      _stream = null;
    }
    if (_autoInterval) {
      clearInterval(_autoInterval);
      _autoInterval = null;
    }
  };

  window.captureFrame = function (videoEl) {
    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  };

  window.startAutoAnalysis = function (intervalSec, captureFn) {
    if (_autoInterval) clearInterval(_autoInterval);
    _autoInterval = setInterval(captureFn, intervalSec * 1000);
    return true;
  };

  window.stopAutoAnalysis = function () {
    if (_autoInterval) {
      clearInterval(_autoInterval);
      _autoInterval = null;
    }
    return false;
  };

  // Enumerate available cameras
  window.listCameras = async function () {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter(d => d.kind === 'videoinput')
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i}` }));
    } catch {
      return [];
    }
  };
})();
