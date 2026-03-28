/**
 * Client-side adaptive jitter buffer.
 *
 * Listens on the LiveKit DataChannel for DubYou sync packets and adjusts
 * dubbed audio playback timing to prevent crackling during fast speech.
 *
 * Usage (inside your LiveKit JS/TS client):
 *
 *   import { AdaptiveJitterBuffer } from './sync_buffer.js';
 *   const jitter = new AdaptiveJitterBuffer();
 *   room.on(RoomEvent.DataReceived, (data) => jitter.onData(data));
 */

export class AdaptiveJitterBuffer {
  constructor({ minBufferMs = 80, maxBufferMs = 160, targetBufferMs = 120 } = {}) {
    this.minBufferMs    = minBufferMs;
    this.maxBufferMs    = maxBufferMs;
    this.targetBufferMs = targetBufferMs;
    this._history       = [];   // last N sync packets
    this._maxHistory    = 10;
  }

  /** Call this with every DataChannel message from the room. */
  onData(rawData) {
    let pkt;
    try {
      const text = typeof rawData === 'string'
        ? rawData
        : new TextDecoder().decode(rawData);
      pkt = JSON.parse(text);
    } catch {
      return;  // not a sync packet
    }
    if (pkt.v !== 1) return;

    this._history.push(pkt);
    if (this._history.length > this._maxHistory) {
      this._history.shift();
    }

    this.targetBufferMs = this._computeBuffer();
  }

  /**
   * Returns the current recommended buffer size in ms.
   * AudioWorklet / Web Audio nodes should query this when scheduling playback.
   */
  get bufferMs() {
    return this.targetBufferMs;
  }

  _computeBuffer() {
    if (this._history.length < 3) return this.targetBufferMs;

    // Estimate speech rate (ms of speech per packet)
    const recent = this._history.slice(-5);
    const avgChunkMs = recent.reduce(
      (sum, p) => sum + (p.orig_end_ms - p.orig_start_ms), 0
    ) / recent.length;

    // Fast speech → shorter buffer (less latency)
    // Slow speech → longer buffer (more stability)
    if (avgChunkMs < 400) return this.minBufferMs;       // fast speaker
    if (avgChunkMs > 900) return this.maxBufferMs;       // slow speaker
    return this.targetBufferMs;                           // normal pace
  }
}
