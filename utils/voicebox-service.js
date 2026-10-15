const axios = require('axios');

const DEFAULT_VOICEBOX_URL = 'http://127.0.0.1:8000';

function normalizeVoiceboxUrl(value = DEFAULT_VOICEBOX_URL) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch (error) {
    throw new Error('Voicebox URL must be a valid HTTP or HTTPS URL');
  }

  if (
    !['http:', 'https:'].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error('Voicebox URL must use HTTP or HTTPS and must not contain credentials, a query, or a fragment');
  }

  return parsed.toString().replace(/\/+$/, '');
}

function responseErrorDetail(data) {
  if (Buffer.isBuffer(data)) {
    const text = data.toString('utf8');
    try {
      return JSON.parse(text).detail || text;
    } catch (error) {
      return text;
    }
  }
  return data?.detail || data?.message || '';
}

class VoiceboxService {
  constructor(options = {}) {
    this.baseUrl = normalizeVoiceboxUrl(options.baseUrl);
    const timeoutMs = Number(options.timeoutMs);
    this.timeoutMs = Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 0;
  }

  async getHealth() {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, { timeout: 5000 });
      return response.data;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not connect to Voicebox at ${this.baseUrl}: ${detail || error.message}`);
    }
  }

  async listProfiles() {
    try {
      const response = await axios.get(`${this.baseUrl}/profiles`, { timeout: this.timeoutMs });
      if (!Array.isArray(response.data)) {
        throw new Error('Voicebox returned an invalid profile list');
      }
      return response.data;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not list Voicebox profiles at ${this.baseUrl}: ${detail || error.message}`);
    }
  }

  async generateSpeech(request) {
    let response;
    try {
      response = await axios.post(`${this.baseUrl}/generate/stream`, request, {
        headers: { Accept: 'audio/wav' },
        responseType: 'arraybuffer',
        timeout: this.timeoutMs
      });
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Voicebox speech generation failed: ${detail || error.message}`);
    }

    const audio = Buffer.from(response.data);
    if (audio.length < 44 || audio.toString('ascii', 0, 4) !== 'RIFF' || audio.toString('ascii', 8, 12) !== 'WAVE') {
      throw new Error('Voicebox returned invalid audio; expected a WAV stream from /generate/stream');
    }
    return audio;
  }
}

module.exports = { VoiceboxService, DEFAULT_VOICEBOX_URL, normalizeVoiceboxUrl };
