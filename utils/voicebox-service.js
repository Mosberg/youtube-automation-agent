const axios = require('axios');
const FormData = require('form-data');

const DEFAULT_VOICEBOX_URL = 'http://127.0.0.1:8000';
const VOICEBOX_LANGUAGES = [
  ['zh', 'Chinese'], ['en', 'English'], ['ja', 'Japanese'], ['ko', 'Korean'],
  ['de', 'German'], ['fr', 'French'], ['ru', 'Russian'], ['pt', 'Portuguese'],
  ['es', 'Spanish'], ['it', 'Italian'], ['he', 'Hebrew'], ['ar', 'Arabic'],
  ['da', 'Danish'], ['el', 'Greek'], ['fi', 'Finnish'], ['hi', 'Hindi'],
  ['ms', 'Malay'], ['nl', 'Dutch'], ['no', 'Norwegian'], ['pl', 'Polish'],
  ['sv', 'Swedish'], ['sw', 'Swahili'], ['tr', 'Turkish']
];

const VOICEBOX_TTS_ENGINES = [
  { id: 'qwen', name: 'Qwen TTS · voice cloning', modelIds: ['qwen-tts-1.7B', 'qwen-tts-0.6B'], modelSizes: ['1.7B', '0.6B'], languages: ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it'], profileType: 'cloned' },
  { id: 'qwen_custom_voice', name: 'Qwen CustomVoice · preset voices', modelIds: ['qwen-custom-voice-1.7B', 'qwen-custom-voice-0.6B'], modelSizes: ['1.7B', '0.6B'], languages: ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it'], profileType: 'preset' },
  { id: 'qwen_voice_design', name: 'Qwen VoiceDesign · designed voices', modelIds: ['qwen-voice-design-1.7B'], modelSizes: ['1.7B'], languages: ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it'], profileType: 'designed' },
  { id: 'chatterbox', name: 'Chatterbox Multilingual · cloning', modelIds: ['chatterbox-tts'], modelSizes: [], languages: VOICEBOX_LANGUAGES.map(([code]) => code), profileType: 'cloned' },
  { id: 'chatterbox_turbo', name: 'Chatterbox Turbo · fast English', modelIds: ['chatterbox-turbo'], modelSizes: [], languages: ['en'], profileType: 'cloned' },
  { id: 'luxtts', name: 'LuxTTS · fast English', modelIds: ['luxtts'], modelSizes: [], languages: ['en'], profileType: 'cloned' },
  { id: 'kokoro', name: 'Kokoro 82M · lightweight presets', modelIds: ['kokoro'], modelSizes: [], languages: ['en', 'es', 'fr', 'hi', 'it', 'pt', 'ja', 'zh'], profileType: 'preset' },
  { id: 'tada', name: 'TADA · multilingual cloning', modelIds: ['tada-1b', 'tada-3b-ml'], modelSizes: ['1B', '3B'], languages: ['en', 'ar', 'zh', 'de', 'es', 'fr', 'it', 'ja', 'pl', 'pt'], profileType: 'cloned' },
  { id: 'omnivoice', name: 'OmniVoice · multilingual cloning', modelIds: ['omnivoice'], modelSizes: [], languages: VOICEBOX_LANGUAGES.map(([code]) => code), profileType: 'cloned', nonCommercial: true }
];

const VOICEBOX_WHISPER_MODELS = [
  { id: 'whisper-base', value: 'base', name: 'Whisper Base' },
  { id: 'whisper-small', value: 'small', name: 'Whisper Small' },
  { id: 'whisper-medium', value: 'medium', name: 'Whisper Medium' },
  { id: 'whisper-large', value: 'large', name: 'Whisper Large v3 · best accuracy' },
  { id: 'whisper-turbo', value: 'turbo', name: 'Whisper Large v3 Turbo · fast' }
];

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

  async getModelStatus() {
    try {
      const response = await axios.get(`${this.baseUrl}/models/status`, {
        timeout: this.timeoutMs || 30000
      });
      if (!Array.isArray(response.data?.models)) {
        throw new Error('Voicebox returned an invalid model status list');
      }
      return response.data.models;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not list Voicebox models at ${this.baseUrl}: ${detail || error.message}`);
    }
  }

  async listPresetVoices(engine) {
    try {
      const response = await axios.get(`${this.baseUrl}/profiles/presets/${encodeURIComponent(engine)}`, {
        timeout: this.timeoutMs || 10000
      });
      return Array.isArray(response.data?.voices) ? response.data.voices : [];
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not list ${engine} preset voices: ${detail || error.message}`);
    }
  }

  async createPresetProfile(profile) {
    try {
      const response = await axios.post(`${this.baseUrl}/profiles`, profile, {
        timeout: this.timeoutMs || 15000
      });
      return response.data;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not create Voicebox preset profile: ${detail || error.message}`);
    }
  }

  async loadModel(modelName) {
    try {
      const response = await axios.post(`${this.baseUrl}/models/load`, { model_name: modelName }, {
        timeout: this.timeoutMs || 0
      });
      return response.data;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not load Voicebox model ${modelName}: ${detail || error.message}`);
    }
  }

  async unloadModel(modelName) {
    try {
      const response = await axios.post(`${this.baseUrl}/models/unload`, { model_name: modelName }, {
        timeout: this.timeoutMs || 15000
      });
      return response.data;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not unload Voicebox model ${modelName}: ${detail || error.message}`);
    }
  }

  async downloadModel(modelName) {
    try {
      const response = await axios.post(`${this.baseUrl}/models/download`, { model_name: modelName }, {
        timeout: this.timeoutMs || 15000
      });
      return response.data;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Could not download Voicebox model ${modelName}: ${detail || error.message}`);
    }
  }

  async transcribeAudio(audio, filename, options = {}) {
    const form = new FormData();
    form.append('file', Buffer.from(audio), {
      filename: filename || 'audio.wav',
      contentType: options.contentType || 'application/octet-stream'
    });
    if (options.model) form.append('model', options.model);
    if (options.language) form.append('language', options.language);

    try {
      const response = await axios.post(`${this.baseUrl}/transcribe`, form, {
        headers: form.getHeaders(),
        maxBodyLength: 200 * 1024 * 1024,
        maxContentLength: 2 * 1024 * 1024,
        timeout: this.timeoutMs || 0
      });
      if (typeof response.data?.text !== 'string') {
        throw new Error('Voicebox returned an invalid transcription response');
      }
      return response.data;
    } catch (error) {
      const detail = responseErrorDetail(error.response?.data);
      throw new Error(`Voicebox transcription failed: ${detail || error.message}`);
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

module.exports = {
  VoiceboxService,
  DEFAULT_VOICEBOX_URL,
  normalizeVoiceboxUrl,
  VOICEBOX_LANGUAGES,
  VOICEBOX_TTS_ENGINES,
  VOICEBOX_WHISPER_MODELS
};
