function getLMStudioNativeBaseURL(compatibilityBaseURL) {
  const url = new URL(compatibilityBaseURL);
  const pathname = url.pathname.replace(/\/+$/, '');
  if (!pathname.endsWith('/api/v1')) {
    url.pathname = `${pathname.replace(/\/v1$/i, '')}/api/v1`;
  } else {
    url.pathname = pathname;
  }
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/+$/, '');
}

class LMStudioService {
  constructor({ baseURL, apiKey = '' }) {
    this.baseURL = getLMStudioNativeBaseURL(baseURL);
    this.apiKey = apiKey;
  }

  async request(endpoint, { method = 'GET', body, timeoutMs = 30000 } = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const headers = {};
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    if (body !== undefined) headers['Content-Type'] = 'application/json';

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method,
        headers,
        ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
        signal: controller.signal,
        redirect: 'error'
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = payload.error?.message || payload.error || response.statusText;
        throw new Error(`LM Studio returned HTTP ${response.status}: ${detail}`);
      }
      return payload;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Timed out connecting to LM Studio.');
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  listModels() {
    return this.request('/models', { timeoutMs: 10000 });
  }

  loadModel(model, options = {}) {
    return this.request('/models/load', {
      method: 'POST',
      body: { model, ...options },
      timeoutMs: 300000
    });
  }

  unloadModel(instanceId) {
    return this.request('/models/unload', {
      method: 'POST',
      body: { instance_id: instanceId },
      timeoutMs: 120000
    });
  }

  downloadModel(model, quantization) {
    const body = { model };
    if (quantization) body.quantization = quantization;
    return this.request('/models/download', {
      method: 'POST',
      body,
      timeoutMs: 30000
    });
  }

  getDownloadStatus(jobId) {
    return this.request(`/models/download/status/${encodeURIComponent(jobId)}`);
  }

  async generateText({ model, input, maxOutputTokens, temperature }) {
    const payload = await this.request('/chat', {
      method: 'POST',
      body: {
        model,
        input,
        max_output_tokens: maxOutputTokens,
        temperature: Math.min(1, Math.max(0, temperature)),
        store: false
      },
      timeoutMs: 300000
    });
    const text = Array.isArray(payload.output)
      ? payload.output.filter(item => item?.type === 'message').map(item => item.content || '').join('\n').trim()
      : '';
    if (!text) throw new Error('LM Studio returned no text message in its chat response.');
    return text;
  }
}

module.exports = { LMStudioService, getLMStudioNativeBaseURL };
