const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const { Logger } = require('./logger');
const { LMStudioService } = require('./lm-studio-service');

const GEMINI_MODELS = [
  'gemini-3.7-flash',
  'gemini-3.1-pro-preview',
  'gemini-3.5-flash-lite',
];
const GEMINI_DEFAULT_MODEL = GEMINI_MODELS[0];
// Floor for the single retry after a thinking model exhausts a small budget.
const GEMINI_THINKING_RETRY_MIN_TOKENS = 1024;

const PROVIDERS = {
  openai: {
    name: 'OpenAI',
    baseURL: 'https://api.openai.com/v1',
    defaultModel: 'gpt-5.6',
    models: ['gpt-5.6', 'gpt-5.6-terra', 'gpt-5.6-luna'],
    envKey: 'OPENAI_API_KEY',
  },
  openrouter: {
    name: 'OpenRouter',
    baseURL: 'https://openrouter.ai/api/v1',
    defaultModel: 'openai/gpt-5.6-sol',
    models: ['openai/gpt-5.6-sol', 'anthropic/claude-fable-5', 'google/gemini-3.7-flash', 'moonshotai/kimi-k3', 'z-ai/glm-5.3'],
    envKey: 'OPENROUTER_API_KEY',
  },
  kimi: {
    name: 'Kimi (Moonshot AI)',
    baseURL: 'https://api.moonshot.ai/v1',
    defaultModel: 'kimi-k3',
    models: ['kimi-k3', 'kimi-k2.7-code', 'kimi-k2.6'],
    envKey: 'MOONSHOT_API_KEY',
  },
  mimo: {
    name: 'MiMo (Xiaomi)',
    baseURL: 'https://api.xiaomimimo.com/v1',
    defaultModel: 'mimo-v2.5-pro',
    models: ['mimo-v2.5-pro', 'mimo-v2.5'],
    envKey: 'MIMO_API_KEY',
  },
  glm: {
    name: 'GLM (Zhipu AI)',
    baseURL: 'https://api.z.ai/api/paas/v4/',
    defaultModel: 'glm-5.3',
    models: ['glm-5.3', 'glm-5.2', 'glm-5.1'],
    envKey: 'GLM_API_KEY',
  },
  anthropic: {
    name: 'Anthropic Claude',
    defaultModel: 'claude-sonnet-5-5',
    models: ['claude-sonnet-5-5', 'claude-opus-5-5', 'claude-haiku-5-5', 'claude-fable-5-1'],
    envKey: 'ANTHROPIC_API_KEY',
    modelEnvKey: 'ANTHROPIC_MODEL',
    api: 'anthropic',
  },
};

class AITextService {
  constructor(credentials = {}) {
    this.logger = new Logger('AITextService');
    this.client = null;
    this.anthropic = null;
    this.gemini = null;
    this.lmStudio = null;
    this.model = null;
    this.providerName = null;

    this._init(credentials);
  }

  _init(credentials) {
    const provider = credentials.aiProvider?.provider;
    const apiKey = credentials.aiProvider?.apiKey;
    const model = credentials.aiProvider?.model;

    if (provider === 'gemini') {
      const key = apiKey || credentials.gemini?.apiKey || process.env.GEMINI_API_KEY;
      if (key) return this._initGemini(key, model || credentials.gemini?.model);
    }

    if (['local', 'lmstudio'].includes(provider) && credentials.aiProvider?.baseURL) {
      if (provider === 'lmstudio') {
        this.lmStudio = new LMStudioService({ baseURL: credentials.aiProvider.baseURL, apiKey });
        this.model = model || 'local-model';
        this.providerName = 'LM Studio';
        this.logger.info(`LM Studio initialized (model: ${this.model})`);
        return;
      }
      return this._initOpenAICompatible({
        name: 'Local GGUF (OpenAI-compatible)',
        baseURL: credentials.aiProvider.baseURL,
        defaultModel: model || 'local-model'
      }, apiKey || 'local', model);
    }

    if (provider && PROVIDERS[provider]) {
      const key = apiKey || process.env[PROVIDERS[provider].envKey];
      if (key) return this._initProvider(PROVIDERS[provider], key, model);
    }

    for (const [, preset] of Object.entries(PROVIDERS)) {
      const key = process.env[preset.envKey];
      if (key) {
        return this._initProvider(preset, key, preset.modelEnvKey && process.env[preset.modelEnvKey]);
      }
    }

    const geminiKey = credentials.gemini?.apiKey || process.env.GEMINI_API_KEY;
    if (geminiKey) {
      return this._initGemini(geminiKey, credentials.gemini?.model);
    }

    this.logger.warn('No AI text provider configured — text generation unavailable');
  }

  _initProvider(preset, apiKey, model) {
    if (preset.api === 'anthropic') {
      this.anthropic = new Anthropic({ apiKey });
      this.model = model || preset.defaultModel;
      this.providerName = preset.name;
      this.logger.info(`${preset.name} initialized (model: ${this.model})`);
      return;
    }

    this._initOpenAICompatible(preset, apiKey, model);
  }

  _initOpenAICompatible(preset, apiKey, model) {
    this.client = new OpenAI({ apiKey, baseURL: preset.baseURL });
    this.model = model || preset.defaultModel;
    this.providerName = preset.name;
    this.logger.info(`${preset.name} initialized (model: ${this.model})`);
  }

  _initGemini(apiKey, model) {
    try {
      const { GoogleGenAI } = require('@google/genai');
      this.gemini = new GoogleGenAI({ apiKey });
      this.model = model || GEMINI_DEFAULT_MODEL;
      this.providerName = 'Google Gemini';
      this.logger.info(`Gemini initialized (model: ${this.model})`);
    } catch (error) {
      this.logger.error('Failed to initialize Gemini:', error.message);
    }
  }

  async generateText(prompt, options = {}) {
    const model = options.model || this.model;
    const maxTokens = options.maxTokens || 2048;
    const temperature = options.temperature ?? 0.7;

    if (this.lmStudio) {
      return this.lmStudio.generateText({
        model,
        input: prompt,
        maxOutputTokens: maxTokens,
        temperature
      });
    }

    if (this.gemini) {
      const config = { maxOutputTokens: maxTokens };
      if (!/^gemini-3\.(?:[5-9]|\d{2,})-/.test(model)) config.temperature = temperature;
      let response = await this.gemini.models.generateContent({
        model,
        contents: prompt,
        config,
      });
      let text = response && response.text;

      // Thinking models (gemini-3.x flash/pro) count thoughts against
      // maxOutputTokens, so a small budget can be spent before any text is
      // written. Retry once with room for the thoughts plus the answer.
      if (!this._hasText(text) && this._geminiFinishReason(response) === 'MAX_TOKENS') {
        const retryTokens = Math.max(maxTokens * 4, GEMINI_THINKING_RETRY_MIN_TOKENS);
        response = await this.gemini.models.generateContent({
          model,
          contents: prompt,
          config: { ...config, maxOutputTokens: retryTokens },
        });
        text = response && response.text;
      }

      if (!this._hasText(text)) {
        throw new Error(this._geminiEmptyResponseMessage(response));
      }
      return text;
    }

    if (this.anthropic) {
      const request = {
        model,
        max_tokens: maxTokens,
        messages: [{ role: 'user', content: prompt }],
        temperature,
      };

      let response;
      try {
        response = await this.anthropic.messages.create(request);
      } catch (error) {
        const unsupportedTemperature =
          error?.status === 400 &&
          /temperature/i.test(error.message || '') &&
          /(unsupported|not supported|not allowed|does not support|only the default|cannot be used)/i.test(error.message || '');
        if (!unsupportedTemperature) throw error;

        const { temperature: _temperature, ...withoutTemperature } = request;
        response = await this.anthropic.messages.create(withoutTemperature);
      }

      const text = response?.content
        ?.filter(block => block.type === 'text')
        .map(block => block.text)
        .join('');
      if (!this._hasText(text)) {
        throw new Error(`${this.providerName} returned an empty response. Check the API key and model quota.`);
      }
      return text;
    }

    if (!this.client) {
      throw new Error('No AI text provider configured');
    }

    const params = {
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature,
    };

    const tokenParams = [
      { max_completion_tokens: maxTokens },
      { max_tokens: maxTokens },
    ];

    let lastError = null;
    for (let index = 0; index < tokenParams.length; index++) {
      const request = { ...params, ...tokenParams[index] };
      try {
        const response = await this.client.chat.completions.create(request);
        return this._extractContent(response);
      } catch (error) {
        lastError = error;
        const message = error?.message || '';
        const unsupportedTemperature =
          error?.status === 400 &&
          /temperature/i.test(message) &&
          /(unsupported value|does not support|only the default)/i.test(message);

        if (unsupportedTemperature) {
          const { temperature: _temperature, ...withoutTemperature } = request;
          try {
            const response = await this.client.chat.completions.create(withoutTemperature);
            return this._extractContent(response);
          } catch (retryError) {
            lastError = retryError;
          }
        }

        // Older models and some providers reject max_completion_tokens with a
        // 400; retry with the legacy max_tokens spelling. This also applies if
        // the temperature-compatible retry exposed the token incompatibility.
        if (
          index === 0 &&
          lastError?.status === 400 &&
          /max(_completion)?_tokens/i.test(lastError.message || '')
        ) {
          continue;
        }
        throw lastError;
      }
    }

    throw lastError;
  }

  _hasText(text) {
    return typeof text === 'string' && text.trim().length > 0;
  }

  _geminiFinishReason(response) {
    return response?.candidates?.[0]?.finishReason || null;
  }

  _geminiEmptyResponseMessage(response) {
    const finishReason = this._geminiFinishReason(response);
    const blockReason = response?.promptFeedback?.blockReason;
    if (finishReason === 'MAX_TOKENS') {
      const thoughts = response?.usageMetadata?.thoughtsTokenCount;
      return `${this.providerName} spent its whole output budget${thoughts ? ` (${thoughts} thinking tokens)` : ''} before writing any text. Raise maxTokens, or choose a model that does not think, such as gemini-3.5-flash-lite.`;
    }
    if (blockReason || ['SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'SPII', 'RECITATION'].includes(finishReason)) {
      return `${this.providerName} blocked the response (${blockReason || finishReason}). Rephrase the prompt or topic and try again.`;
    }
    return `${this.providerName} returned an empty response${finishReason ? ` (finishReason: ${finishReason})` : ''}. Check the API key and model quota — free-tier Gemini keys are rate-limited and can return empty output.`;
  }

  _extractContent(response) {
    const content =
      response &&
      response.choices &&
      response.choices[0] &&
      response.choices[0].message
        ? response.choices[0].message.content
        : null;

    if (typeof content !== 'string' || !content.trim()) {
      // A null/empty body used to surface as cryptic "Unexpected end of JSON input"
      // in the agents' JSON parsers. Report the real cause instead.
      throw new Error(
        `${this.providerName} returned an empty response. Check the API key and model quota.`
      );
    }
    return content;
  }

  isAvailable() {
    return !!(this.client || this.anthropic || this.gemini || this.lmStudio);
  }
}

module.exports = { AITextService, PROVIDERS, GEMINI_MODELS, GEMINI_DEFAULT_MODEL };
