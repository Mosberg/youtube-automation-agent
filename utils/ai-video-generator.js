const OpenAI = require('openai');
const Replicate = require('replicate');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const axios = require('axios');
const sharp = require('sharp');
const { Logger } = require('./logger');
const { runFFmpeg, getMediaDuration, checkFFmpeg, ffmpegInstallHint } = require('./ffmpeg');
const { MediaGenerationService } = require('./media-generation-service');
const { VoiceboxService, DEFAULT_VOICEBOX_URL } = require('./voicebox-service');

const OPENAI_IMAGE_MODEL = 'gpt-image-2';

// Turns an image provider SDK error into one readable line that leads with the
// fix when the cause is recognisable. @google/genai puts the raw JSON error
// body in error.message, which is unreadable on a console.
function describeImageProviderError(provider, model, error) {
  const raw = String(error?.message || error || 'Unknown error');
  let status = Number(error?.status) || null;
  let code = String(error?.code || '');
  let detail = raw;
  try {
    const body = JSON.parse(raw.slice(raw.indexOf('{')));
    if (body?.error) {
      status = status || Number(body.error.code) || null;
      code = body.error.status || code;
      detail = body.error.message || raw;
    }
  } catch (_error) { /* not a JSON error body */ }

  const text = `${code} ${detail}`;
  let fix = null;
  if (provider === 'Gemini') {
    if (/limit:\s*0\b/i.test(text) || (status === 429 && /free.?tier/i.test(text))) {
      fix = `this key has no quota for ${model}. Gemini image models are not on the free tier: turn on billing for the Google Cloud project that owns this API key (AI Studio → API keys → Set up billing). A paid account does not help if the key belongs to a different project.`;
    } else if (status === 429 || /RESOURCE_EXHAUSTED/i.test(code)) {
      fix = 'Gemini rate-limited this key. Wait a minute and retry, or raise the project quota.';
    } else if (/API key not valid|API key expired|API_KEY_INVALID/i.test(text)) {
      fix = 'the Gemini API key was rejected. Create a new key in Google AI Studio and save it in Settings.';
    } else if (/location is not supported|user location/i.test(text)) {
      fix = 'Gemini image generation is not available in your region for this key.';
    } else if (status === 404 || /NOT_FOUND|not supported for generateContent/i.test(text)) {
      fix = `${model} is not available to this key. Set GEMINI_IMAGE_MODEL to an image model your key can use (for example gemini-3.1-flash-image).`;
    } else if (status === 403 || /PERMISSION_DENIED|SERVICE_DISABLED/i.test(text)) {
      fix = "this key's Google Cloud project cannot call the Gemini API. Enable the Generative Language API for that project, or create the key in Google AI Studio.";
    }
  } else if (provider === 'OpenAI') {
    if (status === 401 || /invalid_api_key|Incorrect API key/i.test(text)) {
      fix = 'the OpenAI API key was rejected. Check the key in Settings.';
    } else if (/insufficient_quota|billing/i.test(text)) {
      fix = 'the OpenAI account has no remaining credit. Add billing at platform.openai.com.';
    } else if (status === 403 && /verif/i.test(text)) {
      fix = `${model} requires a verified OpenAI organization (platform.openai.com → Settings → Organization → Verify).`;
    } else if (status === 429) {
      fix = 'OpenAI rate-limited this key. Wait a minute and retry.';
    }
  }

  const label = [status, code && code !== String(status) ? code : null].filter(Boolean).join(' ');
  const summary = `${provider} image generation (${model}) failed${label ? ` [${label}]` : ''}.`;
  const shortDetail = detail.replace(/\s+/g, ' ').trim().slice(0, 220);
  return fix ? `${summary} Fix: ${fix} Details: ${shortDetail}` : `${summary} ${shortDetail}`;
}

class AIVideoGenerator {
  constructor(credentials, options = {}) {
    this.logger = new Logger('AIVideoGenerator');
    const resolvedCredentials = credentials?.credentials || credentials || {};
    this.db = options.db || null;
    this.getMediaDuration = options.getMediaDuration || getMediaDuration;
    this.lastVideoResult = null;
    this.lastNarrationResult = null;
    
    // Initialize AI services with graceful fallback
    const openaiKey = resolvedCredentials.openai?.apiKey || process.env.OPENAI_API_KEY;
    const replicateKey = resolvedCredentials.replicate?.apiKey || process.env.REPLICATE_API_TOKEN || process.env.REPLICATE_API_KEY;
    
    if (openaiKey) {
      this.openai = new OpenAI({ apiKey: openaiKey });
      this.logger.info('OpenAI service initialized');
    } else {
      this.logger.warn('OpenAI API key not found - AI features will be simulated');
    }
    
    if (replicateKey) {
      this.replicate = new Replicate({ auth: replicateKey });
      this.logger.info('Replicate service initialized');
    } else {
      this.logger.warn('Replicate API key not found - advanced video generation unavailable');
    }

    // Gemini media generation (images + native TTS) — free-tier alternative to OpenAI
    const geminiKey = resolvedCredentials.gemini?.apiKey || process.env.GEMINI_API_KEY;
    if (geminiKey) {
      try {
        const { GoogleGenAI } = require('@google/genai');
        this.gemini = new GoogleGenAI({ apiKey: geminiKey });
        this.logger.info('Gemini media service initialized (images + TTS)');
      } catch (error) {
        this.logger.warn('Failed to initialize Gemini media service:', error.message);
      }
    }
    
    // ElevenLabs configuration
    this.elevenLabsApiKey = resolvedCredentials.elevenLabs?.apiKey || process.env.ELEVENLABS_API_KEY;
    this.elevenLabsVoiceId = resolvedCredentials.elevenLabs?.voiceId || process.env.ELEVENLABS_VOICE_ID;
    this.elevenLabsModel = process.env.ELEVENLABS_TTS_MODEL || 'eleven_v3';

    const voicebox = resolvedCredentials.voicebox || {};
    this.voiceboxBaseUrl = voicebox.baseUrl || process.env.VOICEBOX_BASE_URL || DEFAULT_VOICEBOX_URL;
    this.voiceboxProfileId = voicebox.profileId || process.env.VOICEBOX_PROFILE_ID || null;
    this.voiceboxEngine = voicebox.engine || process.env.VOICEBOX_ENGINE || null;
    this.voiceboxModelSize = voicebox.modelSize || process.env.VOICEBOX_MODEL_SIZE || null;
    this.voiceboxLanguage = voicebox.language || process.env.VOICEBOX_LANGUAGE || 'en';
    this.voiceboxInstruct = voicebox.instruct || process.env.VOICEBOX_INSTRUCT || null;
    const maxChunkChars = Number(voicebox.maxChunkChars ?? process.env.VOICEBOX_MAX_CHUNK_CHARS ?? 800);
    this.voiceboxMaxChunkChars = Number.isInteger(maxChunkChars) && maxChunkChars >= 100 && maxChunkChars <= 5000
      ? maxChunkChars : 800;
    const crossfadeMs = Number(voicebox.crossfadeMs ?? process.env.VOICEBOX_CROSSFADE_MS ?? 50);
    this.voiceboxCrossfadeMs = Number.isInteger(crossfadeMs) && crossfadeMs >= 0 && crossfadeMs <= 500
      ? crossfadeMs : 50;
    this.voiceboxNormalize = voicebox.normalize !== false && process.env.VOICEBOX_NORMALIZE !== 'false';
    this.voiceboxTimeoutMs = process.env.VOICEBOX_TIMEOUT_MS || 0;
    this.ttsProvider = process.env.TTS_PROVIDER || resolvedCredentials.ttsProvider || null;
    this.voicebox = null;
    
    // Azure Speech configuration
    this.azureSpeechKey = resolvedCredentials.azure?.speechKey || process.env.AZURE_SPEECH_KEY;
    this.azureSpeechRegion = resolvedCredentials.azure?.speechRegion || process.env.AZURE_SPEECH_REGION;
    this.mediaGeneration = options.mediaGeneration || (this.db
      ? new MediaGenerationService(this.db, credentials || resolvedCredentials, { logger: this.logger })
      : null);
  }

  async generateTTSAudio(text, outputPath) {
    this.logger.info('Generating TTS audio...');
    this.lastNarrationResult = null;
    let provider = 'simulation';
    let model = null;

    try {
      const selected = this.selectTTSProvider();
      provider = selected.provider;
      model = selected.model;
      const generatedPath = await selected.generate(text, outputPath);

      const usable = await this.isUsableAudioFile(generatedPath);
      this.lastNarrationResult = {
        status: usable ? 'ready' : 'unavailable',
        path: generatedPath,
        provider,
        model,
        externalTaskId: null,
        generatedAt: new Date().toISOString(),
        simulated: !usable,
        cost: { provider, amount: null, currency: null, invoiceRequired: !['simulation', 'voicebox'].includes(provider) }
      };
      return generatedPath;
    } catch (error) {
      this.lastNarrationResult = {
        status: 'failed', path: null, provider, model, externalTaskId: null,
        generatedAt: new Date().toISOString(), simulated: false, error: error.message,
        cost: { provider, amount: null, currency: null, invoiceRequired: !['simulation', 'voicebox'].includes(provider) }
      };
      this.logger.error('TTS generation failed:', error);
      throw error;
    }
  }

  selectTTSProvider() {
    const voiceboxConfigured = Boolean(this.voiceboxProfileId);
    const providers = {
      voicebox: {
        configured: voiceboxConfigured,
        model: this.voiceboxEngine || 'profile-default',
        generate: (text, outputPath) => this.generateVoiceboxTTS(text, outputPath)
      },
      elevenlabs: {
        configured: Boolean(this.elevenLabsApiKey && this.elevenLabsVoiceId),
        model: this.elevenLabsModel,
        generate: (text, outputPath) => this.generateElevenLabsTTS(text, outputPath)
      },
      openai: {
        configured: Boolean(this.openai),
        model: 'gpt-4o-mini-tts',
        generate: (text, outputPath) => this.generateOpenAITTS(text, outputPath)
      },
      gemini: {
        configured: Boolean(this.gemini),
        model: process.env.GEMINI_TTS_MODEL || 'gemini-3.1-flash-tts-preview',
        generate: (text, outputPath) => this.generateGeminiTTS(text, outputPath)
      }
    };

    const selectedProvider = this.ttsProvider && this.ttsProvider !== 'auto'
      ? this.ttsProvider
      : null;
    if (selectedProvider) {
      const selected = providers[selectedProvider];
      if (!selected) {
        throw new Error(`Unsupported TTS_PROVIDER "${selectedProvider}". Choose voicebox, elevenlabs, openai, gemini, or auto.`);
      }
      if (!selected.configured) {
        throw new Error(`TTS provider "${selectedProvider}" is selected but not configured.`);
      }
      return { provider: selectedProvider, ...selected };
    }

    for (const provider of ['voicebox', 'elevenlabs', 'openai', 'gemini']) {
      if (providers[provider].configured) {
        return { provider, ...providers[provider] };
      }
    }

    return {
      provider: 'simulation',
      model: null,
      generate: (text, outputPath) => this.simulateTTSGeneration(text, outputPath)
    };
  }

  async generateVoiceboxTTS(text, outputPath) {
    if (!this.voiceboxProfileId) {
      throw new Error('Voicebox requires a voice profile. Run npm run walkthrough or set VOICEBOX_PROFILE_ID.');
    }
    this.voicebox ||= new VoiceboxService({
      baseUrl: this.voiceboxBaseUrl,
      timeoutMs: this.voiceboxTimeoutMs
    });

    const request = {
      profile_id: this.voiceboxProfileId,
      text,
      language: this.voiceboxLanguage,
      max_chunk_chars: this.voiceboxMaxChunkChars,
      crossfade_ms: this.voiceboxCrossfadeMs,
      normalize: this.voiceboxNormalize
    };
    if (this.voiceboxEngine) request.engine = this.voiceboxEngine;
    if (this.voiceboxModelSize) request.model_size = this.voiceboxModelSize;
    if (this.voiceboxInstruct) request.instruct = this.voiceboxInstruct;

    const wavPath = `${outputPath}.${crypto.randomUUID()}.voicebox.wav`;
    try {
      const audio = await this.voicebox.generateSpeech(request);
      await fs.writeFile(wavPath, audio);
      await runFFmpeg([
        '-y', '-i', wavPath,
        '-codec:a', 'libmp3lame', '-q:a', '2',
        outputPath
      ]);
      this.logger.info('Voicebox TTS generation complete');
      return outputPath;
    } finally {
      await fs.unlink(wavPath).catch(() => {});
    }
  }

  async generateElevenLabsTTS(text, outputPath) {
    const url = `https://api.elevenlabs.io/v1/text-to-speech/${this.elevenLabsVoiceId}`;
    
    const data = {
      text: text,
      model_id: this.elevenLabsModel,
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.8,
        style: 0.0,
        use_speaker_boost: true
      }
    };

    const response = await axios({
      method: 'POST',
      url: url,
      data: data,
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': this.elevenLabsApiKey
      },
      responseType: 'stream'
    });

    const writer = require('fs').createWriteStream(outputPath);
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', () => {
        this.logger.info('ElevenLabs TTS generation complete');
        resolve(outputPath);
      });
      writer.on('error', reject);
    });
  }

  async generateOpenAITTS(text, outputPath) {
    const response = await this.openai.audio.speech.create({
      model: "gpt-4o-mini-tts",
      voice: "coral",
      input: text,
      speed: 1.0
    });

    const buffer = Buffer.from(await response.arrayBuffer());
    await fs.writeFile(outputPath, buffer);

    this.logger.info('OpenAI TTS generation complete');
    return outputPath;
  }

  async generateGeminiTTS(text, outputPath) {
    const model = process.env.GEMINI_TTS_MODEL || 'gemini-3.1-flash-tts-preview';
    const voiceName = process.env.GEMINI_TTS_VOICE || 'Kore';

    const response = await this.gemini.models.generateContent({
      model,
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName }
          }
        }
      }
    });

    const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!audioData) {
      throw new Error('Gemini TTS returned no audio data');
    }

    // Gemini returns raw PCM (24kHz, mono, 16-bit); encode to the requested container via FFmpeg
    const pcmPath = outputPath + '.pcm';
    await fs.writeFile(pcmPath, Buffer.from(audioData, 'base64'));
    await runFFmpeg(['-y', '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', pcmPath, outputPath]);
    await fs.unlink(pcmPath).catch(() => {});

    this.logger.info('Gemini TTS generation complete');
    return outputPath;
  }

  async generateVisualAssets(prompt, style = "ethereal", count = 1) {
    this.logger.info(`Generating ${count} visual assets with style: ${style}`);

    try {
      if (!this.openai && !this.gemini) {
        return await this.generateLocalVisualAssets(prompt, style, count);
      }

      const enhancedPrompt = this.enhanceVisualPrompt(prompt, style);
      const localPaths = [];

      for (let i = 0; i < count; i++) {
        const imagePath = path.join(__dirname, '..', 'data', 'assets', `visual_${Date.now()}_${i}.png`);
        await this.generateImage(enhancedPrompt, imagePath);
        localPaths.push(imagePath);
      }

      this.logger.info(`Generated ${localPaths.length} visual assets`);
      return localPaths;
    } catch (error) {
      this.logger.warn(`Cloud visual generation failed; creating local artwork instead: ${error.message}`);
      return this.generateLocalVisualAssets(prompt, style, count);
    }
  }

  async generateLocalVisualAssets(prompt, style = 'local-art', count = 1) {
    await fs.mkdir(path.join(__dirname, '..', 'data', 'assets'), { recursive: true });
    const palette = [
      ['#101b3d', '#256d85', '#f7c873'],
      ['#34205a', '#d5538a', '#f7d06b'],
      ['#0c3b35', '#3e987a', '#f5d68a'],
      ['#25315c', '#556fc4', '#f0a66a']
    ];
    const outputs = [];
    const total = Math.max(1, Math.min(8, Math.floor(Number(count) || 1)));
    const promptText = this.escapeSvgText(String(prompt || 'A new idea').replace(/\s+/g, ' ').slice(0, 95));
    for (let index = 0; index < total; index++) {
      const digest = crypto.createHash('sha256').update(`${prompt}:${style}:${index}`).digest();
      const colors = palette[digest[0] % palette.length];
      const x = 160 + digest[1] * 3;
      const y = 100 + digest[2] * 2;
      const radius = 90 + digest[3];
      const assetPath = path.join(__dirname, '..', 'data', 'assets', `local_art_${Date.now()}_${crypto.randomBytes(4).toString('hex')}.png`);
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
        <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="${colors[0]}"/><stop offset="1" stop-color="${colors[1]}"/></linearGradient>
        <radialGradient id="glow"><stop stop-color="${colors[2]}" stop-opacity=".85"/><stop offset="1" stop-color="${colors[2]}" stop-opacity="0"/></radialGradient>
        <filter id="blur"><feGaussianBlur stdDeviation="28"/></filter></defs>
        <rect width="1280" height="720" fill="url(#bg)"/>
        <circle cx="${x}" cy="${y}" r="${radius * 2}" fill="url(#glow)" opacity=".8"/>
        <path d="M0 565 Q210 410 420 545 T850 510 T1280 520 V720 H0Z" fill="${colors[0]}" opacity=".58"/>
        <circle cx="${900 - digest[4]}" cy="${180 + digest[5]}" r="${80 + digest[6]}" fill="${colors[2]}" opacity=".16" filter="url(#blur)"/>
        <path d="M0 620 Q260 500 520 625 T1040 580 T1280 600" fill="none" stroke="${colors[2]}" stroke-opacity=".45" stroke-width="3"/>
        <rect x="70" y="70" width="10" height="160" rx="5" fill="${colors[2]}"/>
        <text x="112" y="112" fill="#fff" font-family="Arial,sans-serif" font-size="20" font-weight="700" letter-spacing="4">${this.escapeSvgText(String(style).toUpperCase().slice(0, 28))}</text>
        <text x="112" y="170" fill="#fff" font-family="Arial,sans-serif" font-size="28" font-weight="700">${promptText}</text>
        <text x="1190" y="660" text-anchor="end" fill="#fff" fill-opacity=".62" font-family="Arial,sans-serif" font-size="15" letter-spacing="3">LOCAL ARTWORK</text>
      </svg>`;
      await sharp(Buffer.from(svg)).png().toFile(assetPath);
      outputs.push(assetPath);
    }
    return outputs;
  }

  escapeSvgText(value) {
    return String(value).replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;'
    })[character]);
  }

  async generateImage(prompt, imagePath) {
    await fs.mkdir(path.dirname(imagePath), { recursive: true });

    const providers = [];
    if (this.openai) {
      providers.push({ name: 'OpenAI', model: OPENAI_IMAGE_MODEL, run: () => this.generateOpenAIImage(prompt, imagePath) });
    }
    if (this.gemini) {
      providers.push({ name: 'Gemini', model: this.geminiImageModel(), run: () => this.generateGeminiImage(prompt, imagePath) });
    }
    if (!providers.length) {
      throw new Error('No image generation provider configured');
    }

    // Try every configured provider before giving up, so a broken OpenAI key
    // does not hide a working Gemini key (or the reverse).
    const failures = [];
    for (const [index, provider] of providers.entries()) {
      try {
        return await provider.run();
      } catch (error) {
        const description = describeImageProviderError(provider.name, provider.model, error);
        failures.push(description);
        if (index < providers.length - 1) {
          this.logger.warn(`${description} Trying ${providers[index + 1].name} instead.`);
        }
      }
    }

    const error = new Error(failures.join(' | '));
    error.providerFailures = failures;
    throw error;
  }

  geminiImageModel() {
    return process.env.GEMINI_IMAGE_MODEL || 'gemini-3.1-flash-image';
  }

  async generateOpenAIImage(prompt, imagePath) {
    const response = await this.openai.images.generate({
      model: OPENAI_IMAGE_MODEL,
      prompt: prompt,
      n: 1,
      size: "1536x1024",
      quality: "high",
    });

    if (response.data[0].b64_json) {
      const buffer = Buffer.from(response.data[0].b64_json, 'base64');
      await fs.writeFile(imagePath, buffer);
    } else {
      await this.downloadImage(response.data[0].url, imagePath);
    }

    return imagePath;
  }

  async generateGeminiImage(prompt, imagePath) {
    const model = this.geminiImageModel();

    const response = await this.gemini.models.generateContent({
      model,
      contents: prompt,
      config: {
        responseModalities: ['IMAGE'],
        imageConfig: {
          aspectRatio: '16:9',
          imageSize: '1K'
        }
      }
    });

    const parts = response.candidates?.[0]?.content?.parts || [];
    const imageParts = parts.filter(part =>
      part.inlineData?.data && (!part.inlineData.mimeType || part.inlineData.mimeType.startsWith('image/'))
    );
    const renderedImages = imageParts.filter(part => part.thought !== true);
    const imagePart = (renderedImages.length ? renderedImages : imageParts).at(-1);
    if (!imagePart) {
      throw new Error('Gemini image generation returned no image data');
    }

    const imageBuffer = Buffer.from(imagePart.inlineData.data, 'base64');
    const metadata = await sharp(imageBuffer, { failOn: 'error' }).metadata();
    if (!metadata.width || !metadata.height) {
      throw new Error('Gemini image generation returned an invalid image asset');
    }

    const extension = path.extname(imagePath).toLowerCase();
    const output = sharp(imageBuffer, { failOn: 'error' });
    if (extension === '.jpg' || extension === '.jpeg') {
      await output.jpeg({ quality: 92 }).toFile(imagePath);
    } else if (extension === '.webp') {
      await output.webp({ quality: 92 }).toFile(imagePath);
    } else {
      await output.png().toFile(imagePath);
    }
    return imagePath;
  }

  enhanceVisualPrompt(prompt, style) {
    const styleEnhancements = {
      ethereal: "ethereal, dreamy, mystical, soft lighting, floating particles, cosmic background",
      modern: "modern, clean, minimalist, professional, sleek design, contemporary",
      animated: "animated style, cartoon, vibrant colors, expressive, dynamic",
      cinematic: "cinematic lighting, dramatic, movie poster style, high contrast",
      abstract: "abstract art, geometric shapes, gradient colors, artistic composition"
    };

    const normalizedStyle = String(style || '').trim().toLowerCase();
    const enhancement = styleEnhancements[normalizedStyle] || String(style || '').trim() || styleEnhancements.ethereal;
    return `${prompt}, ${enhancement}, high quality, 16:9 aspect ratio, digital art`;
  }

  async downloadImage(url, outputPath) {
    const response = await axios({
      method: 'GET',
      url: url,
      responseType: 'stream'
    });

    const writer = require('fs').createWriteStream(outputPath);
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
  }

  async generateVideo(script, visualAssets, audioPath, outputPath, options = {}) {
    this.logger.info('Generating video from assets...');
    this.lastVideoResult = null;
    try {
      if (this.mediaGeneration && options.productionId) {
        const generated = await this.mediaGeneration.generateClips({
          jobId: options.jobId || null,
          productionId: options.productionId,
          script,
          visualAssets,
          outputDir: path.dirname(outputPath)
        });
        if (generated.requestedProvider === 'comfyui' && generated.actualProvider === 'comfyui' &&
            generated.settings.mode !== 'slideshow' && generated.settings.maxGeneratedSeconds > 0 &&
            generated.clips.length === 0) {
          throw new Error('ComfyUI was selected, but the script did not produce any video scene prompts to render.');
        }
        if (generated.clips.length) {
          const produced = await this.generateHybridVideo(
            generated.clips,
            visualAssets,
            audioPath,
            outputPath,
            options.estimatedDuration || this.calculateScriptDuration(script),
            script.aspectRatio || generated.settings.aspectRatio
          );
          this.lastVideoResult = {
            requestedProvider: generated.requestedProvider,
            actualProvider: generated.actualProvider,
            model: generated.model,
            mode: generated.settings.mode,
            generatedSeconds: generated.clips.reduce((total, clip) => total + clip.duration, 0),
            tasks: generated.clips.map(clip => ({ scene: clip.index, taskId: clip.taskId, provider: clip.provider, model: clip.model })),
            scenes: generated.clips.map(clip => ({
              index: clip.index, label: clip.label, prompt: clip.prompt, duration: clip.duration,
              path: clip.path, taskId: clip.taskId, provider: clip.provider, model: clip.model
            }))
          };
          return produced;
        }
      }

      const produced = await this.generateSlideshowVideo(script, visualAssets, audioPath, outputPath);
      this.lastVideoResult = { requestedProvider: 'slideshow', actualProvider: 'slideshow', model: 'local-ffmpeg', mode: 'slideshow', generatedSeconds: 0, tasks: [], scenes: [] };
      return produced;
    } catch (error) {
      // The Logger's console line only shows the message string, so put the real
      // reason inline. Previously the stack alone went to the file transport and
      // the console printed "Video generation failed:" with no detail.
      const reason = error && error.message ? error.message : String(error);
      const configuredProvider = this.mediaGeneration
        ? (await this.mediaGeneration.settings()).provider
        : null;
      if (configuredProvider === 'comfyui') {
        this.lastVideoResult = {
          requestedProvider: 'comfyui',
          actualProvider: 'failed',
          model: 'ComfyUI API workflow',
          mode: 'local',
          generatedSeconds: 0,
          fallbackReason: reason,
          tasks: [],
          scenes: []
        };
        throw error;
      }
      this.logger.error(`Video provider generation failed; using the local slideshow: ${reason}`, error);
      try {
        const produced = await this.generateSlideshowVideo(script, visualAssets, audioPath, outputPath);
        this.lastVideoResult = {
          requestedProvider: this.lastVideoResult?.requestedProvider || 'configured-provider',
          actualProvider: 'slideshow', model: 'local-ffmpeg', mode: 'fallback', generatedSeconds: 0,
          fallbackReason: reason, tasks: [], scenes: []
        };
        return produced;
      } catch (fallbackError) {
        this.logger.error(`Local slideshow fallback failed: ${fallbackError.message}`, fallbackError);
        const produced = await this.simulateVideoGeneration(script, visualAssets, audioPath, outputPath);
        this.lastVideoResult = {
          requestedProvider: 'configured-provider', actualProvider: 'simulation', model: null,
          mode: 'simulation', generatedSeconds: 0, fallbackReason: `${reason}; ${fallbackError.message}`, tasks: [], scenes: []
        };
        return produced;
      }
    }
  }

  async generateHybridVideo(clips, visualAssets, audioPath, outputPath, totalDuration, aspectRatio = '16:9') {
    if (!(await checkFFmpeg())) throw new Error(ffmpegInstallHint());
    const validImages = await this.filterLocalImageAssets(visualAssets);
    const segments = clips.map(clip => ({ type: 'video', path: clip.path, duration: clip.duration }));
    const generatedDuration = segments.reduce((sum, item) => sum + item.duration, 0);
    const remaining = Math.max(0, this.parseDurationSeconds(totalDuration) - generatedDuration);
    if (remaining && validImages.length) {
      const perImage = Math.max(2, remaining / validImages.length);
      for (const imagePath of validImages) segments.push({ type: 'image', path: imagePath, duration: perImage });
    }
    if (!segments.length) throw new Error('No usable provider clips or still images were generated');

    const visualPath = outputPath.replace(/\.mp4$/i, '_hybrid_visual.mp4');
    await this.renderMediaTimeline(segments, visualPath, aspectRatio);
    await this.addAudioToVideo(visualPath, audioPath, outputPath, { loopVideo: true });
    await fs.unlink(visualPath).catch(() => {});
    return outputPath;
  }

  async renderMediaTimeline(segments, outputPath, aspectRatio = '16:9') {
    const [ratioWidth, ratioHeight] = String(aspectRatio).split(':').map(Number);
    const vertical = ratioWidth > 0 && ratioHeight > ratioWidth;
    const width = vertical ? 1080 : 1920;
    const height = vertical ? 1920 : 1080;
    const args = ['-y'];
    for (const segment of segments) {
      const duration = Number(segment.duration).toFixed(2);
      if (segment.type === 'image') args.push('-loop', '1', '-t', duration, '-framerate', '30', '-i', segment.path);
      // Loop short clips to fill their slot, but bound the input: an unbounded
      // -stream_loop keeps FFmpeg decoding long after trim has what it needs.
      else args.push('-stream_loop', '-1', '-t', duration, '-i', segment.path);
    }
    const filters = segments.map((segment, index) =>
      `[${index}:v]scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2:black,fps=30,format=yuv420p,trim=duration=${Number(segment.duration).toFixed(2)},setpts=PTS-STARTPTS[v${index}]`
    );
    filters.push(`${segments.map((_, index) => `[v${index}]`).join('')}concat=n=${segments.length}:v=1:a=0[vout]`);
    args.push('-filter_complex', filters.join(';'), '-map', '[vout]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', outputPath);
    await runFFmpeg(args);
    return outputPath;
  }

  async filterLocalImageAssets(visualAssets = []) {
    const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp']);
    const images = [];
    for (const asset of visualAssets) {
      if (typeof asset !== 'string' || !imageExtensions.has(path.extname(asset).toLowerCase())) continue;
      try {
        await fs.access(asset);
        images.push(asset);
      } catch (_error) { /* ignore missing assets */ }
    }
    return images;
  }

  parseDurationSeconds(value) {
    if (Number.isFinite(Number(value))) return Math.max(0, Number(value));
    const parts = String(value || '').split(':').map(Number);
    if (parts.length === 2 && parts.every(Number.isFinite)) return Math.max(0, parts[0] * 60 + parts[1]);
    if (parts.length === 3 && parts.every(Number.isFinite)) return Math.max(0, parts[0] * 3600 + parts[1] * 60 + parts[2]);
    return 0;
  }

  async generateReplicateVideo(script, visualAssets, audioPath, outputPath) {
    const output = await this.replicate.run(
      "wan-video/wan-2.7-i2v",
      {
        input: {
          image: visualAssets[0],
          prompt: script.title || "smooth cinematic motion",
          duration: 5,
          resolution: "720p"
        }
      }
    );

    // Download the generated video
    if (output && output.length > 0) {
      await this.downloadVideo(output[0], outputPath);
      
      // Add audio track
      await this.addAudioToVideo(outputPath, audioPath, outputPath);
    }

    return outputPath;
  }

  async generateSlideshowVideo(script, visualAssets, audioPath, outputPath) {
    this.logger.info('Creating slideshow video...');

    if (!(await checkFFmpeg())) {
      throw new Error(ffmpegInstallHint());
    }

    const { chromium } = require('playwright');
    const browser = await chromium.launch();
    const slidesDir = path.join(path.dirname(outputPath), 'slides');

    try {
      const page = await browser.newPage();
      const aspectRatio = script.aspectRatio || '16:9';
      const vertical = aspectRatio === '9:16';
      const dimensions = vertical ? { width: 1080, height: 1920 } : { width: 1920, height: 1080 };
      await page.setViewportSize(dimensions);

      // Create HTML for slideshow (only real image files can be embedded)
      const imageAssets = await this.filterImageAssets(visualAssets);
      await page.setContent(this.createSlideshowHTML(script, imageAssets, dimensions));

      // Freeze CSS transitions/animations so each still is captured fully rendered
      await page.addStyleTag({ content: '* { transition: none !important; animation: none !important; }' });
      await page.waitForTimeout(1000); // Wait for assets to load

      // Capture ONE still per slide instead of screenshotting at 30fps —
      // FFmpeg turns the stills into a crossfaded video in seconds.
      const slideCount = await page.evaluate(() => document.querySelectorAll('.slide').length);
      await fs.mkdir(slidesDir, { recursive: true });

      const stills = [];
      for (let i = 0; i < slideCount; i++) {
        await page.evaluate((index) => {
          document.querySelectorAll('.slide').forEach((slide, s) => {
            slide.classList.toggle('active', s === index);
          });
        }, i);

        const stillPath = path.join(slidesDir, `slide_${String(i).padStart(3, '0')}.png`);
        await page.screenshot({ path: stillPath });
        stills.push(stillPath);
      }

      const videoPath = outputPath.replace('.mp4', '_visual.mp4');
      const duration = await this.resolveSlideshowDuration(script, audioPath);
      await this.renderSlidesToVideo(stills, duration, videoPath);

      // Add audio
      await this.addAudioToVideo(videoPath, audioPath, outputPath);

      return outputPath;
    } finally {
      await browser.close().catch(() => {});
      await this.cleanupDirectory(slidesDir);
    }
  }

  async resolveSlideshowDuration(script, audioPath) {
    try {
      return (await this.getMediaDuration(audioPath)) + 0.5;
    } catch (error) {
      this.logger.warn(`Could not read narration duration, falling back to word-count estimate: ${error.message}`);
      return this.calculateScriptDuration(script);
    }
  }

  async renderSlidesToVideo(stills, totalDuration, videoPath) {
    if (stills.length === 0) {
      throw new Error('No slides to render');
    }

    const fade = 0.5;
    const overlapDuration = fade * Math.max(0, stills.length - 1);
    const perSlide = Math.max(2, (totalDuration + overlapDuration) / stills.length);

    const args = ['-y'];
    for (const still of stills) {
      args.push('-loop', '1', '-t', perSlide.toFixed(2), '-framerate', '30', '-i', still);
    }

    if (stills.length === 1) {
      args.push('-vf', 'format=yuv420p', '-c:v', 'libx264', videoPath);
      await runFFmpeg(args);
      return videoPath;
    }

    // Chain crossfades: transition k starts fade seconds before slide k ends
    const filters = [];
    let prev = '[0:v]';
    for (let i = 1; i < stills.length; i++) {
      const out = `[v${i}]`;
      const offset = (i * (perSlide - fade)).toFixed(2);
      filters.push(`${prev}[${i}:v]xfade=transition=fade:duration=${fade}:offset=${offset}${out}`);
      prev = out;
    }
    filters.push(`${prev}format=yuv420p[vfinal]`);

    args.push(
      '-filter_complex', filters.join(';'),
      '-map', '[vfinal]',
      '-c:v', 'libx264',
      '-r', '30',
      videoPath
    );

    await runFFmpeg(args);
    return videoPath;
  }

  async filterImageAssets(visualAssets = []) {
    const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp']);
    const mimeTypes = {
      jpeg: 'image/jpeg',
      png: 'image/png',
      webp: 'image/webp'
    };
    const images = [];

    for (const asset of visualAssets) {
      if (typeof asset !== 'string' || !imageExtensions.has(path.extname(asset).toLowerCase())) {
        continue;
      }

      try {
        const imageBuffer = await fs.readFile(asset);
        const metadata = await sharp(imageBuffer, { failOn: 'error' }).metadata();
        const mimeType = mimeTypes[metadata.format];
        if (mimeType && metadata.width && metadata.height) {
          images.push(`data:${mimeType};base64,${imageBuffer.toString('base64')}`);
        }
      } catch (_error) {
        // Skip missing or invalid image files
      }
    }

    return images;
  }

  createSlideshowHTML(script, visualAssets, dimensions = { width: 1920, height: 1080 }) {
    return `
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            margin: 0;
            padding: 0;
            width: ${dimensions.width}px;
            height: ${dimensions.height}px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }
        
        .slide {
            position: absolute;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 2s ease-in-out;
        }
        
        .slide.active {
            opacity: 1;
        }
        
        .content {
            text-align: center;
            color: white;
            max-width: 80%;
        }
        
        h1 {
            font-size: 72px;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        h2 {
            font-size: 48px;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        p {
            font-size: 36px;
            line-height: 1.4;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .background-image {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.3;
            z-index: -1;
        }
        
        .particles {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
        }
        
        .particle {
            position: absolute;
            background: rgba(255,255,255,0.8);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }
    </style>
</head>
<body>
    <div class="particles"></div>
    
    <!-- Title Slide -->
    <div class="slide active">
        ${visualAssets[0] ? `<img class="background-image" src="${visualAssets[0]}" />` : ''}
        <div class="content">
            <h1>${script.title}</h1>
            <p>Ethereal Dreamscript</p>
        </div>
    </div>
    
    ${this.generateContentSlides(script, visualAssets).join('')}
    
    <!-- Subscribe Slide -->
    <div class="slide">
        <div class="content">
            <h2>✨ Subscribe for More Stories ✨</h2>
            <p>New content daily at 2:00 PM</p>
        </div>
    </div>
    
    <script>
        // Create floating particles
        function createParticles() {
            const container = document.querySelector('.particles');
            for (let i = 0; i < 20; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.width = (Math.random() * 4 + 2) + 'px';
                particle.style.height = particle.style.width;
                particle.style.animationDelay = Math.random() * 6 + 's';
                container.appendChild(particle);
            }
        }
        
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        
        function advanceAnimation() {
            slides[currentSlide].classList.remove('active');
            currentSlide = (currentSlide + 1) % slides.length;
            slides[currentSlide].classList.add('active');
        }
        
        window.advanceAnimation = advanceAnimation;
        createParticles();
    </script>
</body>
</html>`;
  }

  generateContentSlides(script, visualAssets) {
    const slides = [];
    
    if (script.mainContent && script.mainContent.sections) {
      script.mainContent.sections.forEach((section, index) => {
        const assetIndex = Math.min(index + 1, visualAssets.length - 1);
        
        slides.push(`
        <div class="slide">
            ${visualAssets[assetIndex] ? `<img class="background-image" src="${visualAssets[assetIndex]}" />` : ''}
            <div class="content">
                <h2>${section.title}</h2>
                ${this.formatSectionContent(section)}
            </div>
        </div>`);
      });
    }
    
    return slides;
  }

  formatSectionContent(section) {
    if (section.items && Array.isArray(section.items)) {
      return section.items.slice(0, 3).map(item => 
        `<p>${item.number}. ${item.title}</p>`
      ).join('');
    }
    
    if (section.steps && Array.isArray(section.steps)) {
      return section.steps.slice(0, 3).map(step => 
        `<p>${step.title}</p>`
      ).join('');
    }
    
    if (typeof section.content === 'string') {
      return `<p>${section.content.slice(0, 200)}${section.content.length > 200 ? '...' : ''}</p>`;
    }
    
    return '<p>Content coming soon...</p>';
  }

  calculateScriptDuration(script) {
    // Estimate duration based on word count (average 150 words per minute)
    let totalWords = 0;
    
    if (script.hook) totalWords += script.hook.text.split(' ').length;
    if (script.introduction) {
      totalWords += (script.introduction.greeting || '').split(' ').length;
      totalWords += (script.introduction.topicIntro || '').split(' ').length;
    }
    
    if (script.mainContent && script.mainContent.sections) {
      script.mainContent.sections.forEach(section => {
        if (typeof section.content === 'string') {
          totalWords += section.content.split(' ').length;
        }
        if (section.items) {
          section.items.forEach(item => {
            totalWords += (item.title + ' ' + item.description).split(' ').length;
          });
        }
        if (section.steps) {
          section.steps.forEach(step => {
            totalWords += (step.title + ' ' + step.description).split(' ').length;
          });
        }
      });
    }
    
    if (script.conclusion) {
      totalWords += script.conclusion.finalThought.split(' ').length;
    }
    
    // Convert to duration (150 words per minute)
    return Math.max(30, Math.ceil((totalWords / 150) * 60));
  }

  async addAudioToVideo(videoPath, audioPath, outputPath, options = {}) {
    const hasRealAudio = await this.isUsableAudioFile(audioPath);

    if (!hasRealAudio) {
      if (options.allowSilent === true) {
        this.logger.warn('Creating an intentionally silent video from an operator-confirmed override.');
        if (videoPath !== outputPath) await fs.copyFile(videoPath, outputPath);
        return outputPath;
      }
      const error = new Error('Narration audio is required. Regenerate narration or explicitly confirm an intentional silent video.');
      error.code = 'NARRATION_REQUIRED';
      throw error;
    }

    // FFmpeg cannot write to its own input, so mux to a temp file when paths collide
    const muxPath = outputPath === videoPath
      ? outputPath.replace(/\.mp4$/i, '_muxed.mp4')
      : outputPath;

    const videoInput = options.loopVideo ? ['-stream_loop', '-1', '-i', videoPath] : ['-i', videoPath];
    await runFFmpeg(['-y', ...videoInput, '-i', audioPath, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', muxPath]);

    if (muxPath !== outputPath) {
      await fs.rename(muxPath, outputPath);
    }

    this.logger.info('Audio added to video successfully');
    return outputPath;
  }

  async isUsableAudioFile(audioPath) {
    if (typeof audioPath !== 'string' || audioPath.endsWith('.info')) {
      return false;
    }

    try {
      const stats = await fs.stat(audioPath);
      return stats.isFile() && stats.size > 0;
    } catch (error) {
      return false;
    }
  }

  async downloadVideo(url, outputPath) {
    const response = await axios({
      method: 'GET',
      url: url,
      responseType: 'stream'
    });

    const writer = require('fs').createWriteStream(outputPath);
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
  }

  async cleanupDirectory(dirPath) {
    try {
      const files = await fs.readdir(dirPath);
      for (const file of files) {
        await fs.unlink(path.join(dirPath, file));
      }
      await fs.rmdir(dirPath);
    } catch (error) {
      this.logger.warn('Cleanup failed:', error.message);
    }
  }

  async generateThumbnail(script, style = "ethereal") {
    this.logger.info('Generating custom thumbnail...');

    try {
      if (!this.openai && !this.gemini) {
        return await this.simulateThumbnailGeneration(script, style);
      }

      const prompt = `YouTube thumbnail for "${script.title}", ${style} style, eye-catching, high contrast text, professional design, clickable, engaging`;
      const thumbnailPath = path.join(__dirname, '..', 'uploads', 'thumbnails', `thumbnail_${Date.now()}.png`);

      await this.generateImage(prompt, thumbnailPath);
      const metadata = await sharp(thumbnailPath).metadata();

      return {
        path: thumbnailPath,
        dimensions: { width: metadata.width, height: metadata.height },
        fileSize: await this.getFileSize(thumbnailPath)
      };
    } catch (error) {
      this.logger.error('Thumbnail generation failed:', error);
      return await this.simulateThumbnailGeneration(script, style);
    }
  }

  async getFileSize(filePath) {
    const stats = await fs.stat(filePath);
    return stats.size;
  }

  // Simulation methods for when APIs are not available
  async simulateTTSGeneration(text, outputPath) {
    this.logger.info('Simulating TTS generation...');
    
    const infoPath = outputPath + '.info';
    await fs.writeFile(infoPath, JSON.stringify({
      message: 'AI TTS audio would be generated here',
      text: text.substring(0, 100) + '...',
      timestamp: new Date().toISOString()
    }, null, 2));
    
    return infoPath;
  }

  async simulateVisualAssets(prompt, style, count) {
    this.logger.info(`Creating ${count} local visual assets...`);
    return this.generateLocalVisualAssets(prompt, style, count);
  }

  async simulateVideoGeneration(script, visualAssets, audioPath, outputPath) {
    this.logger.info('Simulating video generation...');
    
    const infoPath = outputPath + '.info';
    await fs.writeFile(infoPath, JSON.stringify({
      message: 'AI video would be generated here',
      script: script.title,
      visualAssets: visualAssets.length,
      audioPath: audioPath,
      timestamp: new Date().toISOString()
    }, null, 2));
    
    return infoPath;
  }

  async simulateThumbnailGeneration(script, style) {
    this.logger.info('Simulating thumbnail generation...');
    
    const thumbnailPath = path.join(__dirname, '..', 'uploads', 'thumbnails', `thumbnail_sim_${Date.now()}.info`);
    await fs.mkdir(path.dirname(thumbnailPath), { recursive: true });
    
    await fs.writeFile(thumbnailPath, JSON.stringify({
      message: 'AI thumbnail would be generated here',
      title: script.title,
      style: style,
      timestamp: new Date().toISOString()
    }, null, 2));
    
    return {
      path: thumbnailPath,
      dimensions: { width: 1792, height: 1024 },
      fileSize: 1024,
      simulated: true
    };
  }
}

module.exports = { AIVideoGenerator, describeImageProviderError };
