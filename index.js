require('dotenv').config();

const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const { Logger } = require('./utils/logger');
const { Database } = require('./database/db');
const { CredentialManager } = require('./utils/credential-manager');
const { ContentStrategyAgent } = require('./agents/content-strategy-agent');
const { ScriptWriterAgent } = require('./agents/script-writer-agent');
const { ThumbnailDesignerAgent } = require('./agents/thumbnail-designer-agent');
const { SEOOptimizerAgent } = require('./agents/seo-optimizer-agent');
const { ProductionManagementAgent } = require('./agents/production-management-agent');
const { PublishingSchedulingAgent } = require('./agents/publishing-scheduling-agent');
const { AnalyticsOptimizationAgent } = require('./agents/analytics-optimization-agent');
const { DailyAutomation } = require('./schedules/daily-automation');
const { OperatorService } = require('./utils/operator-service');
const { AutonomousChannelOperator } = require('./utils/autonomous-channel-operator');
const { ActivationMetrics } = require('./utils/activation-metrics');
const { AnonymousTelemetry } = require('./utils/anonymous-telemetry');
const { ProductionReadinessService } = require('./utils/production-readiness-service');
const { GenerationRecoveryService, GENERATION_STAGES } = require('./utils/generation-recovery-service');
const { ProvenanceService } = require('./utils/provenance-service');
const { SceneRepairService } = require('./utils/scene-repair-service');
const { ShortsRepurposingService } = require('./utils/shorts-repurposing-service');
const { AudienceEngagementService } = require('./utils/audience-engagement-service');
const { GrowthExperimentService } = require('./utils/growth-experiment-service');
const { AITextService, PROVIDERS, GEMINI_MODELS, GEMINI_DEFAULT_MODEL } = require('./utils/ai-text-service');
const { DiscoverabilityService } = require('./utils/discoverability-service');
const {
  VoiceboxService, DEFAULT_VOICEBOX_URL, VOICEBOX_LANGUAGES,
  VOICEBOX_TTS_ENGINES, VOICEBOX_WHISPER_MODELS
} = require('./utils/voicebox-service');
const { LMStudioService } = require('./utils/lm-studio-service');
const { safeModelError } = require('./utils/video-providers');
const { version } = require('./package.json');
const chalk = require('chalk');

function isLoopbackHost(host) {
  return ['127.0.0.1', '::1', 'localhost'].includes(String(host).toLowerCase()) || /^127\./.test(String(host));
}

function normalizeLocalModelBaseURL(value) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch (_error) {
    throw new Error('Local GGUF URL must be a valid HTTP(S) localhost URL');
  }
  if (
    !['http:', 'https:'].includes(parsed.protocol) ||
    !isLoopbackHost(parsed.hostname.replace(/^\[|\]$/g, '')) ||
    parsed.username || parsed.password || parsed.search || parsed.hash
  ) {
    throw new Error('Local GGUF URL must be an HTTP(S) localhost URL without credentials, query, or fragment');
  }
  return parsed.toString().replace(/\/+$/, '');
}

const LM_STUDIO_DEFAULT_URL = 'http://127.0.0.1:1234/v1';

function normalizeLMStudioModelId(value) {
  const model = typeof value === 'string' ? value.trim() : '';
  if (!model || model.length > 500) {
    throw new Error('Enter an LM Studio model ID or a Hugging Face model URL (up to 500 characters)');
  }
  if (/^[\w.-]+(?:\/[\w.-]+)+$/.test(model)) return model;
  let parsed;
  try {
    parsed = new URL(model);
  } catch (_error) {
    throw new Error('Enter an LM Studio model ID or an exact Hugging Face model URL');
  }
  if (
    parsed.protocol !== 'https:' || parsed.hostname !== 'huggingface.co' ||
    parsed.username || parsed.password || parsed.search || parsed.hash ||
    parsed.pathname.split('/').filter(Boolean).length < 2
  ) {
    throw new Error('Downloads accept LM Studio model IDs or exact https://huggingface.co model URLs');
  }
  return model;
}

function validateComfyUIWorkflow(value) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error('Paste a ComfyUI workflow exported in API format');
  }
  if (value.length > 1_000_000) throw new Error('ComfyUI API workflow must be smaller than 1 MB');
  let workflow;
  try {
    workflow = JSON.parse(value);
  } catch (_error) {
    throw new Error('ComfyUI workflow must be valid JSON');
  }
  if (!workflow || typeof workflow !== 'object' || Array.isArray(workflow) || !Object.keys(workflow).length) {
    throw new Error('ComfyUI workflow must be a non-empty API workflow object');
  }
  const nodes = Object.values(workflow);
  if (nodes.some(node => !node || typeof node !== 'object' || typeof node.class_type !== 'string' || !node.inputs || typeof node.inputs !== 'object')) {
    throw new Error('ComfyUI workflow must use API format with node class_type and inputs');
  }
  if (!nodes.some(node => JSON.stringify(node.inputs).includes('{{PROMPT}}'))) {
    throw new Error('Add the {{PROMPT}} placeholder to the workflow text input that should receive each scene prompt');
  }
  return workflow;
}

function normalizeComfyUIBaseURL(value) {
  try {
    return normalizeLocalModelBaseURL(value);
  } catch (_error) {
    throw new Error('ComfyUI URL must be an HTTP(S) localhost URL without credentials, query, or fragment');
  }
}

class YouTubeAutomationAgent {
  constructor() {
    this.logger = new Logger('MainAgent');
    this.db = null;
    this.credentials = null;
    this.agents = {};
    this.app = express();
    this.isInitialized = false;
    this.activeJobs = new Map();
    this.operator = null;
    this.autonomous = null;
    this.activation = null;
    this.telemetry = null;
    this.readiness = null;
    this.recovery = null;
    this.provenance = null;
    this.scenes = null;
    this.shorts = null;
    this.engagement = null;
    this.experiments = null;
    this.discoverability = null;
    this.setupRequired = false;
    this.dashboardAuth = { status: 'idle', error: null };
    this.dashboardConfigurationRestartRequired = false;
  }

  async initialize() {
    try {
      console.log(chalk.cyan.bold(`\n🎬 YouTube Automation Agent v${version}`));
      console.log(chalk.gray('─'.repeat(50)));
      
      // Initialize database
      this.logger.info('Initializing database...');
      this.db = new Database();
      await this.db.initialize();
      await this.db.markInterruptedJobs();
      this.recovery = new GenerationRecoveryService(this.db, {
        logger: this.logger,
        updateJobStage: (...args) => this.updateJobStage(...args)
      });
      this.operator = new OperatorService(this.db);
      this.provenance = new ProvenanceService(this.db);
      this.discoverability = new DiscoverabilityService(this.db, { logger: this.logger });
      this.autonomous = new AutonomousChannelOperator(this.db, {
        researchAndPlan: strategy => {
          if (!this.agents.strategy) throw new Error('The strategy agent is not configured');
          return this.agents.strategy.researchAndPlanChannel(strategy);
        },
        startGenerationJob: input => this.startGenerationJob(input),
        resumeGenerationJob: (jobId, options) => this.resumeGenerationJob(jobId, options),
        waitForGenerationJob: jobId => this.waitForGenerationJob(jobId),
        notify: notification => this.operator.notify(notification)
      });
      this.activation = new ActivationMetrics(this.db);
      this.telemetry = new AnonymousTelemetry(this.db, this.logger);
      
      // Load credentials
      this.logger.info('Loading credentials...');
      this.credentials = new CredentialManager();
      const credentialsValid = await this.credentials.validateAll();
      this.readiness = new ProductionReadinessService(this.db, this.credentials);
      
      if (!credentialsValid) {
        console.log(chalk.yellow('\n⚠️  Some credentials are missing or invalid.'));
        console.log(chalk.yellow('Run: npm run credentials:setup'));
        this.setupRequired = true;
        this.setupAPI();
        this.isInitialized = true;
        this.logger.warn('Dashboard started in setup mode; generation and publishing are disabled');
        return true;
      }
      
      // Initialize agents
      this.logger.info('Initializing agents...');
      await this.initializeAgents();
      this.scenes = this.agents.production?.sceneRepair || new SceneRepairService(
        this.db,
        this.agents.production?.aiVideoGenerator,
        { logger: this.logger }
      );
      this.shorts = new ShortsRepurposingService(this.db, this.agents.publishing, { logger: this.logger });
      this.engagement = new AudienceEngagementService(
        this.db,
        this.credentials,
        new AITextService(this.credentials?.credentials || {}),
        { logger: this.logger }
      );
      this.experiments = new GrowthExperimentService(
        this.db,
        this.agents.analytics,
        this.agents.publishing,
        { logger: this.logger }
      );

      // Show which pipeline stages will run for real vs. be simulated
      const capabilities = await this.logCapabilitySummary();
      if (capabilities.hasText && capabilities.hasFFmpeg && capabilities.hasUpload) {
        await this.activation.markSetupReady(capabilities);
      }
      
      // Setup API endpoints
      this.setupAPI();
      
      // Initialize scheduler
      this.logger.info('Setting up automation scheduler...');
      this.scheduler = new DailyAutomation(this.agents, this.db, {
        generateContent: input => this.queueScheduledContent(input),
        engagement: this.engagement,
        experiments: this.experiments
      });
      await this.scheduler.initialize();

      if (await this.db.getSetting('automation_paused') === 'true') {
        await this.scheduler.pauseAutomation();
      }
      
      this.isInitialized = true;
      this.logger.success('YouTube Automation Agent initialized successfully!');
      
      return true;
    } catch (error) {
      this.logger.error('Failed to initialize:', error);
      return false;
    }
  }

  async initializeAgents() {
    this.agents = {
      strategy: new ContentStrategyAgent(this.db, this.credentials),
      scriptWriter: new ScriptWriterAgent(this.db, this.credentials),
      thumbnailDesigner: new ThumbnailDesignerAgent(this.db, this.credentials),
      seoOptimizer: new SEOOptimizerAgent(this.db, this.credentials),
      production: new ProductionManagementAgent(this.db, this.credentials),
      publishing: new PublishingSchedulingAgent(this.db, this.credentials),
      analytics: new AnalyticsOptimizationAgent(this.db, this.credentials)
    };

    // Initialize each agent
    for (const [name, agent] of Object.entries(this.agents)) {
      await agent.initialize();
      this.logger.info(`✓ ${name} agent initialized`);
    }
  }

  async logCapabilitySummary() {
    const { checkFFmpeg, ffmpegInstallHint } = require('./utils/ffmpeg');
    const creds = this.credentials.credentials || {};

    const hasText = this.credentials.hasAITextProvider();
    const hasGemini = Boolean(creds.gemini?.apiKey || process.env.GEMINI_API_KEY);
    const hasImages = true;
    const hasTTS = Boolean(
      creds.openai?.apiKey || process.env.OPENAI_API_KEY ||
      creds.elevenLabs?.apiKey || process.env.ELEVENLABS_API_KEY ||
      creds.azureSpeech?.subscriptionKey || process.env.AZURE_SPEECH_KEY ||
      creds.voicebox?.profileId || process.env.VOICEBOX_PROFILE_ID ||
      hasGemini
    );
    const hasFFmpeg = await checkFFmpeg();
    const hasUpload = Boolean(creds.youtube && this.credentials.tokens?.youtube);

    const capabilities = [
      { name: 'Script & strategy generation', ok: hasText, hint: 'configure an AI provider (npm run credentials:setup)' },
      { name: 'Image generation (visuals/thumbnails)', ok: hasImages, hint: 'built-in local artwork and thumbnail design work without cloud image API keys' },
      { name: 'Voice narration (TTS)', ok: hasTTS, hint: 'configure OpenAI, Gemini, ElevenLabs, or Azure Speech — otherwise videos are silent' },
      { name: 'Video assembly (FFmpeg)', ok: hasFFmpeg, hint: ffmpegInstallHint() },
      { name: 'YouTube upload', ok: hasUpload, hint: 'run: npm run credentials:setup' }
    ];

    console.log(chalk.cyan('\n🔎 Capability check:'));
    for (const cap of capabilities) {
      if (cap.ok) {
        console.log(chalk.green(`  ✓ ${cap.name}`));
      } else {
        console.log(chalk.yellow(`  ✗ ${cap.name} — ${cap.hint}`));
      }
    }

    if (!hasFFmpeg) {
      this.logger.warn('FFmpeg is missing: no .mp4 files can be produced until it is installed.');
    }
    console.log('');
    return { hasText, hasImages, hasTTS, hasFFmpeg, hasUpload };
  }

  requireAPIKey() {
    return (req, res, next) => {
      if (!process.env.API_KEY) {
        return next();
      }

      if (req.get('x-api-key') !== process.env.API_KEY) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
      }

      return next();
    };
  }

  validateGenerateRequestBody(body = {}) {
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      return { valid: false, status: 400, error: 'Request body must be a JSON object' };
    }

    const value = {
      topic: null,
      style: null,
      length: typeof body.length === 'string' ? body.length.toLowerCase() : 'medium',
      format: typeof body.format === 'string' ? body.format.toLowerCase() : 'video',
      targetDuration: null,
      strategyContext: null
    };

    // JSON has no `undefined`, so clients send `null` to mean "no value provided".
    // Both are treated as "not set" here: topic/style are optional and default to
    // auto-selection, which is exactly what `null` already represents internally.
    if (body.topic !== undefined && body.topic !== null) {
      if (typeof body.topic !== 'string') {
        return { valid: false, status: 400, error: 'topic must be a string' };
      }

      const topic = body.topic.trim();
      if (topic.length > 200) {
        return { valid: false, status: 400, error: 'topic must be 200 characters or less' };
      }
      value.topic = topic || null;
    }

    if (body.style !== undefined && body.style !== null) {
      if (typeof body.style !== 'string') {
        return { valid: false, status: 400, error: 'style must be a string' };
      }

      const allowedStyles = new Set([
        'tutorial',
        'explainer',
        'list',
        'review',
        'story',
        'song',
        'lookfind',
        'activity',
        'educational',
        'informative',
        'engaging',
        'professional',
        'ethereal'
      ]);
      const style = body.style.trim();

      if (style.length > 50) {
        return { valid: false, status: 400, error: 'style must be 50 characters or less' };
      }

      value.style = allowedStyles.has(style.toLowerCase()) ? style.toLowerCase() : style || null;
    }

    if (!['short', 'medium', 'long'].includes(value.length)) {
      return { valid: false, status: 400, error: 'length must be short, medium, or long' };
    }

    if (!['video', 'youtube_short'].includes(value.format)) {
      return { valid: false, status: 400, error: 'format must be video or youtube_short' };
    }
    if (body.targetDuration !== undefined && body.targetDuration !== null && body.targetDuration !== '') {
      const targetDuration = Number(body.targetDuration);
      const minimum = value.format === 'youtube_short' ? 15 : 1;
      const maximum = value.format === 'youtube_short' ? 180 : 60;
      if (!Number.isInteger(targetDuration) || targetDuration < minimum || targetDuration > maximum) {
        const unit = value.format === 'youtube_short' ? 'seconds' : 'minutes';
        return { valid: false, status: 400, error: `targetDuration must be ${minimum}–${maximum} ${unit} for this format` };
      }
      value.targetDuration = targetDuration;
    }

    if (body.strategyContext !== undefined && body.strategyContext !== null) {
      if (typeof body.strategyContext !== 'object' || Array.isArray(body.strategyContext)) {
        return { valid: false, status: 400, error: 'strategyContext must be an object' };
      }
      const limits = { angle: 500, rationale: 1000, audience: 500, objective: 1000, valueProposition: 1000, constraints: 2000, pillar: 100 };
      value.strategyContext = {};
      for (const [key, max] of Object.entries(limits)) {
        if (body.strategyContext[key] === undefined || body.strategyContext[key] === null) continue;
        if (typeof body.strategyContext[key] !== 'string' || body.strategyContext[key].length > max) {
          return { valid: false, status: 400, error: `strategyContext.${key} must be a string of ${max} characters or less` };
        }
        value.strategyContext[key] = body.strategyContext[key].trim();
      }
    }

    return { valid: true, value };
  }

  validateChannelStrategy(body = {}, current = {}) {
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new Error('Channel strategy must be a JSON object');
    }
    const text = (key, fallback, max) => {
      const value = String(body[key] ?? fallback ?? '').trim();
      if (value.length > max) throw new Error(`${key} must be ${max} characters or less`);
      return value;
    };
    const objective = text('objective', current.objective, 1000);
    const audience = text('audience', current.audience, 500);
    if (!objective) throw new Error('A channel objective is required');
    if (!audience) throw new Error('A target audience is required');

    const rawPillars = body.contentPillars ?? current.contentPillars ?? [];
    if (!Array.isArray(rawPillars)) throw new Error('contentPillars must be an array');
    const contentPillars = rawPillars.map(value => String(value).trim()).filter(Boolean);
    if (!contentPillars.length || contentPillars.length > 8 || contentPillars.some(value => value.length > 100)) {
      throw new Error('Provide 1 to 8 content pillars, each 100 characters or less');
    }

    const integer = (key, fallback, min, max) => {
      const value = Number(body[key] ?? fallback);
      if (!Number.isInteger(value) || value < min || value > max) {
        throw new Error(`${key} must be an integer from ${min} to ${max}`);
      }
      return value;
    };
    const defaultFormat = text('defaultFormat', current.default_format || 'explainer', 20).toLowerCase();
    const defaultLength = text('defaultLength', current.default_length || 'medium', 20).toLowerCase();
    const status = text('status', current.status || 'draft', 20).toLowerCase();
    const primaryKpi = text('primaryKpi', current.primary_kpi || 'views', 30).toLowerCase();
    const outcomeCurrency = text('outcomeCurrency', current.outcome_currency || 'USD', 3).toUpperCase();
    if (!['explainer', 'tutorial', 'list', 'review', 'story'].includes(defaultFormat)) {
      throw new Error('defaultFormat is not supported');
    }
    if (!['short', 'medium', 'long'].includes(defaultLength)) throw new Error('defaultLength is not supported');
    if (!['draft', 'active', 'paused'].includes(status)) throw new Error('status must be draft, active, or paused');
    if (!['views', 'watch_hours', 'subscribers', 'engagement', 'revenue'].includes(primaryKpi)) {
      throw new Error('primaryKpi is not supported');
    }
    if (!/^[A-Z]{3}$/.test(outcomeCurrency)) throw new Error('outcomeCurrency must be a three-letter currency code');
    const optionalNumber = (key, fallback, min, max) => {
      const raw = body[key] ?? fallback;
      if (raw === undefined || raw === null || raw === '') return null;
      const value = Number(raw);
      if (!Number.isFinite(value) || value < min || value > max) {
        throw new Error(`${key} must be a number from ${min} to ${max}`);
      }
      return value;
    };

    return {
      objective,
      audience,
      valueProposition: text('valueProposition', current.value_proposition, 1000),
      contentPillars,
      cadencePerWeek: integer('cadencePerWeek', current.cadence_per_week || 1, 1, 7),
      videosPerRun: integer('videosPerRun', current.videos_per_run || 1, 1, 5),
      defaultFormat,
      defaultLength,
      successMetric: text('successMetric', current.success_metric, 300),
      primaryKpi,
      targetValue: optionalNumber('targetValue', current.target_value, 0.01, 1000000000),
      targetWindowDays: integer('targetWindowDays', current.target_window_days || 28, 7, 365),
      monthlyBudget: optionalNumber('monthlyBudget', current.monthly_budget, 0, 10000000),
      outcomeCurrency,
      constraints: text('constraints', current.constraints, 2000),
      status
    };
  }
  setupAPI() {
    this.app.use(express.json({ limit: '1mb' }));
    this.app.use(express.static(path.join(__dirname, 'dashboard')));

    if (!process.env.API_KEY) {
      this.logger.warn('API_KEY is not set; mutating API routes are unprotected');
    }
    
    // Main dashboard route
    this.app.get('/', (req, res) => {
      res.sendFile(path.join(__dirname, 'dashboard', 'index.html'));
    });
    
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: this.setupRequired ? 'setup_required' : 'healthy',
        initialized: this.isInitialized,
        setupRequired: this.setupRequired,
        agents: Object.keys(this.agents),
        uptime: process.uptime(),
        timestamp: new Date().toISOString()
      });
    });

    // Manual content generation
    this.app.post('/generate', this.requireAPIKey(), async (req, res) => {
      try {
        if (this.setupRequired) {
          return res.status(503).json({ success: false, error: 'Finish setup with npm run walkthrough before generating content' });
        }
        const validation = this.validateGenerateRequestBody(req.body);
        if (!validation.valid) {
          return res.status(validation.status).json({ success: false, error: validation.error });
        }

        const { topic, style, length, format, targetDuration } = validation.value;
        const result = await this.startGenerationJob({ topic, style, length, format, targetDuration, source: 'manual' });
        res.status(202).json({ success: true, result });
      } catch (error) {
        res.status(error.status || 500).json({ success: false, error: error.message });
      }
    });

    // Get analytics
    this.app.get('/analytics', async (req, res) => {
      try {
        if (!this.agents.analytics) return res.json({ totalVideos: 0, averagePerformanceScore: 0, topPerformers: [], insights: [], learning: null });
        const analytics = await this.agents.analytics.getRecentAnalytics();
        const learning = await this.agents.analytics.getLearningSummary();
        res.json({ ...analytics, learning });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    this.app.get('/api/analytics/channel', async (req, res) => {
      try {
        if (!this.agents.analytics?.getChannelSnapshot) {
          return res.json({ success: true, result: { available: false, reason: 'Connect YouTube with npm run walkthrough to see channel statistics.' } });
        }
        const result = await this.agents.analytics.getChannelSnapshot({ refresh: req.query.refresh === '1' });
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(500).json({ success: false, error: error.message });
      }
    });

    this.app.get('/api/outcomes', async (_req, res) => {
      try {
        const learning = this.agents.analytics?.getLearningSummary
          ? await this.agents.analytics.getLearningSummary()
          : { outcome: null };
        return res.json({ success: true, result: learning.outcome || null });
      } catch (error) {
        return res.status(500).json({ success: false, error: error.message });
      }
    });

    // Get upcoming schedule
    this.app.get('/schedule', async (req, res) => {
      try {
        const schedule = await this.db.getUpcomingSchedule();
        res.json(schedule);
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // Manual publish
    this.app.post('/publish/:contentId', this.requireAPIKey(), async (req, res) => {
      try {
        if (!this.agents.publishing) return res.status(503).json({ success: false, error: 'YouTube publishing is not configured' });
        const { contentId } = req.params;
        const bundle = await this.db.getProductionBundle(contentId);
        const short = bundle ? null : await this.db.getShortClip(contentId);
        if ((!bundle || bundle.review_status !== 'approved') && (!short || !['scheduled', 'uploading', 'reconciliation_required'].includes(short.status))) {
          return res.status(409).json({ success: false, error: 'Content must pass review and be approved before publishing' });
        }
        const result = await this.agents.publishing.publishContent(contentId);
        res.json({ success: true, result });
      } catch (error) {
        res.status(error.status || 500).json({ success: false, error: error.message });
      }
    });

    this.setupOperatorAPI();
  }

  setupOperatorAPI() {
    const protect = this.requireAPIKey();
    const getLMStudioService = body => {
      const saved = this.credentials?.credentials?.aiProvider;
      const baseURL = typeof body?.baseURL === 'string' && body.baseURL.trim()
        ? body.baseURL.trim()
        : saved?.provider === 'lmstudio' ? saved.baseURL : LM_STUDIO_DEFAULT_URL;
      const apiKey = typeof body?.apiKey === 'string' && body.apiKey.trim()
        ? body.apiKey.trim()
        : saved?.provider === 'lmstudio' ? saved.apiKey || '' : '';
      let normalizedBaseURL;
      try {
        normalizedBaseURL = normalizeLocalModelBaseURL(baseURL);
      } catch (error) {
        error.status = 400;
        throw error;
      }
      return new LMStudioService({
        baseURL: normalizedBaseURL,
        apiKey
      });
    };

    this.app.get('/api/configuration', async (_req, res) => {
      try {
        const credentials = this.credentials?.credentials || {};
        const providerCatalog = {
          local: {
            name: 'Local GGUF (llama.cpp)',
            models: credentials.aiProvider?.provider === 'local' && credentials.aiProvider?.model
              ? [credentials.aiProvider.model] : [],
            defaultModel: credentials.aiProvider?.model || '',
            envKey: null
          },
          lmstudio: {
            name: 'LM Studio (local)',
            models: credentials.aiProvider?.provider === 'lmstudio' && credentials.aiProvider?.model
              ? [credentials.aiProvider.model] : [],
            defaultModel: credentials.aiProvider?.model || '',
            envKey: null
          },
          gemini: {
            name: 'Google Gemini',
            models: GEMINI_MODELS,
            defaultModel: GEMINI_DEFAULT_MODEL,
            envKey: 'GEMINI_API_KEY'
          },
          ...Object.fromEntries(Object.entries(PROVIDERS).map(([id, provider]) => [id, {
            name: provider.name,
            models: provider.models,
            defaultModel: provider.defaultModel,
            envKey: provider.envKey
          }]))
        };
        const savedProvider = credentials.aiProvider?.provider;
        const savedProviderConfigured = ['local', 'lmstudio'].includes(savedProvider)
          ? Boolean(credentials.aiProvider?.baseURL && credentials.aiProvider?.model)
          : savedProvider === 'gemini'
            ? Boolean(credentials.aiProvider?.apiKey || credentials.gemini?.apiKey || process.env.GEMINI_API_KEY)
            : Boolean(providerCatalog[savedProvider] && (
              credentials.aiProvider?.apiKey || process.env[providerCatalog[savedProvider].envKey]
            ));
        const aiProvider = savedProviderConfigured
          ? savedProvider
          : Object.entries(PROVIDERS).find(([id, provider]) =>
            (id === 'openai' && credentials.openai?.apiKey) || process.env[provider.envKey]
          )?.[0] || (credentials.gemini?.apiKey || process.env.GEMINI_API_KEY ? 'gemini' : null);
        const textProviders = Object.entries(providerCatalog).map(([id, provider]) => ({
          id,
          name: provider.name,
          models: provider.models,
          defaultModel: provider.defaultModel,
          configured: Boolean(
            (['local', 'lmstudio'].includes(id) && credentials.aiProvider?.provider === id && credentials.aiProvider?.baseURL && credentials.aiProvider?.model) ||
            (id === 'gemini' && (credentials.aiProvider?.provider === 'gemini' && credentials.aiProvider?.apiKey || credentials.gemini?.apiKey)) ||
            (id === 'openai' && credentials.openai?.apiKey) ||
            (credentials.aiProvider?.provider === id && credentials.aiProvider?.apiKey) ||
            (provider.envKey && process.env[provider.envKey])
          )
        }));
        const elevenLabsConfigured = Boolean(
          (credentials.elevenLabs?.apiKey || process.env.ELEVENLABS_API_KEY) &&
          (credentials.elevenLabs?.voiceId || process.env.ELEVENLABS_VOICE_ID)
        );
        const voicebox = credentials.voicebox || {};
        return res.json({
          success: true,
          result: {
            textProvider: {
              selected: aiProvider,
              localProvider: ['local', 'lmstudio'].includes(credentials.aiProvider?.provider)
                ? credentials.aiProvider.provider : '',
              localBaseUrl: ['local', 'lmstudio'].includes(credentials.aiProvider?.provider) ? credentials.aiProvider.baseURL || '' : '',
              localModel: ['local', 'lmstudio'].includes(credentials.aiProvider?.provider) ? credentials.aiProvider.model || '' : '',
              model: credentials.aiProvider?.model || credentials.gemini?.model ||
                (aiProvider === 'anthropic' ? process.env.ANTHROPIC_MODEL : null) ||
                providerCatalog[aiProvider]?.defaultModel || null,
              providers: textProviders
            },
            tts: {
              selected: process.env.TTS_PROVIDER || credentials.ttsProvider || 'auto',
              configured: {
                voicebox: Boolean(voicebox.profileId || process.env.VOICEBOX_PROFILE_ID),
                elevenlabs: elevenLabsConfigured,
                openai: Boolean(credentials.openai?.apiKey || process.env.OPENAI_API_KEY),
                gemini: Boolean(credentials.gemini?.apiKey || process.env.GEMINI_API_KEY)
              },
              voicebox: {
                baseUrl: voicebox.baseUrl || process.env.VOICEBOX_BASE_URL || DEFAULT_VOICEBOX_URL,
                profileId: voicebox.profileId || process.env.VOICEBOX_PROFILE_ID || '',
                engine: voicebox.engine || process.env.VOICEBOX_ENGINE || '',
                modelSize: voicebox.modelSize || process.env.VOICEBOX_MODEL_SIZE || '',
                language: voicebox.language || process.env.VOICEBOX_LANGUAGE || 'en',
                instruct: voicebox.instruct || process.env.VOICEBOX_INSTRUCT || '',
                maxChunkChars: voicebox.maxChunkChars || Number(process.env.VOICEBOX_MAX_CHUNK_CHARS) || 800,
                crossfadeMs: voicebox.crossfadeMs ?? Number(process.env.VOICEBOX_CROSSFADE_MS || 50),
                normalize: voicebox.normalize !== false && process.env.VOICEBOX_NORMALIZE !== 'false'
              },
              elevenLabs: {
                voiceId: credentials.elevenLabs?.voiceId || process.env.ELEVENLABS_VOICE_ID || ''
              }
            },
            youtube: {
              credentialsConfigured: Boolean(credentials.youtube?.client_id && credentials.youtube?.client_secret),
              connected: Boolean(this.credentials?.tokens?.youtube),
              auth: this.dashboardAuth
            },
            restartRequired: this.setupRequired || this.dashboardConfigurationRestartRequired
          }
        });
      } catch (error) {
        return res.status(500).json({ success: false, error: error.message });
      }
    });

    this.app.put('/api/configuration', protect, async (req, res) => {
      try {
        if (!this.credentials) return res.status(503).json({ success: false, error: 'Credential storage is not initialized' });

        const body = req.body || {};
        const providerCatalog = {
          local: { models: [], defaultModel: '', envKey: null },
          lmstudio: { models: [], defaultModel: '', envKey: null },
          gemini: { models: GEMINI_MODELS, defaultModel: GEMINI_DEFAULT_MODEL, envKey: 'GEMINI_API_KEY' },
          ...Object.fromEntries(Object.entries(PROVIDERS).map(([id, provider]) => [id, provider]))
        };
        const text = body.textProvider;
        const tts = body.tts;
        if (!text || !tts || typeof text !== 'object' || typeof tts !== 'object') {
          return res.status(400).json({ success: false, error: 'Text-provider and TTS settings are required' });
        }

        const provider = providerCatalog[text.provider];
        if (!provider) return res.status(400).json({ success: false, error: 'Unsupported text provider' });
        const textKey = typeof text.apiKey === 'string' ? text.apiKey.trim() : '';
        const current = this.credentials.credentials || {};
        const existingTextKey = text.provider === 'gemini'
          ? current.gemini?.apiKey
          : text.provider === 'openai'
            ? current.aiProvider?.provider === 'openai' ? current.aiProvider.apiKey : current.openai?.apiKey
            : current.aiProvider?.provider === text.provider ? current.aiProvider.apiKey : null;
        const isLocalProvider = ['local', 'lmstudio'].includes(text.provider);
        if (!isLocalProvider && !textKey && !existingTextKey && !process.env[provider.envKey]) {
          return res.status(400).json({ success: false, error: `Enter an API key for ${text.provider}, or configure ${provider.envKey} in .env` });
        }
        const model = typeof text.model === 'string' ? text.model.trim() : '';
        let localBaseURL;
        if (isLocalProvider) {
          if (!model || model.length > 200) {
            return res.status(400).json({ success: false, error: `Enter the model name configured in ${text.provider === 'lmstudio' ? 'LM Studio' : 'the local GGUF server'}` });
          }
          try {
            localBaseURL = normalizeLocalModelBaseURL(text.baseURL);
          } catch (_error) {
            return res.status(400).json({ success: false, error: 'Local AI server URL must be an HTTP(S) localhost URL without credentials, query, or fragment' });
          }
        } else if (!provider.models?.includes(model)) {
          return res.status(400).json({ success: false, error: 'Select a supported model for the chosen text provider' });
        }

        const allowedTTS = ['auto', 'voicebox', 'elevenlabs', 'openai', 'gemini'];
        if (!allowedTTS.includes(tts.provider)) {
          return res.status(400).json({ success: false, error: 'Unsupported narration provider' });
        }
        const voiceboxUrl = typeof tts.voicebox?.baseUrl === 'string' ? tts.voicebox.baseUrl.trim() : '';
        const voiceboxProfileId = typeof tts.voicebox?.profileId === 'string' ? tts.voicebox.profileId.trim() : '';
        const elevenLabsKey = typeof tts.elevenLabs?.apiKey === 'string' ? tts.elevenLabs.apiKey.trim() : '';
        const elevenLabsVoiceId = typeof tts.elevenLabs?.voiceId === 'string' ? tts.elevenLabs.voiceId.trim() : '';
        const ttsKey = typeof tts.apiKey === 'string' ? tts.apiKey.trim() : '';
        const maxChunkChars = Number(tts.voicebox?.maxChunkChars ?? 800);
        const crossfadeMs = Number(tts.voicebox?.crossfadeMs ?? 50);
        const credentials = { ...current };

        if (text.provider === 'gemini') {
          credentials.gemini = {
            ...current.gemini,
            ...(textKey ? { apiKey: textKey } : {}),
            model
          };
          credentials.aiProvider = {
            provider: 'gemini',
            model,
            ...(textKey ? { apiKey: textKey } : {})
          };
        } else if (isLocalProvider) {
          credentials.aiProvider = {
            provider: text.provider,
            baseURL: localBaseURL,
            model,
            ...(textKey || existingTextKey ? { apiKey: textKey || existingTextKey } : {})
          };
        } else {
          const selectedKey = textKey || existingTextKey;
          credentials.aiProvider = { provider: text.provider, apiKey: selectedKey, model };
          if (text.provider === 'openai') {
            credentials.openai = { ...current.openai, apiKey: selectedKey, model };
          }
        }

        if (tts.provider === 'voicebox') {
          if (!voiceboxProfileId) {
            return res.status(400).json({ success: false, error: 'Select or enter a Voicebox voice profile ID' });
          }
          const selectedEngine = typeof tts.voicebox?.engine === 'string' && tts.voicebox.engine.trim()
            ? VOICEBOX_TTS_ENGINES.find(engine => engine.id === tts.voicebox.engine.trim())
            : null;
          if (tts.voicebox?.engine && !selectedEngine) {
            return res.status(400).json({ success: false, error: 'Select a supported Voicebox TTS engine' });
          }
          const modelSize = typeof tts.voicebox?.modelSize === 'string' ? tts.voicebox.modelSize.trim() : '';
          if (modelSize && !selectedEngine?.modelSizes.includes(modelSize)) {
            return res.status(400).json({ success: false, error: 'Select a valid model size for the selected Voicebox engine' });
          }
          const language = typeof tts.voicebox?.language === 'string' ? tts.voicebox.language.trim() : '';
          if (language && !VOICEBOX_LANGUAGES.some(([code]) => code === language)) {
            return res.status(400).json({ success: false, error: 'Select a Voicebox-supported narration language' });
          }
          if (!Number.isInteger(maxChunkChars) || maxChunkChars < 100 || maxChunkChars > 5000) {
            return res.status(400).json({ success: false, error: 'Voicebox chunk size must be between 100 and 5,000 characters' });
          }
          if (!Number.isInteger(crossfadeMs) || crossfadeMs < 0 || crossfadeMs > 500) {
            return res.status(400).json({ success: false, error: 'Voicebox crossfade must be between 0 and 500 milliseconds' });
          }
          const { normalizeVoiceboxUrl } = require('./utils/voicebox-service');
          try {
            normalizeVoiceboxUrl(voiceboxUrl);
          } catch (error) {
            return res.status(400).json({ success: false, error: error.message });
          }
        }
        if (tts.provider === 'openai' && ttsKey) {
          credentials.openai = { ...(credentials.openai || {}), apiKey: ttsKey };
        }
        if (tts.provider === 'gemini' && ttsKey) {
          credentials.gemini = { ...(credentials.gemini || {}), apiKey: ttsKey };
        }
        if (tts.provider === 'elevenlabs' && !(
          (elevenLabsKey || current.elevenLabs?.apiKey || process.env.ELEVENLABS_API_KEY) &&
          (elevenLabsVoiceId || current.elevenLabs?.voiceId || process.env.ELEVENLABS_VOICE_ID)
        )) {
          return res.status(400).json({ success: false, error: 'ElevenLabs requires an API key and Voice ID' });
        }
        if (tts.provider === 'openai' && !(credentials.openai?.apiKey || process.env.OPENAI_API_KEY)) {
          return res.status(400).json({ success: false, error: 'OpenAI narration requires an OpenAI API key' });
        }
        if (tts.provider === 'gemini' && !(credentials.gemini?.apiKey || process.env.GEMINI_API_KEY)) {
          return res.status(400).json({ success: false, error: 'Gemini narration requires a Gemini API key' });
        }
        if (tts.provider === 'voicebox') {
          credentials.voicebox = {
            ...(current.voicebox || {}),
            baseUrl: voiceboxUrl,
            profileId: voiceboxProfileId,
            engine: typeof tts.voicebox?.engine === 'string' ? tts.voicebox.engine.trim() : '',
            modelSize: typeof tts.voicebox?.modelSize === 'string' ? tts.voicebox.modelSize.trim() : '',
            ...(typeof tts.voicebox?.language === 'string' && tts.voicebox.language.trim() ? { language: tts.voicebox.language.trim() } : {}),
            ...(typeof tts.voicebox?.instruct === 'string' && tts.voicebox.instruct.trim() ? { instruct: tts.voicebox.instruct.trim() } : {}),
            maxChunkChars,
            crossfadeMs,
            normalize: tts.voicebox?.normalize !== false
          };
        }
        if (tts.provider === 'elevenlabs') {
          credentials.elevenLabs = {
            ...(current.elevenLabs || {}),
            ...(elevenLabsKey ? { apiKey: elevenLabsKey } : {}),
            ...(elevenLabsVoiceId ? { voiceId: elevenLabsVoiceId } : {})
          };
        }
        if (tts.provider === 'auto') delete credentials.ttsProvider;
        else credentials.ttsProvider = tts.provider;

        this.credentials.credentials = credentials;
        await this.credentials.saveCredentials();
        if (tts.provider === 'auto') delete process.env.TTS_PROVIDER;
        else process.env.TTS_PROVIDER = tts.provider;

        this.dashboardConfigurationRestartRequired = true;
        const restartRequired = true;
        return res.json({ success: true, result: { restartRequired, message: 'Settings saved securely. Restart the application to reinitialize providers.' } });
      } catch (error) {
        return res.status(400).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/configuration/local-model/check', protect, async (req, res) => {
      try {
        const baseURL = typeof req.body?.baseURL === 'string' ? req.body.baseURL.trim() : '';
        const model = typeof req.body?.model === 'string' ? req.body.model.trim() : '';
        const provider = req.body?.provider === 'lmstudio' ? 'lmstudio' : 'local';
        const storedProvider = this.credentials?.credentials?.aiProvider;
        const apiKey = typeof req.body?.apiKey === 'string' && req.body.apiKey.trim()
          ? req.body.apiKey.trim()
          : provider === 'lmstudio' && storedProvider?.provider === 'lmstudio'
            ? storedProvider.apiKey || '' : '';
        if (!model || model.length > 200) {
          return res.status(400).json({ success: false, error: 'Enter a local HTTP(S) server URL and a model name' });
        }
        let normalizedBaseURL;
        try {
          normalizedBaseURL = normalizeLocalModelBaseURL(baseURL);
        } catch (error) {
          return res.status(400).json({ success: false, error: error.message });
        }
        const service = new AITextService({
          aiProvider: {
            provider,
            baseURL: normalizedBaseURL,
            model,
            ...(apiKey ? { apiKey } : {})
          }
        });
        if (!service.isAvailable()) return res.status(503).json({ success: false, error: 'Local model client could not be initialized' });
        await service.generateText('Reply with the single word OK.', { maxTokens: 16, temperature: 0 });
        return res.json({ success: true, result: { message: provider === 'lmstudio' ? 'LM Studio responded successfully.' : 'Local model responded successfully.' } });
      } catch (error) {
        const provider = req.body?.provider === 'lmstudio' ? 'LM Studio' : 'Local model';
        return res.status(502).json({ success: false, error: `${provider} test failed: ${error.message}` });
      }
    });

    this.app.post('/api/configuration/lm-studio/models', protect, async (req, res) => {
      try {
        const service = getLMStudioService(req.body);
        const payload = await service.listModels();
        if (!Array.isArray(payload?.models)) {
          return res.status(502).json({ success: false, error: 'LM Studio returned an invalid model inventory response' });
        }
        const models = payload.models;
        return res.json({
          success: true,
          result: {
            models: models.map(model => ({
              key: typeof model.key === 'string' ? model.key : '',
              type: model.type || '',
              displayName: model.display_name || model.key || '',
              sizeBytes: Number.isFinite(model.size_bytes) ? model.size_bytes : null,
              paramsString: model.params_string || null,
              quantization: model.quantization?.name || null,
              format: model.format || null,
              maxContextLength: model.max_context_length || null,
              loadedInstances: Array.isArray(model.loaded_instances)
                ? model.loaded_instances.map(instance => ({
                  id: instance.id,
                  contextLength: instance.config?.context_length || null
                })) : []
            }))
          }
        });
      } catch (error) {
        return res.status(error.status || 502).json({ success: false, error: `Could not list LM Studio models: ${error.message}` });
      }
    });

    this.app.post('/api/configuration/lm-studio/models/load', protect, async (req, res) => {
      try {
        const model = typeof req.body?.model === 'string' ? req.body.model.trim() : '';
        if (!model || model.length > 200) {
          return res.status(400).json({ success: false, error: 'Select an LM Studio model to load' });
        }
        const options = { echo_load_config: true };
        if (req.body?.contextLength !== undefined && req.body.contextLength !== '') {
          const contextLength = Number(req.body.contextLength);
          if (!Number.isInteger(contextLength) || contextLength < 512 || contextLength > 262144) {
            return res.status(400).json({ success: false, error: 'Context length must be an integer from 512 to 262,144' });
          }
          options.context_length = contextLength;
        }
        for (const [input, key, label, maximum] of [
          ['evalBatchSize', 'eval_batch_size', 'Evaluation batch size', 262144],
          ['numExperts', 'num_experts', 'MoE expert count', 1024]
        ]) {
          if (req.body?.[input] === undefined || req.body[input] === '') continue;
          const value = Number(req.body[input]);
          if (!Number.isInteger(value) || value < 1 || value > maximum) {
            return res.status(400).json({ success: false, error: `${label} must be an integer from 1 to ${maximum.toLocaleString('en-US')}` });
          }
          options[key] = value;
        }
        for (const [input, key] of [
          ['flashAttention', 'flash_attention'],
          ['offloadKVCacheToGPU', 'offload_kv_cache_to_gpu']
        ]) {
          if (typeof req.body?.[input] !== 'boolean') {
            return res.status(400).json({ success: false, error: `Choose whether ${key === 'flash_attention' ? 'Flash Attention' : 'GPU KV cache offload'} is enabled` });
          }
          options[key] = req.body[input];
        }
        const result = await getLMStudioService(req.body).loadModel(model, options);
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 502).json({ success: false, error: `Could not load LM Studio model: ${error.message}` });
      }
    });

    this.app.post('/api/configuration/lm-studio/models/unload', protect, async (req, res) => {
      try {
        const instanceId = typeof req.body?.instanceId === 'string' ? req.body.instanceId.trim() : '';
        if (!instanceId || instanceId.length > 200) {
          return res.status(400).json({ success: false, error: 'Select a loaded LM Studio model instance to unload' });
        }
        const result = await getLMStudioService(req.body).unloadModel(instanceId);
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 502).json({ success: false, error: `Could not unload LM Studio model: ${error.message}` });
      }
    });

    this.app.post('/api/configuration/lm-studio/models/download', protect, async (req, res) => {
      try {
        let model;
        try {
          model = normalizeLMStudioModelId(req.body?.model);
        } catch (error) {
          return res.status(400).json({ success: false, error: error.message });
        }
        const quantization = typeof req.body?.quantization === 'string' ? req.body.quantization.trim() : '';
        if (quantization && !model.startsWith('https://huggingface.co/')) {
          return res.status(400).json({ success: false, error: 'Quantization is supported only when downloading from a Hugging Face model URL' });
        }
        if (quantization && !/^[\w.-]{1,40}$/.test(quantization)) {
          return res.status(400).json({ success: false, error: 'Quantization must be a short model quantization ID (for example Q4_K_M)' });
        }
        const result = await getLMStudioService(req.body).downloadModel(model, quantization);
        if (result.status === 'failed') {
          return res.status(502).json({ success: false, error: result.error || 'LM Studio reported that the model download failed', result });
        }
        const accepted = ['downloading', 'paused'].includes(result.status) && result.job_id;
        return res.status(accepted ? 202 : 200).json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 502).json({ success: false, error: `Could not start LM Studio model download: ${error.message}` });
      }
    });

    this.app.post('/api/configuration/lm-studio/models/download-status', protect, async (req, res) => {
      try {
        const jobId = typeof req.body?.jobId === 'string' ? req.body.jobId.trim() : '';
        if (!/^[\w-]{1,128}$/.test(jobId)) {
          return res.status(400).json({ success: false, error: 'Enter a valid LM Studio download job ID' });
        }
        const result = await getLMStudioService(req.body).getDownloadStatus(jobId);
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 502).json({ success: false, error: `Could not read LM Studio download status: ${error.message}` });
      }
    });

    this.app.post('/api/configuration/voicebox/check', protect, async (req, res) => {
      try {
        const service = new VoiceboxService({ baseUrl: req.body?.baseUrl });
        const [health, profiles] = await Promise.all([service.getHealth(), service.listProfiles()]);
        if (health?.status !== 'healthy') {
          return res.status(502).json({ success: false, error: 'Voicebox did not report a healthy local server' });
        }
        return res.json({
          success: true,
          result: {
            profiles: profiles.map(profile => ({
              id: profile.id, name: profile.name, language: profile.language,
              engine: profile.default_engine || ''
            }))
          }
        });
      } catch (error) {
        return res.status(502).json({ success: false, error: error.message });
      }
    });

    this.app.get('/api/configuration/voicebox/catalog', protect, async (req, res) => {
      try {
        const baseUrl = req.query.baseUrl || this.credentials?.credentials?.voicebox?.baseUrl ||
          process.env.VOICEBOX_BASE_URL || DEFAULT_VOICEBOX_URL;
        const service = new VoiceboxService({ baseUrl });
        const [health, profiles, modelStatuses, kokoroVoices, qwenPresetVoices] = await Promise.all([
          service.getHealth(),
          service.listProfiles(),
          service.getModelStatus(),
          service.listPresetVoices('kokoro'),
          service.listPresetVoices('qwen_custom_voice')
        ]);
        const statuses = new Map(modelStatuses.map(model => [model.model_name, model]));
        return res.json({
          success: true,
          result: {
            health: {
              gpuAvailable: health.gpu_available,
              gpuType: health.gpu_type,
              vramUsedMb: health.vram_used_mb,
              backendType: health.backend_type
            },
            profiles: profiles.map(profile => ({
              id: profile.id,
              name: profile.name,
              language: profile.language,
              voiceType: profile.voice_type || 'cloned',
              presetEngine: profile.preset_engine || null,
              presetVoiceId: profile.preset_voice_id || null,
              defaultEngine: profile.default_engine || null,
              sampleCount: profile.sample_count || 0
            })),
            engines: VOICEBOX_TTS_ENGINES.map(engine => ({
              ...engine,
              models: engine.modelIds.map((id, index) => ({
                id,
                name: statuses.get(id)?.display_name || id,
                modelSize: engine.modelSizes[index] || null,
                downloaded: Boolean(statuses.get(id)?.downloaded),
                downloading: Boolean(statuses.get(id)?.downloading),
                loaded: Boolean(statuses.get(id)?.loaded),
                sizeMb: statuses.get(id)?.size_mb || null
              }))
            })),
            whisperModels: VOICEBOX_WHISPER_MODELS.map(model => ({
              ...model,
              downloaded: Boolean(statuses.get(model.id)?.downloaded),
              downloading: Boolean(statuses.get(model.id)?.downloading),
              loaded: Boolean(statuses.get(model.id)?.loaded),
              sizeMb: statuses.get(model.id)?.size_mb || null
            })),
            languages: VOICEBOX_LANGUAGES,
            presets: {
              kokoro: kokoroVoices,
              qwen_custom_voice: qwenPresetVoices
            }
          }
        });
      } catch (error) {
        return res.status(502).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/configuration/voicebox/profiles/preset', protect, async (req, res) => {
      try {
        const { engine, voiceId } = req.body || {};
        const name = typeof req.body?.name === 'string' ? req.body.name.trim() : '';
        const selectedEngine = VOICEBOX_TTS_ENGINES.find(item => item.id === engine && item.profileType === 'preset');
        if (!selectedEngine || !name || name.length > 100 || typeof voiceId !== 'string') {
          return res.status(400).json({ success: false, error: 'Choose a preset engine and voice, and enter a profile name (up to 100 characters).' });
        }
        const baseUrl = req.body?.baseUrl || this.credentials?.credentials?.voicebox?.baseUrl ||
          process.env.VOICEBOX_BASE_URL || DEFAULT_VOICEBOX_URL;
        const service = new VoiceboxService({ baseUrl });
        const voices = await service.listPresetVoices(engine);
        const voice = voices.find(item => item.voice_id === voiceId);
        if (!voice) return res.status(400).json({ success: false, error: 'That preset voice is not available in Voicebox.' });
        const result = await service.createPresetProfile({
          name,
          language: voice.language || 'en',
          voice_type: 'preset',
          preset_engine: engine,
          preset_voice_id: voiceId,
          default_engine: engine
        });
        return res.status(201).json({ success: true, result });
      } catch (error) {
        return res.status(502).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/configuration/voicebox/models/:modelName/:action', protect, async (req, res) => {
      try {
        const { modelName, action } = req.params;
        if (!['load', 'unload', 'download'].includes(action)) {
          return res.status(404).json({ success: false, error: 'Unsupported model action' });
        }
        const allowedModels = [
          ...VOICEBOX_TTS_ENGINES.flatMap(engine => engine.modelIds),
          ...VOICEBOX_WHISPER_MODELS.map(model => model.id)
        ];
        if (!allowedModels.includes(modelName)) {
          return res.status(400).json({ success: false, error: 'Unknown Voicebox model' });
        }
        const baseUrl = req.body?.baseUrl || this.credentials?.credentials?.voicebox?.baseUrl ||
          process.env.VOICEBOX_BASE_URL || DEFAULT_VOICEBOX_URL;
        const service = new VoiceboxService({ baseUrl });
        if (action === 'load') {
          const status = await service.getModelStatus();
          const model = status.find(item => item.model_name === modelName);
          if (!model?.downloaded) {
            return res.status(409).json({ success: false, error: `${modelName} is not downloaded in Voicebox yet. Download it in Voicebox before loading.` });
          }
          const result = await service.loadModel(modelName);
          return res.json({ success: true, result });
        }
        if (action === 'download') {
          const result = await service.downloadModel(modelName);
          return res.status(202).json({ success: true, result });
        }
        const result = await service.unloadModel(modelName);
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(502).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/configuration/voicebox/preview', protect, async (req, res) => {
      try {
        const { profileId, engine, text, language, modelSize, instruct } = req.body || {};
        const cleanText = typeof text === 'string' ? text.trim() : '';
        if (!cleanText || cleanText.length > 2000) {
          return res.status(400).json({ success: false, error: 'Enter sample text of up to 2,000 characters.' });
        }
        const baseUrl = req.body?.baseUrl || this.credentials?.credentials?.voicebox?.baseUrl ||
          process.env.VOICEBOX_BASE_URL || DEFAULT_VOICEBOX_URL;
        const service = new VoiceboxService({ baseUrl });
        const profiles = await service.listProfiles();
        const profile = profiles.find(item => item.id === profileId);
        if (!profile) return res.status(400).json({ success: false, error: 'Select a profile from the connected Voicebox server.' });
        const selectedEngineId = engine || profile.default_engine || profile.preset_engine || 'qwen';
        const selectedEngine = VOICEBOX_TTS_ENGINES.find(item => item.id === selectedEngineId);
        if (!selectedEngine) return res.status(400).json({ success: false, error: 'Select a supported Voicebox TTS engine.' });
        if (selectedEngine.modelSizes.length && !selectedEngine.modelSizes.includes(modelSize)) {
          return res.status(400).json({ success: false, error: 'Select a supported size for this Voicebox engine.' });
        }
        if (language && !selectedEngine.languages.includes(language)) {
          return res.status(400).json({ success: false, error: `${selectedEngine.name} does not support ${language}.` });
        }
        const profileType = profile.voice_type || 'cloned';
        const profileMatches = profileType === selectedEngine.profileType &&
          (profileType !== 'preset' || profile.preset_engine === selectedEngineId);
        if (!profileMatches) {
          return res.status(400).json({ success: false, error: `${selectedEngine.name} requires a ${selectedEngine.profileType} profile${selectedEngine.profileType === 'preset' ? ` for ${selectedEngineId}` : ''}.` });
        }
        const audio = await service.generateSpeech({
          profile_id: profileId,
          text: cleanText,
          engine: selectedEngineId,
          language: language || profile.language || 'en',
          ...(modelSize ? { model_size: modelSize } : {}),
          ...(typeof instruct === 'string' && instruct.trim() ? { instruct: instruct.trim().slice(0, 500) } : {}),
          max_chunk_chars: Number(req.body?.maxChunkChars) || 800,
          crossfade_ms: Number(req.body?.crossfadeMs) || 0,
          normalize: true
        });
        res.setHeader('Content-Type', 'audio/wav');
        res.setHeader('Content-Length', audio.length);
        return res.send(audio);
      } catch (error) {
        return res.status(502).json({ success: false, error: error.message });
      }
    });

    this.app.post(
      '/api/configuration/voicebox/transcribe',
      protect,
      express.raw({ type: ['audio/*', 'application/octet-stream'], limit: '200mb' }),
      async (req, res) => {
      try {
        const model = typeof req.query.model === 'string' ? req.query.model : 'turbo';
        const language = typeof req.query.language === 'string' ? req.query.language : '';
        const selectedModel = VOICEBOX_WHISPER_MODELS.find(item => item.value === model);
        if (!selectedModel) {
          return res.status(400).json({ success: false, error: 'Select a supported Whisper model.' });
        }
        if (language && !VOICEBOX_LANGUAGES.some(([code]) => code === language)) {
          return res.status(400).json({ success: false, error: 'Select a supported transcription language.' });
        }
        if (!Buffer.isBuffer(req.body) || !req.body.length) {
          return res.status(400).json({ success: false, error: 'Upload a non-empty audio file.' });
        }
        const baseUrl = req.query.baseUrl || this.credentials?.credentials?.voicebox?.baseUrl ||
          process.env.VOICEBOX_BASE_URL || DEFAULT_VOICEBOX_URL;
        const service = new VoiceboxService({ baseUrl });
        const fileName = typeof req.query.filename === 'string' ? path.basename(req.query.filename) : 'voicebox-upload.wav';
        const result = await service.transcribeAudio(req.body, fileName, {
          contentType: req.headers['content-type'] || 'application/octet-stream',
          model,
          language: language || undefined
        });
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.type === 'entity.too.large' ? 413 : 502).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/configuration/youtube/connect', protect, async (req, res) => {
      try {
        if (!this.credentials) return res.status(503).json({ success: false, error: 'Credential storage is not initialized' });
        if (this.dashboardAuth.status === 'pending') {
          return res.status(409).json({ success: false, error: 'YouTube authorization is already in progress' });
        }
        const clientId = typeof req.body?.clientId === 'string' ? req.body.clientId.trim() : '';
        const clientSecret = (typeof req.body?.clientSecret === 'string' ? req.body.clientSecret.trim() : '') ||
          this.credentials.credentials?.youtube?.client_secret;
        if (!clientId || !clientSecret) {
          return res.status(400).json({ success: false, error: 'Google OAuth client ID and client secret are required' });
        }
        this.credentials.credentials = {
          ...(this.credentials.credentials || {}),
          youtube: {
            client_id: clientId,
            client_secret: clientSecret,
            redirect_uris: ['http://127.0.0.1']
          }
        };
        await this.credentials.saveCredentials();
        this.dashboardAuth = { status: 'pending', error: null };
        const auth = new (require('./modern-auth').ModernAuth)();
        void auth.authenticate().then(async tokens => {
          this.credentials.tokens ||= {};
          this.credentials.tokens.youtube = tokens;
          this.dashboardAuth = { status: 'connected', error: null };
          this.dashboardConfigurationRestartRequired = true;
        }).catch(error => {
          this.dashboardAuth = { status: 'failed', error: error.message };
        });
        return res.status(202).json({
          success: true,
          result: { status: 'pending', message: 'Complete YouTube authorization in the browser window that opened.' }
        });
      } catch (error) {
        this.dashboardAuth = { status: 'failed', error: error.message };
        return res.status(400).json({ success: false, error: error.message });
      }
    });

    this.app.get('/api/dashboard', async (_req, res) => {
      try {
        const [stats, jobs, pipeline, schedule, events, notifications, profile, settings, ideas, analytics, learning, activation, channelStrategy, operatorRuns, readiness, engagement, experiments] = await Promise.all([
          this.db.getStats(),
          this.db.listGenerationJobs(20),
          this.db.getPipelineOverview(50),
          this.db.getUpcomingSchedule(30),
          this.db.getRecentAutomationEvents(20),
          this.db.listNotifications(20),
          this.db.getChannelProfile(),
          this.db.getAllSettings(),
          this.db.listContentIdeas(),
          this.agents.analytics
            ? this.agents.analytics.getRecentAnalytics(30)
            : Promise.resolve({ totalVideos: 0, averagePerformanceScore: 0, topPerformers: [], insights: [] }),
          this.agents.analytics?.getLearningSummary
            ? this.agents.analytics.getLearningSummary()
            : Promise.resolve({ measuredVideos: 0, snapshotCount: 0, baseline: {}, recommendations: [], approvedCount: 0, pendingCount: 0 }),
          this.activation
            ? this.activation.getSummary()
            : Promise.resolve({ privacy: 'local-only', counts: {}, milestones: {} }),
          this.db.getChannelStrategy(),
          this.db.listOperatorRuns(10),
          this.readiness
            ? this.readiness.getSummary()
            : Promise.resolve({ status: 'unverified', stale: false, blockingFailures: [], checks: [] }),
          this.engagement
            ? this.engagement.getSummary()
            : Promise.resolve({
                videosTracked: 0, pendingDrafts: 0, postedToday: 0, needsAttentionCount: 0,
                pendingAudienceIdeas: 0, postingEnabled: false, postingDisabledReason: 'setup_required',
                insights: [], recentThemes: [],
                evidencePolicy: 'Comments are fetched read-only from YouTube. Replies post only after operator approval, and fallback analysis never proposes drafts or ideas.'
              }),
          this.experiments
            ? this.experiments.getSummary()
            : Promise.resolve({ experiments: [], candidates: [], activeCount: 0, awaitingDecisionCount: 0, evidencePolicy: 'Finish setup to create a controlled growth experiment.' })
        ]);
        if (this.telemetry) void this.telemetry.sync(activation);
        await this.agents.production?.aiVideoGenerator?.mediaGeneration?.settings();
        res.json({
          stats, jobs, pipeline, schedule, events, notifications, profile, settings, ideas, analytics, learning, activation,
          channelStrategy, operatorRuns, readiness, engagement, experiments,
          system: {
            initialized: this.isInitialized,
            setupRequired: this.setupRequired,
            uptime: process.uptime(),
            activeJobs: this.activeJobs.size,
            automationPaused: this.scheduler ? !this.scheduler.isEnabled : true,
            agents: Object.keys(this.agents),
            autonomousRunning: Boolean(await this.db.getActiveOperatorRun()),
            videoProviders: this.agents.production?.aiVideoGenerator?.mediaGeneration?.listProviders() || []
          }
        });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    this.app.get('/api/jobs/:jobId', async (req, res) => {
      const job = await this.db.getGenerationJob(req.params.jobId);
      if (!job) return res.status(404).json({ error: 'Job not found' });
      job.checkpoints = await this.db.listGenerationCheckpoints(job.id);
      job.mediaTasks = await this.db.listMediaGenerationTasks(job.id);
      job.resumeFrom = this.recovery?.resumePoint(job.checkpoints);
      return res.json(job);
    });

    this.app.post('/api/jobs/:jobId/resume', protect, async (req, res) => {
      try {
        const result = await this.resumeGenerationJob(req.params.jobId, { stage: req.body?.stage });
        return res.status(202).json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message });
      }
    });

    this.app.get('/api/readiness', async (_req, res) => {
      if (!this.readiness) return res.status(503).json({ error: 'Readiness service is not initialized' });
      return res.json(await this.readiness.getSummary());
    });

    this.app.post('/api/readiness/run', protect, async (req, res) => {
      try {
        if (!this.readiness) return res.status(503).json({ error: 'Readiness service is not initialized' });
        const result = await this.readiness.run({
          includePaidMedia: req.body?.includePaidMedia === true,
          includePaidVideo: req.body?.includePaidVideo === true
        });
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 500).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/jobs/:jobId/cancel', protect, async (req, res) => {
      const job = await this.db.getGenerationJob(req.params.jobId);
      if (!job) return res.status(404).json({ error: 'Job not found' });
      if (!['queued', 'running'].includes(job.status)) {
        return res.status(409).json({ error: 'Only queued or running jobs can be cancelled' });
      }
      const updated = await this.db.updateGenerationJob(job.id, { cancelRequested: true, details: { cancelReason: req.body?.reason || 'Cancelled by operator' } });
      return res.json({ success: true, result: updated });
    });

    this.app.get('/api/content/:productionId', async (req, res) => {
      let bundle = await this.db.getProductionBundle(req.params.productionId);
      if (!bundle) return res.status(404).json({ error: 'Content not found' });
      if (this.scenes && !bundle.scenes?.length) {
        await this.scenes.ensureManifest(bundle);
        bundle = await this.db.getProductionBundle(req.params.productionId);
      }
      return res.json(this.decorateContentBundle(bundle));
    });

    this.app.get('/api/content/:productionId/scenes/:sceneId/estimate', async (req, res) => {
      try {
        if (!this.scenes) return res.status(503).json({ error: 'Scene repair requires completed setup' });
        const result = await this.scenes.regenerationEstimate(req.params.productionId, req.params.sceneId, {
          provider: req.query.provider
        });
        return res.json(result);
      } catch (error) {
        return res.status(error.status || 400).json({ error: error.message, code: error.code, details: error.details });
      }
    });

    this.app.patch('/api/content/:productionId/scenes/:sceneId', protect, async (req, res) => {
      try {
        if (!this.scenes) return res.status(503).json({ error: 'Scene repair requires completed setup' });
        const result = await this.scenes.updateScene(req.params.productionId, req.params.sceneId, req.body || {});
        await this.refreshContentReview(req.params.productionId, 'Scene changes require review before scheduling');
        return res.json({ success: true, result: this.scenes.decorateScene(result, req.params.productionId) });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code, details: error.details });
      }
    });

    this.app.post('/api/content/:productionId/scenes/reorder', protect, async (req, res) => {
      try {
        if (!this.scenes) return res.status(503).json({ error: 'Scene repair requires completed setup' });
        const result = await this.scenes.reorder(req.params.productionId, req.body?.sceneIds);
        await this.refreshContentReview(req.params.productionId, 'Timeline order changed; rebuild and review before scheduling');
        return res.json({ success: true, result: result.map(scene => this.scenes.decorateScene(scene, req.params.productionId)) });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code, details: error.details });
      }
    });

    this.app.post('/api/content/:productionId/scenes/:sceneId/regenerate', protect, async (req, res) => {
      try {
        if (!this.scenes) return res.status(503).json({ error: 'Scene repair requires completed setup' });
        const result = await this.scenes.regenerate(req.params.productionId, req.params.sceneId, req.body || {});
        await this.refreshContentReview(req.params.productionId, 'Regenerated scene must be rebuilt and reviewed');
        return res.status(202).json({ success: true, result: {
          ...result,
          scene: this.scenes.decorateScene(result.scene, req.params.productionId)
        } });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code, details: error.details });
      }
    });

    this.app.post('/api/content/:productionId/shorts/propose', protect, async (req, res) => {
      try {
        if (!this.shorts) return res.status(503).json({ error: 'Shorts repurposing requires completed setup' });
        const result = await this.shorts.propose(req.params.productionId, req.body || {});
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.patch('/api/content/:productionId/shorts/:clipId', protect, async (req, res) => {
      try {
        if (!this.shorts) return res.status(503).json({ error: 'Shorts repurposing requires completed setup' });
        const result = await this.shorts.update(req.params.productionId, req.params.clipId, req.body || {});
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.post('/api/content/:productionId/shorts/:clipId/render', protect, async (req, res) => {
      try {
        if (!this.shorts) return res.status(503).json({ error: 'Shorts repurposing requires completed setup' });
        const result = await this.shorts.render(req.params.productionId, req.params.clipId);
        return res.status(202).json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.post('/api/content/:productionId/shorts/:clipId/approve', protect, async (req, res) => {
      try {
        if (!this.shorts) return res.status(503).json({ error: 'Shorts repurposing requires completed setup' });
        const result = await this.shorts.approve(req.params.productionId, req.params.clipId, req.body || {});
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.get('/api/content/:productionId/shorts/:clipId/asset/:kind', async (req, res) => {
      try {
        const clip = await this.db.getShortClip(req.params.clipId);
        if (!clip || clip.productionId !== req.params.productionId) return res.status(404).json({ error: 'Short asset not found' });
        const filePath = req.params.kind === 'video' ? clip.outputPath : req.params.kind === 'captions' ? clip.captionsPath : null;
        if (!filePath) return res.status(404).json({ error: 'Short asset not found' });
        const resolved = path.resolve(filePath);
        const shortsRoot = path.resolve(__dirname, 'data', 'shorts');
        if (!resolved.startsWith(`${shortsRoot}${path.sep}`)) return res.status(403).json({ error: 'Short asset path is not allowed' });
        await fs.access(resolved);
        return res.sendFile(resolved);
      } catch (_error) {
        return res.status(404).json({ error: 'Short asset not found' });
      }
    });

    this.app.post('/api/content/:productionId/scenes/:sceneId/narration', protect, async (req, res) => {
      try {
        if (!this.scenes) return res.status(503).json({ error: 'Narration recovery requires completed setup' });
        const result = await this.scenes.regenerateNarration(req.params.productionId, req.params.sceneId, req.body || {});
        await this.refreshContentReview(req.params.productionId, 'Narration regenerated; rebuild the final video before approval');
        return res.status(202).json({ success: true, result: this.scenes.decorateScene(result, req.params.productionId) });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code, details: error.details });
      }
    });

    this.app.post('/api/content/:productionId/narration/silence', protect, async (req, res) => {
      try {
        if (!this.scenes) return res.status(503).json({ error: 'Narration recovery requires completed setup' });
        const result = await this.scenes.setSilenceOverride(req.params.productionId, req.body || {});
        await this.refreshContentReview(
          req.params.productionId,
          result.enabled ? 'Intentional silence recorded; rebuild and review before approval' : 'Narration is required again; regenerate it before approval'
        );
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code, details: error.details });
      }
    });

    this.app.put(
      '/api/content/:productionId/scenes/:sceneId/asset',
      protect,
      express.raw({ type: ['image/*', 'video/*'], limit: '100mb' }),
      async (req, res) => {
        try {
          if (!this.scenes) return res.status(503).json({ error: 'Scene repair requires completed setup' });
          const result = await this.scenes.replaceAsset(req.params.productionId, req.params.sceneId, {
            buffer: req.body,
            contentType: req.get('content-type'),
            filename: req.get('x-file-name'),
            rightsConfirmed: req.get('x-rights-confirmed') === 'true',
            containsSyntheticMedia: req.get('x-synthetic-media') === 'true'
          });
          await this.refreshContentReview(req.params.productionId, 'Replacement scene asset must be rebuilt and reviewed');
          return res.json({ success: true, result: this.scenes.decorateScene(result, req.params.productionId) });
        } catch (error) {
          return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code, details: error.details });
        }
      }
    );

    this.app.post('/api/content/:productionId/scenes/rebuild', protect, async (req, res) => {
      try {
        if (!this.scenes) return res.status(503).json({ error: 'Scene repair requires completed setup' });
        const result = await this.scenes.rebuild(req.params.productionId);
        await this.refreshContentReview(req.params.productionId, 'Scene repair rebuilt; final approval is required');
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code, details: error.details });
      }
    });

    this.app.get('/api/content/:productionId/scenes/:sceneId/asset', async (req, res) => {
      try {
        const scene = await this.db.getProductionScene(req.params.productionId, req.params.sceneId);
        if (!scene?.assetPath) return res.status(404).json({ error: 'Scene asset not found' });
        const resolved = path.resolve(scene.assetPath);
        const dataRoot = path.resolve(__dirname, 'data');
        if (!resolved.startsWith(`${dataRoot}${path.sep}`)) return res.status(403).json({ error: 'Scene asset path is not allowed' });
        await fs.access(resolved);
        return res.sendFile(resolved);
      } catch (_error) {
        return res.status(404).json({ error: 'Scene asset not found' });
      }
    });

    this.app.patch('/api/content/:productionId', protect, async (req, res) => {
      try {
        const bundle = await this.db.getProductionBundle(req.params.productionId);
        if (!bundle) return res.status(404).json({ error: 'Content not found' });
        if (bundle.review_status === 'approved' && bundle.schedule?.status === 'published') {
          return res.status(409).json({ error: 'Published content cannot be edited here' });
        }
        const editorData = this.validateEditorData(req.body, bundle.editorData);
        const result = await this.db.saveContentReview(bundle.id, {
          status: bundle.review_status || 'needs_review',
          editorData,
          qualityChecks: bundle.qualityChecks,
          reviewNotes: req.body.reviewNotes ?? bundle.review_notes,
          reviewedAt: bundle.reviewed_at
        });
        return res.json({ success: true, result: this.decorateContentBundle(result) });
      } catch (error) {
        return res.status(400).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/content/:productionId/approve', protect, async (req, res) => {
      try {
        const result = await this.approveContent(req.params.productionId, req.body || {});
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, quality: error.quality });
      }
    });

    this.app.post('/api/content/:productionId/reject', protect, async (req, res) => {
      const bundle = await this.db.getProductionBundle(req.params.productionId);
      if (!bundle) return res.status(404).json({ error: 'Content not found' });
      await this.db.saveContentReview(bundle.id, {
        status: 'rejected',
        editorData: bundle.editorData,
        qualityChecks: bundle.qualityChecks,
        reviewNotes: req.body?.notes || 'Rejected by operator',
        reviewedAt: new Date().toISOString()
      });
      await this.db.updateProductionStatus(bundle.id, 'rejected');
      return res.json({ success: true });
    });

    this.app.post('/api/content/:productionId/retry', protect, async (req, res) => {
      const bundle = await this.db.getProductionBundle(req.params.productionId);
      if (!bundle) return res.status(404).json({ error: 'Content not found' });
      const job = await this.startGenerationJob({
        topic: bundle.strategy.topic || bundle.editorData.title || null,
        style: bundle.strategy.requestedStyle || bundle.strategy.contentType || null,
        length: bundle.strategy.requestedLengthKey || 'medium',
        format: bundle.strategy.outputFormat || 'video',
        targetDuration: bundle.strategy.targetDuration || null,
        source: 'retry'
      });
      return res.status(202).json({ success: true, result: job });
    });

    this.app.get('/api/content/:productionId/asset/:kind', async (req, res) => {
      try {
        const bundle = await this.db.getProductionBundle(req.params.productionId);
        if (!bundle) return res.status(404).json({ error: 'Content not found' });
        const allowed = {
          video: bundle.assets?.finalVideo?.path,
          thumbnail: bundle.assets?.thumbnail?.path,
          captions: bundle.assets?.captions?.path,
          script: bundle.assets?.script?.originalPath
        };
        const experimentMatch = req.params.kind.match(/^experiment-thumbnail-(\d+)$/);
        const experimentPath = experimentMatch
          ? bundle.editorData?.packagingExperiment?.thumbnailVariants?.[Number(experimentMatch[1])]?.path
          : null;
        const filePath = allowed[req.params.kind] || experimentPath;
        if (!filePath) return res.status(404).json({ error: 'Asset not found' });
        const resolved = path.resolve(filePath);
        const dataRoot = path.resolve(__dirname, 'data');
        const experimentRoot = path.resolve(__dirname, 'uploads', 'thumbnails');
        const allowedPath = [dataRoot, experimentRoot]
          .some(root => resolved.startsWith(`${root}${path.sep}`));
        if (!allowedPath) return res.status(403).json({ error: 'Asset path is not allowed' });
        await fs.access(resolved);
        return res.sendFile(resolved);
      } catch (_error) {
        return res.status(404).json({ error: 'Asset not found' });
      }
    });

    this.app.put('/api/profile', protect, async (req, res) => {
      try {
        const profile = this.validateProfile(req.body || {});
        return res.json({ success: true, result: await this.db.saveChannelProfile(profile) });
      } catch (error) {
        return res.status(400).json({ success: false, error: error.message });
      }
    });

    this.app.put('/api/operator/strategy', protect, async (req, res) => {
      try {
        const current = await this.db.getChannelStrategy() || {};
        const strategy = this.validateChannelStrategy(req.body || {}, current);
        return res.json({ success: true, result: await this.db.saveChannelStrategy(strategy) });
      } catch (error) {
        return res.status(400).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/operator/start', protect, async (req, res) => {
      try {
        if (this.setupRequired || !this.agents.strategy) {
          return res.status(503).json({ success: false, error: 'Finish setup with npm run walkthrough before activating the autonomous operator' });
        }
        if (this.activeJobs.size) {
          return res.status(409).json({ success: false, error: 'Wait for the current generation job to finish before starting an autonomous run' });
        }
        await this.readiness?.assertReady('Autonomous production');
        const current = await this.db.getChannelStrategy() || {};
        const strategy = this.validateChannelStrategy({ ...(req.body || {}), status: 'active' }, current);
        const saved = await this.db.saveChannelStrategy(strategy);
        const run = await this.autonomous.start(saved);
        return res.status(202).json({ success: true, result: run });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/operator/pause', protect, async (_req, res) => {
      const strategy = await this.db.getChannelStrategy();
      if (!strategy) return res.status(404).json({ error: 'Channel strategy not found' });
      const active = await this.db.getActiveOperatorRun();
      if (active) await this.autonomous.cancel(active.id);
      const saved = await this.db.saveChannelStrategy({ ...strategy, status: 'paused' });
      return res.json({ success: true, result: saved });
    });

    this.app.post('/api/operator/runs/:runId/cancel', protect, async (req, res) => {
      const run = await this.autonomous.cancel(req.params.runId);
      if (!run) return res.status(404).json({ error: 'Operator run not found' });
      return res.json({ success: true, result: run });
    });

    this.app.put('/api/content/:productionId/provenance', protect, async (req, res) => {
      try {
        const bundle = await this.db.getProductionBundle(req.params.productionId);
        if (!bundle) return res.status(404).json({ error: 'Content not found' });
        if (bundle.review_status === 'approved' || bundle.schedule) {
          return res.status(409).json({ error: 'Provenance is locked after content is approved or scheduled' });
        }
        if (!this.provenance) this.provenance = new ProvenanceService(this.db);
        await this.provenance.review(bundle.id, req.body || {});
        const updated = await this.db.getProductionBundle(bundle.id);
        const profile = await this.db.getChannelProfile() || {};
        const quality = await this.operator.runQualityChecks({
          ...updated,
          scheduledPublishTime: updated.scheduled_publish_time
        }, profile);
        const reviewStatus = quality.passed ? 'needs_review' : 'needs_attention';
        const result = await this.db.saveContentReview(bundle.id, {
          status: reviewStatus,
          editorData: updated.editorData,
          qualityChecks: quality.checks,
          reviewNotes: quality.passed ? null : `Blocking checks failed: ${quality.blockingFailures.join(', ')}`,
          reviewedAt: null
        });
        return res.json({ success: true, result: this.decorateContentBundle(result) });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/operator/runs/:runId/resume', protect, async (req, res) => {
      try {
        if (this.setupRequired || !this.agents.strategy) {
          return res.status(503).json({ success: false, error: 'Finish setup before resuming the autonomous operator' });
        }
        await this.readiness?.assertReady('Autonomous production recovery');
        const strategy = await this.db.getChannelStrategy();
        const run = await this.autonomous.resume(req.params.runId, strategy);
        return res.status(202).json({ success: true, result: run });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/learning/recommendations/:recommendationId/:action', protect, async (req, res) => {
      const { recommendationId, action } = req.params;
      if (!['approve', 'reject'].includes(action)) {
        return res.status(400).json({ error: 'Action must be approve or reject' });
      }
      const status = action === 'approve' ? 'approved' : 'rejected';
      const recommendation = await this.db.reviewLearningRecommendation(recommendationId, status);
      if (!recommendation) return res.status(404).json({ error: 'Learning recommendation not found' });
      await this.operator.notify({
        type: 'learning_recommendation_reviewed',
        level: action === 'approve' ? 'success' : 'info',
        title: action === 'approve' ? 'Channel learning approved' : 'Channel learning rejected',
        message: recommendation.title,
        data: { recommendationId, status }
      });
      return res.json({ success: true, result: recommendation });
    });

    this.app.post('/api/content/:productionId/discoverability/run', protect, async (req, res) => {
      try {
        const bundle = await this.db.getProductionBundle(req.params.productionId);
        if (!bundle) return res.status(404).json({ success: false, error: 'Content not found' });
        const profile = await this.db.getChannelProfile() || {};
        const audit = await this.discoverability.auditProduction(bundle, profile, req.body?.platform || 'youtube');
        if (bundle.review_status !== 'approved' && !bundle.schedule) {
          const updated = await this.db.getProductionBundle(bundle.id);
          const quality = await this.operator.runQualityChecks({
            ...updated,
            scheduledPublishTime: updated.scheduled_publish_time
          }, profile);
          await this.db.saveContentReview(bundle.id, {
            status: quality.passed ? 'needs_review' : 'needs_attention',
            editorData: updated.editorData,
            qualityChecks: quality.checks,
            reviewNotes: quality.passed ? null : `Blocking checks failed: ${quality.blockingFailures.join(', ')}`,
            reviewedAt: null
          });
        }
        const result = await this.db.getProductionBundle(bundle.id);
        return res.json({ success: true, result: this.decorateContentBundle(result), audit });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message });
      }
    });

    this.app.patch('/api/content/:productionId/schedule', protect, async (req, res) => {
      try {
        if (!this.agents.publishing) return res.status(503).json({ error: 'Publishing requires completed setup' });
        const result = await this.agents.publishing.rescheduleContent(req.params.productionId, req.body?.publishTime);
        await this.db.updateProductionStatus(req.params.productionId, 'scheduled');
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.post('/api/content/:productionId/publish-now', protect, async (req, res) => {
      try {
        if (!this.agents.publishing) return res.status(503).json({ error: 'Publishing requires completed setup' });
        const result = await this.agents.publishing.emergencyPublish(req.params.productionId);
        await this.db.updateProductionStatus(req.params.productionId, result.status);
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.delete('/api/content/:productionId/schedule', protect, async (req, res) => {
      try {
        if (!this.agents.publishing) return res.status(503).json({ error: 'Publishing requires completed setup' });
        const result = await this.agents.publishing.deleteScheduledContent(req.params.productionId);
        await this.db.updateProductionStatus(req.params.productionId, 'approved');
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.patch('/api/discoverability/findings/:findingId', protect, async (req, res) => {
      try {
        const finding = await this.discoverability.reviewFinding(req.params.findingId, req.body || {});
        const audit = await this.db.getLatestDiscoverabilityAudit(finding.production_id, finding.platform);
        return res.json({ success: true, result: { finding, audit } });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message });
      }
    });

    this.app.get('/api/experiments', async (_req, res) => {
      try {
        const summary = this.experiments
          ? await this.experiments.getSummary()
          : { experiments: [], candidates: [], activeCount: 0, awaitingDecisionCount: 0 };
        return res.json({ success: true, result: summary });
      } catch (error) {
        return res.status(500).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/experiments', protect, async (req, res) => {
      try {
        if (!this.experiments) return res.status(503).json({ error: 'Finish setup before creating experiments' });
        const experiment = await this.experiments.create(req.body || {});
        return res.status(201).json({ success: true, result: experiment });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.post('/api/experiments/:experimentId/:action', protect, async (req, res) => {
      try {
        if (!this.experiments) return res.status(503).json({ error: 'Finish setup before controlling experiments' });
        const actions = {
          approve: () => this.experiments.approve(req.params.experimentId, req.body),
          start: () => this.experiments.start(req.params.experimentId, req.body),
          refresh: () => this.experiments.refresh(req.params.experimentId),
          adopt: () => this.experiments.adoptWinner(req.params.experimentId, req.body),
          cancel: () => this.experiments.cancel(req.params.experimentId, req.body)
        };
        if (!actions[req.params.action]) return res.status(400).json({ error: 'Unsupported experiment action' });
        const experiment = await actions[req.params.action]();
        await this.operator.notify({
          type: 'growth_experiment_updated',
          level: ['adopt', 'approve'].includes(req.params.action) ? 'success' : 'info',
          title: `Growth experiment ${req.params.action}`,
          message: experiment.title,
          data: { experimentId: experiment.id, status: experiment.status }
        });
        return res.json({ success: true, result: experiment });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.get('/api/retention/:videoId', async (req, res) => {
      const videoId = String(req.params.videoId || '').trim();
      if (!/^[A-Za-z0-9_-]{1,100}$/.test(videoId)) {
        return res.status(400).json({ error: 'A valid YouTube video ID is required' });
      }
      const snapshots = await this.db.listRetentionSnapshots({ videoId, limit: 10 });
      return res.json({ success: true, result: snapshots });
    });

    this.app.post('/api/retention/:videoId/refresh', protect, async (req, res) => {
      try {
        const videoId = String(req.params.videoId || '').trim();
        const measurementWindow = String(req.body?.measurementWindow || 'rolling');
        if (!/^[A-Za-z0-9_-]{1,100}$/.test(videoId)) {
          return res.status(400).json({ error: 'A valid YouTube video ID is required' });
        }
        if (!['24h', '7d', 'rolling'].includes(measurementWindow)) {
          return res.status(400).json({ error: 'Measurement window must be 24h, 7d, or rolling' });
        }
        if (!this.agents.analytics) {
          return res.status(503).json({ error: 'YouTube Analytics is not initialized' });
        }
        const report = await this.agents.analytics.analyzeVideoPerformance(videoId, { measurementWindow });
        return res.json({
          success: true,
          result: report.retentionSnapshot || null,
          retention: report.retention
        });
      } catch (error) {
        return res.status(error.status || 400).json({ error: error.message });
      }
    });

    const ENGAGEMENT_VIDEO_ID = /^[A-Za-z0-9_-]{1,100}$/;

    this.app.get('/api/engagement/:videoId', async (req, res) => {
      try {
        const videoId = String(req.params.videoId || '').trim();
        if (!ENGAGEMENT_VIDEO_ID.test(videoId)) {
          return res.status(400).json({ error: 'A valid YouTube video ID is required' });
        }
        const [insight, comments, drafts] = await Promise.all([
          this.db.getEngagementInsight(videoId),
          this.db.listAudienceComments({ videoId, limit: 200 }),
          this.db.listReplyDrafts({ videoId, limit: 100 })
        ]);
        return res.json({ success: true, result: { insight, comments, drafts } });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message });
      }
    });

    this.app.post('/api/engagement/:videoId/sync', protect, async (req, res) => {
      try {
        if (!this.engagement) return res.status(503).json({ error: 'Audience engagement requires completed setup' });
        const videoId = String(req.params.videoId || '').trim();
        if (!ENGAGEMENT_VIDEO_ID.test(videoId)) {
          return res.status(400).json({ error: 'A valid YouTube video ID is required' });
        }
        const sync = await this.engagement.syncVideoComments(videoId, req.body || {});
        const insight = sync.fetched > 0 || req.body?.analyze === true
          ? await this.engagement.analyzeVideo(videoId)
          : sync.insight;
        return res.status(202).json({ success: true, result: { ...sync, insight } });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.post('/api/engagement/:videoId/draft-replies', protect, async (req, res) => {
      try {
        if (!this.engagement) return res.status(503).json({ error: 'Audience engagement requires completed setup' });
        const videoId = String(req.params.videoId || '').trim();
        if (!ENGAGEMENT_VIDEO_ID.test(videoId)) {
          return res.status(400).json({ error: 'A valid YouTube video ID is required' });
        }
        const result = await this.engagement.draftReplies(videoId, req.body || {});
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.patch('/api/engagement/replies/:draftId', protect, async (req, res) => {
      try {
        if (!this.engagement) return res.status(503).json({ error: 'Audience engagement requires completed setup' });
        const result = await this.engagement.updateReplyDraft(req.params.draftId, req.body || {});
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.post('/api/engagement/replies/:draftId/approve', protect, async (req, res) => {
      try {
        if (!this.engagement) return res.status(503).json({ error: 'Audience engagement requires completed setup' });
        const result = await this.engagement.approveReplyDraft(req.params.draftId, req.body || {});
        // The reply is already live on YouTube; a notification failure must not report an error.
        try {
          await this.operator.notify({
            type: 'audience_reply_posted',
            level: 'success',
            title: 'Audience reply posted',
            message: `A reply was posted on video ${result.videoId}`,
            data: { draftId: result.id, videoId: result.videoId, postedCommentId: result.postedCommentId }
          });
        } catch (notifyError) {
          this.logger.warn(`Posted reply notification failed: ${notifyError.message}`);
        }
        return res.json({ success: true, result });
      } catch (error) {
        return res.status(error.status || 400).json({ success: false, error: error.message, code: error.code });
      }
    });

    this.app.post('/api/ideas', protect, async (req, res) => {
      const topic = String(req.body?.topic || '').trim();
      if (!topic || topic.length > 200) return res.status(400).json({ error: 'A topic of 200 characters or less is required' });
      const idea = await this.db.createContentIdea({ ...req.body, topic });
      return res.status(201).json({ success: true, result: idea });
    });

    this.app.patch('/api/ideas/:ideaId', protect, async (req, res) => {
      const idea = await this.db.updateContentIdea(req.params.ideaId, req.body || {});
      if (!idea) return res.status(404).json({ error: 'Idea not found' });
      return res.json({ success: true, result: idea });
    });

    this.app.post('/api/ideas/:ideaId/generate', protect, async (req, res) => {
      const idea = await this.db.updateContentIdea(req.params.ideaId, { status: 'generating' });
      if (!idea) return res.status(404).json({ error: 'Idea not found' });
      const job = await this.startGenerationJob({
        topic: idea.topic, style: idea.style, length: req.body?.length || 'medium',
        format: req.body?.format || 'video', targetDuration: req.body?.targetDuration,
        source: 'idea'
      });
      await this.db.updateContentIdea(idea.id, { status: 'generated' });
      return res.status(202).json({ success: true, result: job });
    });

    this.app.post('/api/automation/:action', protect, async (req, res) => {
      if (!this.scheduler) return res.status(409).json({ error: 'Finish setup before controlling automation' });
      const { action } = req.params;
      if (action === 'pause') {
        await this.scheduler.pauseAutomation();
        await this.db.setSetting('automation_paused', 'true');
      } else if (action === 'resume') {
        await this.scheduler.resumeAutomation();
        await this.db.setSetting('automation_paused', 'false');
      } else {
        return res.status(400).json({ error: 'Action must be pause or resume' });
      }
      return res.json({ success: true, paused: !this.scheduler.isEnabled });
    });

    this.app.put('/api/settings', protect, async (req, res) => {
      const stored = await this.db.getAllSettings();
      const provider = req.body?.video_provider;
      if (provider !== undefined) {
        const supported = ['slideshow', 'auto', 'seedance', 'minimax_h3', 'google_omni', 'kling', 'wan', 'comfyui'];
        if (!supported.includes(provider)) return res.status(400).json({ error: 'Unsupported video provider' });
      }
      let comfyBaseURL;
      if (req.body?.comfyui_base_url !== undefined) {
        try {
          comfyBaseURL = normalizeComfyUIBaseURL(req.body.comfyui_base_url);
        } catch (error) {
          return res.status(400).json({ error: error.message });
        }
      }
      let comfyWorkflow;
      const postedWorkflow = typeof req.body?.comfyui_workflow === 'string' ? req.body.comfyui_workflow.trim() : '';
      const workflowInput = postedWorkflow || stored.comfyui_workflow;
      if (postedWorkflow || provider === 'comfyui') {
        try {
          comfyWorkflow = validateComfyUIWorkflow(
            typeof workflowInput === 'string' ? workflowInput : JSON.stringify(workflowInput)
          );
        } catch (error) {
          return res.status(400).json({ error: error.message });
        }
      }
      let comfyFrameRate;
      if (req.body?.comfyui_frame_rate !== undefined) {
        comfyFrameRate = Number(req.body.comfyui_frame_rate);
        if (!Number.isInteger(comfyFrameRate) || comfyFrameRate < 1 || comfyFrameRate > 60) {
          return res.status(400).json({ error: 'ComfyUI frame rate must be an integer between 1 and 60' });
        }
      }
      const comfyApiKey = typeof req.body?.comfyui_api_key === 'string' ? req.body.comfyui_api_key.trim() : '';
      if (comfyApiKey.length > 500) return res.status(400).json({ error: 'ComfyUI API token must be 500 characters or fewer' });
      if (comfyApiKey && !this.credentials) return res.status(503).json({ error: 'Credential storage is not initialized' });
      const mode = req.body?.video_generation_mode;
      if (mode !== undefined && !['hybrid', 'slideshow'].includes(mode)) {
        return res.status(400).json({ error: 'Unsupported video generation mode' });
      }
      let clipDuration;
      if (req.body?.video_clip_duration !== undefined) {
        clipDuration = Number(req.body.video_clip_duration);
        if (!Number.isInteger(clipDuration) || clipDuration < 3 || clipDuration > 30) {
          return res.status(400).json({ error: 'Clip duration must be between 3 and 30 seconds' });
        }
      }
      let maxGeneratedSeconds;
      if (req.body?.video_max_generated_seconds !== undefined) {
        maxGeneratedSeconds = Number(req.body.video_max_generated_seconds);
        if (!Number.isInteger(maxGeneratedSeconds) || maxGeneratedSeconds < 0 || maxGeneratedSeconds > 600) {
          return res.status(400).json({ error: 'Generated seconds cap must be between 0 and 600' });
        }
      }
      const ageBands = new Set(['general audience', 'under 5', '5-7', '8-12', '13-17', '18+']);
      const ageBand = req.body?.target_age_band;
      if (ageBand !== undefined && !ageBands.has(ageBand)) {
        return res.status(400).json({ error: 'Unsupported target age band' });
      }
      const contentLanguage = req.body?.content_language;
      if (contentLanguage !== undefined &&
          (typeof contentLanguage !== 'string' || !contentLanguage.trim() || contentLanguage.trim().length > 60)) {
        return res.status(400).json({ error: 'Content language must be 1–60 characters' });
      }
      const childSafeMode = req.body?.child_safe_mode;
      if (childSafeMode !== undefined && typeof childSafeMode !== 'boolean') {
        return res.status(400).json({ error: 'child_safe_mode must be a boolean' });
      }
      const imageGenerationMode = req.body?.image_generation_mode;
      if (imageGenerationMode !== undefined && !['local', 'cloud'].includes(imageGenerationMode)) {
        return res.status(400).json({ error: 'Image generation mode must be local or cloud' });
      }
      let defaultVideoMinutes;
      if (req.body?.default_video_minutes !== undefined) {
        defaultVideoMinutes = Number(req.body.default_video_minutes);
        if (!Number.isInteger(defaultVideoMinutes) || defaultVideoMinutes < 1 || defaultVideoMinutes > 60) {
          return res.status(400).json({ error: 'Default video length must be between 1 and 60 minutes' });
        }
      }
      let defaultShortSeconds;
      if (req.body?.default_short_seconds !== undefined) {
        defaultShortSeconds = Number(req.body.default_short_seconds);
        if (!Number.isInteger(defaultShortSeconds) || defaultShortSeconds < 15 || defaultShortSeconds > 180) {
          return res.status(400).json({ error: 'Default Short length must be between 15 and 180 seconds' });
        }
      }
      let shortDraftCount;
      if (req.body?.shorts_draft_count !== undefined) {
        shortDraftCount = Number(req.body.shorts_draft_count);
        if (!Number.isInteger(shortDraftCount) || shortDraftCount < 1 || shortDraftCount > 5) {
          return res.status(400).json({ error: 'Short draft count must be between 1 and 5' });
        }
      }
      let shortsTargetSeconds;
      if (req.body?.shorts_target_seconds !== undefined) {
        shortsTargetSeconds = Number(req.body.shorts_target_seconds);
        if (!Number.isInteger(shortsTargetSeconds) || shortsTargetSeconds < 15 || shortsTargetSeconds > 180) {
          return res.status(400).json({ error: 'Repurposed Short target must be between 15 and 180 seconds' });
        }
      }
      const allowed = ['approval_required', 'notification_enabled', 'channel_timezone', 'max_daily_posts', 'content_buffer_days'];
      for (const key of allowed) {
        if (req.body?.[key] !== undefined) await this.db.setSetting(key, String(req.body[key]));
      }
      if (provider !== undefined) {
        await this.db.setSetting('video_provider', provider);
      }
      if (comfyBaseURL) await this.db.setSetting('comfyui_base_url', comfyBaseURL);
      if (comfyWorkflow) await this.db.setSetting('comfyui_workflow', JSON.stringify(comfyWorkflow));
      if (comfyFrameRate !== undefined) await this.db.setSetting('comfyui_frame_rate', String(comfyFrameRate));
      if (comfyApiKey) {
        this.credentials.credentials = {
          ...(this.credentials.credentials || {}),
          comfyui: { ...(this.credentials.credentials?.comfyui || {}), apiKey: comfyApiKey }
        };
        await this.credentials.saveCredentials();
      }
      if (mode !== undefined) {
        await this.db.setSetting('video_generation_mode', mode);
      }
      if (clipDuration !== undefined) await this.db.setSetting('video_clip_duration', String(clipDuration));
      if (maxGeneratedSeconds !== undefined) await this.db.setSetting('video_max_generated_seconds', String(maxGeneratedSeconds));
      if (ageBand !== undefined) await this.db.setSetting('target_age_band', ageBand);
      if (contentLanguage !== undefined) await this.db.setSetting('content_language', contentLanguage.trim());
      if (childSafeMode !== undefined) await this.db.setSetting('child_safe_mode', String(childSafeMode));
      if (imageGenerationMode !== undefined) await this.db.setSetting('image_generation_mode', imageGenerationMode);
      if (defaultVideoMinutes !== undefined) await this.db.setSetting('default_video_minutes', String(defaultVideoMinutes));
      if (defaultShortSeconds !== undefined) await this.db.setSetting('default_short_seconds', String(defaultShortSeconds));
      if (shortDraftCount !== undefined) await this.db.setSetting('shorts_draft_count', String(shortDraftCount));
      if (shortsTargetSeconds !== undefined) await this.db.setSetting('shorts_target_seconds', String(shortsTargetSeconds));
      return res.json({ success: true, result: await this.db.getAllSettings() });
    });

    this.app.post('/api/video/comfyui/check', protect, async (req, res) => {
      const stored = await this.db.getAllSettings();
      let baseURL;
      let workflow;
      try {
        baseURL = normalizeComfyUIBaseURL(req.body?.baseURL || stored.comfyui_base_url || process.env.COMFYUI_BASE_URL || 'http://127.0.0.1:8188');
        const workflowValue = req.body?.workflow ?? stored.comfyui_workflow;
        workflow = validateComfyUIWorkflow(typeof workflowValue === 'string' ? workflowValue : JSON.stringify(workflowValue));
      } catch (error) {
        return res.status(400).json({ success: false, error: error.message });
      }
      const provider = this.agents.production?.aiVideoGenerator?.mediaGeneration?.registry.get('comfyui');
      if (!provider) return res.status(503).json({ success: false, error: 'Video generation service is not initialized' });
      const apiKey = typeof req.body?.apiKey === 'string' ? req.body.apiKey.trim() : '';
      if (apiKey.length > 500) return res.status(400).json({ success: false, error: 'ComfyUI API token must be 500 characters or fewer' });
      const previousConfig = {
        baseURL: provider.baseURL, workflow: provider.workflow,
        frameRate: provider.frameRate, apiKey: provider.apiKey
      };
      provider.configure({
        baseURL, workflow,
        apiKey: apiKey || this.credentials?.credentials?.comfyui?.apiKey || process.env.COMFYUI_API_KEY || ''
      });
      try {
        const system = await provider.checkConnection();
        return res.json({
          success: true,
          result: {
            message: 'ComfyUI is reachable and the API workflow is valid.',
            workflowReady: true,
            system: { devices: Array.isArray(system.devices) ? system.devices.length : undefined }
          }
        });
      } catch (error) {
        return res.status(502).json({ success: false, error: `Could not connect to ComfyUI: ${safeModelError(error)}` });
      } finally {
        provider.configure(previousConfig);
      }
    });

    this.app.post('/api/notifications/:notificationId/read', protect, async (req, res) => {
      await this.db.markNotificationRead(req.params.notificationId);
      return res.json({ success: true });
    });
  }

  async startGenerationJob(input = {}) {
    if (this.setupRequired || !this.agents.strategy) {
      const error = new Error('Finish setup with npm run walkthrough before generating content');
      error.status = 503;
      throw error;
    }
    if (['scheduler', 'autonomous_operator'].includes(input.source)) {
      await this.readiness?.assertReady('Automated generation');
    }
    const maxConcurrent = Math.max(1, parseInt(process.env.MAX_CONCURRENT_JOBS || '1', 10));
    if (this.activeJobs.size >= maxConcurrent) {
      const error = new Error(`Generation is busy (${this.activeJobs.size}/${maxConcurrent} active jobs). Try again when the current job finishes.`);
      error.status = 429;
      throw error;
    }
    const savedSettings = await this.db.getAllSettings();
    const request = { ...input };
    if (request.targetDuration === undefined || request.targetDuration === null || request.targetDuration === '') {
      request.targetDuration = request.format === 'youtube_short'
        ? Number(savedSettings.default_short_seconds || 45)
        : Number(savedSettings.default_video_minutes || 10);
    }
    const validation = this.validateGenerateRequestBody(request);
    if (!validation.valid) {
      const error = new Error(validation.error);
      error.status = validation.status;
      throw error;
    }

    const job = await this.db.createGenerationJob({
      ...validation.value,
      source: input.source || 'manual'
    });

    const work = this.runGenerationJob(job.id, validation.value)
      .catch(error => this.logger.error(`Generation job ${job.id} failed:`, error))
      .finally(() => this.activeJobs.delete(job.id));
    this.activeJobs.set(job.id, work);
    return job;
  }

  async resumeGenerationJob(jobId, options = {}) {
    if (this.setupRequired || !this.agents.strategy) {
      const error = new Error('Finish setup with npm run walkthrough before resuming content generation');
      error.status = 503;
      throw error;
    }
    const job = await this.db.getGenerationJob(jobId);
    if (!job) {
      const error = new Error('Generation job not found');
      error.status = 404;
      throw error;
    }
    if (!['failed', 'interrupted'].includes(job.status)) {
      const error = new Error('Only failed or interrupted generation jobs can be resumed');
      error.status = 409;
      throw error;
    }
    if (this.activeJobs.has(job.id)) {
      const error = new Error('This generation job is already running');
      error.status = 409;
      throw error;
    }
    const maxConcurrent = Math.max(1, parseInt(process.env.MAX_CONCURRENT_JOBS || '1', 10));
    if (this.activeJobs.size >= maxConcurrent) {
      const error = new Error(`Generation is busy (${this.activeJobs.size}/${maxConcurrent} active jobs). Try again when the current job finishes.`);
      error.status = 429;
      throw error;
    }
    if (['scheduler', 'autonomous_operator'].includes(job.source)) {
      await this.readiness?.assertReady('Automated generation recovery');
    }

    const checkpoints = await this.db.listGenerationCheckpoints(job.id);
    const resumeFrom = options.stage || this.recovery.resumePoint(checkpoints);
    if (!GENERATION_STAGES.includes(resumeFrom)) {
      const error = new Error('Resume stage is not supported');
      error.status = 400;
      throw error;
    }
    if (options.stage) await this.recovery.resetFrom(job.id, resumeFrom);
    const input = {
      topic: job.topic,
      style: job.style,
      length: job.length || 'medium',
      format: job.details?.format || 'video',
      targetDuration: job.details?.targetDuration || null,
      strategyContext: job.details?.strategyContext || {}
    };
    const updated = await this.db.updateGenerationJob(job.id, {
      status: 'queued',
      stage: resumeFrom,
      error: null,
      cancelRequested: false,
      completedAt: null,
      details: {
        resumeCount: Number(job.details?.resumeCount || 0) + 1,
        resumeFrom,
        failedStage: null
      }
    });
    const work = this.runGenerationJob(job.id, input)
      .catch(error => this.logger.error(`Resumed generation job ${job.id} failed:`, error))
      .finally(() => this.activeJobs.delete(job.id));
    this.activeJobs.set(job.id, work);
    return updated;
  }

  async waitForGenerationJob(jobId) {
    const work = this.activeJobs.get(jobId);
    if (work) await work;
    const job = await this.db.getGenerationJob(jobId);
    if (!job) throw new Error(`Generation job ${jobId} was not found after it ran`);
    return job;
  }

  async queueScheduledContent(input = {}) {
    const strategy = await this.db.getChannelStrategy();
    if (strategy?.status === 'active') {
      const weeklyOutput = await this.db.getRow(
        `SELECT COUNT(*) AS count FROM generation_jobs
         WHERE source = 'autonomous_operator' AND status = 'completed'
         AND created_at >= datetime('now', '-7 days')`
      );
      const remaining = Math.max(1, strategy.cadence_per_week - Number(weeklyOutput?.count || 0));
      return this.autonomous.start({
        ...strategy,
        videos_per_run: Math.min(strategy.videos_per_run, remaining)
      });
    }
    return this.startGenerationJob(input);
  }

  async runGenerationJob(jobId, input) {
    try {
      await this.db.updateGenerationJob(jobId, { status: 'running', progress: 2, error: null, completedAt: null });
      const result = await this.generateContent(input.topic, input.style, input.length, {
        jobId,
        strategyContext: input.strategyContext,
        format: input.format || 'video',
        targetDuration: input.targetDuration || null
      });
      await this.db.updateGenerationJob(jobId, {
        status: 'completed',
        stage: result.reviewStatus === 'approved' ? 'scheduled' : result.reviewStatus,
        progress: 100,
        productionId: result.contentId,
        title: result.title,
        details: { reviewStatus: result.reviewStatus, qualityScore: result.qualityScore },
        completedAt: new Date().toISOString()
      });
      await this.db.setSetting('last_content_generation', new Date().toISOString());
      return result;
    } catch (error) {
      const cancelled = error.code === 'JOB_CANCELLED';
      const current = await this.db.getGenerationJob(jobId);
      const failedStage = current?.stage || 'starting';
      await this.db.updateGenerationJob(jobId, {
        status: cancelled ? 'cancelled' : 'failed',
        stage: failedStage,
        error: error.message,
        details: { failedStage },
        completedAt: new Date().toISOString()
      });
      await this.operator.notify({
        type: cancelled ? 'generation_cancelled' : 'generation_failure',
        level: cancelled ? 'warning' : 'error',
        title: cancelled ? 'Generation cancelled' : 'Generation failed',
        message: error.message,
        data: { jobId }
      });
      throw error;
    }
  }

  async updateJobStage(jobId, stage, progress, details = {}) {
    if (!jobId) return;
    const job = await this.db.getGenerationJob(jobId);
    if (job?.cancelRequested) {
      const error = new Error(job.details?.cancelReason || 'Generation cancelled by operator');
      error.code = 'JOB_CANCELLED';
      throw error;
    }
    await this.db.updateGenerationJob(jobId, { stage, progress, details });
  }

  async generateContent(topic = null, style = null, length = 'medium', options = {}) {
    this.logger.info('Starting content generation pipeline...');
    const {
      jobId = null,
      strategyContext: rawStrategyContext = {},
      format = 'video',
      targetDuration = null
    } = options;
    const strategyContext = rawStrategyContext || {};
    const profile = await this.db.getChannelProfile() || {};
    const appSettings = await this.db.getAllSettings();
    const lengthLabels = { short: '2-4 minutes', medium: '8-12 minutes', long: '15-20 minutes' };

    // Step 1: Strategy
    const strategy = await this.runGenerationStage(jobId, 'strategy', 10, async () => {
      const generated = await this.agents.strategy.generateContentStrategy(topic);
      const contentStyles = new Set(['tutorial', 'explainer', 'list', 'review', 'story', 'song', 'lookfind', 'activity']);
      const requestedStyle = style || profile.default_style || null;
      if (requestedStyle && contentStyles.has(requestedStyle.toLowerCase())) {
        generated.contentType = requestedStyle.charAt(0).toUpperCase() + requestedStyle.slice(1).toLowerCase();
      }
      generated.requestedStyle = requestedStyle;
      generated.requestedLengthKey = length;
      generated.outputFormat = format;
      generated.targetDuration = targetDuration ||
        (format === 'youtube_short' ? Number(appSettings.default_short_seconds || 45) : null);
      generated.aspectRatio = format === 'youtube_short' ? '9:16' : '16:9';
      generated.requestedLength = generated.targetDuration
        ? format === 'youtube_short' ? `${generated.targetDuration} seconds (vertical YouTube Short)` : `${generated.targetDuration} minutes`
        : lengthLabels[length] || lengthLabels.medium;
      generated.audienceAgeBand = appSettings.target_age_band || process.env.TARGET_AGE_BAND || '8-12';
      generated.contentLanguage = appSettings.content_language || process.env.CONTENT_LANGUAGE || 'English';
      generated.childSafeMode = appSettings.child_safe_mode === undefined
        ? process.env.CHILD_SAFE_MODE !== 'false'
        : appSettings.child_safe_mode === 'true';
      generated.angle = strategyContext.angle || generated.angle;
      generated.planRationale = strategyContext.rationale || null;
      generated.targetAudience = strategyContext.audience || profile.target_audience || generated.targetAudience;
      generated.brandVoice = profile.brand_voice || null;
      generated.channelGoal = strategyContext.objective || profile.goal || null;
      generated.channelValueProposition = strategyContext.valueProposition || null;
      generated.channelConstraints = strategyContext.constraints || null;
      generated.contentPillar = strategyContext.pillar || null;
      generated.callToAction = profile.call_to_action || null;
      generated.researchSources = Array.isArray(strategyContext.researchSources)
        ? strategyContext.researchSources
        : [];
      return generated;
    });
    this.logger.info(`Strategy generated: ${strategy.topic}`);

    // Step 2: Script Writing
    const script = await this.runGenerationStage(
      jobId,
      'script',
      25,
      () => this.agents.scriptWriter.generateScript(strategy)
    );
    this.logger.info(`Script generated: ${script.title}`);

    // Step 3: Thumbnail Design
    const thumbnail = await this.runGenerationStage(
      jobId,
      'thumbnail',
      40,
      () => this.agents.thumbnailDesigner.generateThumbnail(script)
    );
    this.logger.info('Thumbnail generated');

    // Step 4: SEO Optimization
    const seoData = await this.runGenerationStage(
      jobId,
      'seo',
      52,
      () => this.agents.seoOptimizer.optimize(script, strategy)
    );
    this.logger.info('SEO optimization complete');

    // Step 5: Production Management
    const productionData = await this.runGenerationStage(
      jobId,
      'production',
      62,
      () => this.agents.production.processContent({ strategy, script, thumbnail, seo: seoData, jobId })
    );
    this.logger.info('Production processing complete');

    // Re-persist reused production artifacts in case a restart happened between checkpointing and persistence.
    const contentId = await this.db.saveProductionData(productionData);
    await this.db.saveProductionSnapshot(productionData);
    if (!this.provenance) this.provenance = new ProvenanceService(this.db);
    productionData.provenance = await this.provenance.initialize(contentId, productionData);
    productionData.discoverability = this.discoverability
      ? await this.discoverability.auditProduction(productionData, profile, 'youtube')
      : null;
    this.logger.info(`Content saved with ID: ${contentId}`);

    // Step 6: Quality and approval gate
    return this.runGenerationStage(jobId, 'quality_review', 90, async () => {
      const approvalRequired = await this.db.getSetting('approval_required') !== 'false';
      const packagingExperiment = approvalRequired
        ? await this.preparePackagingExperiment(thumbnail, productionData, seoData, script)
        : null;
      const quality = await this.operator.runQualityChecks(productionData, profile);
      const reviewStatus = quality.passed
        ? (approvalRequired ? 'needs_review' : 'approved')
        : 'needs_attention';
      await this.db.saveContentReview(contentId, {
        status: reviewStatus,
        qualityChecks: quality.checks,
        editorData: packagingExperiment ? {
          packagingExperiment,
          selectedTitleVariant: 0,
          selectedThumbnailVariant: 0
        } : {},
        reviewNotes: quality.passed ? null : `Blocking checks failed: ${quality.blockingFailures.join(', ')}`,
        reviewedAt: approvalRequired ? null : new Date().toISOString()
      });

      let scheduleEntry = null;
      if (reviewStatus === 'approved') {
        scheduleEntry = await this.agents.publishing.scheduleContent(productionData);
        await this.db.updateProductionStatus(contentId, scheduleEntry ? 'scheduled' : productionData.status);
      } else {
        await this.db.updateProductionStatus(contentId, reviewStatus);
        await this.operator.notify({
          type: 'review_required',
          level: quality.passed ? 'info' : 'warning',
          title: quality.passed ? 'Content ready for review' : 'Content needs attention',
          message: `${script.title} ${quality.passed ? 'is ready for approval' : 'failed one or more quality checks'}`,
          data: { contentId, qualityScore: quality.score }
        });
      }

      return {
        contentId,
        title: script.title,
        status: productionData.status,
        reviewStatus,
        qualityScore: quality.score,
        scheduledFor: scheduleEntry ? scheduleEntry.publishTime : null
      };
    });
  }

  async runGenerationStage(jobId, stage, progress, producer) {
    if (!jobId) {
      await this.updateJobStage(jobId, stage, progress);
      return producer();
    }
    if (!this.recovery) {
      this.recovery = new GenerationRecoveryService(this.db, {
        logger: this.logger,
        updateJobStage: (...args) => this.updateJobStage(...args)
      });
    }
    return this.recovery.run(jobId, stage, progress, producer);
  }

  validateEditorData(input = {}, existing = {}) {
    const output = { ...existing };
    if (input.title !== undefined) {
      const title = String(input.title).trim();
      if (!title || title.length > 100) throw new Error('Title must be between 1 and 100 characters');
      output.title = title;
    }
    if (input.description !== undefined) {
      const description = String(input.description).trim();
      if (description.length > 5000) throw new Error('Description must be 5,000 characters or less');
      output.description = description;
    }
    if (input.tags !== undefined) {
      const tags = Array.isArray(input.tags)
        ? input.tags
        : String(input.tags).split(',');
      output.tags = tags.map(tag => String(tag).trim()).filter(Boolean).slice(0, 30);
    }
    if (input.publishTime !== undefined) {
      const date = new Date(input.publishTime);
      if (Number.isNaN(date.getTime())) throw new Error('Publish time must be a valid date');
      output.publishTime = date.toISOString();
    }
    if (input.privacyStatus !== undefined) {
      if (!['private', 'unlisted', 'public'].includes(input.privacyStatus)) throw new Error('Invalid privacy status');
      output.privacyStatus = input.privacyStatus;
    }
    if (input.factChecked !== undefined) output.factChecked = input.factChecked === true;
    if (input.rightsConfirmed !== undefined) output.rightsConfirmed = input.rightsConfirmed === true;
    const experiment = output.packagingExperiment;
    if (input.selectedTitleVariant !== undefined) {
      const selected = Number(input.selectedTitleVariant);
      if (!Number.isInteger(selected) || !experiment?.titleVariants?.[selected]) {
        throw new Error('Selected title variant is invalid');
      }
      output.selectedTitleVariant = selected;
    }
    if (input.selectedThumbnailVariant !== undefined) {
      const selected = Number(input.selectedThumbnailVariant);
      if (!Number.isInteger(selected) || !experiment?.thumbnailVariants?.[selected]) {
        throw new Error('Selected thumbnail variant is invalid');
      }
      output.selectedThumbnailVariant = selected;
    }
    return output;
  }

  buildTitleExperimentVariants(title) {
    const control = String(title || '').trim().slice(0, 100);
    const withoutPunctuation = control.replace(/[.!?]+$/, '');
    return [
      { label: 'Control', title: control },
      { label: 'Step-by-step', title: `${withoutPunctuation}: Step-by-Step`.slice(0, 100) },
      { label: 'Curiosity', title: `${withoutPunctuation}: What Most People Miss`.slice(0, 100) }
    ];
  }

  async preparePackagingExperiment(thumbnail, productionData, seoData, script) {
    const approved = await this.db.listLearningRecommendations({ status: 'approved', limit: 25 });
    const recommendation = approved.find(item => item.proposedChange?.experiment === 'title_thumbnail_variant');
    if (!recommendation) return null;
    try {
      const generated = await this.agents.thumbnailDesigner.generateABVariants(thumbnail.concept);
      return {
        sourceRecommendationId: recommendation.id,
        hypothesis: recommendation.title,
        status: 'draft',
        titleVariants: this.buildTitleExperimentVariants(seoData.title || script.title),
        thumbnailVariants: [
          { label: 'Control', path: productionData.assets?.thumbnail?.path, concept: thumbnail.concept },
          ...generated
        ],
        createdAt: new Date().toISOString()
      };
    } catch (error) {
      this.logger.warn(`Packaging experiment preparation failed without blocking production: ${error.message}`);
      return null;
    }
  }

  validateProfile(input) {
    const textFields = ['channelName', 'goal', 'targetAudience', 'brandVoice', 'defaultStyle', 'callToAction', 'visualStyle', 'timezone'];
    const result = {};
    for (const field of textFields) {
      if (input[field] !== undefined) {
        const value = String(input[field]).trim();
        if (value.length > 500) throw new Error(`${field} is too long`);
        result[field] = value;
      }
    }
    if (input.bannedTopics !== undefined) {
      const topics = Array.isArray(input.bannedTopics) ? input.bannedTopics : String(input.bannedTopics).split(',');
      result.bannedTopics = topics.map(topic => String(topic).trim()).filter(Boolean).slice(0, 50);
    }
    return result;
  }

  decorateContentBundle(bundle) {
    const experiment = bundle.editorData?.packagingExperiment;
    const sceneLabels = new Map((bundle.scenes || []).map(scene => [scene.id, scene.label]));
    return {
      ...bundle,
      scenes: (bundle.scenes || []).map(scene => this.scenes
        ? this.scenes.decorateScene(scene, bundle.id)
        : scene),
      shorts: (bundle.shorts || []).map(clip => ({
        ...clip,
        sourceSceneLabels: clip.sourceSceneIds.map(id => sceneLabels.get(id)).filter(Boolean),
        assetUrls: {
          video: clip.outputPath ? `/api/content/${bundle.id}/shorts/${clip.id}/asset/video` : null,
          captions: clip.captionsPath ? `/api/content/${bundle.id}/shorts/${clip.id}/asset/captions` : null
        }
      })),
      assetUrls: {
        video: bundle.assets?.finalVideo?.path && !bundle.assets?.finalVideo?.simulated ? `/api/content/${bundle.id}/asset/video` : null,
        thumbnail: bundle.assets?.thumbnail?.path ? `/api/content/${bundle.id}/asset/thumbnail` : null,
        experimentThumbnails: (experiment?.thumbnailVariants || []).map((_variant, index) =>
          `/api/content/${bundle.id}/asset/experiment-thumbnail-${index}`
        ),
        captions: bundle.assets?.captions?.path ? `/api/content/${bundle.id}/asset/captions` : null,
        script: bundle.assets?.script?.originalPath ? `/api/content/${bundle.id}/asset/script` : null
      }
    };
  }

  async refreshContentReview(productionId, reviewNotes) {
    const bundle = await this.db.getProductionBundle(productionId);
    if (!bundle) return null;
    const profile = await this.db.getChannelProfile() || {};
    const quality = await this.operator.runQualityChecks({
      ...bundle,
      scheduledPublishTime: bundle.scheduled_publish_time
    }, profile);
    const status = quality.passed ? 'needs_review' : 'needs_attention';
    return this.db.saveContentReview(productionId, {
      status,
      editorData: { ...(bundle.editorData || {}), factChecked: false, rightsConfirmed: false },
      qualityChecks: quality.checks,
      reviewNotes: reviewNotes || (quality.passed ? null : `Blocking checks failed: ${quality.blockingFailures.join(', ')}`),
      reviewedAt: null
    });
  }

  async approveContent(productionId, input) {
    const bundle = await this.db.getProductionBundle(productionId);
    if (!bundle) {
      const error = new Error('Content not found');
      error.status = 404;
      throw error;
    }
    if (bundle.schedule?.status === 'published') {
      const error = new Error('Content is already published');
      error.status = 409;
      throw error;
    }

    const editorData = this.validateEditorData(input, bundle.editorData);
    const packagingExperiment = editorData.packagingExperiment;
    const thumbnailVariant = packagingExperiment?.thumbnailVariants?.[editorData.selectedThumbnailVariant];
    const titleVariant = packagingExperiment?.titleVariants?.[editorData.selectedTitleVariant];
    if (titleVariant && input.title === undefined) editorData.title = titleVariant.title;
    if (!editorData.factChecked || !editorData.rightsConfirmed) {
      const error = new Error('Confirm the factual review and media rights checks before approval');
      error.status = 409;
      throw error;
    }
    const productionData = {
      id: bundle.id,
      status: bundle.status,
      strategy: bundle.strategy,
      script: { ...bundle.script, title: editorData.title || bundle.script.title },
      thumbnail: bundle.thumbnail,
      seo: {
        ...bundle.seo,
        title: editorData.title || bundle.seo.title,
        description: editorData.description || bundle.seo.description,
        tags: editorData.tags || bundle.seo.tags
      },
      assets: thumbnailVariant
        ? { ...bundle.assets, thumbnail: { ...bundle.assets.thumbnail, path: thumbnailVariant.path } }
        : bundle.assets,
      timeline: bundle.timeline,
      scheduledPublishTime: editorData.publishTime || input.publishTime || bundle.scheduled_publish_time,
      priority: bundle.priority,
      estimatedDuration: bundle.estimated_duration,
      privacyStatus: editorData.privacyStatus || process.env.DEFAULT_PRIVACY_STATUS || 'private',
      provenance: bundle.provenance,
      containsSyntheticMedia: bundle.provenance?.containsSyntheticMedia === true,
      scenes: bundle.scenes || []
    };
    const profile = await this.db.getChannelProfile() || {};
    const quality = await this.operator.runQualityChecks(productionData, profile);
    if (!quality.passed) {
      await this.db.saveContentReview(bundle.id, {
        status: 'needs_attention', editorData, qualityChecks: quality.checks,
        reviewNotes: `Blocking checks failed: ${quality.blockingFailures.join(', ')}`
      });
      const error = new Error('Content still has blocking quality failures');
      error.status = 409;
      error.quality = quality;
      throw error;
    }

    let scheduleEntry = bundle.schedule;
    if (!scheduleEntry) {
      scheduleEntry = await this.agents.publishing.scheduleContent(productionData);
    } else if (scheduleEntry.status !== 'published') {
      scheduleEntry.title = productionData.script.title;
      scheduleEntry.publishTime = productionData.scheduledPublishTime;
      scheduleEntry.status = 'scheduled';
      scheduleEntry.metadata = {
        ...scheduleEntry.metadata,
        seo: productionData.seo,
        thumbnail: productionData.assets.thumbnail,
        video: productionData.assets.finalVideo,
        audio: productionData.assets.audio,
        captions: productionData.assets.captions,
        privacyStatus: editorData.privacyStatus || process.env.DEFAULT_PRIVACY_STATUS || 'private',
        containsSyntheticMedia: productionData.containsSyntheticMedia
      };
      await this.db.updateScheduleEntry(scheduleEntry);
      await this.agents.publishing.loadPublishQueue();
    }
    if (!scheduleEntry) {
      const error = new Error('A real MP4 is required before content can be approved for scheduling');
      error.status = 409;
      throw error;
    }

    await this.db.saveContentReview(bundle.id, {
      status: 'approved', editorData, qualityChecks: quality.checks,
      reviewNotes: input.reviewNotes || 'Approved by operator', reviewedAt: new Date().toISOString()
    });
    await this.db.updateProductionStatus(bundle.id, 'scheduled');
    await this.operator.notify({
      type: 'content_approved', level: 'success', title: 'Content approved',
      message: `${productionData.script.title} is scheduled for ${scheduleEntry.publishTime}`,
      data: { productionId, publishTime: scheduleEntry.publishTime }
    });
    return { productionId, reviewStatus: 'approved', qualityScore: quality.score, schedule: scheduleEntry };
  }

  async start() {
    const initialized = await this.initialize();
    
    if (!initialized) {
      console.log(chalk.red('\n❌ Failed to initialize. Please check your configuration.'));
      process.exit(1);
    }
    
    const PORT = process.env.PORT || 3456;
    // Loopback by default: mutating routes are open whenever API_KEY is unset,
    // so the dashboard must not be reachable from the network unless asked.
    const HOST = process.env.HOST || '127.0.0.1';
    this.app.listen(PORT, HOST, () => {
      console.log(chalk.green(`\n✅ YouTube Automation Agent running on ${HOST}:${PORT}`));
      if (!isLoopbackHost(HOST) && !process.env.API_KEY) {
        console.log(chalk.yellow(`⚠️  HOST=${HOST} exposes the dashboard to your network without API_KEY. Set API_KEY in .env to protect generation, approval, and publishing routes.`));
      }
      console.log(chalk.gray('─'.repeat(50)));
      console.log(chalk.white('📊 Dashboard: ') + chalk.cyan(`http://localhost:${PORT}`));
      console.log(chalk.white('🔧 API Health: ') + chalk.cyan(`http://localhost:${PORT}/health`));
      console.log(chalk.white('📅 Schedule: ') + chalk.cyan(`http://localhost:${PORT}/schedule`));
      console.log(chalk.white('📈 Analytics: ') + chalk.cyan(`http://localhost:${PORT}/analytics`));
      console.log(chalk.gray('─'.repeat(50)));
      if (this.setupRequired) {
        console.log(chalk.yellow('\n⚙️  Setup is required. The dashboard is available; run npm run walkthrough to enable generation.'));
      } else {
        console.log(chalk.yellow('\n🤖 Automation is active. Approved content will be published on schedule.'));
      }
    });
  }
}

// Start the agent
if (require.main === module) {
  const agent = new YouTubeAutomationAgent();
  agent.start().catch(error => {
    console.error(chalk.red('Fatal error:'), error);
    process.exit(1);
  });
}

module.exports = { YouTubeAutomationAgent };
