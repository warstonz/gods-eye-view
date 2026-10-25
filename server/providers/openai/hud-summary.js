import {
  HUD_SUMMARY_INSTRUCTIONS,
  keylessHudSummaryResponse,
} from '../../../src/hudSummaryResponse.js';
import { enforceRateLimit, openAiRateLimiter } from './rate-limit.js';
import { readRequestBody } from '../common/request.js';
import { OPENAI_HUD_SUMMARY_MODEL_DEFAULT } from './constants.js';
import { createAnthropicHudSummary } from '../anthropic/hud-summary.js';

function extractOpenAiResponseText(data) {
  if (typeof data?.output_text === 'string' && data.output_text.trim()) {
    return data.output_text.trim();
  }
  if (!Array.isArray(data?.output)) return '';
  return data.output
    .flatMap((item) => (Array.isArray(item?.content) ? item.content : []))
    .map((part) => part?.text || part?.output_text || '')
    .join(' ')
    .trim();
}

function toFiveWordHudSummary(value) {
  return String(value || '')
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 5)
    .join(' ');
}

const PROVIDER_LABELS = { anthropic: 'Claude', openai: 'OpenAI' };

/**
 * Pick which provider writes the HUD line. Claude wins whenever
 * ANTHROPIC_API_KEY is set, unless HUD_SUMMARY_PROVIDER=openai.
 *
 * @param {Record<string, string|undefined>} [env]
 * @returns {'anthropic'|'openai'}
 */
function hudSummaryProvider(env = process.env) {
  const hasAnthropicKey = Boolean(String(env.ANTHROPIC_API_KEY ?? '').trim());
  const preferred = String(env.HUD_SUMMARY_PROVIDER ?? '')
    .trim()
    .toLowerCase();
  return hasAnthropicKey && preferred !== 'openai' ? 'anthropic' : 'openai';
}

async function requestOpenAiHudSummary(context, apiKey) {
  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model:
        process.env.OPENAI_HUD_SUMMARY_MODEL ||
        OPENAI_HUD_SUMMARY_MODEL_DEFAULT,
      instructions: HUD_SUMMARY_INSTRUCTIONS,
      input: JSON.stringify(context),
      reasoning: { effort: 'minimal' },
      max_output_tokens: 100,
    }),
  });
  if (!response.ok) {
    console.warn(`[hud-summary] upstream HTTP ${response.status}`);
    // Never relay `data.error.message`: that is OpenAI's own wording, and
    // it carries request ids, organization hints and quota phrasing.
    return { ok: false, status: response.status || 502, text: '' };
  }
  const data = await response.json().catch(() => ({}));
  return { ok: true, status: 200, text: extractOpenAiResponseText(data) };
}

/**
 * Build the /api/openai/hud-summary handler. The path keeps its old name so
 * the browser and QA scripts need no change; the provider is chosen per
 * request from the environment.
 *
 * @param {{ requestAnthropic?: (context: object, apiKey: string) => Promise<string> }} [options]
 */
function createHudSummaryHandler({
  requestAnthropic = createAnthropicHudSummary(),
} = {}) {
  return async function handleHudSummary(req, res) {
    if (req.method !== 'POST') {
      res.statusCode = 405;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }

    const provider = hudSummaryProvider();
    const apiKey =
      provider === 'anthropic'
        ? process.env.ANTHROPIC_API_KEY
        : process.env.OPENAI_API_KEY;
    const failure = `${PROVIDER_LABELS[provider]} HUD summary request failed`;
    const keyless = keylessHudSummaryResponse(apiKey);
    if (keyless) {
      res.statusCode = keyless.statusCode;
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.setHeader('Cache-Control', 'no-store');
      res.end(JSON.stringify(keyless.payload));
      return;
    }

    // Per-IP throttle (GEV_RATELIMIT_OPENAI_PER_MIN). On by default; 0 disables.
    // It covers this endpoint whichever provider answers. Keyless HUD fallback
    // has no provider cost and resolves above without consuming a paid-endpoint
    // quota slot.
    if (!enforceRateLimit(openAiRateLimiter(), req, res)) return;

    try {
      const body = await readRequestBody(req, 64 * 1024);
      const context = JSON.parse(body || '{}');
      const result =
        provider === 'anthropic'
          ? {
              ok: true,
              status: 200,
              text: await requestAnthropic(context, apiKey),
            }
          : await requestOpenAiHudSummary(context, apiKey);
      const summary = toFiveWordHudSummary(result.text);
      res.statusCode = result.ok && summary ? 200 : result.status || 502;
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.setHeader('Cache-Control', 'no-store');
      res.end(
        JSON.stringify({
          summary: summary || null,
          error: result.ok ? null : failure,
        }),
      );
    } catch (error) {
      // Log only the status: provider error text can carry request ids and
      // account hints, and none of it reaches the browser.
      console.warn(
        `[hud-summary] ${provider} request failed${error?.status ? ` (HTTP ${error.status})` : ''}`,
      );
      res.statusCode = 502;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: failure }));
    }
  };
}

const handleHudSummary = createHudSummaryHandler();

export { createHudSummaryHandler, handleHudSummary, hudSummaryProvider };
