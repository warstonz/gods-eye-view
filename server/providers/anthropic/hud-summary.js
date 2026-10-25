import Anthropic from '@anthropic-ai/sdk';
import { HUD_SUMMARY_INSTRUCTIONS } from '../../../src/hudSummaryResponse.js';

/** Default model for the HUD line. ANTHROPIC_HUD_SUMMARY_MODEL overrides it. */
const ANTHROPIC_HUD_SUMMARY_MODEL_DEFAULT = 'claude-opus-5-5';

/** Re-route a declined request by refusal category instead of failing it. */
const FALLBACK_BETA = 'server-side-fallback-2026-07-01';

/**
 * Build a function that asks Claude for the raw HUD summary text.
 *
 * `createClient` is injectable so tests can stand in for the API.
 *
 * @param {{ createClient?: (apiKey: string) => Anthropic }} [options]
 * @returns {(context: object, apiKey: string) => Promise<string>} Resolves to
 *   the reply text, or '' when Claude declined or wrote nothing.
 */
function createAnthropicHudSummary({
  createClient = (apiKey) => new Anthropic({ apiKey, maxRetries: 1 }),
} = {}) {
  return async function requestAnthropicHudSummary(context, apiKey) {
    const client = createClient(apiKey);
    const response = await client.beta.messages.create({
      model:
        process.env.ANTHROPIC_HUD_SUMMARY_MODEL ||
        ANTHROPIC_HUD_SUMMARY_MODEL_DEFAULT,
      // Thinking is always on for this model, so leave room beyond the five
      // words. Low effort keeps a 15-second refresh fast and cheap.
      max_tokens: 2048,
      output_config: { effort: 'low' },
      betas: [FALLBACK_BETA],
      fallbacks: 'default',
      system: HUD_SUMMARY_INSTRUCTIONS,
      messages: [{ role: 'user', content: JSON.stringify(context) }],
    });
    if (response.stop_reason === 'refusal') return '';
    return response.content
      .filter((block) => block.type === 'text')
      .map((block) => block.text)
      .join(' ')
      .trim();
  };
}

export { ANTHROPIC_HUD_SUMMARY_MODEL_DEFAULT, createAnthropicHudSummary };
