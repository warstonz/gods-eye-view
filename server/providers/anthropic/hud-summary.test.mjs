import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import {
  ANTHROPIC_HUD_SUMMARY_MODEL_DEFAULT,
  createAnthropicHudSummary,
} from './hud-summary.js';
import {
  createHudSummaryHandler,
  hudSummaryProvider,
} from '../openai/hud-summary.js';
import {
  HUD_SUMMARY_INSTRUCTIONS,
  isHudSummaryUnconfigured,
} from '../../../src/hudSummaryResponse.js';

const ENV_NAMES = [
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_HUD_SUMMARY_MODEL',
  'HUD_SUMMARY_PROVIDER',
  'OPENAI_API_KEY',
  'GEV_RATELIMIT_OPENAI_PER_MIN',
];

/** Run `fn` with the given env values, restoring the originals afterwards. */
async function withEnv(values, fn) {
  const saved = Object.fromEntries(ENV_NAMES.map((n) => [n, process.env[n]]));
  for (const name of ENV_NAMES) delete process.env[name];
  // Unthrottled so repeated test requests never hit the shared limiter.
  Object.assign(process.env, { GEV_RATELIMIT_OPENAI_PER_MIN: '0' }, values);
  try {
    return await fn();
  } finally {
    for (const name of ENV_NAMES) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  }
}

/** A fake Anthropic client that records the request and returns `reply`. */
function fakeClient(reply) {
  const calls = [];
  const createClient = (apiKey) => ({
    beta: {
      messages: {
        create: async (params) => {
          calls.push({ apiKey, params });
          if (reply instanceof Error) throw reply;
          return reply;
        },
      },
    },
  });
  return { calls, createClient };
}

function textReply(text, extra = {}) {
  return {
    stop_reason: 'end_turn',
    content: [
      { type: 'thinking', thinking: '' },
      { type: 'text', text },
    ],
    ...extra,
  };
}

function request(body = '{}') {
  const req = new EventEmitter();
  req.method = 'POST';
  req.headers = {};
  req.socket = { remoteAddress: '127.0.0.1' };
  setImmediate(() => {
    req.emit('data', Buffer.from(body));
    req.emit('end');
  });
  return req;
}

function response() {
  let resolve;
  const done = new Promise((r) => (resolve = r));
  const res = {
    statusCode: 200,
    headers: {},
    setHeader(name, value) {
      this.headers[name.toLowerCase()] = value;
    },
    end(text) {
      resolve({ status: this.statusCode, body: JSON.parse(text) });
    },
  };
  return { res, done };
}

async function callHandler(handler, body) {
  const { res, done } = response();
  await handler(request(body), res);
  return done;
}

test('the Claude request carries the HUD instructions, context, and fallback opt-in', async () => {
  await withEnv({}, async () => {
    const { calls, createClient } = fakeClient(
      textReply('Tokyo Shibuya flights live now'),
    );
    const ask = createAnthropicHudSummary({ createClient });
    const context = { place: 'Shibuya', enabledLayerLabels: ['Flights'] };

    assert.equal(
      await ask(context, 'sk-test'),
      'Tokyo Shibuya flights live now',
    );

    assert.equal(calls.length, 1);
    const { apiKey, params } = calls[0];
    assert.equal(apiKey, 'sk-test');
    assert.equal(params.model, ANTHROPIC_HUD_SUMMARY_MODEL_DEFAULT);
    assert.equal(params.system, HUD_SUMMARY_INSTRUCTIONS);
    assert.deepEqual(params.messages, [
      { role: 'user', content: JSON.stringify(context) },
    ]);
    assert.deepEqual(params.output_config, { effort: 'low' });
    assert.equal(params.fallbacks, 'default');
    assert.deepEqual(params.betas, ['server-side-fallback-2026-07-01']);
    assert.equal(
      'thinking' in params,
      false,
      'thinking cannot be disabled on this model',
    );
  });
});

test('ANTHROPIC_HUD_SUMMARY_MODEL overrides the default model', async () => {
  await withEnv(
    { ANTHROPIC_HUD_SUMMARY_MODEL: 'claude-haiku-5-5' },
    async () => {
      const { calls, createClient } = fakeClient(textReply('ok'));
      await createAnthropicHudSummary({ createClient })({}, 'k');
      assert.equal(calls[0].params.model, 'claude-haiku-5-5');
    },
  );
});

test('a refusal yields no summary rather than refusal text', async () => {
  const { createClient } = fakeClient(
    textReply('I cannot help with that', { stop_reason: 'refusal' }),
  );
  assert.equal(await createAnthropicHudSummary({ createClient })({}, 'k'), '');
});

test('Claude answers whenever its key is set, unless OpenAI is pinned', () => {
  assert.equal(hudSummaryProvider({}), 'openai');
  assert.equal(hudSummaryProvider({ OPENAI_API_KEY: 'o' }), 'openai');
  assert.equal(hudSummaryProvider({ ANTHROPIC_API_KEY: 'a' }), 'anthropic');
  assert.equal(hudSummaryProvider({ ANTHROPIC_API_KEY: '  ' }), 'openai');
  assert.equal(
    hudSummaryProvider({ ANTHROPIC_API_KEY: 'a', OPENAI_API_KEY: 'o' }),
    'anthropic',
  );
  assert.equal(
    hudSummaryProvider({
      ANTHROPIC_API_KEY: 'a',
      OPENAI_API_KEY: 'o',
      HUD_SUMMARY_PROVIDER: ' OpenAI ',
    }),
    'openai',
  );
});

test('the endpoint returns a five-word Claude summary', async () => {
  await withEnv({ ANTHROPIC_API_KEY: 'sk-ant' }, async () => {
    const handler = createHudSummaryHandler({
      requestAnthropic: async (context, apiKey) => {
        assert.deepEqual(context, { place: 'Paris' });
        assert.equal(apiKey, 'sk-ant');
        return 'Paris, Eiffel Tower: flights STALE today!';
      },
    });
    const { status, body } = await callHandler(handler, '{"place":"Paris"}');
    assert.equal(status, 200);
    assert.deepEqual(body, {
      summary: 'Paris Eiffel Tower flights STALE',
      error: null,
    });
  });
});

test('a Claude failure is a 502 that never relays provider wording', async () => {
  await withEnv({ ANTHROPIC_API_KEY: 'sk-ant' }, async () => {
    const leak = Object.assign(new Error('org_123 quota exceeded req_abc'), {
      status: 429,
    });
    const handler = createHudSummaryHandler({
      requestAnthropic: async () => {
        throw leak;
      },
    });
    const warn = console.warn;
    const logged = [];
    console.warn = (line) => logged.push(line);
    try {
      const { status, body } = await callHandler(handler, '{}');
      assert.equal(status, 502);
      assert.deepEqual(body, { error: 'Claude HUD summary request failed' });
    } finally {
      console.warn = warn;
    }
    assert.deepEqual(logged, [
      '[hud-summary] anthropic request failed (HTTP 429)',
    ]);
  });
});

test('with no keys at all the endpoint stays a graceful unconfigured 200', async () => {
  await withEnv({}, async () => {
    const handler = createHudSummaryHandler({
      requestAnthropic: () => assert.fail('no provider should be called'),
    });
    const { status, body } = await callHandler(handler, '{}');
    assert.equal(isHudSummaryUnconfigured(status, body), true);
  });
});

test('pinning OpenAI without an OpenAI key does not spend the Claude key', async () => {
  await withEnv(
    { ANTHROPIC_API_KEY: 'sk-ant', HUD_SUMMARY_PROVIDER: 'openai' },
    async () => {
      const handler = createHudSummaryHandler({
        requestAnthropic: () => assert.fail('Claude must not be called'),
      });
      const { status, body } = await callHandler(handler, '{}');
      assert.equal(isHudSummaryUnconfigured(status, body), true);
    },
  );
});
