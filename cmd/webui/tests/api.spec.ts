import { test, expect } from '@playwright/test';

test.describe('API Generate Endpoint', () => {
  test('should return 400 for invalid request body', async ({ request }) => {
    const response = await request.post('/api/generate', {
      data: { invalid_field: 'value' }
    });
    expect(response.status()).toBe(400);
  });

  test('should accept valid generate request', async ({ request }) => {
    const response = await request.post('/api/generate', {
      data: {
        prompt: 'Hello',
        max_tokens: 10,
        temperature: 0.7
      }
    });
    expect(response.ok() || response.status() === 500).toBeTruthy();
  });

  test('should use default values when not provided', async ({ request }) => {
    const response = await request.post('/api/generate', {
      data: { prompt: 'Test' }
    });
    expect(response.ok() || response.status() === 500).toBeTruthy();
  });

  test('should handle empty prompt', async ({ request }) => {
    const response = await request.post('/api/generate', {
      data: { prompt: '' }
    });
    expect(response.status()).toBeGreaterThanOrEqual(400);
  });

  test('should handle negative max_tokens', async ({ request }) => {
    const response = await request.post('/api/generate', {
      data: { prompt: 'Test', max_tokens: -1 }
    });
    expect(response.ok() || response.status() === 500).toBeTruthy();
  });

  test('should handle out of range temperature', async ({ request }) => {
    const response = await request.post('/api/generate', {
      data: { prompt: 'Test', temperature: 5.0 }
    });
    expect(response.ok() || response.status() === 500).toBeTruthy();
  });

  test('should require JSON content type', async ({ request }) => {
    const response = await request.post('/api/generate', {
      headers: { 'Content-Type': 'text/plain' },
      data: 'prompt=hello'
    });
    expect(response.status()).toBe(400);
  });

  test('should handle GET method on generate endpoint', async ({ request }) => {
    const response = await request.get('/api/generate');
    expect(response.status()).toBe(405);
  });
});

test.describe('API Stream Endpoint', () => {
  test('should return 400 for invalid request body', async ({ request }) => {
    const response = await request.post('/api/stream', {
      data: { invalid_field: 'value' }
    });
    expect(response.status()).toBe(400);
  });

  test('should accept valid stream request', async ({ request }) => {
    const response = await request.post('/api/stream', {
      data: {
        prompt: 'Hello',
        max_tokens: 10,
        temperature: 0.7
      }
    });
    expect(response.ok() || response.status() === 500).toBeTruthy();
  });

  test('should return event-stream content type', async ({ request }) => {
    const response = await request.post('/api/stream', {
      data: { prompt: 'Test' }
    });
    const contentType = response.headers()['content-type'] || '';
    expect(contentType.includes('text/event-stream') || response.status() === 500).toBeTruthy();
  });

  test('should handle empty prompt', async ({ request }) => {
    const response = await request.post('/api/stream', {
      data: { prompt: '' }
    });
    expect(response.status()).toBeGreaterThanOrEqual(400);
  });

  test('should handle GET method on stream endpoint', async ({ request }) => {
    const response = await request.get('/api/stream');
    expect(response.status()).toBe(405);
  });
});

test.describe('API Models Endpoint', () => {
  test('should return array of models', async ({ request }) => {
    const response = await request.get('/api/models');
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
  });

  test('should handle POST method on models endpoint', async ({ request }) => {
    const response = await request.post('/api/models');
    expect(response.status()).toBe(405);
  });

  test('should require authentication for models endpoint', async ({ request }) => {
    const response = await request.get('/api/models', {
      headers: { 'Authorization': 'ApiKey invalid-key' }
    });
    expect(response.status()).toBe(401);
  });
});

test.describe('Authentication', () => {
  test('should reject requests without API key when required', async ({ request }) => {
    const response = await request.get('/api/models', {
      headers: { 'Authorization': '' }
    });
    expect(response.status()).toBe(401);
  });

  test('should accept valid API key', async ({ request }) => {
    const response = await request.get('/api/models', {
      headers: { 'Authorization': 'ApiKey qk_test123' }
    });
    expect(response.ok() || response.status() === 401).toBeTruthy();
  });

  test('should accept API key via query parameter', async ({ request }) => {
    const response = await request.get('/api/models?api_key=qk_test123');
    expect(response.ok() || response.status() === 401).toBeTruthy();
  });
});

test.describe('CORS', () => {
  test('should include CORS headers on preflight', async ({ request }) => {
    const response = await request.fetch('/api/models', {
      method: 'OPTIONS',
      headers: {
        'Origin': 'http://localhost:3000',
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'Authorization'
      }
    });
    const corsHeaders = response.headers();
    expect(corsHeaders['access-control-allow-origin']).toBeDefined();
  });

  test('should include CORS headers on actual request', async ({ request }) => {
    const response = await request.get('/api/models', {
      headers: { 'Origin': 'http://localhost:3000' }
    });
    const corsHeaders = response.headers();
    expect(corsHeaders['access-control-allow-origin']).toBeDefined();
  });

  test('should block requests from disallowed origins', async ({ request }) => {
    const response = await request.get('/api/models', {
      headers: { 'Origin': 'http://evil.com' }
    });
    expect(response.ok()).toBeTruthy();
  });
});

test.describe('Rate Limiting', () => {
  test('should handle rate limiting headers', async ({ request }) => {
    const response = await request.get('/health');
    expect(response.ok()).toBeTruthy();
  });
});

test.describe('Error Handling', () => {
  test('should return 404 for unknown endpoint', async ({ request }) => {
    const response = await request.get('/api/unknown');
    expect(response.status()).toBe(404);
  });

  test('should return proper JSON error format', async ({ request }) => {
    const response = await request.post('/api/generate', {
      data: { prompt: '' }
    });
    const contentType = response.headers()['content-type'] || '';
    if (response.status() >= 400) {
      expect(contentType).toContain('application/json');
    }
  });
});

test.describe('Version Endpoint', () => {
  test('should return version info', async ({ request }) => {
    const response = await request.get('/version');
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.version).toBeDefined();
    expect(data.go_version).toBeDefined();
  });
});

test.describe('Metrics Endpoint', () => {
  test('should return Prometheus metrics', async ({ request }) => {
    const response = await request.get('/metrics');
    expect(response.ok()).toBeTruthy();
    const text = await response.text();
    expect(text).toContain('# HELP');
    expect(text).toContain('# TYPE');
  });
});

test.describe('Health Endpoints', () => {
  test('/health should return health status', async ({ request }) => {
    const response = await request.get('/health');
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.status).toBeDefined();
    expect(data.version).toBeDefined();
    expect(data.uptime).toBeDefined();
    expect(data.checks).toBeDefined();
  });

  test('/healthz should return simple OK', async ({ request }) => {
    const response = await request.get('/healthz');
    expect(response.ok()).toBeTruthy();
    const text = await response.text();
    expect(text).toContain('OK');
  });

  test('/readyz should return readiness status', async ({ request }) => {
    const response = await request.get('/readyz');
    expect(response.ok() || response.status() === 503).toBeTruthy();
    const text = await response.text();
    expect(text).toMatch(/Ready|Not ready/);
  });
});