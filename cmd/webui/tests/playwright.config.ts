import { defineConfig, devices } from '@playwright/test';

// The browser tests.
//
// The configuration used to point at a directory that does not exist
// ("./tests" from inside the tests directory) and at a port the server
// does not listen on (8888 instead of 8080), and there was no
// package.json at all, so `npm test` could not run. There was also no
// webServer block, so nothing started the server.
//
// The port and the host come from the environment, so the same
// configuration works locally and in CI.
const port = process.env.WEBUI_PORT ?? '8080';
const host = process.env.WEBUI_HOST ?? '127.0.0.1';
// The server runs with an API key, so the authentication is exercised
// instead of being skipped because the server is unauthenticated.
const apiKey = process.env.WEBUI_API_KEY ?? 'qk_playwright_test';
const baseURL = process.env.WEBUI_BASE_URL ?? `http://${host === '0.0.0.0' ? '127.0.0.1' : host}:${port}`;

export default defineConfig({
  // The specs live next to this file.
  testDir: '.',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list']
  ],
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    // The server is the module in the parent directory.
    command: 'go run -tags webui .',
    cwd: '..',
    url: `${baseURL}/healthz`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      WEBUI_PORT: port,
      WEBUI_HOST: host,
      WEBUI_API_KEY: apiKey,
    },
  },
});
