import { test, expect } from '@playwright/test';

// The key the server under test was started with.
const apiKey = process.env.WEBUI_API_KEY ?? 'qk_playwright_test';

// The server runs with an API key, so every request carries it.
test.use({ extraHTTPHeaders: { Authorization: `ApiKey ${apiKey}` } });

test.describe('Page Load', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the main page successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Longbow-Quarrel/);
  });

  test('should display the header with title and version', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Longbow-Quarrel');
    await expect(page.locator('.version')).toBeVisible();
  });

  test('should show the connection status', async ({ page }) => {
    // The page connects to the WebSocket as soon as it is loaded, so the
    // status is "Connected" once the socket is open.
    const statusText = page.locator('.status-text');
    await expect(statusText).toContainText('Connected');
  });

  test('should display welcome message', async ({ page }) => {
    await expect(page.locator('.welcome-message h2')).toContainText('Welcome to Longbow-Quarrel');
    await expect(page.locator('.welcome-message p')).toContainText('Select a model');
  });

  test('should display feature list', async ({ page }) => {
    await expect(page.locator('.feature')).toHaveCount(4);
    await expect(page.locator('.feature').first()).toContainText('Streaming token generation');
  });

  test('should have all sidebar sections', async ({ page }) => {
    await expect(page.locator('.sidebar-section h2')).toHaveText([
      'Model',
      'Gallery',
      'Adapters',
      'Settings',
      'Conversation',
      'Quick Prompts'
    ]);
  });

  test('should have prompt input area', async ({ page }) => {
    await expect(page.locator('#promptInput')).toBeVisible();
    await expect(page.locator('#sendButton')).toBeVisible();
    await expect(page.locator('#tokenCount')).toContainText('0 tokens');
  });
});

test.describe('Theme Toggle', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
  });

  test('should start with dark theme by default', async ({ page }) => {
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  });

  test('should toggle to light theme when clicked', async ({ page }) => {
    await page.locator('#themeToggle').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  });

  test('should persist theme in localStorage', async ({ page }) => {
    await page.locator('#themeToggle').click();
    const savedTheme = await page.evaluate(() => localStorage.getItem('quarrelTheme'));
    expect(savedTheme).toBe('light');
  });

  test('should toggle back to dark theme', async ({ page }) => {
    await page.locator('#themeToggle').click();
    await page.locator('#themeToggle').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  });
});

test.describe('Model Selection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should have model select dropdown', async ({ page }) => {
    await expect(page.locator('#modelSelect')).toBeVisible();
  });

  test('should have placeholder option', async ({ page }) => {
    await expect(page.locator('#modelSelect option').first()).toContainText('Select a model...');
  });

  test('should show model info when model is selected', async ({ page }) => {
    await expect(page.locator('#modelInfo')).toBeHidden();
  });

  test('should disable send button when no model is selected', async ({ page }) => {
    await expect(page.locator('#sendButton')).toBeDisabled();
  });
});

test.describe('Settings Controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
  });

  test('should have temperature slider', async ({ page }) => {
    const slider = page.locator('#temperature');
    await expect(slider).toBeVisible();
    await expect(slider).toHaveAttribute('min', '0');
    await expect(slider).toHaveAttribute('max', '2');
    await expect(slider).toHaveAttribute('step', '0.1');
    await expect(slider).toHaveValue('0.7');
  });

  test('should have topK input', async ({ page }) => {
    const input = page.locator('#topK');
    await expect(input).toBeVisible();
    await expect(input).toHaveAttribute('type', 'number');
    await expect(input).toHaveValue('40');
  });

  test('should have topP slider', async ({ page }) => {
    const slider = page.locator('#topP');
    await expect(slider).toBeVisible();
    await expect(slider).toHaveAttribute('min', '0');
    await expect(slider).toHaveAttribute('max', '1');
    await expect(slider).toHaveAttribute('step', '0.05');
    await expect(slider).toHaveValue('0.95');
  });

  test('should have max tokens input', async ({ page }) => {
    const input = page.locator('#maxTokens');
    await expect(input).toBeVisible();
    await expect(input).toHaveAttribute('type', 'number');
    await expect(input).toHaveValue('1024');
  });

  test('should update temperature value display when slider changes', async ({ page }) => {
    await page.locator('#temperature').fill('1.5');
    await expect(page.locator('#temperatureValue')).toContainText('1.5');
  });

  test('should update topP value display when slider changes', async ({ page }) => {
    await page.locator('#topP').fill('0.8');
    await expect(page.locator('#topPValue')).toContainText('0.8');
  });

  test('should save settings to localStorage', async ({ page }) => {
    await page.locator('#temperature').fill('1.5');
    await page.locator('#topK').fill('50');
    await page.evaluate(() => {
      localStorage.getItem('quarrelSettings');
    });
    const settings = await page.evaluate(() => JSON.parse(localStorage.getItem('quarrelSettings') || '{}'));
    expect(settings.temperature).toBe('1.5');
    expect(settings.topK).toBe('50');
  });
});

test.describe('Conversation Controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should have new chat button', async ({ page }) => {
    await expect(page.locator('#newChatBtn')).toBeVisible();
  });

  test('should have clear history button', async ({ page }) => {
    await expect(page.locator('#clearHistoryBtn')).toBeVisible();
  });

  test('should have export button', async ({ page }) => {
    await expect(page.locator('#exportBtn')).toBeVisible();
  });

  test('should show alert when exporting empty conversation', async ({ page }) => {
    // The alert blocks the page, so the click and the dialog have to be
    // awaited together.
    // The dialog has to be handled before the click: the alert blocks
    // the page, so the click only completes once the dialog is gone.
    let message = '';
    page.once('dialog', (dialog) => {
      message = dialog.message();
      void dialog.accept();
    });
    await page.locator('#exportBtn').click();
    expect(message).toContain('No conversation');
  });

  test('should clear localStorage on new chat', async ({ page }) => {
    await page.evaluate(() => localStorage.setItem('quarrelConversation', 'test'));
    await page.locator('#newChatBtn').click();
    const conversation = await page.evaluate(() => localStorage.getItem('quarrelConversation'));
    expect(conversation).toBeNull();
  });
});

test.describe('Quick Prompts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display quick prompt buttons', async ({ page }) => {
    await expect(page.locator('.quick-prompt')).toHaveCount(4);
  });

  test('should have Summarize button', async ({ page }) => {
    await expect(page.locator('.quick-prompt').filter({ hasText: 'Summarize' })).toBeVisible();
  });

  test('should have Explain button', async ({ page }) => {
    await expect(page.locator('.quick-prompt').filter({ hasText: 'Explain' })).toBeVisible();
  });

  test('should have Translate button', async ({ page }) => {
    await expect(page.locator('.quick-prompt').filter({ hasText: 'Translate' })).toBeVisible();
  });

  test('should have Code button', async ({ page }) => {
    await expect(page.locator('.quick-prompt').filter({ hasText: 'Code' })).toBeVisible();
  });

  test('should populate prompt input when quick prompt is clicked', async ({ page }) => {
    await page.locator('.quick-prompt').filter({ hasText: 'Summarize' }).click();
    await expect(page.locator('#promptInput')).toHaveValue('Summarize this text');
  });

  test('should focus prompt input when quick prompt is clicked', async ({ page }) => {
    await page.locator('.quick-prompt').filter({ hasText: 'Explain' }).click();
    await expect(page.locator('#promptInput')).toBeFocused();
  });
});

test.describe('Prompt Input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should accept text input', async ({ page }) => {
    await page.locator('#promptInput').fill('Hello, AI!');
    await expect(page.locator('#promptInput')).toHaveValue('Hello, AI!');
  });

  test('should update token count on input', async ({ page }) => {
    await page.locator('#promptInput').fill('Test input');
    await expect(page.locator('#tokenCount')).toContainText('10 chars');
  });

  test('should send on Enter key', async ({ page }) => {
    // The Enter key sends the message. No model is loaded, so the send is
    // refused with an alert, and the alert blocks the page: it has to be
    // handled before the press is asserted.
    let message = '';
    page.once('dialog', (dialog) => {
      message = dialog.message();
      void dialog.accept();
    });
    await page.locator('#promptInput').fill('Test');
    await page.locator('#promptInput').press('Enter');
    expect(message).toContain('model');
  });

  test('should not send on Shift+Enter', async ({ page }) => {
    await page.locator('#promptInput').fill('Test\nmultiline');
    await page.locator('#promptInput').press('Enter+Shift');
    await expect(page.locator('#promptInput')).toHaveValue('Test\nmultiline');
  });
});

test.describe('API Endpoints', () => {
  test('should respond to /api/models', async ({ request }) => {
    const response = await request.get('/api/models');
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
  });

  test('should respond to /health', async ({ request }) => {
    const response = await request.get('/health');
    expect(response.ok()).toBeTruthy();
  });

  test('should respond to /metrics', async ({ request }) => {
    const response = await request.get('/metrics');
    expect(response.ok()).toBeTruthy();
    expect(await response.text()).toContain('quarrel_webui_requests_total');
  });
});

test.describe('Static Assets', () => {
  test('should load main CSS', async ({ page }) => {
    const response = await page.goto('/static/css/main.css');
    expect(response?.status()).toBe(200);
  });

  test('should load main JavaScript', async ({ page }) => {
    const response = await page.goto('/static/js/ui.js');
    expect(response?.status()).toBe(200);
  });

  test('should load websocket JavaScript', async ({ page }) => {
    const response = await page.goto('/static/js/websocket.js');
    expect(response?.status()).toBe(200);
  });

  test('should load marked.js from CDN', async ({ page }) => {
    await page.goto('/');
    // A script element has no box of its own, so it is never "visible":
    // the attribute is what says that it was requested.
    const src = await page.locator('script[src*="marked"]').first().getAttribute('src');
    expect(src).toContain('marked');
  });
});

test.describe('Responsive Design', () => {
  test('should display correctly on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('#promptInput')).toBeVisible();
  });

  test('should display correctly on tablet', async ({ page }) => {
    // The layout hides the sidebar below 769 pixels, which is the phone
    // layout; a tablet in landscape keeps it.
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.sidebar')).toBeVisible();
    await expect(page.locator('#promptInput')).toBeVisible();
  });

  test('should display correctly on desktop', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.sidebar')).toBeVisible();
    await expect(page.locator('#promptInput')).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');
    const h1 = page.locator('h1').first();
    const h2s = page.locator('h2');
    await expect(h1).toHaveCount(1);
    await expect(h2s).toHaveCount(7);
  });

  test('should have buttons with accessible labels', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#themeToggle')).toHaveAttribute('title', 'Toggle Theme');
  });

  test('should have input fields with proper labels', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#promptInput')).toHaveAttribute('placeholder', 'Enter your prompt...');
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');
    await page.keyboard.press('Tab');
    await expect(page.locator('#themeToggle')).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(page.locator('#modelSelect')).toBeFocused();
  });
});

test.describe('Model Gallery and Adapters', () => {
  test('should show the gallery section', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#modelGallery')).toBeVisible();
    await expect(page.locator('#modelGallery .gallery-empty')).toContainText('No models loaded');
  });

  test('should show the adapter section', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#adapterList')).toBeVisible();
    await expect(page.locator('#adapterList .gallery-empty')).toContainText('No adapters loaded');
  });

  test('should draw a card for every loaded model', async ({ page }) => {
    await page.goto('/');
    // The page loads the models when the socket connects, and that
    // redraws the gallery, so the connection has to be settled first.
    await expect(page.locator('.status-text')).toContainText('Connected');
    // The gallery is drawn from the model list, so a card appears as
    // soon as there is a model to show.
    await page.evaluate(() => {
      window.uiController.renderGallery([
        { name: 'test-model', path: '/models/test.gguf', loaded: true, parameters: '1B', quantization: 'Q4_K' },
      ]);
    });
    const card = page.locator('#modelGallery .gallery-card');
    await expect(card).toHaveCount(1);
    await expect(card.first()).toContainText('test-model');
    await expect(card.first()).toContainText('Q4_K');
  });

  test('should list the adapters of the loaded models', async ({ page }) => {
    await page.goto('/');
    // The page loads the adapters when the socket connects, and that
    // redraws the list, so the connection has to be settled first.
    await expect(page.locator('.status-text')).toContainText('Connected');
    await page.evaluate(() => {
      window.uiController.renderAdapters([
        { id: 'adapter-a', path: '/adapters/a.gguf', num_weights: 128 },
      ]);
    });
    const item = page.locator('#adapterList .adapter-item');
    await expect(item).toHaveCount(1);
    await expect(item.first()).toContainText('adapter-a');
    await expect(item.first()).toContainText('128 weights');
  });
});
