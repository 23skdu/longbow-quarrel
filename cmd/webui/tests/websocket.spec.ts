import { test, expect } from '@playwright/test';

test.describe('WebSocket Connection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should have WebSocketClient class defined', async ({ page }) => {
    const hasClass = await page.evaluate(() => typeof WebSocketClient === 'function');
    expect(hasClass).toBe(true);
  });

  test('should attempt to connect to WebSocket', async ({ page }) => {
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'log') {
        consoleMessages.push(msg.text());
      }
    });
    await page.reload();
    const connected = consoleMessages.some(msg => msg.includes('Connecting to'));
    expect(connected).toBe(true);
  });

  test('should have reconnect logic', async ({ page }) => {
    const hasReconnect = await page.evaluate(() => {
      const ws = new WebSocketClient('ws://localhost:8080/ws');
      return typeof ws.attemptReconnect === 'function';
    });
    expect(hasReconnect).toBe(true);
  });

  test('should have max reconnect attempts', async ({ page }) => {
    const maxAttempts = await page.evaluate(() => {
      const ws = new WebSocketClient('ws://localhost:8080/ws');
      return ws.maxReconnectAttempts;
    });
    expect(maxAttempts).toBe(5);
  });

  test('should have send method', async ({ page }) => {
    const hasSend = await page.evaluate(() => {
      const ws = new WebSocketClient('ws://localhost:8080/ws');
      return typeof ws.send === 'function';
    });
    expect(hasSend).toBe(true);
  });

  test('should have inference method', async ({ page }) => {
    const hasInference = await page.evaluate(() => {
      const ws = new WebSocketClient('ws://localhost:8080/ws');
      return typeof ws.inference === 'function';
    });
    expect(hasInference).toBe(true);
  });

  test('should have stop method', async ({ page }) => {
    const hasStop = await page.evaluate(() => {
      const ws = new WebSocketClient('ws://localhost:8080/ws');
      return typeof ws.stop === 'function';
    });
    expect(hasStop).toBe(true);
  });

  test('should have disconnect method', async ({ page }) => {
    const hasDisconnect = await page.evaluate(() => {
      const ws = new WebSocketClient('ws://localhost:8080/ws');
      return typeof ws.disconnect === 'function';
    });
    expect(hasDisconnect).toBe(true);
  });

  test('should send inference payload with correct structure', async ({ page }) => {
    const payloadSent: any[] = [];
    await page.evaluate(() => {
      const ws = new WebSocketClient('ws://localhost:8080/ws');
      const originalSend = ws.send.bind(ws);
      ws.send = (type: string, payload: any) => {
        payloadSent.push({ type, payload });
      };
      ws.inference({ prompt: 'test', temperature: 0.7 });
    });
    expect(payloadSent.length).toBe(1);
    expect(payloadSent[0].type).toBe('inference');
    expect(payloadSent[0].payload.prompt).toBe('test');
  });
});

test.describe('UIController', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should have UIController class defined', async ({ page }) => {
    const hasClass = await page.evaluate(() => typeof UIController === 'function');
    expect(hasClass).toBe(true);
  });

  test('should initialize on DOMContentLoaded', async ({ page }) => {
    const initialized = await page.evaluate(() => typeof window.uiController !== 'undefined');
    expect(initialized).toBe(true);
  });

  test('should have all required elements initialized', async ({ page }) => {
    const elements = await page.evaluate(() => {
      const ui = window.uiController;
      return {
        chatContainer: !!ui.chatContainer,
        promptInput: !!ui.promptInput,
        sendButton: !!ui.sendButton,
        connectionStatus: !!ui.connectionStatus,
        modelSelect: !!ui.modelSelect,
        temperatureSlider: !!ui.temperatureSlider,
        topPSlider: !!ui.topPSlider,
        maxTokensInput: !!ui.maxTokensInput,
        themeToggle: !!ui.themeToggle,
      };
    });
    expect(Object.values(elements).every(v => v)).toBe(true);
  });

  test('should load settings from localStorage', async ({ page }) => {
    await page.evaluate(() => {
      localStorage.setItem('quarrelSettings', JSON.stringify({
        temperature: '1.0',
        topK: '50',
        topP: '0.9',
        maxTokens: '512'
      }));
    });
    await page.reload();
    const settings = await page.evaluate(() => window.uiController.settings);
    expect(settings.temperature).toBe('1.0');
    expect(settings.topK).toBe('50');
  });

  test('should load conversation from localStorage', async ({ page }) => {
    await page.evaluate(() => {
      localStorage.setItem('quarrelConversation', JSON.stringify([
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there!' }
      ]));
    });
    await page.reload();
    const conversation = await page.evaluate(() => window.uiController.conversation);
    expect(conversation.length).toBe(2);
  });

  test('should have conversation management methods', async ({ page }) => {
    const methods = await page.evaluate(() => {
      const ui = window.uiController;
      return {
        newChat: typeof ui.newChat === 'function',
        clearHistory: typeof ui.clearHistory === 'function',
        exportConversation: typeof ui.exportConversation === 'function',
        saveConversation: typeof ui.saveConversation === 'function',
        loadConversation: typeof ui.loadConversation === 'function',
      };
    });
    expect(Object.values(methods).every(v => v)).toBe(true);
  });

  test('should have theme management methods', async ({ page }) => {
    const methods = await page.evaluate(() => {
      const ui = window.uiController;
      return {
        loadTheme: typeof ui.loadTheme === 'function',
        saveTheme: typeof ui.saveTheme === 'function',
        applyTheme: typeof ui.applyTheme === 'function',
        toggleTheme: typeof ui.toggleTheme === 'function',
      };
    });
    expect(Object.values(methods).every(v => v)).toBe(true);
  });

  test('should have message handling methods', async ({ page }) => {
    const methods = await page.evaluate(() => {
      const ui = window.uiController;
      return {
        addMessage: typeof ui.addMessage === 'function',
        handleSend: typeof ui.handleSend === 'function',
        handleKeydown: typeof ui.handleKeydown === 'function',
        handleWebSocketMessage: typeof ui.handleWebSocketMessage === 'function',
        handleInferenceResponse: typeof ui.handleInferenceResponse === 'function',
      };
    });
    expect(Object.values(methods).every(v => v)).toBe(true);
  });

  test('should show alert when sending without model', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.modelSelect.value = '';
    });
    const alertPromise = page.waitForEvent('dialog');
    page.evaluate(() => {
      window.uiController.handleSend();
    });
    const dialog = await alertPromise;
    expect(dialog.message()).toContain('select a model');
    await dialog.accept();
  });
});

test.describe('Message Rendering', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should add user message to chat', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.addMessage('user', 'Hello, world!');
    });
    const messages = page.locator('.message.user');
    await expect(messages).toHaveCount(1);
    await expect(messages.first()).toContainText('Hello, world!');
  });

  test('should add assistant message to chat', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.addMessage('assistant', 'Hello! How can I help?');
    });
    const messages = page.locator('.message.assistant');
    await expect(messages).toHaveCount(1);
    await expect(messages.first()).toContainText('Hello! How can I help?');
  });

  test('should add system message to chat', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.addMessage('system', 'Connection established');
    });
    const messages = page.locator('.message.system');
    await expect(messages).toHaveCount(1);
    await expect(messages.first()).toContainText('Connection established');
  });

  test('should escape HTML in messages', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.addMessage('user', '<script>alert("xss")</script>');
    });
    const messageContent = page.locator('.message.user .message-content').first();
    const html = await messageContent.innerHTML();
    expect(html).not.toContain('<script>');
  });

  test('should show typing indicator', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.showTypingIndicator(true);
    });
    await expect(page.locator('.typing-indicator')).toBeVisible();

    await page.evaluate(() => {
      window.uiController.showTypingIndicator(false);
    });
    await expect(page.locator('.typing-indicator')).toBeHidden();
  });
});

test.describe('Settings Persistence', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should save settings when temperature changes', async ({ page }) => {
    await page.locator('#temperature').fill('1.2');
    const settings = await page.evaluate(() => 
      JSON.parse(localStorage.getItem('quarrelSettings') || '{}')
    );
    expect(settings.temperature).toBe('1.2');
  });

  test('should save settings when topP changes', async ({ page }) => {
    await page.locator('#topP').fill('0.85');
    const settings = await page.evaluate(() => 
      JSON.parse(localStorage.getItem('quarrelSettings') || '{}')
    );
    expect(settings.topP).toBe('0.85');
  });

  test('should save model selection', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.modelSelect.value = 'test-model';
      window.uiController.saveSettings();
    });
    const settings = await page.evaluate(() => 
      JSON.parse(localStorage.getItem('quarrelSettings') || '{}')
    );
    expect(settings.model).toBe('test-model');
  });
});

test.describe('Input Behavior', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should disable input during generation', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.isGenerating = true;
      window.uiController.setInputEnabled(false);
    });
    await expect(page.locator('#promptInput')).toBeDisabled();
    await expect(page.locator('#sendButton')).toBeDisabled();
    await expect(page.locator('#modelSelect')).toBeDisabled();
  });

  test('should enable input when not generating', async ({ page }) => {
    await page.evaluate(() => {
      window.uiController.setInputEnabled(true);
    });
    await expect(page.locator('#promptInput')).toBeEnabled();
    await expect(page.locator('#sendButton')).toBeEnabled();
    await expect(page.locator('#modelSelect')).toBeEnabled();
  });

  test('should update token count dynamically', async ({ page }) => {
    await page.locator('#promptInput').fill('Count me!');
    await expect(page.locator('#tokenCount')).toContainText('10 chars');
  });
});
