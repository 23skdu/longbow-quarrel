import { Page, Locator, expect } from '@playwright/test';

export class WebUIHelper {
  constructor(private page: Page) {}

  async waitForPageLoad(): Promise<void> {
    await this.page.waitForLoadState('domcontentloaded');
    await this.page.waitForSelector('#app', { state: 'visible' });
  }

  async toggleTheme(): Promise<void> {
    await this.page.locator('#themeToggle').click();
  }

  async getCurrentTheme(): Promise<string> {
    return this.page.locator('html').getAttribute('data-theme') || 'dark';
  }

  async fillPrompt(text: string): Promise<void> {
    await this.page.locator('#promptInput').fill(text);
  }

  async sendPrompt(): Promise<void> {
    await this.page.locator('#sendButton').click();
  }

  async selectModel(modelName: string): Promise<void> {
    await this.page.locator('#modelSelect').selectOption(modelName);
  }

  async setTemperature(value: string): Promise<void> {
    await this.page.locator('#temperature').fill(value);
  }

  async setTopP(value: string): Promise<void> {
    await this.page.locator('#topP').fill(value);
  }

  async setTopK(value: string): Promise<void> {
    await this.page.locator('#topK').fill(value);
  }

  async setMaxTokens(value: string): Promise<void> {
    await this.page.locator('#maxTokens').fill(value);
  }

  async clickQuickPrompt(promptText: string): Promise<void> {
    await this.page.locator('.quick-prompt', { hasText: promptText }).click();
  }

  async newChat(): Promise<void> {
    await this.page.locator('#newChatBtn').click();
  }

  async clearHistory(): Promise<void> {
    const dialogPromise = this.page.waitForEvent('dialog');
    await this.page.locator('#clearHistoryBtn').click();
    await (await dialogPromise).accept();
  }

  async exportConversation(): Promise<void> {
    const dialogPromise = this.page.waitForEvent('dialog');
    await this.page.locator('#exportBtn').click();
    const dialog = await dialogPromise;
    if (dialog.message().includes('No conversation')) {
      await dialog.accept();
    }
  }

  async getMessageCount(): Promise<number> {
    return this.page.locator('.message').count();
  }

  async getLastMessage(): Promise<Locator> {
    return this.page.locator('.message').last();
  }

  async getConnectionStatus(): Promise<string> {
    return this.page.locator('.status-text').textContent() || '';
  }

  async waitForConnectionStatus(status: string): Promise<void> {
    await this.page.locator('.status-text', { hasText: status }).waitFor({ state: 'visible' });
  }

  async getWelcomeMessage(): Promise<Locator> {
    return this.page.locator('.welcome-message');
  }

  async isWelcomeMessageVisible(): Promise<boolean> {
    return (await this.getWelcomeMessage().isVisible());
  }

  async getTokenCount(): Promise<string> {
    return this.page.locator('#tokenCount').textContent() || '';
  }

  async pressEnter(): Promise<void> {
    await this.page.locator('#promptInput').press('Enter');
  }
}

export function createWebUIHelper(page: Page): WebUIHelper {
  return new WebUIHelper(page);
}
