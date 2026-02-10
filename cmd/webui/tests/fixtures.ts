import { test as base } from '@playwright/test';
import { WebUIHelper, createWebUIHelper } from './helpers';

interface WebUIFixtures {
  webui: WebUIHelper;
}

export const test = base.extend<WebUIFixtures>({
  webui: async ({ page }, use) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    const helper = createWebUIHelper(page);
    await use(helper);
  },
});

export { expect } from '@playwright/test';
