# Longbow-Quarrel WebUI Tests

This directory contains Playwright tests for the Longbow-Quarrel WebUI.

## Prerequisites

- Node.js 18+ 
- npm or yarn
- Go toolchain (to build the webui server)

## Installation

```bash
cd tests
npm install
```

## Running Tests

### Run all tests
```bash
npm test
```

### Run tests in headed mode (visible browser)
```bash
npm run test:headed
```

### Run tests with debugging
```bash
npm run test:debug
```

### Show test report
```bash
npm run test:report
```

## Test Structure

### `webui.spec.ts`
Main UI tests covering:
- Page load and initial state
- Theme toggle functionality
- Model selection
- Settings controls (temperature, topK, topP, max tokens)
- Conversation controls (new chat, clear history, export)
- Quick prompts
- Prompt input behavior
- API endpoint responses
- Static asset loading
- Responsive design
- Accessibility features

### `websocket.spec.ts`
WebSocket and JavaScript component tests:
- WebSocket connection handling
- Reconnect logic
- Message sending and handling
- UIController initialization
- Message rendering and escaping
- Settings persistence
- Input behavior during generation

### `fixtures.ts`
Custom test fixtures providing:
- `webui` helper for common operations
- Automatic page navigation and cleanup

### `helpers.ts`
Utility functions for:
- Theme management
- Prompt submission
- Model selection
- Settings configuration
- Message counting
- Connection status checks

## Running Individual Tests

### Run a specific test file
```bash
npx playwright test webui.spec.ts
```

### Run tests matching a pattern
```bash
npx playwright test -g "theme"
```

### Run a specific test
```bash
npx playwright test -g "should toggle to light theme"
```

## Configuration

Edit `playwright.config.ts` to modify:
- Base URL
- Test projects (browsers)
- Reporter settings
- Web server configuration

## Test Coverage

The tests cover:
1. **UI Components**: Header, sidebar, chat area, input area
2. **Features**: Theme toggle, model selection, settings, conversation management
3. **API**: `/api/models`, `/health`, `/metrics`
4. **Responsive**: Desktop, tablet, mobile viewport sizes
5. **Accessibility**: Keyboard navigation, proper labels, heading hierarchy

## CI/CD

Tests run automatically in CI with:
- Chromium, Firefox, WebKit browsers
- Mobile Chrome and Safari
- HTML report generation
- Screenshot on failure

## Troubleshooting

### Tests fail to start
Ensure the webui server is not running on port 8080:
```bash
lsof -i :8080
kill <PID>
```

### Tests timeout
Increase timeout in `playwright.config.ts`:
```typescript
timeout: 60000,
```

### Browser not found
Install Playwright browsers:
```bash
npx playwright install
```
