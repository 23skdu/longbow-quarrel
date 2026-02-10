class UIController {
    constructor() {
        this.wsClient = null;
        this.isGenerating = false;
        this.currentAssistantMessage = null;
        this.conversation = [];
        this.settings = this.loadSettings();
        this.theme = this.loadTheme();

        this.initElements();
        this.initEventListeners();
        this.initWebSocket();
        this.applyTheme();
    }

    initElements() {
        this.chatContainer = document.getElementById('chatContainer');
        this.promptInput = document.getElementById('promptInput');
        this.sendButton = document.getElementById('sendButton');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.modelSelect = document.getElementById('modelSelect');
        this.temperatureSlider = document.getElementById('temperature');
        this.temperatureValue = document.getElementById('temperatureValue');
        this.topKInput = document.getElementById('topK');
        this.topPSlider = document.getElementById('topP');
        this.topPValue = document.getElementById('topPValue');
        this.maxTokensInput = document.getElementById('maxTokens');
        this.tokenCount = document.getElementById('tokenCount');
        this.themeToggle = document.getElementById('themeToggle');
    }

    initEventListeners() {
        this.sendButton.addEventListener('click', () => this.handleSend());
        this.promptInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.promptInput.addEventListener('input', () => this.updateTokenCount());

        this.temperatureSlider.addEventListener('input', () => {
            this.temperatureValue.textContent = this.temperatureSlider.value;
            this.saveSettings();
        });

        this.topPSlider.addEventListener('input', () => {
            this.topPValue.textContent = this.topPSlider.value;
            this.saveSettings();
        });

        document.getElementById('modelSelect').addEventListener('change', () => {
            this.saveSettings();
            this.updateModelInfo();
        });

        document.getElementById('newChatBtn').addEventListener('click', () => this.newChat());
        document.getElementById('clearHistoryBtn').addEventListener('click', () => this.clearHistory());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportConversation());
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        document.querySelectorAll('.quick-prompt').forEach(btn => {
            btn.addEventListener('click', () => {
                this.promptInput.value = btn.dataset.prompt;
                this.promptInput.focus();
            });
        });

        window.addEventListener('beforeunload', () => this.saveConversation());
    }

    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.wsClient = new WebSocketClient(wsUrl);

        this.wsClient.onStatusChange = (status) => {
            this.updateConnectionStatus(status);
        };

        this.wsClient.onMessage = (message) => {
            this.handleWebSocketMessage(message);
        };

        this.wsClient.onError = (error) => {
            console.error('WebSocket error:', error);
        };

        this.wsClient.connect();
    }

    updateConnectionStatus(status) {
        this.connectionStatus.className = 'connection-status';
        const statusText = this.connectionStatus.querySelector('.status-text');

        if (status === 'connected') {
            this.connectionStatus.classList.add('connected');
            statusText.textContent = 'Connected';
            this.loadModels();
        } else if (status === 'disconnected') {
            statusText.textContent = 'Disconnected';
        } else if (status === 'generating') {
            statusText.textContent = 'Generating...';
        } else if (status === 'loading') {
            statusText.textContent = 'Loading model...';
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const models = await response.json();

            this.modelSelect.innerHTML = '<option value="">Select a model...</option>';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path || model.name;
                option.textContent = model.name;
                if (model.loaded) {
                    option.textContent += ' (loaded)';
                }
                this.modelSelect.appendChild(option);
            });

            if (this.settings.model) {
                this.modelSelect.value = this.settings.model;
                this.updateModelInfo();
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    updateModelInfo() {
        const modelInfo = document.getElementById('modelInfo');
        const modelParams = document.getElementById('modelParams');
        const modelQuant = document.getElementById('modelQuant');

        if (this.modelSelect.value) {
            modelInfo.style.display = 'block';
            modelParams.textContent = this.settings.parameters || '--';
            modelQuant.textContent = this.settings.quantization || '--';
        } else {
            modelInfo.style.display = 'none';
        }
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'inference':
                this.handleInferenceResponse(message.payload);
                break;
            case 'status':
                this.handleStatusUpdate(message.payload);
                break;
            case 'error':
                this.handleError(message.payload);
                break;
        }
    }

    handleInferenceResponse(payload) {
        if (!this.currentAssistantMessage) {
            this.currentAssistantMessage = this.addMessage('assistant', '', payload.token);
            this.showTypingIndicator(false);

            this.conversation.push({
                role: 'assistant',
                content: ''
            });
        }

        const contentEl = this.currentAssistantMessage.querySelector('.message-content');
        contentEl.textContent += payload.token;
        contentEl.innerHTML = marked.parse(contentEl.textContent);

        if (payload.complete) {
            this.isGenerating = false;
            this.conversation[this.conversation.length - 1].content = payload.token;
            this.currentAssistantMessage = null;
            this.setInputEnabled(true);
            this.sendButton.innerHTML = `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>`;
            this.saveConversation();
        }

        this.updateMetrics(payload);
        this.scrollToBottom();
    }

    updateMetrics(payload) {
        let metaEl = this.currentAssistantMessage.querySelector('.message-meta');
        if (!metaEl) {
            metaEl = document.createElement('div');
            metaEl.className = 'message-meta';
            this.currentAssistantMessage.appendChild(metaEl);
        }
        metaEl.innerHTML = `
            <span class="message-time">${new Date().toLocaleTimeString()}</span>
            <span class="message-tps">${payload.tokens_per_sec?.toFixed(1) || 0} tok/s</span>
        `;
    }

    handleStatusUpdate(payload) {
        console.log('Status update:', payload);

        if (payload.state) {
            const statusText = this.connectionStatus.querySelector('.status-text');
            statusText.textContent = payload.state.charAt(0).toUpperCase() + payload.state.slice(1);
        }

        if (payload.models) {
            this.modelSelect.innerHTML = '<option value="">Select a model...</option>';
            payload.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path || model.name;
                option.textContent = model.name;
                if (model.loaded) {
                    option.textContent += ' (loaded)';
                }
                this.modelSelect.appendChild(option);
            });
        }
    }

    handleError(payload) {
        console.error('Error:', payload);
        this.addMessage('system', `Error: ${payload.message}`);
        this.isGenerating = false;
        this.setInputEnabled(true);
        this.sendButton.innerHTML = `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>`;
    }

    async handleSend() {
        const prompt = this.promptInput.value.trim();
        if (!prompt || this.isGenerating) {
            return;
        }

        if (!this.modelSelect.value) {
            alert('Please select a model first');
            return;
        }

        this.isGenerating = true;
        this.setInputEnabled(false);
        this.sendButton.innerHTML = `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: pulse 1s infinite;">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>`;

        this.addMessage('user', prompt);
        this.conversation.push({
            role: 'user',
            content: prompt
        });

        this.promptInput.value = '';
        this.updateTokenCount();

        this.wsClient.inference({
            prompt: prompt,
            model: this.modelSelect.value,
            temperature: parseFloat(this.temperatureSlider.value),
            topk: parseInt(this.topKInput.value),
            topp: parseFloat(this.topPSlider.value),
            max_tokens: parseInt(this.maxTokensInput.value),
            stream: true
        });
    }

    handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleSend();
        }
    }

    addMessage(role, content, streamingToken = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        let htmlContent = this.escapeHtml(content);
        if (streamingToken) {
            htmlContent += this.escapeHtml(streamingToken);
        }

        messageDiv.innerHTML = `<div class="message-content">${htmlContent}</div>`;
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv;
    }

    showTypingIndicator(show) {
        if (show) {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant typing-indicator';
            typingDiv.id = 'typingIndicator';
            typingDiv.innerHTML = '<span></span><span></span><span></span>';
            this.chatContainer.appendChild(typingDiv);
        } else {
            const typingDiv = document.getElementById('typingIndicator');
            if (typingDiv) {
                typingDiv.remove();
            }
        }
        this.scrollToBottom();
    }

    setInputEnabled(enabled) {
        this.promptInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
        this.modelSelect.disabled = !enabled;
    }

    updateTokenCount() {
        const tokens = this.promptInput.value.length;
        this.tokenCount.textContent = `${tokens} chars`;
    }

    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    newChat() {
        this.conversation = [];
        this.chatContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                        <path d="M2 17l10 5 10-5"></path>
                        <path d="M2 12l10 5 10-5"></path>
                    </svg>
                </div>
                <h2>New Conversation</h2>
                <p>Start chatting with your AI assistant.</p>
            </div>
        `;
        localStorage.removeItem('quarrelConversation');
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear all conversation history?')) {
            this.newChat();
        }
    }

    exportConversation() {
        if (this.conversation.length === 0) {
            alert('No conversation to export');
            return;
        }

        let markdown = '# Conversation Export\n\n';
        markdown += `Date: ${new Date().toLocaleString()}\n\n`;

        this.conversation.forEach(msg => {
            markdown += `## ${msg.role.toUpperCase()}\n\n`;
            markdown += `${msg.content}\n\n`;
        });

        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversation-${Date.now()}.md`;
        a.click();
        URL.revokeObjectURL(url);
    }

    loadSettings() {
        const defaults = {
            temperature: '0.7',
            topP: '0.95',
            topK: '40',
            maxTokens: '1024',
            model: ''
        };

        try {
            return { ...defaults, ...JSON.parse(localStorage.getItem('quarrelSettings') || '{}') };
        } catch {
            return defaults;
        }
    }

    saveSettings() {
        const settings = {
            temperature: this.temperatureSlider.value,
            topP: this.topPSlider.value,
            topK: this.topKInput.value,
            maxTokens: this.maxTokensInput.value,
            model: this.modelSelect.value,
            parameters: this.settings.parameters,
            quantization: this.settings.quantization
        };
        localStorage.setItem('quarrelSettings', JSON.stringify(settings));
    }

    loadConversation() {
        try {
            const saved = localStorage.getItem('quarrelConversation');
            if (saved) {
                this.conversation = JSON.parse(saved);
                this.renderConversation();
            }
        } catch (e) {
            console.error('Failed to load conversation:', e);
        }
    }

    saveConversation() {
        try {
            localStorage.setItem('quarrelConversation', JSON.stringify(this.conversation));
        } catch (e) {
            console.error('Failed to save conversation:', e);
        }
    }

    renderConversation() {
        if (this.conversation.length === 0) return;

        this.chatContainer.innerHTML = '';
        this.conversation.forEach(msg => {
            this.addMessage(msg.role, msg.content);
        });
    }

    loadTheme() {
        return localStorage.getItem('quarrelTheme') || 'dark';
    }

    saveTheme() {
        localStorage.setItem('quarrelTheme', this.theme);
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme();
        this.saveTheme();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.uiController = new UIController();
});
