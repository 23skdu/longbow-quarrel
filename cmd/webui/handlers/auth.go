//go:build webui

package handlers

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

type APIKey struct {
	Key       string
	Name      string
	CreatedAt time.Time
	ExpiresAt time.Time
	RateLimit int
	Active    bool
}

type APIKeyManager struct {
	mu      sync.RWMutex
	keys    map[string]*APIKey
	byName  map[string]*APIKey
	rateLim map[string]int
}

var (
	apiKeyManager *APIKeyManager
	apiKeyOnce    sync.Once
)

func GetAPIKeyManager() *APIKeyManager {
	apiKeyOnce.Do(func() {
		apiKeyManager = &APIKeyManager{
			keys:    make(map[string]*APIKey),
			byName:  make(map[string]*APIKey),
			rateLim: make(map[string]int),
		}
	})
	return apiKeyManager
}

func GenerateAPIKey() (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return "qk_" + hex.EncodeToString(bytes), nil
}

func (m *APIKeyManager) AddKey(key, name string, ttl time.Duration, rateLimit int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.byName[name]; exists {
		return ErrAPIKeyExists
	}

	apiKey := &APIKey{
		Key:       key,
		Name:      name,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(ttl),
		RateLimit: rateLimit,
		Active:    true,
	}

	m.keys[key] = apiKey
	m.byName[name] = apiKey
	return nil
}

func (m *APIKeyManager) RevokeKey(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	key, exists := m.byName[name]
	if !exists {
		return ErrAPIKeyNotFound
	}

	delete(m.keys, key.Key)
	delete(m.byName, name)
	return nil
}

func (m *APIKeyManager) ValidateKey(key string) (*APIKey, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	apiKey, exists := m.keys[key]
	if !exists {
		return nil, ErrInvalidAPIKey
	}

	if !apiKey.Active {
		return nil, ErrInactiveAPIKey
	}

	if time.Now().After(apiKey.ExpiresAt) {
		return nil, ErrExpiredAPIKey
	}

	return apiKey, nil
}

func (m *APIKeyManager) CheckRateLimit(key string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now().Unix()
	minuteKey := key + "_" + strconv.FormatInt(now/60, 10)

	count := m.rateLim[minuteKey]
	if count >= 100 {
		return false
	}

	m.rateLim[minuteKey]++
	if len(m.rateLim) > 10000 {
		m.cleanupRateLim()
	}
	return true
}

func (m *APIKeyManager) cleanupRateLim() {
	now := time.Now().Unix()
	for k := range m.rateLim {
		minuteStr := k[len(k)-len(strconv.FormatInt(now/60, 10)):]
		minute, _ := strconv.ParseInt(minuteStr, 10, 64)
		if now/60-minute > 5 {
			delete(m.rateLim, k)
		}
	}
}

type AuthMiddleware struct {
	APIKey string
}

func NewAuthMiddleware(apiKey string) *AuthMiddleware {
	return &AuthMiddleware{APIKey: apiKey}
}

func (m *AuthMiddleware) Authenticate(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if m.APIKey == "" {
			next.ServeHTTP(w, r)
			return
		}

		apiKey := extractAPIKey(r)
		if apiKey == "" {
			http.Error(w, `{"error": "API key required"}`, http.StatusUnauthorized)
			return
		}

		if subtle.ConstantTimeCompare([]byte(apiKey), []byte(m.APIKey)) != 1 {
			http.Error(w, `{"error": "Invalid API key"}`, http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	}
}

func extractAPIKey(r *http.Request) string {
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "ApiKey ") {
		return strings.TrimPrefix(authHeader, "ApiKey ")
	}

	apiKey := r.URL.Query().Get("api_key")
	if apiKey != "" {
		return apiKey
	}

	return ""
}

var (
	ErrAPIKeyExists      = errors.New("API key already exists")
	ErrAPIKeyNotFound    = errors.New("API key not found")
	ErrInvalidAPIKey     = errors.New("invalid API key")
	ErrInactiveAPIKey    = errors.New("inactive API key")
	ErrExpiredAPIKey     = errors.New("expired API key")
	ErrRateLimitExceeded = errors.New("rate limit exceeded")
)
