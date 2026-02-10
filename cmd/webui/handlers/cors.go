//go:build webui

package handlers

import (
	"net/http"
	"strconv"
	"strings"
)

type CORSMiddleware struct {
	AllowedOrigins []string
	AllowedMethods []string
	AllowedHeaders []string
	MaxAge         int
}

func NewCORSMiddleware(allowedOrigins []string) *CORSMiddleware {
	if len(allowedOrigins) == 0 {
		allowedOrigins = []string{"*"}
	}

	return &CORSMiddleware{
		AllowedOrigins: allowedOrigins,
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"Content-Type", "Authorization", "X-Requested-With", "Accept"},
		MaxAge:         86400,
	}
}

func (m *CORSMiddleware) Middleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		if m.isOriginAllowed(origin) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		} else if len(m.AllowedOrigins) == 1 && m.AllowedOrigins[0] == "*" {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}

		w.Header().Set("Access-Control-Allow-Methods", strings.Join(m.AllowedMethods, ", "))
		w.Header().Set("Access-Control-Allow-Headers", strings.Join(m.AllowedHeaders, ", "))
		w.Header().Set("Access-Control-Max-Age", strconv.Itoa(m.MaxAge))
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	}
}

func (m *CORSMiddleware) isOriginAllowed(origin string) bool {
	if len(m.AllowedOrigins) == 0 {
		return false
	}

	for _, allowed := range m.AllowedOrigins {
		if allowed == "*" {
			return true
		}
		if strings.EqualFold(allowed, origin) {
			return true
		}
	}

	return false
}
