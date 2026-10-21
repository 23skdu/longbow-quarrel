//go:build webui

package handlers

import (
	"log"
	"net/http"

	"github.com/23skdu/longbow-quarrel/cmd/webui/templates"
)

func IndexHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if err := templates.RenderIndex(w); err != nil {
			// The page cannot be rendered. Saying so is better than
			// panicking in a request handler, which takes the server
			// down.
			log.Printf("ERROR: failed to render the index: %v", err)
			http.Error(w, "The page could not be rendered", http.StatusInternalServerError)
		}
	}
}
