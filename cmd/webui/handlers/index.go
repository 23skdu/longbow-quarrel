package handlers

import (
	"net/http"

	"github.com/23skdu/longbow-quarrel/cmd/webui/templates"
)

func IndexHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		templates.RenderIndex(w)
	}
}
