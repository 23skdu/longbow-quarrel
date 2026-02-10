package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/cmd/webui/config"
	"github.com/23skdu/longbow-quarrel/cmd/webui/engine"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type WSMessage struct {
	Type    string      `json:"type"`
	Payload interface{} `json:"payload"`
}

type InferenceRequest struct {
	Prompt      string  `json:"prompt"`
	Model       string  `json:"model,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	TopK        int     `json:"topk,omitempty"`
	TopP        float64 `json:"topp,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Stream      bool    `json:"stream"`
}

type InferenceResponse struct {
	Token        string  `json:"token"`
	TokenID      int     `json:"token_id"`
	Complete     bool    `json:"complete"`
	TokensPerSec float64 `json:"tokens_per_sec,omitempty"`
}

type Connection struct {
	conn     *websocket.Conn
	cfg      config.Config
	send     chan []byte
	mu       sync.Mutex
	stopChan chan struct{}
	adapter  *engine.EngineAdapter
}

func WebSocketHandler(cfg config.Config) http.HandlerFunc {
	adapter := engine.GetAdapter()

	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("WebSocket upgrade error: %v", err)
			return
		}

		client := &Connection{
			conn:     conn,
			cfg:      cfg,
			send:     make(chan []byte, 256),
			stopChan: make(chan struct{}),
			adapter:  adapter,
		}

		go client.writePump()
		go client.readPump()
	}
}

func (c *Connection) readPump() {
	defer func() {
		c.conn.Close()
		close(c.send)
	}()

	c.conn.SetReadLimit(512 * 1024)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		var wsMsg WSMessage
		if err := json.Unmarshal(message, &wsMsg); err != nil {
			log.Printf("Invalid JSON: %v", err)
			c.sendError("INVALID_REQUEST", "Invalid JSON format")
			continue
		}

		c.handleMessage(wsMsg)
	}
}

func (c *Connection) writePump() {
	ticker := time.NewTicker(30 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			c.mu.Lock()
			err := c.conn.WriteMessage(websocket.TextMessage, message)
			c.mu.Unlock()

			if err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}

		case <-c.stopChan:
			return
		}
	}
}

func (c *Connection) handleMessage(msg WSMessage) {
	switch msg.Type {
	case "inference":
		c.handleInference(msg)
	case "stop":
		close(c.stopChan)
	case "status":
		c.sendStatus()
	default:
		c.sendError("UNKNOWN_TYPE", "Unknown message type: "+msg.Type)
	}
}

func (c *Connection) handleInference(msg WSMessage) {
	var req InferenceRequest
	data, _ := json.Marshal(msg.Payload)
	if err := json.Unmarshal(data, &req); err != nil {
		c.sendError("INVALID_REQUEST", "Invalid inference request")
		return
	}

	if req.MaxTokens <= 0 {
		req.MaxTokens = 100
	}
	if req.Temperature <= 0 {
		req.Temperature = 0.7
	}
	if req.TopK <= 0 {
		req.TopK = 40
	}
	if req.TopP <= 0 {
		req.TopP = 0.95
	}

	log.Printf("Inference request: prompt=%s, model=%s, max_tokens=%d", req.Prompt, req.Model, req.MaxTokens)

	c.sendStatusUpdate("loading")

	adapterReq := &engine.InferenceRequest{
		Prompt:      req.Prompt,
		Model:       req.Model,
		Temperature: req.Temperature,
		TopK:        req.TopK,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
	}

	responseChanChan, err := c.adapter.Infer(nil, adapterReq)
	if err != nil {
		c.sendError("INFERENCE_ERROR", "Failed to queue inference")
		return
	}

	c.sendStatusUpdate("generating")

	if responseChanChan == nil {
		c.sendError("QUEUE_FULL", "Request queue is full")
		return
	}

	responseChan := <-responseChanChan
	startTime := time.Now()
	tokensGenerated := 0

	for resp := range responseChan {
		tokensGenerated++

		tokensPerSec := float64(tokensGenerated) / time.Since(startTime).Seconds()

		wsResp := WSMessage{
			Type: "inference",
			Payload: InferenceResponse{
				Token:        resp.Token,
				TokenID:      resp.TokenID,
				Complete:     resp.Complete,
				TokensPerSec: tokensPerSec,
			},
		}

		data, _ := json.Marshal(wsResp)
		c.send <- data

		if resp.Complete {
			break
		}
	}
}

func (c *Connection) sendStatus() {
	models := c.adapter.ListModels()
	status := map[string]interface{}{
		"connected": true,
		"models":    models,
	}

	data, _ := json.Marshal(WSMessage{
		Type:    "status",
		Payload: status,
	})
	c.send <- data
}

func (c *Connection) sendStatusUpdate(state string) {
	status := map[string]interface{}{
		"state": state,
	}

	data, _ := json.Marshal(WSMessage{
		Type:    "status",
		Payload: status,
	})
	c.send <- data
}

func (c *Connection) sendError(code, message string) {
	err := map[string]interface{}{
		"code":    code,
		"message": message,
	}

	data, _ := json.Marshal(WSMessage{
		Type:    "error",
		Payload: err,
	})
	c.send <- data
}
