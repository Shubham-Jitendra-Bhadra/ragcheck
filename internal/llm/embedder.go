package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
)

const voyageAPIURL = "https://api.voyageai.com/v1/embeddings"

// AnthropicEmbedder calls the Voyage AI embeddings API to convert
// text into vectors. Voyage AI is Anthropic's recommended embedding provider.
// Implements the embedding.Embedder interface.
type AnthropicEmbedder struct {
	apiKey string
	model  string
	cache  map[string][]float64
	mu     sync.RWMutex
	http   *http.Client
}

// NewAnthropicEmbedder creates a new embedder with an empty cache.
// apiKey should be your Voyage AI API key from voyageai.com
// model should be "voyage-2" or "voyage-large-2"
func NewAnthropicEmbedder(apiKey, model string) *AnthropicEmbedder {
	return &AnthropicEmbedder{
		apiKey: apiKey,
		model:  model,
		cache:  make(map[string][]float64),
		http:   &http.Client{},
	}
}

type voyageRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type voyageResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Embed converts text to a vector using the Voyage AI embeddings API.
// Returns cached result if this text has been embedded before.
func (e *AnthropicEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	// check cache first
	e.mu.RLock()
	if vec, ok := e.cache[text]; ok {
		e.mu.RUnlock()
		return vec, nil
	}
	e.mu.RUnlock()

	// build request
	body, err := json.Marshal(voyageRequest{
		Input: []string{text},
		Model: e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, voyageAPIURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed api call: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embed response: %w", err)
	}

	var result voyageResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	if result.Error != nil {
		return nil, fmt.Errorf("voyage api error: %s", result.Error.Message)
	}

	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding response for text: %.50s", text)
	}

	vec := result.Data[0].Embedding

	// store in cache
	e.mu.Lock()
	e.cache[text] = vec
	e.mu.Unlock()

	return vec, nil
}

// CacheSize returns how many texts are currently cached.
func (e *AnthropicEmbedder) CacheSize() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.cache)
}