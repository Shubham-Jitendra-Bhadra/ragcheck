package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Client wraps the Anthropic SDK into a clean two-method interface.
// Create once at startup, pass into any scorer that needs an LLM judge.
type Client struct {
	inner      *anthropic.Client
	model      string
	maxRetries int
}

// New creates a new Anthropic client.
func New(apiKey, model string) *Client {
	return &Client{
		inner:      anthropic.NewClient(option.WithAPIKey(apiKey)),
		model:      model,
		maxRetries: 3,
	}
}

// Complete sends a system + user prompt to the Anthropic API and returns
// the text response. Retries up to 3 times with backoff on transient errors.
func (c *Client) Complete(ctx context.Context, system, user string) (string, error) {
	var lastErr error

	for attempt := 0; attempt < c.maxRetries; attempt++ {
		if attempt > 0 {
			wait := time.Duration(attempt) * time.Second
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}

		result, err := c.call(ctx, system, user)
		if err == nil {
			return result, nil
		}

		lastErr = err
	}

	return "", fmt.Errorf("all %d attempts failed: %w", c.maxRetries, lastErr)
}

// call makes a single API request with a 30 second timeout.
func (c *Client) call(ctx context.Context, system, user string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	msg, err := c.inner.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.F(c.model),
		MaxTokens: anthropic.F(int64(256)),
		System: anthropic.F([]anthropic.TextBlockParam{
			anthropic.NewTextBlock(system),
		}),
		Messages: anthropic.F([]anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(user)),
		}),
	})
	if err != nil {
		return "", fmt.Errorf("anthropic api call: %w", err)
	}

	if len(msg.Content) == 0 {
		return "", fmt.Errorf("empty response from model %s", c.model)
	}

	return msg.Content[0].Text, nil
}
