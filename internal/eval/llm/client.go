package llm

import "context"

// LLMClient is the interface every judge scorer depends on.
// The user provides a concrete implementation — Anthropic, OpenAI, or any other.
// This keeps scorers testable and model-agnostic.
type LLMClient interface {
	Complete(ctx context.Context, system, user string) (string, error)
}