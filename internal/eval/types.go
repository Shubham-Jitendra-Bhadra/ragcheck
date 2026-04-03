package eval

import (
	"context"
	"time"
)

// EvalCase is one test case — the input to any scorer.
type EvalCase struct {
	ID              string            `json:"id"`
	Query           string            `json:"query"`
	RetrievedChunks []string          `json:"retrieved_chunks"`
	Answer          string            `json:"answer"`
	Reference       string            `json:"reference"`
	Model           string            `json:"model"`
	Metadata        map[string]string `json:"metadata"`
}

// Score is the result from one scorer on one EvalCase.
type Score struct {
	Metric    string  `json:"metric"`
	Value     float64 `json:"value"`
	Reasoning string  `json:"reasoning"`
	Error     error   `json:"error,omitempty"`
}

// Result bundles all scores for one EvalCase.
type Result struct {
	CaseID    string            `json:"case_id"`
	Model     string            `json:"model"`
	Scores    []Score           `json:"scores"`
	Metadata  map[string]string `json:"metadata"`
	RunID     int64             `json:"run_id"`
	CreatedAt time.Time         `json:"created_at"`
	HasErrors bool              `json:"has_errors"`
}

// Scorer is the interface every approach implements.
type Scorer interface {
	Name() string
	Score(ctx context.Context, c EvalCase) (Score, error)
}