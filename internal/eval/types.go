package eval

import "context"

// EvalCase is one test case — the input to any scorer.
type EvalCase struct {
	ID              string   
	Query           string   // the user's question
	RetrievedChunks []string // chunks retriever returned
	Answer          string   // the generated answer
	Reference       string   // ground truth — needed for overlap + recall scorers
}

// Score is the result from one scorer on one EvalCase.
type Score struct {
	Metric    string
	Value     float64 // always 0.0 – 1.0
	Reasoning string  // explanation — populated by LLM scorer
	Error     error
}

// Result bundles all scores for one EvalCase.
type Result struct {
	CaseID string
	Scores []Score
}

// Scorer is the interface every approach implements.

type Scorer interface {
	Name() string
	Score(ctx context.Context, c EvalCase) (Score, error)
}