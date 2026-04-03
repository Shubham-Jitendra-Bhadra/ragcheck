package heuristic

import (
	"context"
	"fmt"
	"strconv"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

// LatencyScorer reads latency_ms from EvalCase.Metadata
// and converts it to a 0.0–1.0 score based on speed thresholds.
// Not a quality metric — a performance metric stored in the same format.
type LatencyScorer struct {
	FastMs       int // below this = score 1.0
	AcceptableMs int // below this = score 0.5
	              // above AcceptableMs = score 0.0
}

func NewLatencyScorer() *LatencyScorer {
	return &LatencyScorer{
		FastMs:       500,
		AcceptableMs: 2000,
	}
}

func (s *LatencyScorer) Name() string { return "latency_ms" }

func (s *LatencyScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	raw, ok := c.Metadata["latency_ms"]
	if !ok || raw == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "latency_ms not found in metadata — skipping",
		}, nil
	}

	ms, err := strconv.Atoi(raw)
	if err != nil {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: fmt.Sprintf("could not parse latency_ms value: %s", raw),
		}, nil
	}

	var score float64
	var reasoning string

	switch {
	case ms < s.FastMs:
		score = 1.0
		reasoning = fmt.Sprintf("fast response — %dms", ms)
	case ms < s.AcceptableMs:
		score = 0.5
		reasoning = fmt.Sprintf("acceptable response time — %dms", ms)
	default:
		score = 0.0
		reasoning = fmt.Sprintf("slow response — %dms exceeds %dms threshold", ms, s.AcceptableMs)
	}

	return eval.Score{
		Metric:    s.Name(),
		Value:     score,
		Reasoning: reasoning,
	}, nil
}