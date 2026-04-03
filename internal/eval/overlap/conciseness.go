package overlap

import (
	"context"
	"fmt"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

// ConcisenessScorer measures how close the answer length is to the reference length.
// Penalises answers that are too short or too long.
type ConcisenessScorer struct{}

func NewConcisenessScorer() *ConcisenessScorer { return &ConcisenessScorer{} }

func (s *ConcisenessScorer) Name() string { return "conciseness" }

func (s *ConcisenessScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	if c.Reference == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no reference provided — skipping conciseness",
		}, nil
	}

	answerWords := len(tokenize(c.Answer))
	refWords := len(tokenize(c.Reference))

	if refWords == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "reference is empty",
		}, nil
	}

	if answerWords == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer is empty",
		}, nil
	}

	ratio := float64(answerWords) / float64(refWords)

	var score float64
	var reasoning string

	switch {
	case ratio < 0.5:
		score = 0.0
		reasoning = fmt.Sprintf("answer too short — %.0f%% of reference length", ratio*100)
	case ratio <= 1.5:
		score = 1.0
		reasoning = fmt.Sprintf("answer length is appropriate — %.0f%% of reference length", ratio*100)
	case ratio <= 2.5:
		score = 0.5
		reasoning = fmt.Sprintf("answer is wordy — %.0f%% of reference length", ratio*100)
	default:
		score = 0.0
		reasoning = fmt.Sprintf("answer is far too long — %.0f%% of reference length", ratio*100)
	}

	return eval.Score{
		Metric:    s.Name(),
		Value:     score,
		Reasoning: reasoning,
	}, nil
}