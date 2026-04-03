package llm

import (
	"context"
	"fmt"
	"strings"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

const faithfulnessSystem = `You are a RAG evaluation assistant scoring answer faithfulness.

Faithfulness measures whether every claim in the answer is supported by the provided context.
A score of 1.0 means every claim can be traced directly to the context.
A score of 0.0 means the answer contains claims not present in the context at all.

Respond in exactly this format with nothing else:
SCORE: <number between 0.0 and 1.0>
REASONING: <one sentence explaining the score>`

type FaithfulnessScorer struct {
	client LLMClient
}

func NewFaithfulnessScorer(client LLMClient) *FaithfulnessScorer {
	return &FaithfulnessScorer{client: client}
}

func (s *FaithfulnessScorer) Name() string { return "faithfulness" }

func (s *FaithfulnessScorer) Score(ctx context.Context, c eval.EvalCase) (eval.Score, error) {
	if len(c.RetrievedChunks) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no chunks retrieved — cannot assess faithfulness",
		}, nil
	}

	if c.Answer == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer is empty",
		}, nil
	}

	chunks := strings.Join(c.RetrievedChunks, "\n\n---\n\n")
	user := fmt.Sprintf(
		"CONTEXT:\n%s\n\nANSWER:\n%s\n\nIs every claim in the answer supported by the context?",
		chunks, c.Answer,
	)

	raw, err := s.client.Complete(ctx, faithfulnessSystem, user)
	if err != nil {
		return eval.Score{Metric: s.Name(), Error: err}, err
	}

	return parseScore(s.Name(), raw), nil
}