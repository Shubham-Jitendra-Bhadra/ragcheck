package heuristic

// Each retrieved chunk gets checked individually — does the answer contain any meaningful words from this chunk? If yes, that chunk is "used". Final score is used chunks / total chunks. Low score means you're retrieving more chunks than your LLM actually uses — wasted tokens and wasted cost. High score means your retriever is fetching relevant information that the LLM is incorporating into the answer. This is a crucial metric for RAG systems — it directly measures how well your retriever and reader are working together.

import (
	"context"
	"fmt"
	"strings"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

// UtilisationScorer measures what percentage of retrieved chunks
// contributed at least one meaningful word to the answer.
// Low score = retriever is fetching more chunks than the LLM uses.
type UtilisationScorer struct {
	MinWordLength int 
}

func NewUtilisationScorer() *UtilisationScorer {
	return &UtilisationScorer{MinWordLength: 4}
}

func (s *UtilisationScorer) Name() string { return "chunk_utilisation" }

func (s *UtilisationScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	if len(c.RetrievedChunks) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no chunks were retrieved",
		}, nil
	}

	if c.Answer == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer is empty",
		}, nil
	}

	answerLower := strings.ToLower(c.Answer)
	used := 0

	for _, chunk := range c.RetrievedChunks {
		chunkWords := strings.Fields(strings.ToLower(chunk))
		for _, word := range chunkWords {
			word = strings.Trim(word, ".,!?;:\"'()")
			if len(word) >= s.MinWordLength && strings.Contains(answerLower, word) {
				used++
				break 
			}
		}
	}

	score := float64(used) / float64(len(c.RetrievedChunks))
	reasoning := fmt.Sprintf(
		"%d of %d chunks contributed to the answer",
		used, len(c.RetrievedChunks),
	)

	return eval.Score{
		Metric:    s.Name(),
		Value:     score,
		Reasoning: reasoning,
	}, nil
}