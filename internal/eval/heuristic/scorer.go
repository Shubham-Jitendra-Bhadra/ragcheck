package heuristic

// The three heuristics ragcheck uses:

// 1. Length check — counts the words in the answer. If someone asks "what is our refund policy?" and the answer is 2 words, something is clearly wrong. If it's 2000 words, it's probably hallucinating irrelevant content. A healthy answer for most RAG apps sits between 10 and 500 words. This scorer returns 1.0 if in range, 0.0 if too short, 0.5 if suspiciously long.

// 2. Refusal detection — scans the answer for phrases like "I don't know", "I cannot find", "I'm not sure". These mean your RAG pipeline retrieved chunks but the LLM still refused to answer — which means either the chunks were irrelevant or your system prompt is too cautious. Score 0.0 if a refusal phrase is found, 1.0 if clean.

// 3. Chunk coverage — counts how many meaningful words in the answer also appear in the retrieved chunks. If the answer shares almost no words with the chunks, the LLM likely ignored the context and went off its training data instead — which is exactly the hallucination problem RAG is supposed to solve. Score is a ratio from 0.0 to 1.0.

import (
	"context"
	"strings"
	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

type LengthScorer struct {
	MinWords int
	MaxWords int
}

func NewLengthScorer() *LengthScorer {
	return &LengthScorer{MinWords: 10, MaxWords: 500}
}

func (s *LengthScorer) Name() string { return "answer_length" }

func (s *LengthScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	words := len(strings.Fields(c.Answer))
	if words < s.MinWords {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer too short — likely a refusal or empty response",
		}, nil
	}
	if words > s.MaxWords {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.5,
			Reasoning: "answer unusually long — may contain irrelevant content",
		}, nil
	}
	return eval.Score{
		Metric:    s.Name(),
		Value:     1.0,
		Reasoning: "answer length looks reasonable",
	}, nil
}

type RefusalScorer struct{}

func NewRefusalScorer() *RefusalScorer { return &RefusalScorer{} }

func (s *RefusalScorer) Name() string { return "refusal_detection" }

var refusalPhrases = []string{
	"i don't know",
	"i cannot",
	"i can't",
	"i'm not sure",
	"i do not have",
	"no information",
	"not able to answer",
	"cannot find",
	"unable to find",
	"don't have enough",
}

func (s *RefusalScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	lower := strings.ToLower(c.Answer)
	for _, phrase := range refusalPhrases {
		if strings.Contains(lower, phrase) {
			return eval.Score{
				Metric:    s.Name(),
				Value:     0.0,
				Reasoning: "refusal phrase found: " + phrase,
			}, nil
		}
	}
	return eval.Score{
		Metric:    s.Name(),
		Value:     1.0,
		Reasoning: "no refusal phrases detected",
	}, nil
}

type ChunkCoverageScorer struct{}

func NewChunkCoverageScorer() *ChunkCoverageScorer { return &ChunkCoverageScorer{} }

func (s *ChunkCoverageScorer) Name() string { return "chunk_coverage" }

func (s *ChunkCoverageScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	if len(c.RetrievedChunks) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no chunks were retrieved",
		}, nil
	}

	chunkText := strings.ToLower(strings.Join(c.RetrievedChunks, " "))
	answerWords := strings.Fields(strings.ToLower(c.Answer))

	if len(answerWords) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer is empty",
		}, nil
	}

	matched := 0
	for _, word := range answerWords {
		if len(word) > 4 && strings.Contains(chunkText, word) {
			matched++
		}
	}

	score := float64(matched) / float64(len(answerWords))
	if score > 1.0 {
		score = 1.0
	}

	return eval.Score{
		Metric:    s.Name(),
		Value:     score,
		Reasoning: "word overlap between answer and retrieved chunks",
	}, nil
}