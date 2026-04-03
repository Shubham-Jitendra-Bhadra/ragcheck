package embedding

import (
	"context"
	"math"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

// Embedder is the interface any embedding provider must satisfy.
// This lets you swap Anthropic, OpenAI, or a local model freely.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float64, error)
}

// cosineSimilarity measures the angle between two vectors.
// Returns 1.0 for identical meaning, 0.0 for unrelated.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	var dot, magA, magB float64
	for i := range a {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}

	if magA == 0 || magB == 0 {
		return 0.0
	}

	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

// AnswerSimilarityScorer embeds the answer and reference,
// then measures cosine similarity between them.
// Catches paraphrasing that token overlap misses.
type AnswerSimilarityScorer struct {
	embedder Embedder
}

func NewAnswerSimilarityScorer(e Embedder) *AnswerSimilarityScorer {
	return &AnswerSimilarityScorer{embedder: e}
}

func (s *AnswerSimilarityScorer) Name() string { return "answer_similarity" }

func (s *AnswerSimilarityScorer) Score(ctx context.Context, c eval.EvalCase) (eval.Score, error) {
	if c.Reference == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no reference provided — skipping answer similarity",
		}, nil
	}

	answerVec, err := s.embedder.Embed(ctx, c.Answer)
	if err != nil {
		return eval.Score{Metric: s.Name(), Error: err}, err
	}

	refVec, err := s.embedder.Embed(ctx, c.Reference)
	if err != nil {
		return eval.Score{Metric: s.Name(), Error: err}, err
	}

	similarity := cosineSimilarity(answerVec, refVec)

	return eval.Score{
		Metric:    s.Name(),
		Value:     similarity,
		Reasoning: "semantic similarity between answer and reference",
	}, nil
}

// ChunkRelevanceScorer embeds the query and each retrieved chunk,
// then returns the average cosine similarity across all chunks.
// Low score means the retriever fetched irrelevant documents.
type ChunkRelevanceScorer struct {
	embedder Embedder
}

func NewChunkRelevanceScorer(e Embedder) *ChunkRelevanceScorer {
	return &ChunkRelevanceScorer{embedder: e}
}

func (s *ChunkRelevanceScorer) Name() string { return "chunk_relevance" }

func (s *ChunkRelevanceScorer) Score(ctx context.Context, c eval.EvalCase) (eval.Score, error) {
	if len(c.RetrievedChunks) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no chunks were retrieved",
		}, nil
	}

	queryVec, err := s.embedder.Embed(ctx, c.Query)
	if err != nil {
		return eval.Score{Metric: s.Name(), Error: err}, err
	}

	var total float64
	for _, chunk := range c.RetrievedChunks {
		chunkVec, err := s.embedder.Embed(ctx, chunk)
		if err != nil {
			return eval.Score{Metric: s.Name(), Error: err}, err
		}
		total += cosineSimilarity(queryVec, chunkVec)
	}

	avg := total / float64(len(c.RetrievedChunks))

	return eval.Score{
		Metric:    s.Name(),
		Value:     avg,
		Reasoning: "average semantic similarity between query and retrieved chunks",
	}, nil
}