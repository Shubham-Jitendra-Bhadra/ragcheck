package llm
// Context recall answers one question: did your retriever fetch the right documents?

// This is the only core scorer that evaluates your retriever rather than your generator. Faithfulness and relevancy both look at the answer. Context recall ignores the answer completely — it only looks at whether the retrieved chunks contain enough information to produce the reference answer.

// Why this matters
// Your RAG pipeline has two separate components that can fail independently:
// retriever   →   fetches chunks from your knowledge base
// generator   →   reads chunks and produces an answer

// If your final answer is bad, you need to know which component failed. Context recall tells you if the retriever is the problem. If context recall is high but faithfulness is low, the retriever did its job — the generator is hallucinating. If context recall is low, fix your retriever first — better chunks will fix everything downstream.

// What causes low context recall:

// Wrong embedding model — your embedding model doesn't capture domain-specific terminology well so semantically relevant documents aren't surfaced.
// Chunk size too large — important information is buried in a large chunk alongside irrelevant content, diluting the embedding signal.
// Top-k too low — you're only fetching 3 chunks when the answer requires information spread across 5 documents.
// Missing documents — the knowledge base simply doesn't contain the relevant information yet.

// How it differs from chunk coverage:
// chunk_coverage    →  heuristic  →  word overlap between answer and chunks
// context_recall    →  LLM judge  →  do chunks contain enough to answer the question?

// Chunk coverage is a surface-level signal — it checks word overlap. Context recall is a deep signal — it asks Claude whether the information needed is actually present in the chunks, even if expressed differently.

// Needs Reference
// This is the only core LLM scorer that requires a ground truth answer. Without a reference you have no way to ask "do the chunks contain enough to produce this answer?" — because you don't know what the ideal answer looks like.
// If Reference is empty the scorer skips gracefully with a clear message, same pattern as overlap scorers.

// What the output looks like:
// context_recall   0.91   chunks contain sufficient information to answer the question
// context_recall   0.34   chunks are missing key details present in the reference answer

import (
	"context"
	"fmt"
	"strings"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

const recallSystem = `You are a RAG evaluation assistant scoring context recall.

Context recall measures whether the retrieved context contains enough information
to produce the reference answer. You are evaluating the retriever, not the answer.

A score of 1.0 means the context contains all the information needed to produce the reference answer.
A score of 0.0 means the context is missing key information present in the reference answer.

Do not consider the quality of the answer — only whether the context contains the right information.

Respond in exactly this format with nothing else:
SCORE: <number between 0.0 and 1.0>
REASONING: <one sentence explaining the score>`

type ContextRecallScorer struct {
	client LLMClient
}

func NewContextRecallScorer(client LLMClient) *ContextRecallScorer {
	return &ContextRecallScorer{client: client}
}

func (s *ContextRecallScorer) Name() string { return "context_recall" }

func (s *ContextRecallScorer) Score(ctx context.Context, c eval.EvalCase) (eval.Score, error) {
	if c.Reference == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no reference provided — skipping context recall",
		}, nil
	}

	if len(c.RetrievedChunks) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no chunks retrieved — cannot assess context recall",
		}, nil
	}

	chunks := strings.Join(c.RetrievedChunks, "\n\n---\n\n")
	user := fmt.Sprintf(
		"QUESTION:\n%s\n\nREFERENCE ANSWER:\n%s\n\nRETRIEVED CONTEXT:\n%s\n\nDoes the context contain enough information to produce the reference answer?",
		c.Query, c.Reference, chunks,
	)

	raw, err := s.client.Complete(ctx, recallSystem, user)
	if err != nil {
		return eval.Score{Metric: s.Name(), Error: err}, err
	}

	return parseScore(s.Name(), raw), nil
}