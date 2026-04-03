package llm

// Relevancy answers one question: does the answer actually address what the user asked?

// It's the simplest of the three core scorers because it only needs two things — the Query and the Answer. It completely ignores the retrieved chunks. That's intentional.

// Here's why chunks don't matter here — relevancy is purely about the relationship between question and answer. Whether the retriever did a good job is a separate concern. You could have perfect chunks and still get an irrelevant answer if your generation prompt is badly written.

// What causes low relevancy scores in real RAG apps:
// The LLM went off-topic — user asked about refund policy, LLM answered about shipping policy because both were in the chunks.

// The system prompt is too restrictive — "only answer questions about product features" causes the LLM to deflect any other question with a canned response that scores 0.0 on relevancy.

// The query was ambiguous — "tell me about the account" could mean billing, security, or profile settings. The LLM picks one and ignores the others.

// Prompt template confusion — your prompt template injects so much context that the LLM loses track of what was actually asked.

// How it differs from faithfulness:
// faithfulness  →  answer vs chunks    "is the answer grounded?"
// relevancy     →  answer vs query     "does the answer address the question?"

// You can have high faithfulness and low relevancy — the answer is perfectly grounded in the chunks but answers the wrong question. You can also have high relevancy and low faithfulness — the answer addresses the question but adds hallucinated details not in the chunks.

// Both scorers together give you complete coverage of generation quality.
// What the output looks like:
// relevancy   0.92   answer directly addresses the question about refund policy
// relevancy   0.21   answer discusses shipping but question was about returns
import (
	"context"
	"fmt"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

const relevancySystem = `You are a RAG evaluation assistant scoring answer relevancy.

Relevancy measures whether the answer directly addresses what the user asked.
A score of 1.0 means the answer fully addresses the question.
A score of 0.0 means the answer is completely off-topic or answers a different question.

Do not consider whether the answer is factually correct — only whether it addresses the question.

Respond in exactly this format with nothing else:
SCORE: <number between 0.0 and 1.0>
REASONING: <one sentence explaining the score>`

type RelevancyScorer struct {
	client LLMClient
}

func NewRelevancyScorer(client LLMClient) *RelevancyScorer {
	return &RelevancyScorer{client: client}
}

func (s *RelevancyScorer) Name() string { return "relevancy" }

func (s *RelevancyScorer) Score(ctx context.Context, c eval.EvalCase) (eval.Score, error) {
	if c.Query == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "query is empty — cannot assess relevancy",
		}, nil
	}

	if c.Answer == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer is empty",
		}, nil
	}

	user := fmt.Sprintf(
		"QUESTION:\n%s\n\nANSWER:\n%s\n\nDoes the answer directly address the question?",
		c.Query, c.Answer,
	)

	raw, err := s.client.Complete(ctx, relevancySystem, user)
	if err != nil {
		return eval.Score{Metric: s.Name(), Error: err}, err
	}

	return parseScore(s.Name(), raw), nil
}