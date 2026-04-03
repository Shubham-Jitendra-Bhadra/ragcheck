package llm

// Hallucination answers one question: which specific claim in the answer is not supported by the context?

// You already have faithfulness.go which gives you a score like 0.4 meaning "60% of claims are not grounded." That's useful but not actionable enough. When you get a low faithfulness score you still don't know what went wrong — you have to read the entire answer and all the chunks yourself to find the problem.

// Hallucination detection is different. Instead of a score it asks Claude to identify the exact sentence or claim that isn't supported. The reasoning field becomes the primary output, not the score.

// A real example of the difference:
// faithfulness      0.3    some claims are not supported by context
// hallucination     0.0    "the refund window is 60 days" is not supported
//                          by any retrieved chunk — context only mentions 30 days

// The second output tells you exactly what to fix. You can go straight to your knowledge base and check whether "60 days" appears anywhere. If it doesn't, your LLM is hallucinating. If it does, your retriever missed the relevant document.

// How the score works here
// Unlike other scorers where high score = good, hallucination scoring is:
// 1.0   no hallucinations detected — every claim is supported
// 0.5   minor unsupported details — present but not critical
// 0.0   clear hallucination detected — specific claim has no basis in context
// The score is still 0.0–1.0 to stay consistent with the Scorer interface, but the Reasoning field is where the real value is. Your report should always print the reasoning for hallucination scores below 0.8.

// How it differs from faithfulness:
// faithfulness    →   overall grounding score        "how faithful is this answer overall?"
// hallucination   →   specific claim identification  "which exact claim is wrong?"
// Run faithfulness first as a cheap filter. If it scores below 0.7, run hallucination to pinpoint exactly what went wrong. In your runner you can even configure hallucination to only run when faithfulness is low — saving API calls.

// What the output looks like:
// hallucination   1.0   all claims in the answer are supported by the retrieved context
// hallucination   0.0   claim "available in 14 countries" not found in any retrieved chunk
// That second line is a direct pointer to a bug in your RAG pipeline.

// Needs RetrievedChunks and Answer
// Same requirements as faithfulness. No reference needed — comparing answer against chunks, not against a ground truth.

import (
	"context"
	"fmt"
	"strings"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

const hallucinationSystem = `You are a RAG evaluation assistant detecting hallucinations.

A hallucination is any specific claim in the answer that cannot be traced to the provided context.
Your job is to identify the single most problematic unsupported claim if one exists.

Scoring guide:
1.0 — every claim in the answer is supported by the context
0.5 — minor unsupported details present but not critically wrong
0.0 — a clear hallucination exists — a specific claim has no basis in the context

Do not penalise for paraphrasing or summarising — only penalise for claims
that introduce new information not present anywhere in the context.

Respond in exactly this format with nothing else:
SCORE: <number between 0.0 and 1.0>
REASONING: <one sentence — if hallucination found, quote the exact claim>`

type HallucinationScorer struct {
	client LLMClient
}

func NewHallucinationScorer(client LLMClient) *HallucinationScorer {
	return &HallucinationScorer{client: client}
}

func (s *HallucinationScorer) Name() string { return "hallucination" }

func (s *HallucinationScorer) Score(ctx context.Context, c eval.EvalCase) (eval.Score, error) {
	if len(c.RetrievedChunks) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no chunks retrieved — cannot detect hallucinations",
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
		"CONTEXT:\n%s\n\nANSWER:\n%s\n\nIdentify any hallucinated claims in the answer.",
		chunks, c.Answer,
	)

	raw, err := s.client.Complete(ctx, hallucinationSystem, user)
	if err != nil {
		return eval.Score{Metric: s.Name(), Error: err}, err
	}

	return parseScore(s.Name(), raw), nil
}