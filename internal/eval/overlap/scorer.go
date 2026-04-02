package overlap

// Token overlap measures how many words the answer shares with the reference (ground truth). It's the same technique used to evaluate machine translation and summarization for decades. Two metrics:

// F1 token match — treats both the answer and reference as a bag of words. Counts how many words appear in both, then computes a balance between precision and recall:

// precision = shared words / total words in answer
// recall    = shared words / total words in reference
// F1        = 2 × (precision × recall) / (precision + recall)

// Example — reference says "refunds take 5 to 7 business days", answer says "refunds are processed in 5 to 7 days". Shared words: refunds, 5, 7, days. F1 would be around 0.75 — not perfect but clearly in the right ballpark.

// ROUGE-L — instead of a bag of words, it finds the longest sequence of words that appear in both texts in the same order. "5 to 7 days" appearing in both in the same order scores higher than random word overlap. More sensitive to structure than F1.

// How is this different from heuristics?
// Heuristics don't need the Reference field at all — they only look at the answer itself. Token overlap scorers compare the answer against the reference — so they need a ground truth. This means they only run when you have a Reference in your EvalCase.

import (
	"context"
	"strings"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

// tokenize splits text into lowercase words, filtering empty strings.
func tokenize(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	tokens := make([]string, 0, len(words))
	for _, w := range words {
		w = strings.Trim(w, ".,!?;:\"'()")
		if len(w) > 0 {
			tokens = append(tokens, w)
		}
	}
	return tokens
}

// countMap builds a word frequency map from a token slice.
func countMap(tokens []string) map[string]int {
	m := make(map[string]int, len(tokens))
	for _, t := range tokens {
		m[t]++
	}
	return m
}

// F1Scorer computes token-level F1 between answer and reference.
type F1Scorer struct{}

func NewF1Scorer() *F1Scorer { return &F1Scorer{} }

func (s *F1Scorer) Name() string { return "f1_token_match" }

func (s *F1Scorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	if c.Reference == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no reference provided — skipping F1",
		}, nil
	}

	answerTokens := tokenize(c.Answer)
	refTokens := tokenize(c.Reference)

	if len(answerTokens) == 0 || len(refTokens) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer or reference is empty",
		}, nil
	}

	answerCounts := countMap(answerTokens)
	refCounts := countMap(refTokens)

	// count shared tokens (overlap)
	shared := 0
	for word, refCount := range refCounts {
		if answerCount, ok := answerCounts[word]; ok {
			if answerCount < refCount {
				shared += answerCount
			} else {
				shared += refCount
			}
		}
	}

	precision := float64(shared) / float64(len(answerTokens))
	recall := float64(shared) / float64(len(refTokens))

	if precision+recall == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no token overlap between answer and reference",
		}, nil
	}

	f1 := 2 * (precision * recall) / (precision + recall)

	return eval.Score{
		Metric:    s.Name(),
		Value:     f1,
		Reasoning: "token overlap between answer and reference",
	}, nil
}

// ROUGELScorer computes ROUGE-L using longest common subsequence.
type ROUGELScorer struct{}

func NewROUGELScorer() *ROUGELScorer { return &ROUGELScorer{} }

func (s *ROUGELScorer) Name() string { return "rouge_l" }

func (s *ROUGELScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	if c.Reference == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no reference provided — skipping ROUGE-L",
		}, nil
	}

	answerTokens := tokenize(c.Answer)
	refTokens := tokenize(c.Reference)

	if len(answerTokens) == 0 || len(refTokens) == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "answer or reference is empty",
		}, nil
	}

	lcsLen := lcs(answerTokens, refTokens)

	precision := float64(lcsLen) / float64(len(answerTokens))
	recall := float64(lcsLen) / float64(len(refTokens))

	if precision+recall == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no common subsequence found",
		}, nil
	}

	rougeL := 2 * (precision * recall) / (precision + recall)

	return eval.Score{
		Metric:    s.Name(),
		Value:     rougeL,
		Reasoning: "longest common subsequence score",
	}, nil
}

// lcs computes the length of the longest common subsequence
// between two token slices using dynamic programming.
func lcs(a, b []string) int {
	m, n := len(a), len(b)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else if dp[i-1][j] > dp[i][j-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}
	return dp[m][n]
}