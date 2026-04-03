package overlap
// BLEU stands for Bilingual Evaluation Understudy. Originally built for machine translation but works great for RAG.

// The key difference from ROUGE and F1 is that BLEU looks at n-grams — sequences of words — not just individual words. So "New York City" as a 3-word phrase scores higher than matching "New", "York", "City" separately. This matters in RAG because domain-specific phrases like "terms and conditions", "next business day", "as per policy" carry meaning as a unit.

// How it works:
// 1-gram  =  single words          "refund"
// 2-gram  =  word pairs            "refund policy"
// 3-gram  =  word triples          "refund policy applies"
// 4-gram  =  four word sequences   "refund policy applies to"

// BLEU computes precision for each n-gram size, then takes a geometric mean of all four. It also applies a brevity penalty — if your answer is much shorter than the reference, the score gets penalized. This prevents a one-word answer from scoring high just because that one word matches.

// Output looks like:
// bleu_score   0.68   n-gram precision across 1 to 4 gram sizes

// Needs Reference. Returns 0.0 with a message if missing.

import (
	"context"
	"math"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

// BLEUScorer computes BLEU score between answer and reference.
// Measures n-gram precision across 1 to 4 gram sizes with brevity penalty.
type BLEUScorer struct{}

func NewBLEUScorer() *BLEUScorer { return &BLEUScorer{} }

func (s *BLEUScorer) Name() string { return "bleu_score" }

func (s *BLEUScorer) Score(_ context.Context, c eval.EvalCase) (eval.Score, error) {
	if c.Reference == "" {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "no reference provided — skipping BLEU",
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

	// compute precision for n-gram sizes 1 through 4
	var logSum float64
	validN := 0

	for n := 1; n <= 4; n++ {
		answerNgrams := buildNgrams(answerTokens, n)
		refNgrams := buildNgrams(refTokens, n)

		if len(answerNgrams) == 0 {
			continue
		}

		matched := 0
		for ngram, answerCount := range answerNgrams {
			if refCount, ok := refNgrams[ngram]; ok {
				if answerCount < refCount {
					matched += answerCount
				} else {
					matched += refCount
				}
			}
		}

		precision := float64(matched) / float64(len(answerTokens)-n+1)
		if precision == 0 {
			return eval.Score{
				Metric:    s.Name(),
				Value:     0.0,
				Reasoning: "no matching n-grams found",
			}, nil
		}

		logSum += math.Log(precision)
		validN++
	}

	if validN == 0 {
		return eval.Score{
			Metric:    s.Name(),
			Value:     0.0,
			Reasoning: "could not compute any n-gram precision",
		}, nil
	}

	// geometric mean of all n-gram precisions
	score := math.Exp(logSum / float64(validN))

	// brevity penalty — punish answers shorter than the reference
	bp := 1.0
	if len(answerTokens) < len(refTokens) {
		bp = math.Exp(1 - float64(len(refTokens))/float64(len(answerTokens)))
	}

	score = bp * score
	if score > 1.0 {
		score = 1.0
	}

	return eval.Score{
		Metric:    s.Name(),
		Value:     score,
		Reasoning: "n-gram precision across 1 to 4 gram sizes with brevity penalty",
	}, nil
}

// buildNgrams builds a frequency map of n-grams from a token slice.
func buildNgrams(tokens []string, n int) map[string]int {
	ngrams := make(map[string]int)
	for i := 0; i <= len(tokens)-n; i++ {
		key := ""
		for j := 0; j < n; j++ {
			if j > 0 {
				key += " "
			}
			key += tokens[i+j]
		}
		ngrams[key]++
	}
	return ngrams
}