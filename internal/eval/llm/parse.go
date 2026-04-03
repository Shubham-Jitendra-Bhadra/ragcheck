package llm

import (
	"strconv"
	"strings"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
)

// parseScore extracts SCORE and REASONING from raw LLM response text.
// Expected format:
//
//	SCORE: 0.91
//	REASONING: every claim is supported by the retrieved context

func parseScore(metric, raw string) eval.Score {
	s := eval.Score{Metric: metric}

	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		upper := strings.ToUpper(line)

		if strings.HasPrefix(upper, "SCORE:") {
			val := strings.TrimSpace(line[6:])
			val = strings.Trim(val, "`*_") 
			if f, err := strconv.ParseFloat(val, 64); err == nil {
				if f < 0.0 {
					f = 0.0
				}
				if f > 1.0 {
					f = 1.0
				}
				s.Value = f
			}
		}

		if strings.HasPrefix(upper, "REASONING:") {
			s.Reasoning = strings.TrimSpace(line[10:])
		}
	}

	return s
}