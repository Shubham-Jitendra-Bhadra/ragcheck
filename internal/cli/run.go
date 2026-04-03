package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval/embedding"
	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval/heuristic"
	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval/llm"
	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval/overlap"
	ragllm "github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/llm"
	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/store"
	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(runCmd)

	runCmd.Flags().String("input", "", "path to JSON file of EvalCases (required)")
	runCmd.Flags().StringSlice("scorers", nil, "comma separated scorer names to run (default: all)")
	runCmd.Flags().Float64("fail-below", 0, "exit non-zero if average score below this threshold")

	runCmd.MarkFlagRequired("input")
}

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Evaluate a JSON file of RAG outputs",
	Example: `  ragcheck run --input cases.json
  ragcheck run --input cases.json --model claude-sonnet-4-6
  ragcheck run --input cases.json --scorers faithfulness,relevancy
  ragcheck run --input cases.json --fail-below 0.75`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()

		// read flags
		inputFile, _ := cmd.Flags().GetString("input")
		requestedScorers, _ := cmd.Flags().GetStringSlice("scorers")
		failBelow, _ := cmd.Flags().GetFloat64("fail-below")
		dbPath, _ := cmd.Root().PersistentFlags().GetString("db")
		model, _ := cmd.Root().PersistentFlags().GetString("model")

		// resolve API keys
		apiKey, _ := cmd.Root().PersistentFlags().GetString("api-key")
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		voyageKey := os.Getenv("VOYAGE_API_KEY")

		// parse input file
		cases, err := loadCases(inputFile)
		if err != nil {
			return fmt.Errorf("load input: %w", err)
		}
		fmt.Printf("loaded %d cases from %s\n", len(cases), inputFile)

		// build scorers
		scorers := buildScorers(apiKey, voyageKey, model)

		// filter to requested scorers if specified
		if len(requestedScorers) > 0 {
			scorers = filterScorers(scorers, requestedScorers)
		}

		fmt.Printf("running %d scorers\n\n", len(scorers))

		// open store
		s, err := store.New(dbPath)
		if err != nil {
			return fmt.Errorf("open store: %w", err)
		}

		// create run record
		runID, err := s.CreateRun(ctx, inputFile, model, len(cases))
		if err != nil {
			return fmt.Errorf("create run: %w", err)
		}

		// run evaluation with progress
		progress := make(chan eval.Progress, len(cases))
		runner := eval.NewRunner(scorers)

		// print progress in background
		go func() {
			for p := range progress {
				fmt.Printf("\revaluating... %d/%d cases complete", p.Done, p.Total)
			}
		}()

		results, err := runner.Run(ctx, cases, progress)
		close(progress)
		fmt.Println()

		if err != nil {
			return fmt.Errorf("evaluation failed: %w", err)
		}

		// save results
		for _, result := range results {
			result.RunID = runID
			if err := s.SaveResult(ctx, result); err != nil {
				fmt.Fprintf(os.Stderr, "warning: failed to save result %s: %v\n", result.CaseID, err)
			}
		}

		// print summary
		printSummary(results, runID)
		

		// check fail-below threshold
		if failBelow > 0 {
			avg := averageScore(results)
			if avg < failBelow {
				fmt.Fprintf(os.Stderr, "\nquality gate failed — average score %.2f is below threshold %.2f\n", avg, failBelow)
				os.Exit(1)
			}
		}

		return nil
	},
}

// loadCases reads and parses a JSON file into []eval.EvalCase.
func loadCases(path string) ([]eval.EvalCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var cases []eval.EvalCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("parse json: %w", err)
	}

	if len(cases) == 0 {
		return nil, fmt.Errorf("no cases found in %s", path)
	}

	return cases, nil
}

// buildScorers constructs the full scorer list based on available API keys.
// Heuristic and overlap scorers always run — free and instant.
// LLM scorers only run if Anthropic key is present.
// Embedding scorers only run if Voyage key is present.
func buildScorers(apiKey, voyageKey, model string) []eval.Scorer {
	scorers := []eval.Scorer{
		// heuristic — always run
		heuristic.NewLengthScorer(),
		heuristic.NewRefusalScorer(),
		heuristic.NewChunkCoverageScorer(),
		heuristic.NewUtilisationScorer(),
		heuristic.NewLatencyScorer(),

		// overlap — always run
		overlap.NewF1Scorer(),
		overlap.NewROUGELScorer(),
		overlap.NewBLEUScorer(),
		overlap.NewConcisenessScorer(),
	}

	// embedding scorers — only if Voyage key present
	if voyageKey != "" {
		embedder := ragllm.NewAnthropicEmbedder(voyageKey, "voyage-2")
		scorers = append(scorers,
			embedding.NewAnswerSimilarityScorer(embedder),
			embedding.NewChunkRelevanceScorer(embedder),
		)
	} else {
		fmt.Println("note: VOYAGE_API_KEY not set — skipping embedding scorers")
	}

	// LLM judge scorers — only if Anthropic key present
	if apiKey != "" {
		client := ragllm.New(apiKey, model)
		scorers = append(scorers,
			llm.NewFaithfulnessScorer(client),
			llm.NewRelevancyScorer(client),
			llm.NewContextRecallScorer(client),
			llm.NewHallucinationScorer(client),
		)
	} else {
		fmt.Println("note: ANTHROPIC_API_KEY not set — skipping LLM judge scorers")
	}

	return scorers
}

// filterScorers returns only scorers whose Name() is in the requested list.
func filterScorers(scorers []eval.Scorer, requested []string) []eval.Scorer {
	set := make(map[string]bool, len(requested))
	for _, name := range requested {
		set[strings.TrimSpace(name)] = true
	}

	filtered := make([]eval.Scorer, 0)
	for _, s := range scorers {
		if set[s.Name()] {
			filtered = append(filtered, s)
		}
	}
	return filtered
}

// averageScore computes the mean score across all results and scorers.
func averageScore(results []eval.Result) float64 {
	if len(results) == 0 {
		return 0.0
	}

	var total float64
	var count int

	for _, r := range results {
		for _, s := range r.Scores {
			if s.Error == nil {
				total += s.Value
				count++
			}
		}
	}

	if count == 0 {
		return 0.0
	}

	return total / float64(count)
}

// printSummary prints a simple score summary to the terminal.
// Full table rendering happens in the report command.
func printSummary(results []eval.Result, runID int64) {
	fmt.Printf("\nrun complete — run ID: %d\n", runID)
	fmt.Printf("total cases: %d\n\n", len(results))

	// aggregate scores per metric
	totals := make(map[string]float64)
	counts := make(map[string]int)

	for _, r := range results {
		for _, s := range r.Scores {
			if s.Error == nil {
				totals[s.Metric] += s.Value
				counts[s.Metric]++
			}
		}
	}

	fmt.Println("avg scores:")
	for metric, total := range totals {
		avg := total / float64(counts[metric])
		fmt.Printf("  %-25s %.2f\n", metric, avg)
	}

	fmt.Printf("\nrun 'ragcheck report --run %d' for full details\n", runID)
}
