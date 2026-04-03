package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/store"
	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(reportCmd)

	reportCmd.Flags().Int64("run", 0, "run ID to report on (required)")
	reportCmd.Flags().String("format", "table", "output format: table or json")

	reportCmd.MarkFlagRequired("run")
}

var reportCmd = &cobra.Command{
	Use:   "report",
	Short: "Print detailed results for an evaluation run",
	Example: `  ragcheck report --run 3
  ragcheck report --run 3 --format json
  ragcheck report --run 3 --format json > results.json`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()

		runID, _ := cmd.Flags().GetInt64("run")
		format, _ := cmd.Flags().GetString("format")
		dbPath, _ := cmd.Root().PersistentFlags().GetString("db")

		// open store
		s, err := store.New(dbPath)
		if err != nil {
			return fmt.Errorf("open store: %w", err)
		}

		// fetch run metadata
		run, err := s.GetRun(ctx, runID)
		if err != nil {
			return fmt.Errorf("fetch run %d: %w", runID, err)
		}

		// fetch all results for this run
		results, err := s.GetResults(ctx, runID)
		if err != nil {
			return fmt.Errorf("fetch results: %w", err)
		}

		if len(results) == 0 {
			return fmt.Errorf("no results found for run %d", runID)
		}

		switch format {
		case "json":
			return printJSON(*run, results)
		default:
			return printTable(*run, results)
		}	
	},
}

// printJSON outputs run and results as JSON to stdout.
func printJSON(run store.Run, results []store.ResultRow) error {
	output := map[string]any{
		"run":     run,
		"results": results,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(output)
}

// printTable prints a human readable breakdown of every case and score.
func printTable(run store.Run, results []store.ResultRow) error {
	// print run header
	fmt.Printf("\nrun %d — %s — %d cases — model: %s\n\n",
		run.ID,
		run.CreatedAt.Format("2006-01-02 15:04"),
		run.TotalCases,
		run.Model,
	)

	// print each case
	for _, r := range results {
		if r.HasErrors {
			fmt.Printf("case: %s  ⚠ has errors\n", r.CaseID)
		} else {
			fmt.Printf("case: %s\n", r.CaseID)
		}

		for _, score := range r.Scores {
			if score.Error != "" {
				fmt.Printf("  %-25s  %.2f   scorer error: %s\n",
					score.Metric, score.Value, score.Error)
			} else {
				fmt.Printf("  %-25s  %.2f   %s\n",
					score.Metric, score.Value, score.Reasoning)
			}
		}

		// print metadata if present
		if len(r.Metadata) > 0 {
			fmt.Println("  metadata:")
			for k, v := range r.Metadata {
				fmt.Printf("    %s: %s\n", k, v)
			}
		}

		fmt.Println()
	}

	// print aggregate summary at bottom
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("averages:")

	totals := make(map[string]float64)
	counts := make(map[string]int)

	for _, r := range results {
		for _, score := range r.Scores {
			if score.Error == "" {
				totals[score.Metric] += score.Value
				counts[score.Metric]++
			}
		}
	}

	for metric, total := range totals {
		avg := total / float64(counts[metric])
		fmt.Printf("  %-25s  %.2f\n", metric, avg)
	}

	fmt.Println()
	return nil
}
