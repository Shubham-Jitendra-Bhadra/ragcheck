package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "ragcheck",
	Short: "A fast, Go-native RAG evaluation tool",
	Long: `ragcheck evaluates your RAG pipeline by scoring outputs
across multiple metrics — heuristic, token overlap, embedding similarity,
and LLM-as-judge. Works as a CLI or importable Go SDK.`,
}

// Execute is called by main.go — starts cobra's argument parser.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().String(
		"db", "./ragcheck.db",
		"path to ragcheck SQLite database",
	)
	rootCmd.PersistentFlags().String(
		"model", "claude-haiku-4-5-20251001",
		"judge model for LLM scorers",
	)
	rootCmd.PersistentFlags().String(
		"api-key", "",
		"Anthropic API key (defaults to ANTHROPIC_API_KEY env var)",
	)
}