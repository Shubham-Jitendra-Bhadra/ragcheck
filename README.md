# ragcheck

A fast, Go-native RAG evaluation tool. Works as a CLI binary or an importable SDK.
Point it at your RAG pipeline outputs and get structured quality scores back — no Python, no heavy dependencies, single binary.

---

## Why ragcheck

Most RAG eval tools are Python-only and require you to restructure your pipeline around them.
ragcheck is a single Go binary that works with any stack — Python, Node, Java, Go — as long as you can produce a JSON file.
It also embeds cleanly as an SDK inside Go RAG apps for real-time inline scoring.

---

## How it works

ragcheck takes three things from each request your RAG app handles:

- the **query** — what the user asked
- the **retrieved chunks** — what your retriever fetched
- the **answer** — what your LLM generated

It runs those through up to 11 scorers across 4 approaches and returns a score between 0.0 and 1.0 for each metric.

---

## Scoring approaches

### Heuristic — free, instant, no API
| Metric | What it checks |
|---|---|
| `answer_length` | Is the answer a reasonable length? |
| `refusal_detection` | Did the LLM refuse to answer? |
| `chunk_coverage` | Does the answer use words from the retrieved chunks? |
| `chunk_utilisation` | What percentage of chunks contributed to the answer? |
| `latency_ms` | How fast did your pipeline respond? |

### Token overlap — free, instant, needs reference
| Metric | What it checks |
|---|---|
| `f1_token_match` | Word-level overlap between answer and reference |
| `rouge_l` | Longest common subsequence score |
| `bleu_score` | N-gram phrase matching with brevity penalty |
| `conciseness` | Answer length ratio vs reference |

### Embedding similarity — cheap API call, catches paraphrasing
| Metric | What it checks |
|---|---|
| `answer_similarity` | Semantic similarity between answer and reference |
| `chunk_relevance` | How relevant are retrieved chunks to the query |

### LLM judge — most accurate, understands reasoning
| Metric | What it checks |
|---|---|
| `faithfulness` | Is every claim grounded in the retrieved context? |
| `relevancy` | Does the answer address what was asked? |
| `context_recall` | Did the retriever fetch the right documents? |
| `hallucination` | Which specific claim is not supported by context? |


---

## Install

```bash
go install github.com/yourusername/ragcheck/cmd/ragcheck@latest
```

Or download a prebuilt binary from the releases page.

---

## CLI usage

```bash
export ANTHROPIC_API_KEY=your_key

# run evaluation against a JSON file
ragcheck run --input testdata/example.json

# choose your judge model
ragcheck run --input testdata/example.json --model claude-haiku-4-5-20251001

# run only specific scorers
ragcheck run --input testdata/example.json --scorers faithfulness,relevancy,bleu_score

# compare two runs
ragcheck compare --run-a 1 --run-b 2

# export results
ragcheck report --run 3 --format json
ragcheck report --run 3 --format table
```

### Input file format

```json
[
  {
    "id": "q1",
    "query": "What is our refund policy?",
    "retrieved_chunks": [
      "Refunds are processed within 5 to 7 business days.",
      "Items must be returned in original condition."
    ],
    "answer": "You can get a refund within 30 days of purchase.",
    "reference": "Refunds are processed within 5 to 7 business days for items in original condition.",
    "model": "claude-sonnet-4-6",
    "metadata": {
      "latency_ms": "342",
      "chunk_size": "512",
      "temperature": "0.7"
    }
  }
]
```

`reference` is optional — scorers that need it skip gracefully when missing.
`model` and `metadata` are optional — used for comparison and reporting.

---

## SDK usage

For Go RAG apps — import the eval package directly and score inline:

```go
import (
    "github.com/yourusername/ragcheck/internal/eval"
    "github.com/yourusername/ragcheck/internal/eval/heuristic"
    "github.com/yourusername/ragcheck/internal/eval/overlap"
    "github.com/yourusername/ragcheck/internal/eval/llm"
    ragllm "github.com/yourusername/ragcheck/internal/llm"
)

// set up scorers once at startup
client := ragllm.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-haiku-4-5-20251001")

scorers := []eval.Scorer{
    heuristic.NewLengthScorer(),
    heuristic.NewRefusalScorer(),
    overlap.NewF1Scorer(),
    overlap.NewBLEUScorer(),
    llm.NewFaithfulnessScorer(client),
    llm.NewHallucinationScorer(client),
}

// in your RAG handler — score async so user response is not blocked
go func() {
    c := eval.EvalCase{
        ID:              requestID,
        Query:           userQuery,
        RetrievedChunks: chunks,
        Answer:          answer,
        Model:           "claude-sonnet-4-6",
        Metadata:        map[string]string{"latency_ms": "342"},
    }
    runner.Evaluate(ctx, c, scorers)
}()
```

---

## CI/CD quality gate

Add ragcheck to GitHub Actions to block deploys when quality drops:

```yaml
- name: Install ragcheck
  run: go install github.com/yourusername/ragcheck/cmd/ragcheck@latest

- name: Run quality gate
  run: ragcheck run --input testdata/golden.json --fail-below 0.75
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## Compare models

Run the same test cases through different models and compare:

```bash
ragcheck run --input cases.json --model claude-haiku-4-5-20251001
ragcheck run --input cases.json --model claude-sonnet-4-6
ragcheck compare --run-a 1 --run-b 2
```

Output:

```
metric              haiku     sonnet    delta
faithfulness        0.81      0.94      +0.13
relevancy           0.88      0.87      -0.01
bleu_score          0.71      0.79      +0.08
chunk_utilisation   0.60      0.62      +0.02
latency_ms          1.00      0.50      -0.50
```

---

## Project layout

```
ragcheck/
├── cmd/ragcheck/            binary entrypoint
├── internal/
│   ├── cli/                 cobra commands
│   ├── eval/
│   │   ├── types.go         EvalCase, Score, Scorer interface
│   │   ├── heuristic/       length, refusal, coverage, utilisation, latency
│   │   ├── overlap/         F1, ROUGE-L, BLEU, conciseness
│   │   ├── embedding/       answer similarity, chunk relevance
│   │   └── llm/             faithfulness, relevancy, recall, hallucination,
│   │                        completeness, coherence, toxicity
│   ├── llm/                 Anthropic client wrapper
│   ├── store/               SQLite persistence
│   └── report/              terminal tables, JSON export
└── testdata/                example eval cases
```

---

## Requirements

- Go 1.22+
- Anthropic API key (only required for LLM judge and embedding scorers)
- Heuristic and overlap scorers work fully offline with no API key

---

## License

MIT
