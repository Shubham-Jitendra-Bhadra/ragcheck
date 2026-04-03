package eval

import (
	"context"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
)

// Progress reports how many cases have been evaluated so far.
type Progress struct {
	Done  int
	Total int
	Error error
}

// Runner executes all scorers against all cases concurrently.
type Runner struct {
	Scorers     []Scorer
	Concurrency int // max simultaneous API calls, default 10
}

func NewRunner(scorers []Scorer) *Runner {
	return &Runner{
		Scorers:     scorers,
		Concurrency: 10,
	}
}

// Run evaluates all cases against all scorers concurrently.
// Progress updates are sent to the progress channel as cases complete.
// Returns all results once every case and scorer has finished.
func (r *Runner) Run(ctx context.Context, cases []EvalCase, progress chan<- Progress) ([]Result, error) {
	total := len(cases)
	results := make([]Result, total)
	sem := make(chan struct{}, r.Concurrency)

	g, ctx := errgroup.WithContext(ctx)
	var mu sync.Mutex
	done := 0

	for i, c := range cases {
		i, c := i, c

		g.Go(func() error {
			// acquire semaphore slot
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return ctx.Err()
			}
			defer func() { <-sem }() // release on exit

			// run all scorers concurrently for this case
			scores, err := r.scoreCase(ctx, c)
			if err != nil {
				return err
			}

			hasErrors := false
			for _, s := range scores {
				if s.Error != nil {
					hasErrors = true
					break
				}
			}

			results[i] = Result{
				CaseID:    c.ID,
				Model:     c.Model,
				Scores:    scores,
				Metadata:  c.Metadata,
				CreatedAt: time.Now(),
				HasErrors: hasErrors,
			}

			// report progress
			mu.Lock()
			done++
			if progress != nil {
				progress <- Progress{Done: done, Total: total}
			}
			mu.Unlock()

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return results, nil
}

// scoreCase runs all scorers against a single EvalCase concurrently.
// Scorer errors are stored in Score.Error — they do not fail the whole run.
func (r *Runner) scoreCase(ctx context.Context, c EvalCase) ([]Score, error) {
	scores := make([]Score, len(r.Scorers))
	var wg sync.WaitGroup

	for i, scorer := range r.Scorers {
		i, scorer := i, scorer // capture loop variables

		wg.Add(1)
		go func() {
			defer wg.Done()

			score, err := scorer.Score(ctx, c)
			if err != nil {
				// store error in score but don't fail the run
				scores[i] = Score{
					Metric:    scorer.Name(),
					Value:     0.0,
					Reasoning: "scorer error: " + err.Error(),
					Error:     err,
				}
				return
			}
			scores[i] = score
		}()
	}

	wg.Wait()
	return scores, nil
}
