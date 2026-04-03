package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Shubham-Jitendra-Bhadra/ragcheck/internal/eval"
	_ "modernc.org/sqlite"
)

// Store handles all SQLite persistence for ragcheck.
type Store struct {
	db *sql.DB
}

// Run represents one ragcheck eval run.
type Run struct {
	ID         int64
	CreatedAt  time.Time
	InputFile  string
	Model      string
	TotalCases int
}

// ScoreRow is one scorer result within a comparison.
type ScoreRow struct {
	CaseID    string
	Metric    string
	ValueA    float64
	ValueB    float64
	Delta     float64
	ReasoningA string
	ReasoningB string
}
// ScoreDetail is one scorer result within a ResultRow.
type ScoreDetail struct {
	Metric    string  `json:"metric"`
	Value     float64 `json:"value"`
	Reasoning string  `json:"reasoning"`
	Error     string  `json:"error,omitempty"`
}

// ResultRow is a flat representation of one EvalCase result
// with all its scores — used for reporting.
type ResultRow struct {
	CaseID    string            `json:"case_id"`
	Model     string            `json:"model"`
	HasErrors bool              `json:"has_errors"`
	CreatedAt time.Time         `json:"created_at"`
	Metadata  map[string]string `json:"metadata"`
	Scores    []ScoreDetail     `json:"scores"`
}

// New opens or creates a SQLite database at the given path.
func New(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	s := &Store{db: db}
	if err := s.init(); err != nil {
		return nil, fmt.Errorf("init schema: %w", err)
	}

	return s, nil
}

// init creates all tables if they do not exist yet.
func (s *Store) init() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS runs (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at  DATETIME NOT NULL,
			input_file  TEXT NOT NULL,
			model       TEXT NOT NULL,
			total_cases INTEGER NOT NULL
		);

		CREATE TABLE IF NOT EXISTS results (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			run_id     INTEGER NOT NULL REFERENCES runs(id),
			case_id    TEXT NOT NULL,
			model      TEXT,
			has_errors INTEGER NOT NULL DEFAULT 0,
			created_at DATETIME NOT NULL,
			metadata   TEXT
		);

		CREATE TABLE IF NOT EXISTS scores (
			id        INTEGER PRIMARY KEY AUTOINCREMENT,
			result_id INTEGER NOT NULL REFERENCES results(id),
			metric    TEXT NOT NULL,
			value     REAL NOT NULL,
			reasoning TEXT,
			error     TEXT
		);
	`)
	return err
}

// CreateRun inserts a new run row and returns the run ID.
func (s *Store) CreateRun(ctx context.Context, inputFile, model string, totalCases int) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`INSERT INTO runs (created_at, input_file, model, total_cases) VALUES (?, ?, ?, ?)`,
		time.Now(), inputFile, model, totalCases,
	)
	if err != nil {
		return 0, fmt.Errorf("create run: %w", err)
	}
	return res.LastInsertId()
}

// SaveResult saves one EvalCase result and all its scores in a transaction.
func (s *Store) SaveResult(ctx context.Context, result eval.Result) error {
	meta, _ := json.Marshal(result.Metadata)

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	res, err := tx.ExecContext(ctx,
		`INSERT INTO results (run_id, case_id, model, has_errors, created_at, metadata)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		result.RunID,
		result.CaseID,
		result.Model,
		result.HasErrors,
		result.CreatedAt,
		string(meta),
	)
	if err != nil {
		return fmt.Errorf("insert result: %w", err)
	}

	resultID, err := res.LastInsertId()
	if err != nil {
		return fmt.Errorf("get result id: %w", err)
	}

	for _, score := range result.Scores {
		errMsg := ""
		if score.Error != nil {
			errMsg = score.Error.Error()
		}
		_, err = tx.ExecContext(ctx,
			`INSERT INTO scores (result_id, metric, value, reasoning, error)
			 VALUES (?, ?, ?, ?, ?)`,
			resultID,
			score.Metric,
			score.Value,
			score.Reasoning,
			errMsg,
		)
		if err != nil {
			return fmt.Errorf("insert score: %w", err)
		}
	}

	return tx.Commit()
}

// GetRun fetches a single run by ID.
func (s *Store) GetRun(ctx context.Context, runID int64) (*Run, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, created_at, input_file, model, total_cases FROM runs WHERE id = ?`,
		runID,
	)

	var r Run
	if err := row.Scan(&r.ID, &r.CreatedAt, &r.InputFile, &r.Model, &r.TotalCases); err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("run %d not found", runID)
		}
		return nil, fmt.Errorf("get run: %w", err)
	}

	return &r, nil
}

// ListRuns returns all runs ordered by most recent first.
func (s *Store) ListRuns(ctx context.Context) ([]Run, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, created_at, input_file, model, total_cases
		 FROM runs ORDER BY created_at DESC`,
	)
	if err != nil {
		return nil, fmt.Errorf("list runs: %w", err)
	}
	defer rows.Close()

	var runs []Run
	for rows.Next() {
		var r Run
		if err := rows.Scan(&r.ID, &r.CreatedAt, &r.InputFile, &r.Model, &r.TotalCases); err != nil {
			return nil, fmt.Errorf("scan run: %w", err)
		}
		runs = append(runs, r)
	}

	return runs, nil
}

// CompareRuns fetches scores for two runs and computes deltas.
func (s *Store) CompareRuns(ctx context.Context, runAID, runBID int64) ([]ScoreRow, error) {
	query := `
		SELECT
			r.case_id,
			s.metric,
			s.value,
			s.reasoning,
			r.run_id
		FROM results r
		JOIN scores s ON s.result_id = r.id
		WHERE r.run_id IN (?, ?)
		ORDER BY r.case_id, s.metric, r.run_id
	`

	rows, err := s.db.QueryContext(ctx, query, runAID, runBID)
	if err != nil {
		return nil, fmt.Errorf("compare runs query: %w", err)
	}
	defer rows.Close()

	// group by case_id + metric
	type key struct{ caseID, metric string }
	type pair struct {
		valueA, valueB         float64
		reasoningA, reasoningB string
	}
	pairs := make(map[key]*pair)

	for rows.Next() {
		var caseID, metric, reasoning string
		var value float64
		var runID int64

		if err := rows.Scan(&caseID, &metric, &value, &reasoning, &runID); err != nil {
			return nil, fmt.Errorf("scan comparison row: %w", err)
		}

		k := key{caseID, metric}
		if pairs[k] == nil {
			pairs[k] = &pair{}
		}

		if runID == runAID {
			pairs[k].valueA = value
			pairs[k].reasoningA = reasoning
		} else {
			pairs[k].valueB = value
			pairs[k].reasoningB = reasoning
		}
	}

	var result []ScoreRow
	for k, p := range pairs {
		result = append(result, ScoreRow{
			CaseID:     k.caseID,
			Metric:     k.metric,
			ValueA:     p.valueA,
			ValueB:     p.valueB,
			Delta:      p.valueB - p.valueA,
			ReasoningA: p.reasoningA,
			ReasoningB: p.reasoningB,
		})
	}

	return result, nil
}
// GetResults fetches all results and their scores for a given run.
func (s *Store) GetResults(ctx context.Context, runID int64) ([]ResultRow, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT
			r.case_id,
			r.model,
			r.has_errors,
			r.created_at,
			r.metadata,
			s.metric,
			s.value,
			s.reasoning,
			s.error
		FROM results r
		JOIN scores s ON s.result_id = r.id
		WHERE r.run_id = ?
		ORDER BY r.case_id, s.metric
	`, runID)
	if err != nil {
		return nil, fmt.Errorf("query results: %w", err)
	}
	defer rows.Close()

	resultMap := make(map[string]*ResultRow)
	var order []string

	for rows.Next() {
		var (
			caseID    string
			model     string
			hasErrors bool
			createdAt time.Time
			metaRaw   string
			metric    string
			value     float64
			reasoning string
			scoreErr  string
		)

		if err := rows.Scan(
			&caseID, &model, &hasErrors, &createdAt,
			&metaRaw, &metric, &value, &reasoning, &scoreErr,
		); err != nil {
			return nil, fmt.Errorf("scan row: %w", err)
		}

		if _, exists := resultMap[caseID]; !exists {
			var meta map[string]string
			json.Unmarshal([]byte(metaRaw), &meta)

			resultMap[caseID] = &ResultRow{
				CaseID:    caseID,
				Model:     model,
				HasErrors: hasErrors,
				CreatedAt: createdAt,
				Metadata:  meta,
				Scores:    []ScoreDetail{},
			}
			order = append(order, caseID)
		}

		resultMap[caseID].Scores = append(resultMap[caseID].Scores, ScoreDetail{
			Metric:    metric,
			Value:     value,
			Reasoning: reasoning,
			Error:     scoreErr,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate rows: %w", err)
	}

	results := make([]ResultRow, 0, len(order))
	for _, id := range order {
		results = append(results, *resultMap[id])
	}

	return results, nil
}

// Close closes the database connection.
func (s *Store) Close() error {
	return s.db.Close()
}