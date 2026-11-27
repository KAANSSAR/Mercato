# Mercato

What is a footballer actually worth?

A live football stats platform and transfer valuation engine for Europe's top five leagues, the Champions League and the Europa League. Every player gets a model-driven fair value with a P10 to P90 range and SHAP drivers, benchmarked against Transfermarkt.

Live at [mercato.kanishka-sarkar.com](https://mercato.kanishka-sarkar.com) (from 31 Dec 2026).

## How it works

```
API-Football ─┐
Understat ────┤
TM-datasets ──┼─► GitHub Actions jobs ─► Supabase (Postgres) ─► Next.js on Vercel ─► Visitors
ClubElo ──────┘
```

- Jobs poll for finished matches every 15 to 30 minutes on matchdays and pull stats for those games only.
- Visitors only ever read the database. They never trigger an API call.
- Valuations are recomputed nightly.

## Methodology

raw stats → per 90 with shrinkage → opponent-adjusted (ClubElo) → position percentiles → XGBoost on inflation-adjusted log fees

Real past fees train the model. Mercato sets the value. Transfermarkt is only the benchmark.

## Layout

| Path | What |
|---|---|
| `mercato/sources/` | One client per data source |
| `mercato/jobs/` | Scheduled jobs (post-match, backfill, nightly) |
| `mercato/features/` | Per 90, shrinkage, PAdj, Elo weighting, form, fee index |
| `mercato/model/` | Valuation model |
| `mercato/scouting/` | Twin finder |
| `mercato/api/` | FastAPI, on-demand endpoints only |
| `db/schema.sql` | Postgres schema |
| `web/` | Next.js frontend |

## Dev

```bash
uv sync --extra dev
cp .env.example .env
uv run pytest
uv run uvicorn mercato.api.main:app --reload
```

## Data credits

API-Football, Understat, ClubElo, and [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (CC0). Not affiliated with any club, league or Transfermarkt.
