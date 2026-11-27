# Sources

| Module | Source | Cadence | Status |
|---|---|---|---|
| `api_football.py` | API-Football | After every game | Client done |
| `understat.py` | Understat via soccerdata | Nightly | Parked: endpoints broken |
| `tm_datasets.py` | dcaribou/transfermarkt-datasets (CSVs in `data/tm/`) | Weekly | Done |
| `../features/elo.py` | Our own Elo from Transfermarkt games (ClubElo feed closed) | Weekly, then after each matchday | Done |
| `transfermarkt.py` | Transfermarkt, one player on demand, 7-day cache | On demand | To do |

## transfermarkt-datasets

Download from Kaggle (`davidcariboo/player-scores`) and put `players`, `clubs`, `transfers` and `player_valuations` CSVs in `data/tm/`. That folder is gitignored.
