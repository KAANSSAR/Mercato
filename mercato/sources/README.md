# Sources

| Module | Source | Cadence | Status |
|---|---|---|---|
| `api_football.py` | API-Football | After every game | Client done |
| `understat.py` | Understat via soccerdata | Nightly | To do |
| `tm_datasets.py` | dcaribou/transfermarkt-datasets (CSVs in `data/tm/`) | Weekly | Done |
| `clubelo.py` | ClubElo via soccerdata | Daily | To do |
| `transfermarkt.py` | Transfermarkt, one player on demand, 7-day cache | On demand | To do |

## transfermarkt-datasets

Download from Kaggle (`davidcariboo/player-scores`) and put `players`, `clubs`, `transfers` and `player_valuations` CSVs in `data/tm/`. That folder is gitignored.
