"""Global Football Rankings parsing and matching (rows as captured from the live pages)."""

import pandas as pd

LEAGUE_ROWS = [
    {"href": "https://globalfootballrankings.com/rankings/leagues/men/england/premier-league",
     "text": "1 Premier League 91.0"},
    {"href": "https://globalfootballrankings.com/rankings/leagues/men/germany/bundesliga",
     "text": "2 Bundesliga 86.3"},
    {"href": "https://globalfootballrankings.com/rankings/leagues/men/portugal/liga-portugal",
     "text": "8 Liga Portugal 81.8"},
    {"href": "https://globalfootballrankings.com/rankings/leagues/men/england/championship",
     "text": "10 Championship 81.1"},
    {"href": "https://globalfootballrankings.com/rankings/leagues/men/turkiye/super-lig",
     "text": "19 Super Lig 79.5"},
    {"href": "https://globalfootballrankings.com/rankings/leagues/men/germany/2-bundesliga",
     "text": "24 2. Bundesliga 78.8"},
    {"href": "https://globalfootballrankings.com/rankings/leagues/men/brazil/serie-a",
     "text": "6 Serie A 84.0"},
    {"href": "https://example.com/broken", "text": None},
]
CLUB_ROWS = [
    {"href": "https://globalfootballrankings.com/rankings/teams/men/france/psg",
     "text": "2 PSG 96.5"},
    {"href": "https://globalfootballrankings.com/rankings/teams/men/germany/bayern-munich",
     "text": "6 Bayern Munich 95.5"},
    {"href": "https://globalfootballrankings.com/rankings/teams/men/portugal/sporting-cp",
     "text": "32 Sporting CP 90.1"},
    {"href": "https://globalfootballrankings.com/rankings/teams/men/germany/koln",
     "text": "144 Koln 83.7"},
    {"href": "https://globalfootballrankings.com/rankings/teams/men/spain/sporting-gijon",
     "text": "527 Sporting Gijon 78.1"},
]
TM_CLUBS = pd.DataFrame({
    "club_id": [583, 27, 336, 3, 2448, 294],
    "name": ["Paris Saint-Germain Football Club", "FC Bayern München",
             "Sporting Clube de Portugal", "1. FC Köln", "Real Sporting de Gijón",
             "Sport Lisboa e Benfica"],
    "domestic_competition_id": ["FR1", "L1", "PO1", "L1", "ES2", "PO1"],
})


def test_parse_rows_and_updated():
    from mercato.sources.gfr import parse_rows, parse_updated
    lg = parse_rows(LEAGUE_ROWS)
    assert len(lg) == 7  # the broken row is skipped
    assert lg.iloc[0].to_dict() == {"rank": 1, "name": "Premier League", "rating": 91.0,
                                    "country": "england", "slug": "premier-league"}
    assert lg.loc[lg["slug"] == "2-bundesliga", "name"].item() == "2. Bundesliga"
    assert str(parse_updated("Premier League #1 · Updated Nov 27, 2026")) == "2026-11-27"


def test_top_flight_per_country_gets_our_league_id():
    from mercato.sources.gfr import map_leagues, parse_rows
    lg = map_leagues(parse_rows(LEAGUE_ROWS)).set_index("slug")
    assert lg.loc["premier-league", "tm_competition_id"] == "GB1"
    assert lg.loc["championship", "tm_competition_id"] is None
    assert lg.loc["liga-portugal", "tm_competition_id"] == "PO1"
    assert lg.loc["super-lig", "tm_competition_id"] == "TR1"
    assert lg.loc["2-bundesliga", "tm_competition_id"] is None


def test_clubs_match_within_country():
    from mercato.sources.gfr import map_clubs, parse_rows
    c = map_clubs(parse_rows(CLUB_ROWS), TM_CLUBS).set_index("slug")["tm_club_id"]
    assert c["psg"] == 583           # initials
    assert c["bayern-munich"] == 27  # München -> Munich
    assert c["sporting-cp"] == 336   # same country only: not Sporting Gijon
    assert c["koln"] == 3
    assert pd.isna(c["sporting-gijon"])  # Segunda: not in our countries' top flights


def test_overrides_win():
    from mercato.sources.gfr import map_clubs, parse_rows
    o = pd.DataFrame({"country": ["germany"], "slug": ["koln"], "tm_club_id": [999]})
    c = map_clubs(parse_rows(CLUB_ROWS), TM_CLUBS, o).set_index("slug")["tm_club_id"]
    assert c["koln"] == 999


def test_ambiguous_names_need_an_override():
    from mercato.sources.gfr import map_clubs, parse_rows
    braga = pd.DataFrame({"club_id": [1075], "name": ["Sporting Clube de Braga"],
                          "domestic_competition_id": ["PO1"]})
    tm = pd.concat([TM_CLUBS, braga], ignore_index=True)
    c = map_clubs(parse_rows(CLUB_ROWS), tm).set_index("slug")["tm_club_id"]
    assert pd.isna(c["sporting-cp"])  # Sporting CP or Sporting Braga: not guessed
    o = pd.read_csv("db/overrides/gfr_clubs.csv")
    c = map_clubs(parse_rows(CLUB_ROWS), tm, o).set_index("slug")["tm_club_id"]
    assert c["sporting-cp"] == 336


def test_parse_fifa():
    from mercato.sources.gfr import parse_fifa
    raw = [{"rank": "1", "team": "Spain", "points": "2004\n-", "conf": "UEFA logo"},
           {"rank": "7", "team": "Morocco", "points": "1791\n-18", "conf": "CAF logo"},
           {"rank": "211", "team": "San Marino", "points": "713\n-", "conf": None},
           {"rank": "", "team": "header", "points": "", "conf": None}]
    f = parse_fifa(raw)
    assert list(f["team"]) == ["Spain", "Morocco", "San Marino"]
    assert f.loc[1, "points"] == 1791.0 and f.loc[1, "confederation"] == "CAF"
    assert pd.isna(f.loc[2, "confederation"])
