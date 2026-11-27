import pytest

from mercato.mapping.ids import age_on, player_score
from mercato.mapping.match import assign
from mercato.mapping.names import club_similarity, player_similarity, position_compatible


@pytest.mark.parametrize("tm,api", [
    ("Arsenal FC", "Arsenal"),
    ("Manchester City", "Manchester City"),
    ("Bayern Munich", "Bayern München"),
    ("FC Barcelona", "Barcelona"),
    ("Inter Milan", "Inter"),
    ("Paris Saint-Germain", "Paris Saint Germain"),
    ("Tottenham Hotspur", "Tottenham"),
    ("1.FC Köln", "1. FC Köln"),
    ("Wolverhampton Wanderers", "Wolves"),
    ("Brighton & Hove Albion", "Brighton"),
    ("Borussia Mönchengladbach", "Borussia Monchengladbach"),
    ("Atlético de Madrid", "Atletico Madrid"),
    ("Nott'm Forest", "Nottingham Forest"),
    ("Stade Rennais", "Rennes"),
])
def test_club_names_match(tm, api):
    assert club_similarity(tm, api) >= 85


def test_club_assignment_is_one_to_one():
    tm = [{"name": "Manchester City"}, {"name": "Manchester United"}]
    api = [{"name": "Manchester United"}, {"name": "Manchester City"}]
    out = assign(tm, api, lambda a, b: club_similarity(a["name"], b["name"]))
    assert [m.right["name"] for m in out] == ["Manchester City", "Manchester United"]


@pytest.mark.parametrize("api,tm,ok", [
    ("B. Saka", "Bukayo Saka", True),
    ("Kepa", "Kepa Arrizabalaga", True),
    ("Vinícius Júnior", "Vinicius Junior", True),
    ("M. Ødegaard", "Martin Ødegaard", True),
    ("G. Martinelli", "Gabriel Martinelli", True),
    ("B. Saka", "Bryan Mbeumo", False),
    ("J. Timber", "Jurriën Timber", True),
    ("J. Timber", "Quinten Timber", False),
    ("M. Đurić", "Milan Djuric", True),
    ("M. Bokele Mputu", "Morgan Bokele", True),
    ("C. Mawissa Elebi", "Christian Mawissa", True),
    ("E. Scott", "Scott Dann", False),
    ("A. Dieng", "Bamba Dieng", False),
    ("Joël Monteiro", "Javi Montero", False),
])
def test_player_names(api, tm, ok):
    assert (player_similarity(api, tm) >= 85) == ok


def test_age_and_position_break_ties():
    api = {"name": "J. Timber", "age": 25, "position": "Defender"}
    jurrien = {"name": "Jurriën Timber", "age": 25, "group": "CB"}
    namesake = {"name": "J. Timber", "age": 31, "group": "CM"}
    assert player_score(jurrien, api) > player_score(namesake, api)


def test_position_compatible():
    assert position_compatible("Defender", "FB")
    assert position_compatible("Attacker", "W")
    assert not position_compatible("Goalkeeper", "ST")
    assert position_compatible(None, "ST")


def test_squad_assignment_flags_uncertain():
    api = [{"name": "B. Saka", "age": 25, "position": "Attacker"},
           {"name": "Nobody Here", "age": 19, "position": "Midfielder"}]
    tm = [{"name": "Bukayo Saka", "age": 25, "group": "W"},
          {"name": "Declan Rice", "age": 27, "group": "CM"}]
    out = assign(api, tm, lambda a, t: player_score(t, a))
    assert out[0].status == "auto" and out[0].right["name"] == "Bukayo Saka"
    assert out[1].status == "unmatched"


def test_age_on():
    import datetime as dt
    assert age_on(dt.date(2001, 9, 5), dt.date(2026, 9, 4)) == 24
    assert age_on(dt.date(2001, 9, 5), dt.date(2026, 9, 5)) == 25
    assert age_on(None, dt.date(2026, 1, 1)) is None


def test_same_club_different_surname_passes_within_club_bar():
    from mercato.mapping.ids import WITHIN_CLUB_AUTO
    api = {"name": "Álvaro Fernández", "age": 23, "position": "Defender"}
    tm = {"name": "Álvaro Carreras", "age": 23, "group": "FB"}
    assert player_score(tm, api) >= WITHIN_CLUB_AUTO


def test_nicknames_pass_within_a_club_only():
    from mercato.mapping.ids import WITHIN_CLUB_AUTO
    api = {"name": "T. Douvikas", "age": 27, "position": "Attacker"}
    tm = {"name": "Anastasios Douvikas", "age": 27, "group": "ST"}
    assert player_score(tm, api, same_club=True) >= WITHIN_CLUB_AUTO
    assert player_score(tm, api) < WITHIN_CLUB_AUTO
    # brothers in one squad: the initial still decides
    jt = {"name": "Jurriën Timber", "age": 25, "group": "CB"}
    qt = {"name": "Quinten Timber", "age": 25, "group": "CM"}
    api_j = {"name": "J. Timber", "age": 25, "position": "Defender"}
    assert player_score(jt, api_j, same_club=True) > player_score(qt, api_j, same_club=True)


def test_single_name_spelling():
    api = {"name": "Brugui", "age": 29, "position": "Midfielder"}
    tm = {"name": "Roger Brugué", "age": 29, "group": "W"}
    from mercato.mapping.ids import WITHIN_CLUB_AUTO
    assert player_score(tm, api, same_club=True) >= WITHIN_CLUB_AUTO


def test_common_surname_found_in_wide_shortlist():
    from mercato.mapping.ids import global_candidates
    from mercato.mapping.names import player_key
    pool = [{"name": f"{first} Jackson"} for first in
            ["Ben", "Brad", "Louis", "Jackson", "Cameron", "Adam", "Aaron", "Bob", "Carl", "Dan",
             "Ed", "Fred", "Greg", "Hal", "Ian", "Jack", "Kyle", "Leo", "Mo"]]
    pool.append({"name": "Nicolas Jackson"})
    found = global_candidates({"name": "N. Jackson"}, pool, [player_key(p["name"]) for p in pool])
    assert {"name": "Nicolas Jackson"} in found
