import pandas as pd

from mercato.sources.tm_datasets import (
    coverage_report,
    paid_moves,
    position_group,
    relevant_players,
)


def test_position_group():
    assert position_group("Centre-Back", "Defender") == "CB"
    assert position_group("Right Winger", "Attack") == "W"
    assert position_group(float("nan"), "Midfield") == "CM"
    assert position_group(None, None) is None


PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4],
    "name": ["A", "B", "C", "D"],
    "current_club_id": [10, 99, 99, 10],
    "last_season": [2025, 2025, 2025, 2012],
    "sub_position": ["Centre-Forward", "Goalkeeper", "Left-Back", None],
    "position": ["Attack", "Goalkeeper", "Defender", "Attack"],
    "contract_expiration_date": ["2028-06-30", None, None, None],
    "date_of_birth": ["2000-01-01"] * 4,
})
CLUBS = pd.DataFrame({"club_id": [10, 99, 11], "name": ["Top", "Other", "Relegated"],
                      "domestic_competition_id": ["GB1", "NL1", "GB1"],
                      "last_season": [2025, 2025, 2022]})
TRANSFERS = pd.DataFrame({
    "player_id": [2, 3, 1],
    "transfer_date": ["2019-07-01", "2012-07-01", "2022-07-01"],
    "from_club_id": [10, 10, 99],
    "to_club_id": [99, 99, 10],
    "transfer_fee": [5e6, 1e6, 0],
})


def test_relevant_players():
    rel = relevant_players(PLAYERS, TRANSFERS, {10})
    # 1 current top 5; 2 moved out of a top-5 club in 2019; 3 active elsewhere this season
    # (a possible summer signing); 4 retired in 2012
    assert sorted(rel["player_id"]) == [1, 2, 3]


def test_paid_moves_needs_fee_and_top5_buyer():
    assert paid_moves(TRANSFERS, {10}).empty  # only move into club 10 was free
    assert len(paid_moves(TRANSFERS, {99})) == 1  # 2019 move, 2012 one is too early


def test_coverage_report(tmp_path):
    PLAYERS.to_csv(tmp_path / "players.csv", index=False)
    CLUBS.to_csv(tmp_path / "clubs.csv", index=False)
    TRANSFERS.to_csv(tmp_path / "transfers.csv", index=False)
    r = coverage_report(tmp_path)
    assert r["top5_clubs_ever"] == 2
    assert r["current_top5_clubs"] == 1
    assert r["current_top5_players"] == 1  # player 4 retired, player 2 at a non-top-5 club
    assert r["position_group_coverage"] == 1.0


def test_current_top5():
    from mercato.sources.tm_datasets import current_top5_club_ids, current_top5_player_mask
    assert current_top5_club_ids(CLUBS) == {10}
    assert list(current_top5_player_mask(PLAYERS, CLUBS)) == [True, False, False, False]
