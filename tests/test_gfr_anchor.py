"""Anchoring Mercato Elo to GFR league levels."""

import numpy as np
import pandas as pd


def _world():
    """Two leagues; our Elo puts league B level with A, GFR puts it 5 points lower."""
    rows, gid = [], 0
    clubs = {**{c: "A" for c in range(1, 11)}, **{c: "B" for c in range(11, 21)}}
    strength = {c: (c % 10) * 20 for c in clubs}
    for day in range(30):
        for lg, ids in (("A", range(1, 11)), ("B", range(11, 21))):
            ids = list(ids)
            for h, a in zip(ids[::2], ids[1::2]):
                gid += 1
                for club, opp in ((h, a), (a, h)):
                    rows.append({"game_id": gid, "date": pd.Timestamp("2026-01-01")
                                 + pd.Timedelta(days=day), "competition_id": lg,
                                 "tm_club_id": club, "opponent_tm_club_id": opp,
                                 "elo_pre": 1500 + strength[club],
                                 "opp_elo_pre": 1500 + strength[opp],
                                 "elo_post": 1500 + strength[club]})
    te = pd.DataFrame(rows)
    gfr_clubs = pd.DataFrame({"tm_club_id": list(clubs), "country": "x",
                              "slug": [str(c) for c in clubs],
                              "rating": [80 + strength[c] / 10 - (5 if clubs[c] == "B" else 0)
                                         for c in clubs]})
    gfr_leagues = pd.DataFrame({"tm_competition_id": ["A", "B"],
                                "rating": [gfr_clubs.rating[:10].mean(),
                                           gfr_clubs.rating[10:].mean()]})
    return te, gfr_clubs, gfr_leagues


def test_offsets_follow_gfr_league_gap():
    from mercato.features.gfr_anchor import fit_offsets
    te, gc, gl = _world()
    gc = pd.concat([gc] * 2, ignore_index=True)  # enough matched clubs
    fit = fit_offsets(te, gc, gl, {"A", "B"})
    assert fit["scale"] == 10.0  # 1 GFR point = 10 of our Elo points within a league
    # B moves 50 Elo below A
    assert fit["offsets"]["A"] - fit["offsets"]["B"] == 50.0


def test_anchor_shifts_history_by_league_at_the_time():
    from mercato.features.gfr_anchor import anchor
    te, _, _ = _world()
    out = anchor(te, {"A": 10.0, "B": -40.0}, {"A", "B"})
    b = out["competition_id"] == "B"
    assert np.allclose(out.loc[b, "elo_pre"] - te.loc[b, "elo_pre"], -40)
    assert np.allclose(out.loc[b, "opp_elo_pre"] - te.loc[b, "opp_elo_pre"], -40)
    assert np.allclose(out.loc[~b, "elo_post"] - te.loc[~b, "elo_post"], 10)


def test_promoted_club_takes_its_new_league_offset():
    from mercato.features.gfr_anchor import club_league
    te = pd.DataFrame({"game_id": [1, 2, 3], "tm_club_id": [7, 7, 7],
                       "date": ["2024-01-01", "2024-09-01", "2024-10-01"],
                       "competition_id": ["B", "A", "CUP"]})
    assert list(club_league(te, {"A", "B"})) == ["B", "A", "A"]


def test_no_snapshot_leaves_elo_alone(tmp_path):
    from mercato.features.gfr_anchor import anchored
    te, _, _ = _world()
    clubs = pd.DataFrame({"club_id": [1], "domestic_competition_id": ["A"]})
    out, fit = anchored(te, clubs, data_dir=tmp_path)
    assert out is te and fit == {}
