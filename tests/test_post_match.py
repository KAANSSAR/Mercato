from mercato.jobs.post_match import finished_tracked


def fx(fid, league, status):
    return {"fixture": {"id": fid, "status": {"short": status}}, "league": {"id": league}}


def test_only_new_finished_tracked_matches():
    fixtures = [
        fx(1, 39, "FT"),    # EPL, finished: pull
        fx(2, 39, "2H"),    # in play: skip
        fx(3, 999, "FT"),   # untracked league: skip
        fx(4, 2, "PEN"),    # UCL on penalties: pull
        fx(5, 140, "FT"),   # already pulled: skip
    ]
    got = [f["fixture"]["id"] for f in finished_tracked(fixtures, already_pulled={5})]
    assert got == [1, 4]
