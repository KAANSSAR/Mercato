"""Fold placeholder records (created during ingestion for unknown ids) into real ones."""


def merge_club(cur, from_id: int, into_id: int) -> None:
    """Move everything pointing at club `from_id` onto `into_id`, then delete `from_id`."""
    if from_id == into_id:
        return
    cur.execute("select api_football_id from clubs where id = %s", (from_id,))
    row = cur.fetchone()
    api_id = row[0] if row else None
    for table, col in [("matches", "home_club_id"), ("matches", "away_club_id"),
                       ("player_match_stats", "club_id"), ("players", "club_id"),
                       ("transfers", "from_club_id"), ("transfers", "to_club_id")]:
        cur.execute(f"update {table} set {col} = %s where {col} = %s", (into_id, from_id))
    # tables keyed on club: drop rows that would collide, then move the rest
    for table, key in [("team_match_stats", "match_id"),
                       ("standings", "competition_id, season_id, as_of"),
                       ("competition_entries", "competition_id, season_id")]:
        cur.execute(f"""delete from {table} t where club_id = %s and exists (
                            select 1 from {table} u where u.club_id = %s
                            and ({key}) = ({', '.join('t.' + k.strip() for k in key.split(','))}))""",
                    (from_id, into_id))
        cur.execute(f"update {table} set club_id = %s where club_id = %s", (into_id, from_id))
    cur.execute("delete from clubs where id = %s", (from_id,))
    if api_id is not None:
        cur.execute("update clubs set api_football_id = %s where id = %s "
                    "and api_football_id is null", (api_id, into_id))


def merge_player(cur, from_mid: int, into_mid: int) -> None:
    """Move a placeholder player's stats onto the real record, then delete the placeholder."""
    if from_mid == into_mid:
        return
    cur.execute("""delete from player_match_stats p where mercato_id = %s and exists (
                       select 1 from player_match_stats q
                       where q.mercato_id = %s and q.match_id = p.match_id)""",
                (from_mid, into_mid))
    cur.execute("update player_match_stats set mercato_id = %s where mercato_id = %s",
                (into_mid, from_mid))
    cur.execute("update api_player_season set mercato_id = %s where mercato_id = %s",
                (into_mid, from_mid))
    cur.execute("delete from id_map where mercato_id = %s", (from_mid,))
    cur.execute("delete from players where mercato_id = %s", (from_mid,))


def set_player_api_id(cur, mercato_id: int, api_id: int, method: str, score: float) -> str:
    """Point api_id at mercato_id. A placeholder holding it is merged in; a real record
    holding it gives it up (the caller is the more confident source). Returns what happened."""
    cur.execute("select i.mercato_id, i.match_method from id_map i where i.api_football_id = %s",
                (api_id,))
    holder = cur.fetchone()
    outcome = "set"
    if holder and holder[0] != mercato_id:
        if holder[1] == "api_only":
            merge_player(cur, holder[0], mercato_id)
            outcome = "merged"
        else:
            cur.execute("update id_map set api_football_id = null where mercato_id = %s",
                        (holder[0],))
            outcome = "moved"
    elif holder and holder[0] == mercato_id:
        return "unchanged"
    cur.execute("update id_map set api_football_id = %s, match_method = %s, match_score = %s "
                "where mercato_id = %s", (api_id, method, min(score, 100.0), mercato_id))
    return outcome
