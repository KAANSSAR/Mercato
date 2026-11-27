"""Tracks whether live updates can run, so the site can hide live features cleanly.

Free plan: no current-season access, so live is off.
Pro or higher and active: live is on.
A PlanRestricted error from the API also switches live off, whatever the plan says.
"""

from mercato.config import api_seasons

FREE_REASON = "Live match updates are paused. Stats are shown as of the last update."
INACTIVE_REASON = "Live match updates are paused while the data subscription is inactive."


def evaluate(status: dict) -> dict:
    """Turn an API-Football /status response into a service_status row."""
    sub = status.get("subscription") or {}
    req = status.get("requests") or {}
    plan = str(sub.get("plan") or "unknown")
    active = bool(sub.get("active"))
    is_free = plan.lower() == "free"
    live = active and not is_free
    reason = None if live else (FREE_REASON if active else INACTIVE_REASON)
    return {
        "plan": plan,
        "live_enabled": live,
        "seasons": list(api_seasons(plan)),
        "reason": reason,
        "requests_today": req.get("current"),
        "request_limit": req.get("limit_day"),
    }


def transition(prev_live: bool | None, live: bool) -> str | None:
    if prev_live is None or prev_live == live:
        return None
    return "upgrade" if live else "downgrade"


def save(conn, row: dict) -> str | None:
    """Upsert the single status row. Returns 'upgrade', 'downgrade' or None."""
    with conn.cursor() as cur:
        cur.execute("select live_enabled from service_status where id = 1")
        prev = cur.fetchone()
        change = transition(prev[0] if prev else None, row["live_enabled"])
        cur.execute(
            """
            insert into service_status (id, plan, live_enabled, seasons, reason,
                                        requests_today, request_limit, checked_at, changed_at)
            values (1, %(plan)s, %(live_enabled)s, %(seasons)s, %(reason)s,
                    %(requests_today)s, %(request_limit)s, now(), now())
            on conflict (id) do update set
                plan = excluded.plan,
                live_enabled = excluded.live_enabled,
                seasons = excluded.seasons,
                reason = excluded.reason,
                requests_today = excluded.requests_today,
                request_limit = excluded.request_limit,
                checked_at = now(),
                changed_at = case when service_status.live_enabled <> excluded.live_enabled
                                  then now() else service_status.changed_at end
            """,
            row,
        )
    conn.commit()
    return change


def check(conn, api) -> tuple[dict, str | None]:
    row = evaluate(api.status())
    return row, save(conn, row)


def mark_restricted(conn, detail: str) -> str | None:
    """Called when the API refuses a request on plan grounds."""
    with conn.cursor() as cur:
        cur.execute("select plan, seasons, requests_today, request_limit "
                    "from service_status where id = 1")
        r = cur.fetchone()
    row = {
        "plan": r[0] if r else "unknown",
        "live_enabled": False,
        "seasons": r[1] if r else [],
        "reason": FREE_REASON,
        "requests_today": r[2] if r else None,
        "request_limit": r[3] if r else None,
    }
    print(f"Plan restriction: {detail}")
    return save(conn, row)


def mark_live_update(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("update service_status set last_live_update = now() where id = 1")
    conn.commit()


def announce(change: str | None, row: dict) -> bool:
    """Print GitHub Actions annotations. Returns True if the run should fail to send an alert."""
    if change == "downgrade":
        print(f"::error title=Mercato live updates OFF::Plan is {row['plan']}. "
              "Site now shows the paused banner and hides live features.")
        return True
    if change == "upgrade":
        print(f"::notice title=Mercato live updates ON::Plan is {row['plan']}. Live features restored.")
    return False


if __name__ == "__main__":
    import sys

    from mercato.db import connect
    from mercato.sources.api_football import ApiFootball

    with connect() as conn:
        row, change = check(conn, ApiFootball())
    print(row)
    sys.exit(1 if announce(change, row) else 0)
