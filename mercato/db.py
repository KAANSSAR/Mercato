"""Postgres helpers: connection, bulk upserts, job run logging."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import psycopg

from mercato.config import settings


def connect() -> psycopg.Connection:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(settings.database_url)


def upsert(
    conn: psycopg.Connection,
    table: str,
    rows: Iterable[dict],
    conflict: list[str],
    update: list[str] | None = None,
) -> int:
    """Insert rows; on conflict update the given columns (or do nothing). Returns rows sent."""
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0])
    col_sql = ", ".join(cols)
    ph = ", ".join(f"%({c})s" for c in cols)
    if update:
        action = "do update set " + ", ".join(f"{c} = excluded.{c}" for c in update)
    else:
        action = "do nothing"
    sql = (
        f"insert into {table} ({col_sql}) values ({ph}) "
        f"on conflict ({', '.join(conflict)}) {action}"
    )
    # Pipeline mode batches round trips, which matters over the network to Supabase
    with conn.pipeline(), conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


@contextmanager
def job_run(conn: psycopg.Connection, job: str) -> Iterator[dict]:
    """Logs a row in job_runs. Mutate the yielded dict to record requests and rows."""
    with conn.cursor() as cur:
        cur.execute("insert into job_runs (job) values (%s) returning id", (job,))
        run_id = cur.fetchone()[0]
    conn.commit()
    stats = {"requests": 0, "rows_written": 0}
    try:
        yield stats
    except Exception as e:
        conn.rollback()
        _finish(conn, run_id, stats, ok=False, error=repr(e)[:2000])
        raise
    else:
        conn.commit()
        _finish(conn, run_id, stats, ok=True, error=None)


def _finish(conn, run_id, stats, ok, error):
    with conn.cursor() as cur:
        cur.execute(
            "update job_runs set finished_at = now(), requests = %s, rows_written = %s, "
            "ok = %s, error = %s where id = %s",
            (stats["requests"], stats["rows_written"], ok, error, run_id),
        )
    conn.commit()
