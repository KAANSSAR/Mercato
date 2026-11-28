"""Player photos, club and competition crests, flags and the colours behind the colour layer.

    uv run python -m mercato.jobs.media            # new players and clubs only
    uv run python -m mercato.jobs.media --refresh  # everything again (new kits, new photos)
    uv run python -m mercato.jobs.media --dry      # download and read colours, write nothing

Photos and crests come from API-Football's image server (free, not counted against the
daily quota), for every player and club we have an API-Football id for. Each is copied once
into Supabase Storage (bucket "media", public) so the site never depends on their server;
without storage keys the source link is stored instead. Players without an id yet get
initials on the site until the Pro backfill maps them.

Colours: a club's two main colours from its crest, a nation's from its flag (flagcdn.com),
unless db/overrides/colours.csv says otherwise. Credit "Images: API-Football" on the site.
"""

import argparse
import csv
import time
from pathlib import Path

import httpx

from mercato.config import settings
from mercato.media.colours import from_bytes
from mercato.media.nations import flag_code

SOURCE = "https://media.api-sports.io/football"
FLAGS = "https://flagcdn.com/w160/{code}.png"
BUCKET = "media"
INTERVAL = 0.15
OVERRIDES = Path("db/overrides/colours.csv")  # kind (club|nation), name, primary, secondary
USER_AGENT = "MercatoBot/1.0 (+https://mercato.kanishka-sarkar.com)"


class Media:
    def __init__(self, dry: bool = False):
        self.http = httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True)
        self.dry = dry
        self.store = bool(settings.supabase_url and settings.supabase_service_key)
        self._last = 0.0
        self.stats = {"downloaded": 0, "missing": 0, "uploaded": 0}

    def get(self, url: str) -> bytes | None:
        wait = INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self.http.get(url)
        if r.status_code != 200 or not r.content:
            self.stats["missing"] += 1
            return None
        self.stats["downloaded"] += 1
        return r.content

    def keep(self, path: str, data: bytes, source: str) -> str:
        """Copy into storage and return its public link, or the source link."""
        if not self.store or self.dry:
            return source
        base = settings.supabase_url.rstrip("/")
        r = self.http.post(f"{base}/storage/v1/object/{BUCKET}/{path}", content=data,
                           headers={**auth_headers(settings.supabase_service_key),
                                    "Content-Type": "image/png", "x-upsert": "true"})
        r.raise_for_status()
        self.stats["uploaded"] += 1
        return f"{base}/storage/v1/object/public/{BUCKET}/{path}"


def auth_headers(key: str) -> dict:
    """New Supabase secret keys (sb_secret_...) go in the apikey header only; the gateway
    rejects them as a Bearer token because they aren't JWTs. Legacy service_role keys are
    JWTs and go in both."""
    if key.startswith("sb_"):
        return {"apikey": key}
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def overrides() -> dict:
    if not OVERRIDES.exists():
        return {}
    with OVERRIDES.open() as f:
        return {(r["kind"], r["name"]): (r["primary"] or None, r["secondary"] or None)
                for r in csv.DictReader(f)}


def run(refresh: bool = False, dry: bool = False, limit: int | None = None) -> dict:
    from mercato.db import connect, job_run

    m = Media(dry)
    manual = overrides()
    new_only = "" if refresh else " and {col} is null"
    with connect() as conn, job_run(conn, "media") as stats, conn.cursor() as cur:
        # clubs: crest and colours
        cur.execute("select id, name, api_football_id from clubs where api_football_id is not null"
                    + new_only.format(col="logo_url") + " order by id" +
                    (f" limit {int(limit)}" if limit else ""))
        clubs = cur.fetchall()
        for cid, name, api in clubs:
            src = f"{SOURCE}/teams/{api}.png"
            data = m.get(src)
            if data is None:
                continue
            p, s = manual.get(("club", name)) or from_bytes(data)
            how = "manual" if ("club", name) in manual else "crest"
            url = m.keep(f"clubs/{cid}.png", data, src)
            if not dry:
                cur.execute("update clubs set logo_url = %s, colour_primary = %s, "
                            "colour_secondary = %s, colours_from = %s where id = %s",
                            (url, p, s, how, cid))
            else:
                print(f"  {name}: {p} {s}")

        # competitions: crest
        cur.execute("select id, api_football_id from competitions"
                    + (" where logo_url is null" if not refresh else ""))
        for comp_id, api in cur.fetchall():
            src = f"{SOURCE}/leagues/{api}.png"
            data = m.get(src)
            if data is not None and not dry:
                cur.execute("update competitions set logo_url = %s where id = %s",
                            (m.keep(f"competitions/{comp_id}.png", data, src), comp_id))

        # nations: flag code and colours from the flag
        cur.execute("select id, name from nations" + new_only.format(col="flag_code")
                    .replace(" and", " where"))
        unknown = []
        for nid, name in cur.fetchall():
            code = flag_code(name)
            if code is None:
                unknown.append(name)
                continue
            if ("nation", name) in manual:
                (p, s), how = manual[("nation", name)], "manual"
            else:
                data = m.get(FLAGS.format(code=code))
                (p, s), how = (from_bytes(data) if data else (None, None)), "flag"
            if not dry:
                cur.execute("update nations set flag_code = %s, colour_primary = %s, "
                            "colour_secondary = %s, colours_from = %s where id = %s",
                            (code, p, s, how, nid))

        # players: photo (mapped to API-Football only)
        cur.execute("select p.mercato_id, i.api_football_id from players p join id_map i "
                    "using (mercato_id) where i.api_football_id is not null and p.current_top5"
                    + new_only.format(col="p.photo_url") +
                    (f" limit {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        for k, (pid, api) in enumerate(rows, 1):
            src = f"{SOURCE}/players/{api}.png"
            data = m.get(src)
            if data is not None and not dry:
                cur.execute("update players set photo_url = %s where mercato_id = %s",
                            (m.keep(f"players/{pid}.png", data, src), pid))
            if k % 250 == 0:
                conn.commit()
                print(f"  players {k}/{len(rows)}", flush=True)
        stats["requests"] = m.stats["downloaded"] + m.stats["missing"]
        stats["rows_written"] = m.stats["downloaded"]
        if dry:
            conn.rollback()
    out = {**m.stats, "storage": m.store, "clubs": len(clubs), "players": len(rows),
           "nations_without_flag": unknown}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="clubs and players, for a test")
    args = ap.parse_args()
    print(run(args.refresh, args.dry, args.limit))


if __name__ == "__main__":
    main()
