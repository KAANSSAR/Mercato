"""On-demand endpoints only. Valuations are precomputed nightly and read from Postgres."""

from fastapi import FastAPI

app = FastAPI(title="Mercato")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/compare/{mercato_id}")
def compare(mercato_id: int):
    # TODO: 7-day cache hit, else scrape one Transfermarkt page, store dated value
    return {"mercato_id": mercato_id, "transfermarkt": None, "status": "unavailable"}
