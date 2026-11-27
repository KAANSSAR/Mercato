import os

import httpx
import pytest

from mercato import service_status as ss
from mercato.sources.api_football import ApiFootball, PlanRestricted


def status(plan="Free", active=True):
    return {"subscription": {"plan": plan, "active": active},
            "requests": {"current": 4, "limit_day": 100}}


def test_free_plan_is_not_live():
    row = ss.evaluate(status("Free"))
    assert row["live_enabled"] is False
    assert row["seasons"] == [2022, 2023, 2024]
    assert row["reason"] == ss.FREE_REASON


def test_pro_plan_is_live():
    row = ss.evaluate(status("Pro"))
    assert row["live_enabled"] is True
    assert row["reason"] is None
    assert 2026 in row["seasons"] and 2015 in row["seasons"]


def test_inactive_paid_plan_is_not_live():
    row = ss.evaluate(status("Pro", active=False))
    assert row["live_enabled"] is False
    assert row["reason"] == ss.INACTIVE_REASON


def test_transition():
    assert ss.transition(None, False) is None
    assert ss.transition(True, True) is None
    assert ss.transition(True, False) == "downgrade"
    assert ss.transition(False, True) == "upgrade"


def test_downgrade_alerts():
    assert ss.announce("downgrade", ss.evaluate(status("Free"))) is True
    assert ss.announce("upgrade", ss.evaluate(status("Pro"))) is False
    assert ss.announce(None, ss.evaluate(status("Pro"))) is False


def test_plan_error_raises_plan_restricted():
    def handler(request):
        return httpx.Response(200, json={
            "errors": {"plan": "Free plans do not have access to this season, try from 2022 to 2024."},
            "response": []})
    api = ApiFootball(key="x", budget=5, transport=httpx.MockTransport(handler))
    api.min_interval = 0
    with pytest.raises(PlanRestricted):
        api.get("players", league=39, season=2026)


class FakeApi:
    def __init__(self, plan, restrict=False):
        self.plan, self.restrict, self.requests_made = plan, restrict, 0

    def status(self):
        return status(self.plan)

    def fixtures_by_date(self, date):
        self.requests_made += 1
        if self.restrict:
            raise PlanRestricted("Free plans do not have access to this season")
        return []


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_post_match_flow_against_postgres():
    import psycopg

    from mercato.jobs.post_match import run

    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as conn:
        conn.execute("delete from service_status")
        conn.commit()

        # Free: no fixture requests at all
        api = FakeApi("Free")
        r = run(conn, api, "2026-11-22")
        assert "skipped" in r and api.requests_made == 0

        # Upgrade to Pro: live on, announced
        r = run(conn, FakeApi("Pro"), "2026-11-22")
        assert r["change"] == "upgrade" and r["pulled"] == 0

        # Pro but the API refuses on plan grounds: live switched off, downgrade alert
        r = run(conn, FakeApi("Pro", restrict=True), "2026-11-22")
        assert r["change"] == "downgrade"
        live, reason = conn.execute(
            "select live_enabled, reason from service_status").fetchone()
        assert live is False and reason == ss.FREE_REASON
