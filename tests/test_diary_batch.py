"""add_food_entries / remove_food_entries (and the singular tools on top).

A real CronometerClient runs against FakeDiaryServer, so every request the
tools send is counted: validation must fail before any request, and a
mid-batch failure must report exactly what was applied.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from test_client import FakeFoodServer, make_server_client

from cronometer_api_mcp import server

EGG = {
    "id": 1234,
    "name": "Egg",
    "measures": [{"id": 3773, "name": "large", "value": 50, "amount": 1}],
}
CHILI = {
    "id": 85013904,
    "name": "Batch – Lentil chili v3",
    "measures": [{"id": 313190451, "name": "serving (1/8 batch)", "value": 570}],
}
D1, D2, D3 = "2027-03-15", "2027-03-16", "2027-03-17"


class FakeDiaryServer(FakeFoodServer):
    """FakeFoodServer plus the diary: get_diary, add_serving, and the v3
    DELETE. Days are kept as YYYY-MM-DD, as get_diary returns them; requests
    send "2027-3-15". `fail_add_at` (an item index) makes every attempt to log
    that item fail; `fail_delete_at` (a DELETE call number) answers it 500."""

    def __init__(self, fail_add_at=None, fail_delete_at=None):
        super().__init__(ingredient_foods=[EGG, CHILI])
        self.diary: dict[str, list[dict]] = {}
        self.next_serving_id = 5000
        self.logged = 0
        self.deletes: list[list[dict]] = []
        self.fail_add_at = fail_add_at
        self.fail_delete_at = fail_delete_at

    def handle(self, endpoint, body):
        if endpoint == "/api/v2/get_diary":
            day = _iso(body["day"])
            return {"summary": {}, "diary": copy.deepcopy(self.diary.get(day, []))}
        if endpoint == "/api/v2/add_serving":
            if self.logged == self.fail_add_at:
                return {"result": "FAIL", "error": "nope"}
            self.logged += 1
            day = _iso(body["serving"]["day"])
            serving = {**body["serving"], "day": day, "servingId": self.next_serving_id}
            self.next_serving_id += 1
            self.diary.setdefault(day, []).append(serving)
            return {"result": "SUCCESS", "id": serving["servingId"]}
        return super().handle(endpoint, body)

    def delete(self, method, url, json=None, headers=None):
        assert (method, url) == ("DELETE", "/api/v3/user/42/diary-entries")
        entries = json["diaryEntries"]
        failed = len(self.deletes) == self.fail_delete_at
        self.deletes.append(entries)
        if failed:
            return SimpleNamespace(status_code=500, text="boom")
        for e in entries:
            self.diary[e["day"]].remove(e)
        return SimpleNamespace(status_code=204, text="")


def _iso(api_day: str) -> str:
    return date(*map(int, api_day.split("-"))).isoformat()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    def build(**kwargs):
        fake = FakeDiaryServer(**kwargs)
        client, state = make_server_client(tmp_path, fake)
        client._http.request = fake.delete  # type: ignore[method-assign]
        monkeypatch.setattr(server, "_client", client)
        fake.state = state
        return fake

    return build


def _item(**overrides):
    return {
        "food_id": 1234,
        "measure_id": 3773,
        "grams": 100,
        "date": D1,
        "diary_group": "breakfast",
        **overrides,
    }


def _add(entries):
    return json.loads(server.add_food_entries(entries))


def _remove(entries):
    return json.loads(server.remove_food_entries(entries))


def _logged(fake, day):
    return [
        (e["foodId"], e["order"] >> 16, e["grams"]) for e in fake.diary.get(day, [])
    ]


# ---------------------------------------------------------------------------
# add_food_entries
# ---------------------------------------------------------------------------


def test_add_logs_every_item_in_order_with_one_food_lookup(fake):
    f = fake()
    result = _add(
        [
            _item(),
            _item(
                food_id=85013904,
                measure_id=313190451,
                grams=570,
                date=D2,
                diary_group=" Dinner ",
                translation_id=7,
            ),
            _item(grams=50, diary_group="SNACKS"),
        ]
    )

    assert result == {
        "status": "success",
        "added": [
            {"date": D1, "serving_id": 5000},
            {"date": D2, "serving_id": 5001},
            {"date": D1, "serving_id": 5002},
        ],
        "count": 3,
    }
    assert f.state["endpoints"] == ["get_foods"] + ["add_serving"] * 3
    assert f.state["payloads"][0]["ids"] == [1234, 85013904]
    sent = [p["serving"] for p in f.state["payloads"][1:]]
    assert [
        (s["day"], s["order"], s["measureId"], s["translationId"]) for s in sent
    ] == [
        ("2027-3-15", (1 << 16) | 1, 3773, 0),
        ("2027-3-16", (3 << 16) | 1, 313190451, 7),
        ("2027-3-15", (4 << 16) | 1, 3773, 0),
    ]
    assert _logged(f, D1) == [(1234, 1, 100), (1234, 4, 50)]
    assert _logged(f, D2) == [(85013904, 3, 570)]


@pytest.mark.parametrize(
    "bad,match",
    [
        ("not an object", "expected an object"),
        (
            {k: v for k, v in _item().items() if k != "grams"},
            r"missing keys \['grams'\]",
        ),
        (_item(group="lunch"), r"unknown keys \['group'\]"),
        (_item(food_id=True), "food_id must be a positive integer"),
        (_item(food_id=0), "food_id must be a positive integer"),
        (_item(food_id="1234"), "food_id must be a positive integer"),
        (_item(food_id=12.0), "food_id must be a positive integer"),
        (_item(measure_id=0), "measure_id must be a positive integer"),
        (_item(measure_id=-3), "measure_id must be a positive integer"),
        (_item(grams=0), "grams must be a positive number"),
        (_item(grams=-1), "grams must be a positive number"),
        (_item(grams=float("nan")), "grams must be a positive number"),
        (_item(grams=float("inf")), "grams must be a positive number"),
        (_item(grams=True), "grams must be a positive number"),
        (_item(grams="100"), "grams must be a positive number"),
        (_item(date="2027-3-15"), "date must be YYYY-MM-DD"),
        (_item(date="20270315"), "date must be YYYY-MM-DD"),
        (_item(date="today"), "date must be YYYY-MM-DD"),
        (_item(date=None), "date must be YYYY-MM-DD"),
        (_item(date="2027-02-30"), "must be in range"),
        (_item(diary_group="auto"), "diary_group must be breakfast"),
        (_item(diary_group="brunch"), "diary_group must be breakfast"),
        (_item(diary_group=3), "diary_group must be breakfast"),
        (_item(diary_group="group 4"), "diary_group must be breakfast"),
        (_item(diary_group="group 0"), "diary_group must be breakfast"),
        (_item(diary_group="group 05"), "diary_group must be breakfast"),
        (_item(diary_group="group"), "diary_group must be breakfast"),
        (_item(diary_group="group five"), "diary_group must be breakfast"),
        (_item(diary_group="group -5"), "diary_group must be breakfast"),
        (_item(diary_group="group5"), "diary_group must be breakfast"),
        (_item(diary_group=5), "diary_group must be breakfast"),
        (_item(translation_id=-1), "translation_id must be a non-negative"),
        (_item(translation_id=False), "translation_id must be a non-negative"),
        (_item(translation_id="0"), "translation_id must be a non-negative"),
    ],
)
def test_add_rejects_a_bad_item_by_index_before_any_request(fake, bad, match):
    f = fake()
    result = _add([_item(), _item(), bad])

    assert result["status"] == "error"
    assert result["message"].startswith("ValueError: entries[2]: ")
    assert re.search(match, result["message"])
    assert f.state["endpoints"] == []


@pytest.mark.parametrize("entries", [[], None, {"food_id": 1234}])
def test_add_needs_a_non_empty_list(fake, entries):
    f = fake()
    result = _add(entries)

    assert result == {
        "status": "error",
        "message": "ValueError: entries must be a non-empty list",
    }
    assert f.state["endpoints"] == []


def test_add_logs_a_gram_only_custom_food(fake):
    f = fake()
    f.ingredient_foods[4242] = {
        "id": 4242,
        "name": "Custom oats",
        "defaultMeasureId": 0,
        "measures": [{"id": 9000, "name": "g", "value": 1, "amount": 1}],
    }

    result = _add([_item(food_id=4242, measure_id=9000, grams=60)])

    assert result["added"] == [{"date": D1, "serving_id": 5000}]
    assert _logged(f, D1) == [(4242, 1, 60)]


def test_add_refuses_an_unknown_food_before_logging(fake):
    f = fake()
    result = _add([_item(), _item(food_id=999)])

    assert result == {
        "status": "error",
        "message": "CronometerError: Cronometer API error: "
        "WebServiceV2 doGetFoods(): Food 999 not found",
    }
    assert f.state["endpoints"] == ["get_foods"]
    assert f.diary == {}


def test_add_refuses_a_measure_of_another_food_before_logging(fake):
    f = fake()
    result = _add([_item(measure_id=313190451), _item()])

    assert result["status"] == "error"
    assert (
        "entries[0]: measure_id 313190451 is not a measure of 'Egg'"
        in (result["message"])
    )
    assert "{3773: 'large'}" in result["message"]
    assert f.state["endpoints"] == ["get_foods"]


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_add_stops_at_a_failure_and_reports_what_was_added(fake, fail_at):
    f = fake(fail_add_at=fail_at)
    result = _add([_item(), _item(date=D2), _item(date=D3)])

    assert result["status"] == "error"
    assert result["message"].startswith(f"entries[{fail_at}] failed: CronometerError")
    expected = [
        {"date": D1, "serving_id": 5000},
        {"date": D2, "serving_id": 5001},
    ][:fail_at]
    assert result["added"] == expected
    assert result["failed"] == [fail_at]
    assert result["not_attempted"] == list(range(fail_at + 1, 3))
    # A FAIL body is re-sent once after re-authenticating, then raises.
    assert f.state["endpoints"] == ["get_foods"] + ["add_serving"] * (fail_at + 2)
    assert sum(len(v) for v in f.diary.values()) == fail_at

    rollback = _remove(result["added"]) if expected else None
    if rollback:
        assert rollback["count"] == fail_at
    assert sum(len(v) for v in f.diary.values()) == 0


def test_add_reports_a_response_without_an_id_as_a_failure(fake, monkeypatch):
    f = fake()
    monkeypatch.setattr(
        server._client, "add_serving", lambda **kw: {"result": "SUCCESS"}
    )
    result = _add([_item(), _item()])

    assert result["status"] == "error"
    assert result["added"] == []
    assert result["failed"] == [0]
    assert result["not_attempted"] == [1]
    assert "may still have been applied" in result["message"]
    assert f.state["endpoints"] == ["get_foods"]


# ---------------------------------------------------------------------------
# add_food_entry: one entry, through the same checks
# ---------------------------------------------------------------------------


def _add_one(**overrides):
    kwargs = {"food_id": 1234, "measure_id": 3773, "grams": 100, "date": D1}
    return json.loads(server.add_food_entry(**{**kwargs, **overrides}))


def _at_hour(monkeypatch, hour):
    monkeypatch.setattr(
        server._client,
        "now",
        lambda: datetime(
            2026, 10, 13, hour, 30, tzinfo=ZoneInfo("America/Los_Angeles")
        ),
    )


def test_add_food_entry_logs_through_the_batch_path(fake):
    f = fake()
    result = _add_one(
        food_id=85013904,
        measure_id=313190451,
        grams=570,
        diary_group=" Dinner ",
        translation_id=7,
    )

    assert result == {
        "status": "success",
        "added": [{"date": D1, "serving_id": 5000}],
        "count": 1,
    }
    assert f.state["endpoints"] == ["get_foods", "add_serving"]
    assert f.state["payloads"][0]["ids"] == [85013904]
    sent = f.state["payloads"][1]["serving"]
    assert (sent["day"], sent["order"], sent["measureId"], sent["translationId"]) == (
        "2027-3-15",
        (3 << 16) | 1,
        313190451,
        7,
    )
    assert _logged(f, D1) == [(85013904, 3, 570)]


@pytest.mark.parametrize(
    "group,number",
    [
        ("breakfast", 1),
        ("Lunch", 2),
        (" DINNER ", 3),
        ("snacks", 4),
        ("group 5", 5),
        (" Group 6 ", 6),
        ("group 12", 12),
    ],
)
def test_add_food_entry_explicit_groups(fake, group, number):
    f = fake()
    assert _add_one(diary_group=group)["count"] == 1
    assert _logged(f, D1) == [(1234, number, 100)]


@pytest.mark.parametrize(
    "hour,number",
    [
        (0, 4),
        (3, 4),
        (4, 1),
        (9, 1),
        (10, 2),
        (13, 2),
        (14, 3),
        (20, 3),
        (21, 4),
        (23, 4),
    ],
)
def test_add_food_entry_auto_resolves_by_account_hour(fake, monkeypatch, hour, number):
    f = fake()
    _at_hour(monkeypatch, hour)

    assert _add_one(diary_group="auto")["count"] == 1
    assert _logged(f, D1) == [(1234, number, 100)]
    assert f.state["payloads"][1]["serving"]["order"] == (number << 16) | 1


def test_add_food_entry_defaults_to_auto(fake, monkeypatch):
    f = fake()
    _at_hour(monkeypatch, 12)

    assert _add_one()["count"] == 1
    assert _logged(f, D1) == [(1234, 2, 100)]


@pytest.mark.parametrize("group", [" AUTO ", "Auto"])
def test_add_food_entry_auto_is_case_insensitive(fake, monkeypatch, group):
    f = fake()
    _at_hour(monkeypatch, 7)

    assert _add_one(diary_group=group)["count"] == 1
    assert _logged(f, D1) == [(1234, 1, 100)]


def test_add_food_entry_defaults_to_today(fake, monkeypatch):
    f = fake()
    monkeypatch.setattr(server._client, "today", lambda: date(2027, 3, 16))

    result = json.loads(
        server.add_food_entry(
            food_id=1234, measure_id=3773, grams=100, diary_group="lunch"
        )
    )

    assert result["added"] == [{"date": D2, "serving_id": 5000}]
    assert _logged(f, D2) == [(1234, 2, 100)]


def test_add_food_entry_refuses_a_measure_of_another_food(fake):
    f = fake()
    result = _add_one(measure_id=313190451, diary_group="lunch")

    assert result["status"] == "error"
    assert "measure_id 313190451 is not a measure of 'Egg'" in result["message"]
    assert "{3773: 'large'}" in result["message"]
    assert f.state["endpoints"] == ["get_foods"]
    assert f.diary == {}


def test_add_food_entry_refuses_an_unknown_food(fake):
    f = fake()
    result = _add_one(food_id=999, diary_group="lunch")

    assert result["status"] == "error"
    assert "Food 999 not found" in result["message"]
    assert f.state["endpoints"] == ["get_foods"]
    assert f.diary == {}


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"measure_id": 0}, "measure_id must be a positive integer"),
        ({"food_id": 0}, "food_id must be a positive integer"),
        ({"grams": 0}, "grams must be a positive number"),
        ({"grams": float("nan")}, "grams must be a positive number"),
        ({"date": "2027-3-15"}, "date must be YYYY-MM-DD"),
        ({"translation_id": -1}, "translation_id must be a non-negative"),
        ({"date": ""}, "date must be YYYY-MM-DD"),
        ({"diary_group": "brunch"}, "diary_group must be breakfast"),
        ({"diary_group": ""}, "diary_group must be breakfast"),
        ({"diary_group": "1"}, "diary_group must be breakfast"),
        ({"diary_group": 3}, "diary_group must be breakfast"),
        ({"diary_group": "group 4"}, "diary_group must be breakfast"),
        ({"diary_group": "group 0"}, "diary_group must be breakfast"),
    ],
)
def test_add_food_entry_rejects_bad_fields_before_any_request(
    fake, monkeypatch, overrides, match
):
    f = fake()
    monkeypatch.setattr(
        server._client, "now", lambda: pytest.fail("now() must not be called")
    )
    overrides.setdefault("diary_group", "lunch")
    result = _add_one(**overrides)

    assert result["status"] == "error"
    assert re.search(match, result["message"])
    assert f.state["endpoints"] == []


# ---------------------------------------------------------------------------
# remove_food_entries
# ---------------------------------------------------------------------------


def _seed(f, *days):
    """Log one egg per listed day (repeats allowed); returns their entries."""
    added = _add([_item(date=d) for d in days])["added"]
    f.state["endpoints"].clear()
    return added


def test_remove_groups_by_date_with_one_lookup_and_delete_per_date(fake):
    f = fake()
    a, b, c, d = _seed(f, D1, D2, D1, D3)
    keep = _seed(f, D2)

    result = _remove([b, a, c, d])

    assert result == {"status": "success", "removed": [b, a, c, d], "count": 4}
    assert f.state["endpoints"] == ["get_diary"] * 3
    assert [_iso(p["day"]) for p in f.state["payloads"][-3:]] == [D2, D1, D3]
    assert [[e["servingId"] for e in batch] for batch in f.deletes] == [
        [b["serving_id"]],
        [a["serving_id"], c["serving_id"]],
        [d["serving_id"]],
    ]
    # The v3 endpoint takes whole entry objects, as get_diary returned them.
    assert f.deletes[0][0]["foodId"] == 1234
    assert f.deletes[0][0]["order"] == (1 << 16) | 1
    assert [e["servingId"] for e in f.diary[D2]] == [keep[0]["serving_id"]]
    assert f.diary[D1] == f.diary[D3] == []


def test_remove_refuses_unknown_ids_before_deleting_anything(fake):
    f = fake()
    a, b = _seed(f, D1, D2)

    result = _remove(
        [
            a,
            {"date": D2, "serving_id": 777},
            {"date": D1, "serving_id": b["serving_id"]},  # logged on D2
        ]
    )

    assert result["status"] == "error"
    assert result["message"] == (
        "ValueError: Not in the diary, so nothing was removed: "
        f"entries[2] (serving_id {b['serving_id']} on {D1}); "
        f"entries[1] (serving_id 777 on {D2})"
    )
    assert f.state["endpoints"] == ["get_diary"] * 2
    assert f.deletes == []
    assert len(f.diary[D1]) == len(f.diary[D2]) == 1


@pytest.mark.parametrize("with_id", [False, True])
def test_remove_ignores_non_food_entries(fake, with_id):
    f = fake()
    (a,) = _seed(f, D1)
    run = {"type": "Exercise", "name": "Running", "day": D1, "order": 1}
    if with_id:
        run["servingId"] = 77
    f.diary[D1].append(run)

    refused = _remove([{"date": D1, "serving_id": 77}])
    result = _remove([a])

    assert "entries[0] (serving_id 77 on 2027-03-15)" in refused["message"]
    assert result == {"status": "success", "removed": [a], "count": 1}
    assert f.diary[D1] == [run]


def test_remove_refuses_a_repeated_id_before_any_request(fake):
    f = fake()
    (a,) = _seed(f, D1)

    result = _remove([a, {"date": D2, "serving_id": a["serving_id"]}])

    assert result["status"] == "error"
    assert (
        f"entries[1] repeats serving_id {a['serving_id']} from entries[0]"
        in (result["message"])
    )
    assert f.state["endpoints"] == []


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"date": D1}, r"missing keys \['serving_id'\]"),
        ({"date": D1, "serving_id": 1, "food_id": 2}, r"unknown keys \['food_id'\]"),
        ({"date": D1, "serving_id": "5000"}, "serving_id must be a positive integer"),
        ({"date": D1, "serving_id": True}, "serving_id must be a positive integer"),
        ({"date": D1, "serving_id": 0}, "serving_id must be a positive integer"),
        ({"date": "yesterday", "serving_id": 1}, "date must be YYYY-MM-DD"),
        ([D1, 1], "expected an object"),
    ],
)
def test_remove_rejects_a_bad_item_by_index_before_any_request(fake, bad, match):
    f = fake()
    result = _remove([{"date": D1, "serving_id": 9}, bad])

    assert result["status"] == "error"
    assert result["message"].startswith("ValueError: entries[1]: ")
    assert re.search(match, result["message"])
    assert f.state["endpoints"] == []


@pytest.mark.parametrize("entries", [[], None])
def test_remove_needs_a_non_empty_list(fake, entries):
    f = fake()
    assert _remove(entries)["message"] == "ValueError: entries must be a non-empty list"
    assert f.state["endpoints"] == []


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_remove_stops_at_a_failed_date_and_reports_what_was_removed(fake, fail_at):
    f = fake(fail_delete_at=fail_at)
    a, b, c, d = _seed(f, D1, D2, D2, D3)

    result = _remove([a, b, c, d])

    groups = [[0], [1, 2], [3]]
    assert result["status"] == "error"
    assert result["message"].startswith(
        f"entries{groups[fail_at]} failed: CronometerError: Delete failed with status 500"
    )
    assert result["removed"] == [[], [a], [a, b, c]][fail_at]
    assert result["failed"] == groups[fail_at]
    assert result["not_attempted"] == [i for g in groups[fail_at + 1 :] for i in g]
    assert len(f.deletes) == fail_at + 1
    remaining = sum(len(v) for v in f.diary.values())
    assert remaining == 4 - len(result["removed"])


# ---------------------------------------------------------------------------
# remove_food_entry: one date, through the same checks
# ---------------------------------------------------------------------------


def test_remove_food_entry_removes_ids_of_one_date(fake):
    f = fake()
    a, b, _ = _seed(f, D1, D1, D1)

    result = json.loads(
        server.remove_food_entry([str(a["serving_id"]), str(b["serving_id"])], date=D1)
    )

    assert result == {"status": "success", "removed": [a, b], "count": 2}
    assert f.state["endpoints"] == ["get_diary"]
    assert len(f.deletes) == 1
    assert len(f.diary[D1]) == 1


def test_remove_food_entry_defaults_to_today(fake, monkeypatch):
    f = fake()
    (a,) = _seed(f, D2)
    monkeypatch.setattr(server._client, "today", lambda: date(2027, 3, 16))

    result = json.loads(server.remove_food_entry([str(a["serving_id"])]))

    assert result["removed"] == [a]
    assert f.diary[D2] == []


def test_remove_food_entry_unknown_id_is_an_error_not_a_silent_zero(fake):
    f = fake()
    (a,) = _seed(f, D1)

    result = json.loads(server.remove_food_entry([str(a["serving_id"]), "4"], date=D1))

    assert result["status"] == "error"
    assert "entries[1] (serving_id 4 on 2027-03-15)" in result["message"]
    assert f.deletes == []


def test_remove_food_entry_refuses_an_empty_date(fake):
    f = fake()
    result = json.loads(server.remove_food_entry(["1"], date=""))

    assert result["status"] == "error"
    assert "date must be YYYY-MM-DD" in result["message"]
    assert f.state["endpoints"] == []


@pytest.mark.parametrize("bad", ["12a", "", "-5", 5])
def test_remove_food_entry_rejects_non_digit_ids_before_any_request(fake, bad):
    f = fake()
    result = json.loads(server.remove_food_entry(["1", bad], date=D1))

    assert result["status"] == "error"
    assert "entry_ids must be digit strings" in result["message"]
    assert f.state["endpoints"] == []


def test_extra_group_round_trips_through_the_food_log_label(fake):
    """An entry written to "group 5" reads back under the same label."""
    f = fake()
    assert _add([_item(diary_group="group 5")])["count"] == 1
    assert _logged(f, D1) == [(1234, 5, 100)]
    assert server._group_name(5) == "group 5"
