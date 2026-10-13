"""Regression tests for CronometerClient auth-failure handling.

Covers issue #18: an expired session comes back as an HTTP 200 body of
{"result": "FAIL", ...} which the client previously failed to detect
(it only matched "FAILURE"), so a stale token was never invalidated and
the failure was surfaced to callers as a success.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from pathlib import Path

import pytest

from cronometer_api_mcp import server as mcp_server
from cronometer_api_mcp.client import CronometerClient, CronometerError

# Real response body observed from an expired session.
REAL_FAIL_BODY = {"result": "FAIL", "error": "Token Authorization failed"}
# Synthetic/defensive variant. Never observed in real traffic, but the
# client keeps handling it just in case some endpoint uses it.
SYNTHETIC_FAILURE_BODY = {"result": "FAILURE", "error": "synthetic"}
# Captured live (Oct 2026): bogus/null token or wrong userId, then two
# pre-dispatch errors. None carry "messages", unlike a handler's FAIL.
AUTH_FAILED_BODY = {"result": "FAIL", "error": "Authentication Failed"}
INVALID_COMMAND_BODY = {"result": "FAIL", "error": "Invalid Command"}
BAD_PAYLOAD_BODY = {"result": "FAIL", "error": 'JSONObject["ids"] not found.'}
SYNC_MESSAGES = [
    {"text": '{"id":85096127,"type":"ADD"}', "id": 20043751833, "type": "FOOD_CHANGED"}
]


def not_found_body(endpoint: str, food_id: int, messages=()) -> dict:
    """The real (live-captured) FAIL for an unknown id; get_foods names only
    the first unknown id."""
    where = {
        "/api/v2/get_food": "doGetFood()",
        "/api/v2/get_foods": "WebServiceV2 doGetFoods()",
    }[endpoint]
    return {
        "result": "FAIL",
        "messages": list(messages),
        "error": f"{where}: Food {food_id} not found",
    }


class FakeResp:
    def __init__(self, body, status: int = 200) -> None:
        self._body = body
        self.status_code = status

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        pass

    def json(self):
        return self._body


def make_client(tmp_path: Path, responses: list[dict]):
    """Build a client whose HTTP POST returns `responses` in order and whose
    login() just swaps in a fresh token without hitting the network.

    Returns (client, state) where state tracks login/post call counts.
    """
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = 123
    client._token = "STALE_TOKEN"

    state = {"login": 0, "post": 0}

    def fake_login() -> None:
        state["login"] += 1
        client._user_id = 123
        client._token = f"FRESH_TOKEN_{state['login']}"

    def fake_post(endpoint, json=None):
        idx = state["post"]
        state["post"] += 1
        body = responses[min(idx, len(responses) - 1)]
        return FakeResp(body)

    client.login = fake_login  # type: ignore[method-assign]
    client._http.post = fake_post  # type: ignore[method-assign]
    return client, state


def test_fail_invalidates_token_logs_in_and_retries_once(tmp_path):
    """HTTP 200 + result:"FAIL" -> invalidate token, login, retry once, succeed."""
    client, state = make_client(
        tmp_path, [REAL_FAIL_BODY, {"result": "SUCCESS", "id": 999}]
    )

    result = client._request("/api/v2/get_diary", {})

    assert result == {"result": "SUCCESS", "id": 999}
    assert state["login"] == 1
    assert state["post"] == 2


def test_failure_variant_still_retries(tmp_path):
    """HTTP 200 + result:"FAILURE" retains the original retry behavior."""
    client, state = make_client(
        tmp_path, [SYNTHETIC_FAILURE_BODY, {"result": "SUCCESS", "id": 1}]
    )

    result = client._request("/api/v2/get_diary", {})

    assert result == {"result": "SUCCESS", "id": 1}
    assert state["login"] == 1
    assert state["post"] == 2


def test_second_failure_raises(tmp_path):
    """A second failed response raises CronometerError rather than returning it."""
    client, state = make_client(tmp_path, [REAL_FAIL_BODY, REAL_FAIL_BODY])

    with pytest.raises(CronometerError):
        client._request("/api/v2/get_diary", {})

    assert state["login"] == 1
    assert state["post"] == 2


def test_add_serving_does_not_return_success_wrapper_on_failure(tmp_path):
    """add_serving must not surface an API failure body as a success result."""
    client, state = make_client(tmp_path, [REAL_FAIL_BODY, REAL_FAIL_BODY])

    with pytest.raises(CronometerError):
        client.add_serving(food_id=1, measure_id=2, grams=100.0, diary_group=2)

    assert state["post"] == 2


def test_stale_cache_file_is_removed_on_failure(tmp_path):
    """The stale cached session file is deleted when a FAIL is detected."""
    session_path = tmp_path / "session.json"
    session_path.write_text('{"username": "", "user_id": 123, "token": "STALE"}')

    client, _ = make_client(tmp_path, [REAL_FAIL_BODY, {"result": "SUCCESS", "id": 5}])
    # make_client overwrote session_path arg via same tmp_path
    assert client._session_path == session_path

    client._request("/api/v2/get_diary", {})

    # _invalidate_session() should have unlinked the stale file.
    assert not session_path.exists()


@pytest.mark.parametrize(
    "body",
    [
        REAL_FAIL_BODY,
        AUTH_FAILED_BODY,
        INVALID_COMMAND_BODY,
        BAD_PAYLOAD_BODY,
        SYNTHETIC_FAILURE_BODY,
    ],
    ids=["expired", "auth-failed", "invalid-command", "bad-payload", "failure"],
)
def test_fail_without_messages_reauthenticates(tmp_path, body):
    """A FAIL without "messages" may be an auth failure, so it re-logs in."""
    client, state = make_client(tmp_path, [body, {"result": "SUCCESS", "id": 3}])

    assert client._request("/api/v2/get_foods", {}) == {"result": "SUCCESS", "id": 3}
    assert (state["login"], state["post"]) == (1, 2)


@pytest.mark.parametrize(
    "body",
    [
        not_found_body("/api/v2/get_foods", 999999999, SYNC_MESSAGES),
        not_found_body("/api/v2/get_foods", 999999998),
        not_found_body("/api/v2/get_food", 999999999),
        {"result": "FAIL", "messages": [], "error": "Missing food id"},
        {"result": "FAILURE", "messages": [], "error": "synthetic"},
    ],
    ids=["foods-with-sync", "foods-empty", "food", "missing-id", "failure"],
)
def test_fail_with_messages_raises_without_reauthenticating(tmp_path, body):
    """A handler's FAIL raises its error as-is: no re-login, session kept."""
    session_path = tmp_path / "session.json"
    session_path.write_text('{"username": "", "user_id": 123, "token": "STALE"}')
    client, state = make_client(tmp_path, [body, {"result": "SUCCESS"}])

    with pytest.raises(CronometerError) as exc:
        client._request("/api/v2/get_foods", {})

    assert str(exc.value) == f"Cronometer API error: {body['error']}"
    assert "FOOD_CHANGED" not in str(exc.value)
    assert (state["login"], state["post"]) == (0, 1)
    assert client._token == "STALE_TOKEN"
    assert session_path.exists()


def test_fail_with_messages_after_reauth_raises_its_error(tmp_path):
    """Re-auth on an ambiguous FAIL, then a handler FAIL: raise that one."""
    body = not_found_body("/api/v2/get_foods", 7, SYNC_MESSAGES)
    client, state = make_client(tmp_path, [REAL_FAIL_BODY, body])

    with pytest.raises(
        CronometerError,
        match=r"^Cronometer API error: WebServiceV2 doGetFoods\(\): Food 7 not found$",
    ):
        client._request("/api/v2/get_foods", {})

    assert (state["login"], state["post"]) == (1, 2)


GET_FOODS_FOODS = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]


@pytest.mark.parametrize(
    "ids,unknown",
    [
        ([999], 999),
        ([999, 1], 999),
        ([1, 999], 999),
        ([1, 2, 7, 8], 7),  # the API names only the first unknown id
        ([8, 7], 8),
    ],
)
def test_get_foods_unknown_id_fails_the_call_naming_it(tmp_path, ids, unknown):
    client, state = make_server_client(
        tmp_path, FakeFoodServer(ingredient_foods=GET_FOODS_FOODS)
    )

    with pytest.raises(CronometerError) as exc:
        client.get_foods(ids)

    assert str(exc.value) == (
        f"Cronometer API error: WebServiceV2 doGetFoods(): Food {unknown} not found"
    )
    assert state["endpoints"] == ["get_foods"]


@pytest.mark.parametrize("ids", [[1], [2, 1], [1, 2], [1, 1]])
def test_get_foods_returns_every_requested_food(tmp_path, ids):
    client, state = make_server_client(
        tmp_path, FakeFoodServer(ingredient_foods=GET_FOODS_FOODS)
    )

    assert [f["id"] for f in client.get_foods(ids)] == ids
    assert state["endpoints"] == ["get_foods"]


def test_get_foods_empty_sends_nothing(tmp_path):
    client, state = make_server_client(tmp_path, FakeFoodServer())

    assert client.get_foods([]) == []
    assert state["endpoints"] == []


def test_get_foods_fails_fast_on_a_body_without_foods(tmp_path):
    client, _ = make_client(tmp_path, [{"messages": []}])

    with pytest.raises(KeyError, match="foods"):
        client.get_foods([1])


def test_get_food_unknown_id_raises_without_reauthenticating(tmp_path):
    client, state = make_server_client(tmp_path, FakeFoodServer())

    with pytest.raises(
        CronometerError,
        match=r"^Cronometer API error: doGetFood\(\): Food 5 not found$",
    ):
        client.get_food(5)

    assert state["endpoints"] == ["get_food"]


# ---------------------------------------------------------------------------
# Cold-start auth ordering (issues #30 / #31)
#
# On a client that hasn't logged in yet, add_serving() must not embed a null
# userId: reading identity has to trigger login first, so the payload is
# correct on the first attempt and the retry never re-sends stale state.
# ---------------------------------------------------------------------------


def make_cold_client(tmp_path: Path, responses: list[dict]):
    """Like make_client but starts unauthenticated and records posted payloads."""
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = None
    client._token = None

    state = {"login": 0, "post": 0, "payloads": []}

    def fake_login() -> None:
        state["login"] += 1
        client._user_id = 42
        client._token = f"FRESH_TOKEN_{state['login']}"

    def fake_post(endpoint, json=None):
        state["payloads"].append(json)
        idx = state["post"]
        state["post"] += 1
        body = responses[min(idx, len(responses) - 1)]
        return FakeResp(body)

    client.login = fake_login  # type: ignore[method-assign]
    client._http.post = fake_post  # type: ignore[method-assign]
    return client, state


_SERVER_FRACTION = re.compile(r"(?:^|.*?)\b([1-5])/([2-5])\s+(\S.*)$")
_SERVER_LEADING_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\S.*)$")


def server_reparse(name: str, amount: float) -> tuple[str, float]:
    """How add_food stores a NEW measure's name (observed live): only when
    amount is 1, a leading number or an n/d fraction (d in 2-5) followed by a
    space is split off into the amount. Existing measures are kept verbatim."""
    if amount != 1:
        return name, amount
    led = _SERVER_LEADING_NUMBER.match(name)
    if led:
        return led.group(2), float(led.group(1))
    frac = _SERVER_FRACTION.match(name)
    if frac:
        return frac.group(3), int(frac.group(1)) / int(frac.group(2))
    return name, amount


class FakeFoodServer:
    """A stateful stand-in for add_food / get_food / get_foods.

    Unknown ids get the real not-found body (not_found_body).

    add_food upserts like the real endpoint: new foods and measures get ids,
    the first measure becomes the default (unless it's a lone "g", which
    keeps defaultMeasureId 0, as seen live), new measure names are re-parsed
    (server_reparse), and a recipe's "full recipe" weight and per-100g
    nutrients are derived from its ingredients, leaving out the grams of an
    ingredient with a non-"Weight" measure as the live server does
    (derive_nutrients=False keeps whatever nutrients were sent, simulating a
    save that derived nothing).
    `save_responses` replaces the next add_food answers.
    """

    def __init__(
        self, foods=(), ingredient_foods=(), save_responses=(), derive_nutrients=True
    ):
        self.foods = {f["id"]: copy.deepcopy(f) for f in foods}
        self.ingredient_foods = {f["id"]: f for f in ingredient_foods}
        self.save_responses = list(save_responses)
        self.derive_nutrients = derive_nutrients
        self.next_id = 4242
        self.next_measure_id = 9000

    def handle(self, endpoint: str, body: dict) -> dict:
        if endpoint == "/api/v2/get_food":
            if body["id"] not in self.foods:
                return not_found_body(endpoint, body["id"])
            food = copy.deepcopy(self.foods[body["id"]])
            # A recipe without a stored "full recipe" reads back with a
            # virtual id-0 one (seen live).
            names = [m["name"] for m in food.get("measures", [])]
            if food.get("ingredients") and "full recipe" not in names:
                food["measures"].insert(
                    0,
                    {
                        "id": 0,
                        "name": "full recipe",
                        "amount": 1,
                        "value": sum(i["grams"] for i in food["ingredients"]),
                        "type": "Weight",
                    },
                )
            return {**food, "messages": []}
        if endpoint == "/api/v2/get_foods":
            for i in body["ids"]:
                if i not in self.ingredient_foods:
                    return not_found_body(endpoint, i, SYNC_MESSAGES)
            return {
                "messages": [],
                "foods": [self.ingredient_foods[i] for i in body["ids"]],
            }
        if endpoint == "/api/v2/add_food":
            if self.save_responses:
                return self.save_responses.pop(0)
            return self._save(copy.deepcopy(body["data"]))
        raise AssertionError(f"unexpected endpoint {endpoint}")

    def _save(self, food: dict) -> dict:
        if not food["id"]:
            food["id"] = self.next_id
            self.next_id += 1
            food["source"] = "Custom"
        new_food = food["id"] == self.next_id - 1 and food["id"] not in self.foods
        if new_food and all(m["name"] != "g" for m in food["measures"]):
            # Live: a new food without a "g" measure gets one.
            food["measures"].append(
                {"id": 0, "name": "g", "value": 1, "amount": 1, "type": "Weight"}
            )
        for m in food["measures"]:
            if not m["id"]:
                m["id"] = self.next_measure_id
                self.next_measure_id += 1
                m["name"], m["amount"] = server_reparse(m["name"], m["amount"])
        if not food.get("defaultMeasureId") and not _only_gram(food):
            food["defaultMeasureId"] = food["measures"][0]["id"]
        if food.get("ingredients"):
            # Live: an ingredient adds its grams only if all its food's
            # measures are type "Weight", but always adds its nutrients.
            batch = sum(
                i["grams"]
                for i in food["ingredients"]
                if all(
                    m["type"] == "Weight"
                    for m in self.ingredient_foods[i["foodId"]]["measures"]
                )
            )
            for m in food["measures"]:
                if m["name"] == "full recipe":
                    m["value"] = batch
            if self.derive_nutrients:
                totals: dict[int, float] = {}
                for i in food["ingredients"]:
                    for n in self.ingredient_foods[i["foodId"]]["nutrients"]:
                        totals[n["id"]] = (
                            totals.get(n["id"], 0) + n["amount"] * i["grams"] / 100
                        )
                # Live: a 0 g batch stores the totals as the per-100 g amounts.
                food["nutrients"] = [
                    {"id": nid, "amount": amount * 100 / (batch or 100)}
                    for nid, amount in sorted(totals.items())
                ]
        self.foods[food["id"]] = food
        return {"result": "SUCCESS", "messages": [], "id": food["id"]}


def _only_gram(food: dict) -> bool:
    return [(m["name"], m["value"]) for m in food["measures"]] == [("g", 1)]


def make_server_client(tmp_path: Path, server: FakeFoodServer):
    """A logged-in client backed by `server`; state records every request."""
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = 42
    client._token = "TOKEN"
    state = {"endpoints": [], "payloads": []}

    def fake_post(endpoint, json=None):
        state["endpoints"].append(endpoint.removeprefix("/api/v2/"))
        state["payloads"].append(copy.deepcopy(json))
        return FakeResp(server.handle(endpoint, json))

    client.login = lambda: None  # type: ignore[method-assign]
    client._http.post = fake_post  # type: ignore[method-assign]
    return client, state


def test_add_serving_cold_start_embeds_real_user_id(tmp_path):
    """First write on a cold client logs in once and sends the real userId."""
    client, state = make_cold_client(tmp_path, [{"result": "SUCCESS", "id": 7}])

    client.add_serving(food_id=1, measure_id=2, grams=100.0, diary_group=2)

    assert state["login"] == 1
    assert state["post"] == 1  # no stale-payload double failure (#31)
    assert state["payloads"][0]["serving"]["userId"] == 42


def test_user_id_property_triggers_login(tmp_path):
    """Reading user_id on a cold client authenticates and returns the real id."""
    client, state = make_cold_client(tmp_path, [{"result": "SUCCESS"}])

    assert client.user_id == 42
    assert state["login"] == 1


# ---------------------------------------------------------------------------
# Diary enrichment (food names / per-entry nutrients)
# ---------------------------------------------------------------------------


def _enrich_client(tmp_path: Path):
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = 123
    client._token = "TOKEN"
    return client


SAMPLE_DIARY = {
    "diary": [
        {
            "type": "Serving",
            "foodId": 100,
            "measureId": 10,
            "grams": 200,
            "order": 65537,
        },
        {
            "type": "Serving",
            "foodId": 200,
            "measureId": 999,  # a measure the food no longer has
            "grams": 50,
            "order": 65538,
        },
        {
            "type": "Serving",
            "foodId": 300,
            "measureId": 30,
            "grams": 1.1,  # Recipe: "grams" is a serving count, not grams
            "order": 65539,
        },
        {"type": "Exercise", "name": "Running", "order": 1},
    ]
}

SAMPLE_FOODS = [
    {
        "id": 100,
        "name": "Oats",
        "source": "USDA",
        "category": "Grains",
        "defaultMeasureId": 10,
        "measures": [{"id": 10, "name": "cup", "value": 100.0}],
        # per-100g
        "nutrients": [{"id": 208, "amount": 389.0}, {"id": 203, "amount": 16.9}],
    },
    {
        "id": 200,
        "name": "Milk",
        "source": "Custom",
        "defaultMeasureId": 20,
        "measures": [{"id": 20, "name": "glass", "value": 244.0}],
        "nutrients": [{"id": 208, "amount": 42.0}],
    },
    {
        "id": 300,
        "name": "Recipe Food",
        "source": "Custom",
        "defaultMeasureId": 30,
        "measures": [
            {"id": 30, "name": "serving", "value": 1, "amount": 1, "type": "Recipe"},
            {"id": 31, "name": "g", "value": 233.4, "amount": 1, "type": "Recipe"},
        ],
        # stored per one reference serving (not per-100g)
        "nutrients": [{"id": 208, "amount": 708.538}, {"id": 203, "amount": 56.03}],
    },
]


def test_enrich_diary_merges_names_measures_and_scaled_nutrients(tmp_path):
    client = _enrich_client(tmp_path)
    client.get_foods = lambda ids: SAMPLE_FOODS  # type: ignore[method-assign]

    out = client.enrich_diary_servings(
        {"diary": [dict(e) for e in SAMPLE_DIARY["diary"]]}
    )
    entries = out["diary"]

    oats = entries[0]
    assert oats["name"] == "Oats"
    assert "source" not in oats and "category" not in oats
    assert oats["serving_size"] == {
        "measure_id": 10,
        "quantity": 1,
        "unit": "cup",
        "grams": 100.0,
    }
    assert oats["servings"] == 2.0  # 200g / 100g per cup
    # per-100g energy 389 scaled to 200g -> 778
    assert oats["nutrients"] == {208: 778.0, 203: 33.8}

    milk = entries[1]
    # Measure 999 is gone: no serving (not the default's), still per 100 g.
    assert "serving_size" not in milk
    assert "servings" not in milk
    assert milk["nutrients"] == {208: 21.0}

    recipe = entries[2]
    assert recipe["name"] == "Recipe Food"
    assert recipe["serving_size"] == {
        "measure_id": 30,
        "quantity": 1,
        "unit": "serving",
        "grams": None,
    }
    assert recipe["servings"] == 1.1
    # Recipe measure: nutrients are per-serving and "grams" is a serving count,
    # so 708.538 kcal/serving * 1.1 servings -> 779.39 (not grams/100)
    assert recipe["nutrients"] == {208: 779.3918, 203: 61.633}

    # Non-Serving entry untouched
    assert entries[3] == {"type": "Exercise", "name": "Running", "order": 1}


@pytest.mark.parametrize(
    ("measure_type", "quantity", "value", "expected_quantity", "servings"),
    [
        ("Weight", 100, 100, 100, 0.6),
        ("Atomic", 0.5, 120, 0.5, 0.5),
        ("Atomic", None, 120, None, 0.5),
        ("Atomic", 0, 120, None, 0.5),
        ("Atomic", -1, 120, None, 0.5),
        ("Atomic", "bad", 120, None, 0.5),
        ("Atomic", float("nan"), 120, None, 0.5),
        ("Atomic", float("inf"), 120, None, 0.5),
    ],
)
def test_enrich_diary_measure_quantity(
    tmp_path, monkeypatch, measure_type, quantity, value, expected_quantity, servings
):
    client = _enrich_client(tmp_path)
    food = {
        "id": 100,
        "defaultMeasureId": 10,
        "measures": [
            {"id": 10, "type": measure_type, "amount": quantity, "value": value}
        ],
    }
    monkeypatch.setattr(client, "get_foods", lambda ids: [food])

    out = client.enrich_diary_servings(
        {"diary": [{"type": "Serving", "foodId": 100, "measureId": 10, "grams": 60}]}
    )

    entry = out["diary"][0]
    assert entry["serving_size"]["quantity"] == expected_quantity
    assert entry["serving_size"]["grams"] == value
    assert entry["servings"] == servings


# The serving-based recipe probed live; expected kcal are the server's own.
LIVE_SERVING_RECIPE = {
    "id": 85100983,
    "name": "ZZ test serving recipe",
    "defaultMeasureId": 313595231,
    "measures": [
        {"id": 313595763, "name": "g", "value": 200, "amount": 1, "type": "Recipe"},
        {"id": 313595231, "name": "full recipe", "value": 4, "amount": 1,
         "type": "Recipe"},
        {"id": 313595764, "name": "portion", "value": 1, "amount": 1,
         "type": "Recipe"},
    ],
    "nutrients": [{"id": 208, "amount": 71.5}],
}  # fmt: skip


def _enrich_one(tmp_path, food, measure_id, grams):
    client = _enrich_client(tmp_path)
    client.get_foods = lambda ids: [food]  # type: ignore[method-assign]
    entry = {
        "type": "Serving",
        "servingId": 7,
        "foodId": food["id"],
        "measureId": measure_id,
        "grams": grams,
    }
    return client.enrich_diary_servings({"diary": [entry]})["diary"][0]


@pytest.mark.parametrize(
    ("measure_id", "grams", "server_kcal"),
    [
        (313595763, 1.5, 0.5362),  # moved onto "g" when "serving" was deleted
        (313595231, 4, 71.5),  # "full recipe", value 4
        (313595764, 2, 143.0),  # value 1: grams is a serving count
    ],
)
def test_enrich_recipe_scales_by_the_measure_value_like_the_server(
    tmp_path, measure_id, grams, server_kcal
):
    entry = _enrich_one(tmp_path, LIVE_SERVING_RECIPE, measure_id, grams)

    assert entry["nutrients"][208] == pytest.approx(server_kcal, abs=1e-4)
    assert entry["serving_size"]["grams"] is None
    assert entry["recipe"] is True
    assert entry["servings"] == grams


@pytest.mark.parametrize("measure_id", [0, 999, 313595230])
def test_enrich_recipe_entry_without_its_measure_raises(tmp_path, measure_id):
    # Live, measure 0 counted 71.5 * 2 / 200, not the default's 71.5 * 2 / 4.
    with pytest.raises(CronometerError, match=f"measure {measure_id}, which"):
        _enrich_one(tmp_path, LIVE_SERVING_RECIPE, measure_id, 2)


@pytest.mark.parametrize("value", [0, -1, None, "1", float("nan"), float("inf")])
def test_enrich_recipe_measure_without_a_usable_value_raises(tmp_path, value):
    food = {
        **LIVE_SERVING_RECIPE,
        "measures": [{"id": 1, "name": "bowl", "value": value, "type": "Recipe"}],
    }

    with pytest.raises(CronometerError, match="no usable value"):
        _enrich_one(tmp_path, food, 1, 2)


def test_enrich_mixed_recipe_and_weight_measures_raises(tmp_path):
    food = {
        **LIVE_SERVING_RECIPE,
        "measures": [
            *LIVE_SERVING_RECIPE["measures"],
            {"id": 5, "name": "oz", "value": 28.35, "type": "Weight"},
        ],
    }

    with pytest.raises(CronometerError, match="mixes Recipe"):
        _enrich_one(tmp_path, food, 5, 28.35)


@pytest.mark.parametrize("measure_id", [0, 999])
@pytest.mark.parametrize("default", [0, 10, 999])
def test_enrich_weight_entry_without_its_measure_ignores_the_default(
    tmp_path, measure_id, default
):
    # Live, such entries counted per 100 g.
    food = {
        "id": 100,
        "name": "Bar",
        "defaultMeasureId": default,
        "measures": [{"id": 10, "name": "chunk", "value": 40, "type": "Atomic"}],
        "nutrients": [{"id": 208, "amount": 400}],
    }

    entry = _enrich_one(tmp_path, food, measure_id, 30)

    assert "serving_size" not in entry
    assert "servings" not in entry
    assert entry["nutrients"] == {208: 120.0}


@pytest.mark.parametrize("measures", [[], [{"id": 10, "name": "x", "value": 40}]])
def test_enrich_food_without_recipe_measures_is_per_100g(tmp_path, measures):
    food = {"id": 100, "name": "Bar", "measures": measures,
            "nutrients": [{"id": 208, "amount": 400}]}  # fmt: skip

    entry = _enrich_one(tmp_path, food, 10, 30)

    assert entry["recipe"] is False
    assert entry["nutrients"] == {208: 120.0}


def test_enrich_recipe_entry_ignores_the_default_measure(tmp_path):
    entry = _enrich_one(tmp_path, LIVE_SERVING_RECIPE, 313595764, 1)

    assert entry["serving_size"]["unit"] == "portion"
    assert entry["nutrients"] == {208: 71.5}


def test_enrich_diary_raises_when_get_foods_fails(tmp_path):
    client = _enrich_client(tmp_path)

    def boom(ids):
        raise CronometerError("network down")

    client.get_foods = boom  # type: ignore[method-assign]

    with pytest.raises(CronometerError, match="network down"):
        client.enrich_diary_servings(
            {"diary": [dict(e) for e in SAMPLE_DIARY["diary"]]}
        )


def test_enrich_diary_unknown_food_raises_the_not_found_error(tmp_path):
    client, _ = make_server_client(
        tmp_path, FakeFoodServer(ingredient_foods=SAMPLE_FOODS[1:])
    )

    with pytest.raises(CronometerError, match=r"doGetFoods\(\): Food 100 not found$"):
        client.enrich_diary_servings(
            {"diary": [dict(e) for e in SAMPLE_DIARY["diary"]]}
        )


# ---------------------------------------------------------------------------
# create_recipe
#
# Recipes post to the same /api/v2/add_food endpoint as custom foods; the
# `ingredients` array is what makes a food a recipe. The server derives the
# per-100g nutrients from the ingredients, so the client sends none.
# ---------------------------------------------------------------------------

RECIPE_INGREDIENT_FOODS = [
    {
        "id": 1000,
        "name": "Sardines",
        "defaultMeasureId": 11,
        "measures": [
            {"id": 10, "name": "oz", "value": 28.3495231, "type": "Weight"},
            {"id": 11, "name": "g", "value": 1, "type": "Weight"},
        ],
        "translations": [{"translationId": 555, "name": "Sardines"}],
        # per-100g
        "nutrients": [{"id": 208, "amount": 200.0}, {"id": 203, "amount": 25.0}],
    },
    {
        "id": 2000,
        "name": "Marshmallow Creme",
        "defaultMeasureId": 21,
        "measures": [{"id": 21, "name": "g", "value": 1, "type": "Weight"}],
        # no translations -> translationId falls back to 0
        "nutrients": [{"id": 208, "amount": 300.0}, {"id": 205, "amount": 60.0}],
    },
]


def first_save(state) -> dict:
    return state["payloads"][state["endpoints"].index("add_food")]["data"]


def _recipe_client(tmp_path: Path):
    return make_server_client(
        tmp_path, FakeFoodServer(ingredient_foods=RECIPE_INGREDIENT_FOODS)
    )


def test_create_recipe_payload_shape(tmp_path):
    """Ingredients and weight-based measures; nutrients are left to the server."""
    client, state = _recipe_client(tmp_path)

    result = client.create_recipe(
        "Test Recipe", ingredients=[(1000, 100.0), (2000, 50.0)]
    )

    assert result == {
        "food_id": 4242,
        "measure_id": 9000,
        "serving_grams": 150.0,
        "total_grams": 150.0,
        "ingredient_count": 2,
    }
    data = first_save(state)

    # A new food, with ingredients -- the marker that makes it a recipe.
    assert data["id"] == 0
    assert data["defaultMeasureId"] == 0
    assert data["properties"] == {"advancedServingSize": "false"}
    assert data["ingredients"] == [
        {
            "id": 0,
            "foodId": 1000,
            "measureId": 11,  # resolved to the 1-gram measure, not defaultMeasureId
            "translationId": 555,
            "grams": 100.0,
            "value": 100.0,
        },
        {
            "id": 0,
            "foodId": 2000,
            "measureId": 21,
            "translationId": 0,  # no translations on this food
            "grams": 50.0,
            "value": 50.0,
        },
    ]

    # Weight-based: every measure is type "Weight" (not "Recipe"), so the diary
    # treats logged amounts as real grams rather than a serving count.
    assert [m["type"] for m in data["measures"]] == ["Weight"] * 4
    by_name = {m["name"]: m["value"] for m in data["measures"]}
    assert by_name == {
        "Serving": 150.0,  # defaults to the full batch
        "g": 1.0,
        "oz": 28.3495231,
        "full recipe": 150.0,
    }
    # The serving measure leads, since the server makes the first one default.
    assert data["measures"][0]["name"] == "Serving"

    # The server derives nutrients from the ingredients (verified live with an
    # empty list), so none are sent.
    assert data["nutrients"] == []


def test_create_recipe_explicit_serving_and_comments(tmp_path):
    client, state = _recipe_client(tmp_path)

    client.create_recipe(
        "Test Recipe",
        ingredients=[(1000, 100.0), (2000, 50.0)],
        serving_name="bowl",
        serving_grams=75.0,
        comments="wildly inadvisable",
    )

    data = first_save(state)
    assert data["measures"][0] == {
        "id": 0,
        "name": "bowl",
        "value": 75.0,
        "amount": 1.0,
        "type": "Weight",
    }
    # An explicit serving size must not change the batch weight.
    assert {m["name"]: m["value"] for m in data["measures"]}["full recipe"] == 150.0
    assert data["comments"] == "wildly inadvisable"


def test_create_recipe_measure_id_override(tmp_path):
    """A 3-tuple overrides the auto-resolved display measure."""
    client, state = _recipe_client(tmp_path)

    client.create_recipe("Test Recipe", ingredients=[(1000, 100.0, 10)])

    assert first_save(state)["ingredients"][0]["measureId"] == 10


def test_create_recipe_rejects_empty_and_malformed(tmp_path):
    client, state = _recipe_client(tmp_path)

    with pytest.raises(ValueError):
        client.create_recipe("Test Recipe", ingredients=[])

    with pytest.raises(ValueError):
        client.create_recipe("Test Recipe", ingredients=[(1000,)])

    with pytest.raises(ValueError):
        client.create_recipe("Test Recipe", ingredients=[(1000, 0)])

    # Validation happens before any network call.
    assert state["endpoints"] == []


def test_create_recipe_missing_ingredient_food_raises(tmp_path):
    """An unresolvable food ID fails loudly rather than silently dropping it."""
    client, state = _recipe_client(tmp_path)

    with pytest.raises(CronometerError):
        client.create_recipe("Test Recipe", ingredients=[(1000, 100.0), (9999, 10.0)])

    assert state["endpoints"] == ["get_foods"]


# ---------------------------------------------------------------------------
# import_recipe
#
# Async flow: import_recipe returns a future id, poll_async_result is polled
# until it attaches a result, then matches are saved via add_food. Fixtures are
# the real bodies captured from the app for "one hot dog\nketchup\nbun".
# Verified against the live API (food 81033679).
# ---------------------------------------------------------------------------

IMPORT_JOB_ID = "69326e05-6f0e-4227-bd13-7a21e0fee9d6"

IMPORT_STARTED = {
    "isError": False,
    "progress": 0,
    "messages": [],
    "id": IMPORT_JOB_ID,
    "message": "Preparing your recipe for import.",
    "userId": 6407976,
}

IMPORT_IN_PROGRESS = {
    "isError": False,
    "progress": 10,
    "messages": [],
    "id": IMPORT_JOB_ID,
    "message": "Searching our database for the ingredients...",
    "userId": 6407976,
}


def _import_entry(raw, description, food_id, measure_id, grams, top_k):
    return {
        "amountSearchString": "Optional[1.0]",
        "unitSearchString": "",
        "ingredient": {
            "measureId": measure_id,
            "translationId": 0,
            "foodId": food_id,
            "id": 0,
            "grams": grams,
            "value": grams,
        },
        "rawIngredientImport": raw,
        "description": description,
        "topKFoodIds": top_k,
        "servingSizeMatchFound": True,
    }


IMPORT_ENTRIES = [
    _import_entry(
        "one hot dog", "hot dog", 4572, 12695, 45, [4572, 449773, 459646, 461995]
    ),
    _import_entry("ketchup", "ketchup", 449782, 992859, 1, [449782, 11239521, 4387294]),
    _import_entry("bun", "bun", 456806, 1030841, 43, [456806, 461989, 33638819]),
]

IMPORT_COMPLETE = {
    "result": {
        "entries": IMPORT_ENTRIES,
        "recipe": {
            "name": "Classic Hot Dog Bun",
            "id": 0,
            "ingredients": [e["ingredient"] for e in IMPORT_ENTRIES],
        },
    },
    "isError": False,
    "progress": 100,
    "messages": [],
    "id": IMPORT_JOB_ID,
    "message": "Finishing Up",
    "userId": 6407976,
}

# Ingredient foods the saved recipe resolves against, keyed to the captured IDs.
IMPORT_INGREDIENT_FOODS = [
    {
        "id": 4572,
        "name": "Hot Dog",
        "measures": [{"id": 12695, "name": "g", "value": 1, "type": "Weight"}],
        "nutrients": [{"id": 208, "amount": 290.0}],
    },
    {
        "id": 449782,
        "name": "Ketchup",
        "measures": [{"id": 992859, "name": "g", "value": 1, "type": "Weight"}],
        "nutrients": [{"id": 208, "amount": 100.0}],
    },
    {
        "id": 456806,
        "name": "Bun",
        "measures": [{"id": 1030841, "name": "g", "value": 1, "type": "Weight"}],
        "nutrients": [{"id": 208, "amount": 280.0}],
    },
]


def _import_client(tmp_path: Path, monkeypatch, poll_bodies):
    """Client whose async-job endpoints replay `poll_bodies` in order.

    Routes by endpoint so the trailing add_food save is recorded separately
    from polling traffic. time.sleep is neutered to keep the loop instant.
    """
    monkeypatch.setattr("cronometer_api_mcp.client.time.sleep", lambda _s: None)

    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = 42
    client._token = "TOKEN"
    client.get_foods = lambda ids: IMPORT_INGREDIENT_FOODS  # type: ignore[method-assign]

    state = {"calls": [], "polls": 0, "add_food": []}
    remaining = list(poll_bodies)
    server = FakeFoodServer(ingredient_foods=IMPORT_INGREDIENT_FOODS)
    server.next_id = 76533895

    def fake_post(endpoint, json=None):
        state["calls"].append(endpoint)
        if endpoint == "/api/v2/add_food":
            state["add_food"].append(json)
        if endpoint in ("/api/v2/add_food", "/api/v2/get_food"):
            return FakeResp(server.handle(endpoint, json))
        if endpoint in ("/api/v2/import_recipe", "/api/v2/poll_async_result"):
            if endpoint == "/api/v2/poll_async_result":
                state["polls"] += 1
            return FakeResp(remaining.pop(0) if remaining else poll_bodies[-1])
        raise AssertionError(f"unexpected endpoint {endpoint}")

    client._http.post = fake_post  # type: ignore[method-assign]
    return client, state


def test_import_recipe_parses_polls_and_saves(tmp_path, monkeypatch):
    """Happy path: start, poll past an in-progress tick, then save the match."""
    client, state = _import_client(
        tmp_path,
        monkeypatch,
        [IMPORT_STARTED, IMPORT_IN_PROGRESS, IMPORT_COMPLETE],
    )

    result = client.import_recipe("one hot dog\nketchup\nbun")

    # Name comes from the server when the caller doesn't supply one.
    assert result["recipe_name"] == "Classic Hot Dog Bun"
    assert result["food_id"] == 76533895
    assert result["ingredient_count"] == 3
    assert result["total_grams"] == 89.0  # 45 + 1 + 43
    assert result["unmatched"] == []

    started = state["calls"][0]
    assert started == "/api/v2/import_recipe"
    assert state["polls"] == 2

    # One save, carrying the parser's food/measure/gram triples.
    assert len(state["add_food"]) == 1
    ingredients = state["add_food"][0]["data"]["ingredients"]
    assert [(i["foodId"], i["measureId"], i["grams"]) for i in ingredients] == [
        (4572, 12695, 45.0),
        (449782, 992859, 1.0),
        (456806, 1030841, 43.0),
    ]

    # Matches are reported back for review, alternates included.
    assert result["ingredients"][0]["raw_text"] == "one hot dog"
    assert result["ingredients"][0]["description"] == "hot dog"
    assert result["ingredients"][0]["alternate_food_ids"] == [
        4572,
        449773,
        459646,
        461995,
    ]


def test_import_recipe_sends_expected_payloads(tmp_path, monkeypatch):
    """Request bodies match the captured app traffic."""
    client, _state = _import_client(
        tmp_path, monkeypatch, [IMPORT_STARTED, IMPORT_COMPLETE]
    )
    sent = []
    inner = client._http.post

    def recording_post(endpoint, json=None):
        sent.append((endpoint, json))
        return inner(endpoint, json=json)

    client._http.post = recording_post  # type: ignore[method-assign]

    client.import_recipe("one hot dog\nketchup\nbun")

    _, start_body = sent[0]
    assert start_body["url"] == ""
    assert start_body["ingredients"] == "one hot dog\nketchup\nbun"
    assert start_body["enable_async"] is True

    _, poll_body = sent[1]
    assert poll_body["futureId"] == IMPORT_JOB_ID
    assert poll_body["resultType"] == "recipe"


def test_import_recipe_save_false_skips_persistence(tmp_path, monkeypatch):
    """Parse-only mode returns matches without creating a food."""
    client, state = _import_client(
        tmp_path, monkeypatch, [IMPORT_STARTED, IMPORT_COMPLETE]
    )

    result = client.import_recipe("one hot dog\nketchup\nbun", save=False)

    assert state["add_food"] == []
    assert "food_id" not in result
    assert len(result["ingredients"]) == 3


def test_import_recipe_explicit_name_overrides_server_name(tmp_path, monkeypatch):
    client, state = _import_client(
        tmp_path, monkeypatch, [IMPORT_STARTED, IMPORT_COMPLETE]
    )

    result = client.import_recipe("one hot dog\nketchup\nbun", name="Ballpark Dog")

    assert result["recipe_name"] == "Ballpark Dog"
    assert state["add_food"][0]["data"]["name"] == "Ballpark Dog"


def test_import_recipe_result_on_first_response_skips_polling(tmp_path, monkeypatch):
    """A job that finishes immediately needs no poll round trip."""
    immediate = IMPORT_STARTED | {
        "progress": 100,
        "result": IMPORT_COMPLETE["result"],
    }
    client, state = _import_client(tmp_path, monkeypatch, [immediate])

    result = client.import_recipe("one hot dog")

    assert state["polls"] == 0
    assert result["food_id"] == 76533895


def test_import_recipe_server_error_raises(tmp_path, monkeypatch):
    """isError on the kickoff response fails loudly and saves nothing."""
    failed = {
        "isError": True,
        "progress": 0,
        "id": IMPORT_JOB_ID,
        "message": "Could not parse those ingredients.",
    }
    client, state = _import_client(tmp_path, monkeypatch, [failed])

    with pytest.raises(CronometerError, match="Could not parse those ingredients"):
        client.import_recipe("asdfgh")

    assert state["add_food"] == []


def test_import_recipe_poll_error_raises(tmp_path, monkeypatch):
    """isError surfacing mid-poll aborts the import."""
    failed = {"isError": True, "id": IMPORT_JOB_ID, "message": "Import failed"}
    client, state = _import_client(tmp_path, monkeypatch, [IMPORT_STARTED, failed])

    with pytest.raises(CronometerError, match="Import failed"):
        client.import_recipe("one hot dog")

    assert state["add_food"] == []


def test_import_recipe_times_out(tmp_path, monkeypatch):
    """A job that never completes raises rather than polling forever."""
    client, state = _import_client(
        tmp_path, monkeypatch, [IMPORT_STARTED, IMPORT_IN_PROGRESS]
    )

    # Advance the clock a minute per reading so the deadline trips quickly.
    ticks = iter(range(0, 10_000, 60))
    monkeypatch.setattr(
        "cronometer_api_mcp.client.time.monotonic", lambda: float(next(ticks))
    )

    with pytest.raises(CronometerError, match="did not finish within"):
        client.import_recipe("one hot dog", timeout=30.0)

    assert state["add_food"] == []


def test_import_recipe_finished_without_result_raises(tmp_path, monkeypatch):
    """100% progress with no payload is a server bug, not a silent success."""
    empty = {"isError": False, "progress": 100, "id": IMPORT_JOB_ID, "messages": []}
    client, state = _import_client(tmp_path, monkeypatch, [IMPORT_STARTED, empty])

    with pytest.raises(CronometerError, match="without a result"):
        client.import_recipe("one hot dog")

    assert state["add_food"] == []


def test_import_recipe_separates_unmatched_lines(tmp_path, monkeypatch):
    """Unresolvable lines are reported, not fed to create_recipe."""
    partial = {
        "isError": False,
        "progress": 100,
        "id": IMPORT_JOB_ID,
        "result": {
            "recipe": {"name": "Partial Recipe"},
            "entries": [
                IMPORT_ENTRIES[0],
                # No food matched.
                _import_entry("a pinch of unobtainium", "unobtainium", 0, 0, 0, []),
                # Matched a food but weighed nothing.
                _import_entry("air", "air", 999, 111, 0, [999]),
            ],
        },
    }
    client, state = _import_client(tmp_path, monkeypatch, [IMPORT_STARTED, partial])

    result = client.import_recipe("one hot dog\na pinch of unobtainium\nair")

    assert [i["raw_text"] for i in result["ingredients"]] == ["one hot dog"]
    assert [u["raw_text"] for u in result["unmatched"]] == [
        "a pinch of unobtainium",
        "air",
    ]
    # Only the usable line reaches the save.
    saved = state["add_food"][0]["data"]["ingredients"]
    assert [i["foodId"] for i in saved] == [4572]


def test_import_recipe_all_unmatched_raises(tmp_path, monkeypatch):
    """Nothing usable means no empty recipe gets created."""
    nothing = {
        "isError": False,
        "progress": 100,
        "id": IMPORT_JOB_ID,
        "result": {
            "recipe": {"name": "Empty"},
            "entries": [_import_entry("unobtainium", "unobtainium", 0, 0, 0, [])],
        },
    }
    client, state = _import_client(tmp_path, monkeypatch, [IMPORT_STARTED, nothing])

    with pytest.raises(CronometerError, match="no usable ingredients"):
        client.import_recipe("unobtainium")

    assert state["add_food"] == []


def test_import_recipe_rejects_blank_input(tmp_path, monkeypatch):
    """Validation happens before any network call."""
    client, state = _import_client(tmp_path, monkeypatch, [IMPORT_STARTED])

    with pytest.raises(ValueError):
        client.import_recipe("   \n  ")

    assert state["calls"] == []


# ---------------------------------------------------------------------------
# create_custom_food: extra_nutrients
# ---------------------------------------------------------------------------


def test_create_custom_food_extra_nutrients_payload(tmp_path):
    """extra_nutrients are appended, normalized to per-100g like the macros."""
    client, state = make_server_client(tmp_path, FakeFoodServer())

    client.create_custom_food(
        "Test Food",
        calories=200,
        protein_g=10,
        fat_g=5,
        carbs_g=20,
        serving_grams=200.0,  # scale = 0.5
        extra_nutrients={601: 30.0, 430: 10.0},  # cholesterol, vitamin K
    )

    nutrients = state["payloads"][0]["data"]["nutrients"]
    by_id = {n["id"]: n["amount"] for n in nutrients}
    assert by_id[601] == 15.0  # 30 * 0.5
    assert by_id[430] == 5.0  # 10 * 0.5


def test_create_custom_food_extra_nutrients_overlap_raises(tmp_path):
    """Reusing an ID the named macro args already write is rejected, not
    silently duplicated or shadowed."""
    client, _ = make_client(tmp_path, [{"result": "SUCCESS", "id": 555}])

    with pytest.raises(ValueError):
        client.create_custom_food(
            "Test Food",
            calories=200,
            protein_g=10,
            fat_g=5,
            carbs_g=20,
            extra_nutrients={204: 5.0},  # fat -- already set by fat_g
        )


def test_enrich_diary_no_servings_skips_lookup(tmp_path):
    client = _enrich_client(tmp_path)
    calls = {"n": 0}

    def track(ids):
        calls["n"] += 1
        return []

    client.get_foods = track  # type: ignore[method-assign]

    out = client.enrich_diary_servings(
        {"diary": [{"type": "Exercise", "name": "Running"}]}
    )
    assert calls["n"] == 0
    assert out["diary"] == [{"type": "Exercise", "name": "Running"}]


# ---------------------------------------------------------------------------
# Concurrent auth (MCP SDK 2.x)
#
# The SDK runs sync tool handlers on worker threads, so tool calls share one
# client concurrently where 1.x serialized them. Auth is a read-modify-write
# over (_user_id, _token, _timezone) plus a session-file write, so an unguarded
# burst logs in once per thread against a rate-limited endpoint (#3).
# ---------------------------------------------------------------------------

THREADS = 8


def test_concurrent_cold_start_logs_in_once(tmp_path):
    """A burst of first-calls on a cold client triggers exactly one login."""
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = None
    client._token = None

    state = {"login": 0}
    start = threading.Barrier(THREADS)

    def fake_login() -> None:
        # Widen the race window: without the lock every thread has already
        # passed the `_token is None` check by the time the first one stores.
        state["login"] += 1
        time.sleep(0.05)
        client._user_id = 42
        client._token = "FRESH_TOKEN"

    client.login = fake_login  # type: ignore[method-assign]

    def worker() -> None:
        start.wait()
        client._ensure_auth()

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["login"] == 1
    assert client._token == "FRESH_TOKEN"


def test_concurrent_expired_token_logs_in_once(tmp_path):
    """Concurrent 401s on one stale token collapse into a single re-login."""
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = 123
    client._token = "STALE_TOKEN"

    state = {"login": 0}
    lock = threading.Lock()
    start = threading.Barrier(THREADS)

    def fake_login() -> None:
        with lock:
            state["login"] += 1
            n = state["login"]
        time.sleep(0.05)
        client._user_id = 123
        client._token = f"FRESH_TOKEN_{n}"

    def fake_post(endpoint, json=None):
        # Reject the stale token, accept anything issued by a login.
        if json["auth"]["token"] == "STALE_TOKEN":
            return FakeResp(REAL_FAIL_BODY)
        return FakeResp({"result": "SUCCESS"})

    client.login = fake_login  # type: ignore[method-assign]
    client._http.post = fake_post  # type: ignore[method-assign]

    results = []

    def worker() -> None:
        start.wait()
        results.append(client._request("/api/v2/get_diary", {}))

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["login"] == 1
    assert results == [{"result": "SUCCESS"}] * THREADS


def test_reauthenticate_skips_login_when_token_already_refreshed(tmp_path):
    """A caller holding an already-replaced token must not force a new login."""
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = 123
    client._token = "FRESH_TOKEN"

    state = {"login": 0}

    def fake_login() -> None:
        state["login"] += 1

    client.login = fake_login  # type: ignore[method-assign]

    client._reauthenticate("STALE_TOKEN")

    assert state["login"] == 0
    assert client._token == "FRESH_TOKEN"


def test_reauthenticate_logs_in_when_token_is_still_stale(tmp_path):
    """The thread that actually holds the current (stale) token does re-login."""
    client = CronometerClient(session_path=tmp_path / "session.json")
    client._user_id = 123
    client._token = "STALE_TOKEN"

    state = {"login": 0}

    def fake_login() -> None:
        state["login"] += 1
        client._token = "FRESH_TOKEN"

    client.login = fake_login  # type: ignore[method-assign]

    client._reauthenticate("STALE_TOKEN")

    assert state["login"] == 1
    assert client._token == "FRESH_TOKEN"


LEADING_QUANTITY_NAMES = [
    "2 bowls",
    "1.5 cups",
    "1/2 cup",
    "1 1/2 cups",
    "  3 plates",
    "½ cup",
    "1½ cups",
    ".5 cup",
    "2x bowls",
    "2-3 bowls",
    "1,5 cups",
    "2",
    "⅓ batch",
    "⅞ cup",
    "¼",
    "⅟2 cup",
    "↉",
]


def _create(client, **kwargs):
    return client.create_custom_food(
        "Test Food", calories=200, protein_g=10, fat_g=5, carbs_g=20, **kwargs
    )


@pytest.mark.parametrize(
    ("kwargs", "name", "amount"),
    [
        ({}, "serving", 1),
        ({"serving_name": " cookies ", "serving_quantity": 2}, "cookies", 2),
    ],
)
def test_create_custom_food_serving_measure(tmp_path, kwargs, name, amount):
    client, state = make_server_client(tmp_path, FakeFoodServer())

    result = _create(client, serving_grams=30, **kwargs)

    assert state["endpoints"] == ["add_food", "get_food"]
    (measure,) = state["payloads"][0]["data"]["measures"]
    assert (measure["name"], measure["amount"], measure["value"]) == (name, amount, 30)
    # As the web client saves a weighed serving, so it counts in recipes.
    assert measure["type"] == "Weight"
    assert result == {"food_id": 4242, "measure_id": 9000}


@pytest.mark.parametrize(
    "serving_name", ["serving (1/2 batch)", "cup 1/3 full", "slice (1/4 pie)"]
)
def test_create_custom_food_resends_reparsed_serving(tmp_path, serving_name):
    """A new measure's name can be re-parsed by the server (server_reparse);
    the serving is re-sent once, verbatim, and the nutrients are kept."""
    server = FakeFoodServer()
    client, state = make_server_client(tmp_path, server)

    result = _create(client, serving_name=serving_name, serving_grams=250)

    assert state["endpoints"] == ["add_food", "get_food", "add_food", "get_food"]
    sent = state["payloads"][2]["data"]
    assert sent["nutrients"] == state["payloads"][0]["data"]["nutrients"]
    measure, gram = server.foods[result["food_id"]]["measures"]
    assert (measure["name"], measure["amount"], measure["value"]) == (
        serving_name,
        1,
        250,
    )
    assert (gram["name"], gram["value"], gram["type"]) == ("g", 1, "Weight")


def test_create_custom_food_raises_with_id_when_serving_wont_settle(tmp_path):
    """A serving still rewritten after the re-send is an error naming the
    food id, since the food already exists."""
    server = FakeFoodServer()
    client, state = make_server_client(tmp_path, server)
    save = server._save

    def reparse_every_save(food):
        food["measures"][0].update(name="batch)", amount=0.5)
        return save(food)

    server._save = reparse_every_save

    with pytest.raises(CronometerError, match=r"exists \(id 4242\).*reads back as"):
        _create(client, serving_name="serving (1/2 batch)")
    assert state["endpoints"] == ["add_food", "get_food", "add_food", "get_food"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"serving_name": n}, "pass the number as serving_quantity")
        for n in LEADING_QUANTITY_NAMES
    ]
    + [({"serving_name": n}, "serving_name must not be empty") for n in ("", "  ")]
    + [
        ({field: v}, field)
        for field in ("serving_grams", "serving_quantity")
        for v in (0, -1, float("nan"), float("inf"))
    ],
)
def test_create_custom_food_rejects_bad_serving(tmp_path, kwargs, match):
    client, state = make_cold_client(tmp_path, [])

    with pytest.raises(ValueError, match=match):
        _create(client, **kwargs)
    assert state["payloads"] == []


def _failing_server(fail_on: str, call: int) -> FakeFoodServer:
    """A FakeFoodServer whose `call`-th request to `fail_on` raises."""
    fake = FakeFoodServer()
    handle, seen = fake.handle, {"n": 0}

    def flaky(endpoint, body):
        if endpoint == f"/api/v2/{fail_on}":
            seen["n"] += 1
            if seen["n"] == call:
                raise CronometerError(f"{fail_on} timed out")
        return handle(endpoint, body)

    fake.handle = flaky
    return fake


@pytest.mark.parametrize(
    ("fail_on", "call", "serving_name", "add_foods"),
    [
        ("get_food", 1, "serving", 1),
        ("add_food", 2, "serving (1/2 batch)", 2),
    ],
)
def test_create_custom_food_failure_after_save_names_the_id(
    tmp_path, fail_on, call, serving_name, add_foods
):
    client, state = make_server_client(tmp_path, _failing_server(fail_on, call))

    with pytest.raises(CronometerError, match=r"exists \(id 4242\)"):
        _create(client, serving_name=serving_name)
    assert state["endpoints"].count("add_food") == add_foods


@pytest.mark.parametrize(
    ("break_read_back", "match"),
    [
        (lambda food: food.update(measures=[]), r"read back with measures \[\];"),
        (
            lambda food: food["measures"].append({**food["measures"][0], "id": 1}),
            (
                r"read back with measures \[\(9000, 'serving', 100.0\), "
                r"\(9001, 'g', 1\), \(1, 'serving', 100.0\)\]; expected its "
                r"serving 'serving' and at most Cronometer's g"
            ),
        ),
        (
            lambda food: food["measures"].pop(0),
            r"read back with measures \[\(9001, 'g', 1\)\]",
        ),
    ],
)
def test_create_custom_food_read_back_without_one_measure(
    tmp_path, break_read_back, match
):
    fake = FakeFoodServer()
    client, state = make_server_client(tmp_path, fake)
    handle = fake.handle

    def broken(endpoint, body):
        out = handle(endpoint, body)
        if endpoint == "/api/v2/get_food":
            break_read_back(out)
        return out

    fake.handle = broken

    with pytest.raises(CronometerError, match=rf"exists \(id 4242\).*{match}"):
        _create(client)
    assert state["endpoints"] == ["add_food", "get_food"]


@pytest.mark.parametrize("default", [0, 9000, None])
def test_create_custom_food_ignores_default_measure_id(tmp_path, default):
    """The one measure sent is the serving, whatever defaultMeasureId says."""
    fake = FakeFoodServer()
    client, state = make_server_client(tmp_path, fake)
    handle = fake.handle

    def read_back(endpoint, body):
        out = handle(endpoint, body)
        if endpoint == "/api/v2/get_food":
            out["defaultMeasureId"] = default
        return out

    fake.handle = read_back

    assert _create(client)["measure_id"] == 9000
    assert state["endpoints"] == ["add_food", "get_food"]


def _create_gram_food(tmp_path):
    fake = FakeFoodServer()
    client, state = make_server_client(tmp_path, fake)
    result = _create(client, serving_name="g", serving_grams=1)
    return client, state, fake, result


def test_create_gram_custom_food_with_zero_default_measure(tmp_path):
    """Live, a lone "g" serving is stored with defaultMeasureId 0."""
    _, state, fake, result = _create_gram_food(tmp_path)

    assert fake.foods[4242]["defaultMeasureId"] == 0
    assert result == {"food_id": 4242, "measure_id": 9000}
    assert state["endpoints"] == ["add_food", "get_food"]


def test_update_custom_food_gram_food_takes_per_gram_nutrients(tmp_path):
    client, state, fake, _ = _create_gram_food(tmp_path)

    client.update_custom_food(4242, name="Renamed", calories=3)

    assert state["endpoints"] == ["add_food", "get_food", "get_food", "add_food"]
    food = fake.foods[4242]
    assert food["name"] == "Renamed"
    assert {n["id"]: n["amount"] for n in food["nutrients"]}[208] == 300.0
    assert food["defaultMeasureId"] == 0
    assert [(m["id"], m["name"], m["value"]) for m in food["measures"]] == [
        (9000, "g", 1)
    ]


_CUP = {"id": 9001, "name": "cup", "value": 240, "amount": 1, "type": "Atomic"}


@pytest.mark.parametrize("kwargs", [{"calories": 3}, {"serving_grams": 2}])
def test_update_custom_food_two_measures_need_measure_id(tmp_path, kwargs):
    client, state, fake, _ = _create_gram_food(tmp_path)
    fake.foods[4242]["measures"].append(dict(_CUP))

    with pytest.raises(
        CronometerError,
        match=r"has 2 measures .*\(9000, 'g', 1\), \(9001, 'cup', 240\)\]; pass measure_id",
    ):
        client.update_custom_food(4242, **kwargs)
    assert state["endpoints"] == ["add_food", "get_food", "get_food"]


def test_update_custom_food_two_measures_still_rename(tmp_path):
    """A rename never touches a measure, so an ambiguous serving doesn't matter."""
    client, state, fake, _ = _create_gram_food(tmp_path)
    fake.foods[4242]["measures"].append(dict(_CUP))

    client.update_custom_food(4242, name="Renamed")

    assert state["endpoints"] == ["add_food", "get_food", "get_food", "add_food"]
    assert fake.foods[4242]["name"] == "Renamed"


def test_update_custom_food_lone_measure_is_the_serving(tmp_path):
    client, state, fake, _ = _create_gram_food(tmp_path)
    fake.foods[4242]["measures"] = [dict(_CUP)]

    client.update_custom_food(4242, calories=120)

    assert state["endpoints"] == ["add_food", "get_food", "get_food", "add_food"]
    assert {n["id"]: n["amount"] for n in fake.foods[4242]["nutrients"]}[208] == 50.0


def test_add_custom_food_tool_sends_serving_quantity(tmp_path, monkeypatch):
    client, state = make_server_client(tmp_path, FakeFoodServer())
    monkeypatch.setattr(mcp_server, "_get_client", lambda: client)

    out = json.loads(
        mcp_server.add_custom_food(
            "Cookies",
            calories=100,
            protein_g=2,
            fat_g=5,
            carbs_g=12,
            serving_name="cookies",
            serving_grams=30,
            serving_quantity=2,
        )
    )

    assert out["status"] == "success"
    assert (out["food_id"], out["measure_id"]) == (4242, 9000)
    (measure,) = state["payloads"][0]["data"]["measures"]
    assert (measure["name"], measure["amount"], measure["value"]) == ("cookies", 2, 30)


def test_update_custom_food_tool_passes_measure_id(tmp_path, monkeypatch):
    client, _, fake, _ = _create_gram_food(tmp_path)
    fake.foods[4242]["measures"].append(dict(_CUP))
    monkeypatch.setattr(mcp_server, "_get_client", lambda: client)

    refused = json.loads(mcp_server.update_custom_food(4242, calories=120))
    assert refused["status"] == "error"
    assert "pass measure_id" in refused["message"]

    out = json.loads(mcp_server.update_custom_food(4242, calories=120, measure_id=9001))
    assert out["status"] == "success"
    assert {n["id"]: n["amount"] for n in fake.foods[4242]["nutrients"]}[208] == 50.0
