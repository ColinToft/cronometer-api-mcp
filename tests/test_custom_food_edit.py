"""update_custom_food / retire_custom_food.

Both ride on /api/v2/add_food, which upserts: re-sending a food object with
its existing id edits it in place, and flipping `retired` hides it from search
and the Custom Foods list (the app's own delete hard-deletes via an endpoint we
haven't found; retiring is the closest the known API offers).
"""

from __future__ import annotations

import threading

import pytest
from test_client import LEADING_QUANTITY_NAMES, make_cold_client

from cronometer_api_mcp.client import CronometerError

FOOD_ID = 81460655
OK = {"result": "SUCCESS", "id": FOOD_ID}


def custom_food(**overrides) -> dict:
    """A get_food body shaped like the real one for a user-created food.

    Nutrients are per-100g; the default measure is a 50 g serving, so
    per-serving values are half of what is stored.
    """
    food = {
        "id": FOOD_ID,
        "name": "Test Food",
        "source": "Custom",
        "owner": 42,
        "retired": False,
        "category": 0,
        "labelType": "AMERICAN_2016",
        "foodTags": [],
        "barcodes": [],
        "properties": {},
        "meal": False,
        "defaultMeasureId": 900,
        "measures": [
            {
                "id": 900,
                "name": "serving",
                "amount": 1,
                "value": 50,
                "type": "Atomic",
                "hidden": False,
                "derivable": False,
                "derived": False,
            }
        ],
        "translations": [
            {"translationId": 1, "name": "Test Food", "languageCode": "en"}
        ],
        "nutrients": [
            {"id": 208, "amount": 200, "type": "PRIMARY"},
            {"id": 203, "amount": 20, "type": "PRIMARY"},
            {"id": 204, "amount": 10, "type": "PRIMARY"},
            {"id": 205, "amount": 40, "type": "PRIMARY"},
            {"id": 291, "amount": 4, "type": "PRIMARY"},
            {"id": 269, "amount": 0, "type": "PRIMARY"},
            {"id": 307, "amount": 0, "type": "PRIMARY"},
            {"id": 606, "amount": 0, "type": "PRIMARY"},
            {"id": -203, "amount": 20, "type": "PRIMARY"},
            {"id": -204, "amount": 10, "type": "PRIMARY"},
            {"id": -205, "amount": 40, "type": "PRIMARY"},
            {"id": -221, "amount": 0, "type": "PRIMARY"},
            {"id": -1205, "amount": 36, "type": "PRIMARY"},
            {"id": 601, "amount": 30, "type": "PRIMARY"},  # cholesterol (extra)
        ],
        # Server-side sync events that ride along on get_food; not food data.
        "messages": [
            {"id": 1, "type": "FOOD_CHANGED", "text": '{"id":1,"type":"ADD"}'}
        ],
    }
    food.update(overrides)
    return food


def recipe(**overrides) -> dict:
    """A user-created recipe: same shape as a custom food (get_food reports
    source "Custom" for both) plus an `ingredients` array."""
    return custom_food(
        name="Test Recipe",
        ingredients=[
            {"id": 1, "foodId": 12345, "grams": 100, "measureId": 0, "amount": 1}
        ],
        **overrides,
    )


def nutrients_by_id(payload: dict) -> dict[int, float]:
    return {n["id"]: n["amount"] for n in payload["data"]["nutrients"]}


# ---------------------------------------------------------------------------
# update_custom_food
# ---------------------------------------------------------------------------


def test_update_custom_food_resends_food_with_same_id_and_new_name(tmp_path):
    """The fetched food goes back to add_food with its id intact, the new name
    applied to both the name and its translation, and the sync messages
    stripped."""
    client, state = make_cold_client(tmp_path, [custom_food(), OK])

    result = client.update_custom_food(FOOD_ID, name="Renamed")

    get_payload, add_payload = state["payloads"]
    assert get_payload["id"] == FOOD_ID
    data = add_payload["data"]
    assert data["id"] == FOOD_ID
    assert data["name"] == "Renamed"
    assert [t["name"] for t in data["translations"]] == ["Renamed"]
    assert data["retired"] is False
    assert "messages" not in data
    assert add_payload["config"] == {"call_version": 1}
    assert result == {"food_id": FOOD_ID, "name": "Renamed"}


def test_update_custom_food_scales_macros_by_default_serving(tmp_path):
    """Nutrient args are per serving; storage is per-100g, so a 50 g default
    measure doubles them. Nutrients not passed keep their stored values."""
    client, state = make_cold_client(tmp_path, [custom_food(), OK])

    client.update_custom_food(FOOD_ID, calories=150, sodium_mg=30)

    by_id = nutrients_by_id(state["payloads"][1])
    assert by_id[208] == 300.0  # 150 kcal per 50 g serving
    assert by_id[307] == 60.0
    assert by_id[203] == 20  # untouched
    assert by_id[204] == 10


def test_update_custom_food_refreshes_derived_fields(tmp_path):
    """The app mirrors protein/fat/carbs into -203/-204/-205 and keeps
    net carbs (-1205) = carbs - fiber; an edit must keep those consistent."""
    client, state = make_cold_client(tmp_path, [custom_food(), OK])

    client.update_custom_food(FOOD_ID, protein_g=15, carbs_g=10, fiber_g=4)

    by_id = nutrients_by_id(state["payloads"][1])
    assert by_id[203] == 30.0
    assert by_id[-203] == 30.0
    assert by_id[205] == 20.0
    assert by_id[-205] == 20.0
    assert by_id[291] == 8.0
    assert by_id[-1205] == 12.0  # 20 - 8
    assert by_id[-204] == 10  # fat untouched


def test_update_custom_food_serving_overrides_measure_and_scale(tmp_path):
    """serving_grams / serving_name edit the default measure, and nutrient
    args are interpreted against the new serving size."""
    client, state = make_cold_client(tmp_path, [custom_food(), OK])

    client.update_custom_food(
        FOOD_ID, serving_name="cup", serving_grams=200, calories=400
    )

    data = state["payloads"][1]["data"]
    (measure,) = data["measures"]
    assert measure["id"] == 900
    assert measure["name"] == "cup"
    assert measure["amount"] == 1.0
    assert measure["value"] == 200
    assert nutrients_by_id(state["payloads"][1])[208] == 200.0  # 400 per 200 g


def weight_default_food() -> dict:
    """A food first saved as "400 g": its default measure is 400 units of a
    Weight measure, which the app renders as "400 <name>"."""
    food = custom_food()
    food["measures"][0].update(name="1 serving", amount=400, value=250, type="Weight")
    return food


def test_update_custom_food_serving_grams_resets_quantity_and_type(tmp_path):
    """Setting the weight makes the default measure one named serving, so a
    food first saved as "400 g" stops rendering as "400 <name>"."""
    client, state = make_cold_client(tmp_path, [weight_default_food(), OK])

    client.update_custom_food(FOOD_ID, serving_name="serving", serving_grams=250)

    (measure,) = state["payloads"][1]["data"]["measures"]
    assert measure["id"] == 900
    assert measure["name"] == "serving"
    assert measure["value"] == 250
    assert measure["amount"] == 1.0
    assert measure["type"] == "Atomic"


def test_update_custom_food_serving_quantity(tmp_path):
    """serving_quantity sets how many units the default serving is, alone or
    together with the weight."""
    client, state = make_cold_client(
        tmp_path, [weight_default_food(), OK, custom_food(), OK]
    )

    client.update_custom_food(FOOD_ID, serving_quantity=1)
    (measure,) = state["payloads"][1]["data"]["measures"]
    assert (measure["amount"], measure["value"], measure["type"]) == (1, 250, "Atomic")

    client.update_custom_food(FOOD_ID, serving_grams=30, serving_quantity=2)
    (measure,) = state["payloads"][3]["data"]["measures"]
    assert (measure["amount"], measure["value"]) == (2, 30)


@pytest.mark.parametrize(
    ("measure_id", "quantity", "unit", "grams", "counts"),
    [
        (900, 2, "cookies", 30, (0.5, 1, 1.5)),
        (901, 1, "cookie", 15, (1, 2, 3)),
    ],
)
def test_update_custom_food_serving_size_in_diary(
    tmp_path, monkeypatch, measure_id, quantity, unit, grams, counts
):
    food = custom_food()
    food["measures"].append(
        {"id": 901, "name": "cookie", "amount": 1, "value": 15, "type": "Atomic"}
    )
    client, state = make_cold_client(tmp_path, [food, OK])
    client.update_custom_food(
        FOOD_ID,
        serving_grams=30,
        serving_name="cookies",
        serving_quantity=2,
        measure_id=900,
    )

    saved = state["payloads"][1]["data"]
    monkeypatch.setattr(client, "get_foods", lambda ids: [saved])
    diary = client.enrich_diary_servings(
        {
            "diary": [
                {
                    "type": "Serving",
                    "foodId": FOOD_ID,
                    "measureId": measure_id,
                    "grams": consumed_grams,
                }
                for consumed_grams in (15, 30, 45)
            ]
        }
    )

    for entry, count, calories in zip(diary["diary"], counts, (30, 60, 90)):
        assert entry["serving_size"] == {
            "measure_id": measure_id,
            "quantity": quantity,
            "unit": unit,
            "grams": grams,
        }
        assert entry["servings"] == count
        assert "measure" not in entry
        assert "quantity" not in entry
        assert entry["nutrients"][208] == calories


@pytest.mark.parametrize("kwargs", [{}, {"serving_quantity": 2}])
@pytest.mark.parametrize("serving_name", LEADING_QUANTITY_NAMES)
def test_update_custom_food_rejects_leading_quantity(tmp_path, serving_name, kwargs):
    client, state = make_cold_client(tmp_path, [])

    with pytest.raises(ValueError, match="pass the number as serving_quantity"):
        client.update_custom_food(FOOD_ID, serving_name=serving_name, **kwargs)
    assert state["payloads"] == []


@pytest.mark.parametrize("serving_name", ["", "   "])
def test_update_custom_food_rejects_blank_serving_name(tmp_path, serving_name):
    client, state = make_cold_client(tmp_path, [])

    with pytest.raises(ValueError, match="serving_name must not be empty"):
        client.update_custom_food(FOOD_ID, serving_name=serving_name)
    assert state["payloads"] == []


@pytest.mark.parametrize(
    "serving_name", ["cookie (2 per pack)", "serving #2", "half (½) cup"]
)
def test_update_custom_food_keeps_inner_numbers(tmp_path, serving_name):
    client, state = make_cold_client(tmp_path, [weight_default_food(), OK])

    client.update_custom_food(FOOD_ID, serving_name=f"  {serving_name} ")

    (measure,) = state["payloads"][1]["data"]["measures"]
    assert (measure["name"], measure["amount"]) == (serving_name, 400)


def test_update_custom_food_serving_name_alone_keeps_quantity(tmp_path):
    client, state = make_cold_client(tmp_path, [weight_default_food(), OK])

    client.update_custom_food(FOOD_ID, serving_name="portion")

    (measure,) = state["payloads"][1]["data"]["measures"]
    assert (measure["name"], measure["amount"], measure["type"]) == (
        "portion",
        400,
        "Weight",
    )


@pytest.mark.parametrize("field", ["serving_grams", "serving_quantity"])
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_update_custom_food_rejects_bad_serving_numbers(tmp_path, field, bad):
    client, state = make_cold_client(tmp_path, [])

    with pytest.raises(ValueError, match=field):
        client.update_custom_food(FOOD_ID, **{field: bad})
    assert state["payloads"] == []


def test_update_custom_food_quantity_alone_keeps_total_weight(tmp_path):
    """Without serving_grams, a new quantity keeps the serving's total weight:
    "2 cookies = 30 g" becomes "3 cookies = 30 g"."""
    food = custom_food()
    food["measures"][0].update(name="cookies", amount=2, value=30)
    client, state = make_cold_client(tmp_path, [food, OK])

    client.update_custom_food(FOOD_ID, serving_quantity=3)

    (measure,) = state["payloads"][1]["data"]["measures"]
    assert (measure["name"], measure["amount"], measure["value"]) == (
        "cookies",
        3,
        30,
    )


def test_update_custom_food_extra_nutrients(tmp_path):
    """extra_nutrients are set (scaled) by id, replacing an existing entry or
    appending a new one, and may not overlap the named macro args."""
    client, state = make_cold_client(tmp_path, [custom_food(), OK])

    client.update_custom_food(FOOD_ID, extra_nutrients={601: 2.5, 430: 10.0})

    by_id = nutrients_by_id(state["payloads"][1])
    assert by_id[601] == 5.0  # cholesterol, replaced in place
    assert by_id[430] == 20.0  # vitamin K, appended
    ids = [n["id"] for n in state["payloads"][1]["data"]["nutrients"]]
    assert ids.count(601) == 1

    client2, _ = make_cold_client(tmp_path, [custom_food(), OK])
    with pytest.raises(ValueError):
        client2.update_custom_food(FOOD_ID, extra_nutrients={204: 1.0})


def test_update_custom_food_refuses_database_foods(tmp_path):
    """Only user-created foods may be edited; a database food is rejected
    before anything is sent to add_food."""
    client, state = make_cold_client(
        tmp_path, [custom_food(source="CRDB", owner=None), OK]
    )

    with pytest.raises(CronometerError, match="not a custom food"):
        client.update_custom_food(FOOD_ID, name="Renamed")

    assert len(state["payloads"]) == 1  # only the get_food


def test_update_custom_food_refuses_recipes(tmp_path):
    """A recipe also reports source "Custom", but its nutrients are derived
    from its ingredients and its measures mean something else, so editing it
    as a plain food would corrupt it. Reject before anything is sent."""
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    with pytest.raises(CronometerError, match="recipe"):
        client.update_custom_food(FOOD_ID, calories=100)

    assert len(state["payloads"]) == 1  # only the get_food


def test_update_custom_food_raises_when_server_returns_other_id(tmp_path):
    """add_food answering with a different id means it created a copy instead
    of editing in place; surface that rather than report success."""
    client, _ = make_cold_client(
        tmp_path, [custom_food(), {"result": "SUCCESS", "id": FOOD_ID + 1}]
    )

    with pytest.raises(CronometerError):
        client.update_custom_food(FOOD_ID, name="Renamed")


# ---------------------------------------------------------------------------
# retire_custom_food
# ---------------------------------------------------------------------------


def test_retire_custom_food_resends_food_with_retired_flag(tmp_path):
    client, state = make_cold_client(tmp_path, [custom_food(), OK])

    result = client.retire_custom_food(FOOD_ID)

    data = state["payloads"][1]["data"]
    assert data["id"] == FOOD_ID
    assert data["retired"] is True
    assert data["name"] == "Test Food"
    assert "messages" not in data
    assert result == {"food_id": FOOD_ID, "name": "Test Food", "retired": True}


def test_retire_custom_food_refuses_database_foods(tmp_path):
    client, state = make_cold_client(tmp_path, [custom_food(source="USDA"), OK])

    with pytest.raises(CronometerError, match="not a custom food"):
        client.retire_custom_food(FOOD_ID)

    assert len(state["payloads"]) == 1


def test_retire_custom_food_also_retires_recipes(tmp_path):
    """Retiring is the same flag for both, so recipes are accepted."""
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    result = client.retire_custom_food(FOOD_ID)

    data = state["payloads"][1]["data"]
    assert data["retired"] is True
    assert data["ingredients"] == recipe()["ingredients"]
    assert result == {"food_id": FOOD_ID, "name": "Test Recipe", "retired": True}


@pytest.mark.parametrize("name", ["", "   "])
def test_update_custom_food_rejects_blank_name(tmp_path, name):
    client, state = make_cold_client(tmp_path, [])

    with pytest.raises(ValueError, match="name must not be empty"):
        client.update_custom_food(FOOD_ID, name=name)

    assert state["payloads"] == []


def test_update_custom_food_refuses_retired_foods(tmp_path):
    client, state = make_cold_client(tmp_path, [custom_food(retired=True), OK])

    with pytest.raises(CronometerError, match="deleted"):
        client.update_custom_food(FOOD_ID, name="X")

    assert len(state["payloads"]) == 1


def test_retire_custom_food_accepts_already_retired_foods(tmp_path):
    client, state = make_cold_client(tmp_path, [custom_food(retired=True), OK])

    assert client.retire_custom_food(FOOD_ID)["retired"] is True
    assert state["payloads"][1]["data"]["retired"] is True


# ---------------------------------------------------------------------------
# Per-food locking
# ---------------------------------------------------------------------------


def test_edit_holds_the_food_lock_from_fetch_through_save(tmp_path):
    """Both the get_food and the add_food of an update or retire happen with
    the food's lock held, and it is released afterwards."""
    client, _ = make_cold_client(tmp_path, [custom_food(), OK, custom_food(), OK])
    lock = client._food_lock(FOOD_ID)
    seen: list[tuple[str, bool]] = []
    real_post = client._http.post

    def spying_post(endpoint, json=None):
        seen.append((endpoint, lock.locked()))
        return real_post(endpoint, json=json)

    client._http.post = spying_post  # type: ignore[method-assign]

    client.update_custom_food(FOOD_ID, name="Renamed")
    client.retire_custom_food(FOOD_ID)

    assert seen == [
        ("/api/v2/get_food", True),
        ("/api/v2/add_food", True),
        ("/api/v2/get_food", True),
        ("/api/v2/add_food", True),
    ]
    assert not lock.locked()
    assert client._food_lock(FOOD_ID) is lock  # one lock per food id


def test_concurrent_edits_of_one_food_are_serialized(tmp_path):
    """add_food replaces the whole object, so two interleaved
    fetch-modify-save cycles would clobber each other (or resurrect a food
    the other just retired). The second caller must not fetch until the first
    has saved."""
    client, state = make_cold_client(tmp_path, [custom_food(), OK, custom_food(), OK])
    first_fetched = threading.Event()
    release_first = threading.Event()
    calls: list[tuple[str, str]] = []
    real_post = client._http.post

    def stalling_post(endpoint, json=None):
        me = threading.current_thread().name
        calls.append((me, endpoint))
        if me == "first" and endpoint == "/api/v2/get_food":
            first_fetched.set()
            release_first.wait(timeout=5)
        return real_post(endpoint, json=json)

    client._http.post = stalling_post  # type: ignore[method-assign]

    first = threading.Thread(
        target=client.update_custom_food,
        args=(FOOD_ID,),
        kwargs={"name": "A"},
        name="first",
    )
    second = threading.Thread(
        target=client.retire_custom_food, args=(FOOD_ID,), name="second"
    )
    first.start()
    assert first_fetched.wait(timeout=5)
    second.start()
    second.join(timeout=0.2)
    assert second.is_alive()  # blocked on the lock, has not fetched
    assert calls == [("first", "/api/v2/get_food")]

    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert calls == [
        ("first", "/api/v2/get_food"),
        ("first", "/api/v2/add_food"),
        ("second", "/api/v2/get_food"),
        ("second", "/api/v2/add_food"),
    ]
    assert state["payloads"][1]["data"]["name"] == "A"
    assert state["payloads"][3]["data"]["retired"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"serving_name": "cup"},
        {"serving_grams": 30},
        {"serving_quantity": 2},
        {"calories": 100},
        {"extra_nutrients": {601: 5}},
    ],
)
def test_update_custom_food_without_measure_refuses_serving_edits(tmp_path, kwargs):
    client, state = make_cold_client(tmp_path, [custom_food(measures=[])])

    with pytest.raises(CronometerError, match="has no serving measure"):
        client.update_custom_food(FOOD_ID, **kwargs)
    assert len(state["payloads"]) == 1  # the read, no save


def test_update_custom_food_without_measure_still_renames(tmp_path):
    client, state = make_cold_client(tmp_path, [custom_food(measures=[]), OK])

    client.update_custom_food(FOOD_ID, name="Renamed")

    assert state["payloads"][1]["data"]["name"] == "Renamed"


@pytest.mark.parametrize("value", [0, -5, None, float("nan"), float("inf"), "50"])
def test_update_custom_food_refuses_unusable_serving_weight(tmp_path, value):
    """Per-serving nutrients need a real serving weight to convert to per-100g;
    there is no fallback scale."""
    food = custom_food()
    food["measures"][0]["value"] = value
    client, state = make_cold_client(tmp_path, [food, food, OK])

    with pytest.raises(CronometerError, match="per-serving nutrients"):
        client.update_custom_food(FOOD_ID, calories=100)
    assert len(state["payloads"]) == 1

    client.update_custom_food(FOOD_ID, calories=100, serving_grams=50)
    assert nutrients_by_id(state["payloads"][2])[208] == 200.0


GRAM_MEASURE = {"id": 901, "name": "g", "amount": 1, "value": 1, "type": "Weight"}


def two_measure_food(default: int) -> dict:
    food = custom_food(defaultMeasureId=default)
    food["measures"].append(dict(GRAM_MEASURE))
    return food


SERVING_EDITS = [
    {"calories": 100},
    {"extra_nutrients": {601: 5}},
    {"serving_name": "cup"},
    {"serving_grams": 30},
    {"serving_quantity": 2},
]


@pytest.mark.parametrize("default", [900, 901, 0, 12345])
@pytest.mark.parametrize("kwargs", SERVING_EDITS)
def test_update_custom_food_with_several_measures_requires_measure_id(
    tmp_path, default, kwargs
):
    client, state = make_cold_client(tmp_path, [two_measure_food(default)])

    with pytest.raises(
        CronometerError,
        match=r"has 2 measures \(id, name, grams\) \[\(900, 'serving', 50\), "
        r"\(901, 'g', 1\)\]; pass measure_id",
    ):
        client.update_custom_food(FOOD_ID, **kwargs)
    assert len(state["payloads"]) == 1  # the read, no save


@pytest.mark.parametrize("default", [900, 901, 0])
def test_update_custom_food_measure_id_picks_the_serving(tmp_path, default):
    client, state = make_cold_client(tmp_path, [two_measure_food(default), OK])

    client.update_custom_food(FOOD_ID, measure_id=900, calories=150, serving_grams=75)

    data = state["payloads"][1]["data"]
    assert data["defaultMeasureId"] == default
    serving, gram = data["measures"]
    assert (serving["id"], serving["value"], serving["amount"]) == (900, 75, 1.0)
    assert gram == GRAM_MEASURE
    assert nutrients_by_id(state["payloads"][1])[208] == 200.0


def test_update_custom_food_measure_id_can_pick_the_gram_measure(tmp_path):
    client, state = make_cold_client(tmp_path, [two_measure_food(900), OK])

    client.update_custom_food(FOOD_ID, measure_id=901, calories=1.3, protein_g=0.11)

    by_id = nutrients_by_id(state["payloads"][1])
    assert (by_id[208], by_id[203], by_id[-203]) == (130.0, 11.0, 11.0)
    assert state["payloads"][1]["data"]["measures"][1] == GRAM_MEASURE


@pytest.mark.parametrize("default", [900, 0, 12345])
def test_update_custom_food_sole_measure_ignores_default_measure_id(tmp_path, default):
    food = custom_food(defaultMeasureId=default)
    client, state = make_cold_client(tmp_path, [food, OK, food, OK])

    client.update_custom_food(FOOD_ID, calories=100)
    client.update_custom_food(FOOD_ID, calories=100, measure_id=900)

    for payload in state["payloads"][1], state["payloads"][3]:
        assert nutrients_by_id(payload)[208] == 200.0
        assert payload["data"]["defaultMeasureId"] == default


@pytest.mark.parametrize("food", [custom_food(), two_measure_food(900)])
def test_update_custom_food_refuses_unknown_measure_id(tmp_path, food):
    client, state = make_cold_client(tmp_path, [food])

    with pytest.raises(CronometerError, match="has no measure 777; its measures"):
        client.update_custom_food(FOOD_ID, calories=100, measure_id=777)
    assert len(state["payloads"]) == 1  # the read, no save


def test_update_custom_food_refuses_measure_id_without_serving_edit(tmp_path):
    client, state = make_cold_client(tmp_path, [])

    with pytest.raises(ValueError, match="pass it only with those"):
        client.update_custom_food(FOOD_ID, name="Renamed", measure_id=900)
    assert state["payloads"] == []


@pytest.mark.parametrize("unit", ["g", "oz", " G "])
@pytest.mark.parametrize("kwargs", SERVING_EDITS[2:])
def test_update_custom_food_refuses_renaming_or_resizing_weight_unit(
    tmp_path, unit, kwargs
):
    food = custom_food()
    food["measures"][0].update(name=unit, value=1)
    client, state = make_cold_client(tmp_path, [food])

    with pytest.raises(CronometerError, match="is the weight unit"):
        client.update_custom_food(FOOD_ID, **kwargs)
    assert len(state["payloads"]) == 1  # the read, no save


@pytest.mark.parametrize("unit", ["g", "oz", " G "])
def test_update_custom_food_refuses_renaming_serving_to_weight_unit(tmp_path, unit):
    client, state = make_cold_client(tmp_path, [])

    with pytest.raises(ValueError, match="is a weight unit"):
        client.update_custom_food(FOOD_ID, serving_name=unit)
    assert state["payloads"] == []
