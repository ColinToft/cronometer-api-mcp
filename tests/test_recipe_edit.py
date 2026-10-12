"""update_recipe / retire_recipe, and the update_recipe / delete_recipe tools.

Like custom foods, recipes are edited by re-sending the get_food object to
/api/v2/add_food, which upserts by id. Verified against the live API on
throwaway recipes:

- the food id, measure ids, and untouched fields survive a re-send;
- a re-sent `ingredients` list replaces the old one wholesale (rows sent with
  id 0 get fresh row ids, rows left out are dropped);
- the server re-derives the nutrients and the "full recipe" weight from the
  ingredients on every save (a deliberately wrong energy value read back as
  the derived one);
- an existing measure's name is stored verbatim, while a *new* measure named
  "serving (1/2 batch)" was stored as amount 0.5, name "batch)";
- retiring hides the recipe from find_food (after a few seconds of index lag)
  but get_food still reads it.

No hard-delete endpoint exists among the plausible v2/v3 names, so delete is
a retire, as for custom foods.
"""

from __future__ import annotations

import copy
import json
import math

import pytest
from test_client import make_cold_client

from cronometer_api_mcp import server
from cronometer_api_mcp.client import CronometerError

RECIPE_ID = 85008439
OK = {"result": "SUCCESS", "messages": [], "id": RECIPE_ID}

# Ingredient foods as get_foods returns them (per-100g nutrients).
BANANA = {
    "id": 450856,
    "name": "Banana, Fresh",
    "defaultMeasureId": 998940,
    "measures": [
        {"id": 998940, "name": "medium", "value": 118, "amount": 1, "type": "Atomic"},
        {"id": 998949, "name": "g", "value": 1, "amount": 1, "type": "Weight"},
    ],
    "translations": [{"translationId": 398492, "name": "Banana, Fresh"}],
    "nutrients": [{"id": 208, "amount": 89.0}, {"id": 203, "amount": 1.1}],
}
EGG = {
    "id": 1234,
    "name": "Egg, whole",
    "defaultMeasureId": 3770,
    "measures": [
        {"id": 3770, "name": "large", "value": 50, "amount": 1, "type": "Atomic"},
        {"id": 3776, "name": "g", "value": 1, "amount": 1, "type": "Weight"},
    ],
    "translations": [{"translationId": 1893, "name": "Egg, whole"}],
    "nutrients": [{"id": 208, "amount": 143.0}, {"id": 203, "amount": 12.6}],
}
PB = {
    "id": 619,
    "name": "Peanut Butter",
    "defaultMeasureId": 1960,
    # No gram measure: the ingredient falls back to the default measure.
    "measures": [
        {"id": 1960, "name": "tbsp", "value": 16, "amount": 1, "type": "Atomic"}
    ],
    # No translations: translationId falls back to 0.
    "nutrients": [{"id": 208, "amount": 588.0}, {"id": 204, "amount": 50.0}],
}


def foods_response(*foods: dict) -> dict:
    return {"foods": [copy.deepcopy(f) for f in foods]}


def recipe(**overrides) -> dict:
    """A get_food body shaped like the live one for a create_recipe recipe:
    banana 120 g + peanut butter 32 g, serving 76 g, batch 152 g."""
    food = {
        "id": RECIPE_ID,
        "owner": 42,
        "name": "Test Recipe",
        "comments": "old notes",
        "labelType": "AMERICAN_2016",
        "foodTags": [],
        "source": "Custom",
        "barcodes": [],
        "meal": False,
        "retired": False,
        "category": 0,
        "properties": {"advancedServingSize": "false"},
        "defaultMeasureId": 313167825,
        "measures": [
            {
                "id": 313167828,
                "name": "full recipe",
                "amount": 1,
                "value": 152,
                "type": "Weight",
                "hidden": False,
                "derivable": False,
                "derived": False,
            },
            {
                "id": 313167825,
                "name": "serving (1/2 batch)",
                "amount": 1,
                "value": 76,
                "type": "Weight",
                "hidden": False,
                "derivable": False,
                "derived": False,
            },
            {
                "id": 313167827,
                "name": "oz",
                "amount": 1,
                "value": 28.3495231,
                "type": "Weight",
                "hidden": False,
                "derivable": False,
                "derived": False,
            },
            {
                "id": 313167826,
                "name": "g",
                "amount": 1,
                "value": 1,
                "type": "Weight",
                "hidden": False,
                "derivable": False,
                "derived": False,
            },
        ],
        "translations": [
            {"translationId": 94417409, "name": "Test Recipe", "languageCode": "en"}
        ],
        "ingredients": [
            {
                "id": 459217357,
                "foodId": 450856,
                "measureId": 998949,
                "translationId": 398492,
                "grams": 120,
                "value": 120,
            },
            {
                "id": 459217358,
                "foodId": 619,
                "measureId": 1960,
                "translationId": 0,
                "grams": 32,
                "value": 32,
            },
        ],
        "nutrients": [
            {"id": 208, "amount": 194.0, "type": "PRIMARY"},
            {"id": 203, "amount": 0.87, "type": "PRIMARY"},
            {"id": 204, "amount": 10.53, "type": "PRIMARY"},
            {"id": 512, "amount": 0.158, "type": "PRIMARY"},
        ],
        # Server-side sync events that ride along on get_food; not food data.
        "messages": [{"id": 1, "type": "FOOD_CHANGED", "text": "{}"}],
    }
    food.update(overrides)
    return food


def saved(state) -> dict:
    """The food object sent to add_food (always the last request)."""
    payload = state["payloads"][-1]
    assert payload["config"] == {"call_version": 1}
    return payload["data"]


def endpoints(state) -> list[str]:
    # make_cold_client records payloads only; get_food sends "id", get_foods
    # "ids", add_food "data".
    out = []
    for p in state["payloads"]:
        if "data" in p:
            out.append("add_food")
        elif "ids" in p:
            out.append("get_foods")
        else:
            out.append("get_food")
    return out


def measures_by_id(food: dict) -> dict[int, dict]:
    return {m["id"]: m for m in food["measures"]}


def without(food: dict, *keys: str) -> dict:
    return {k: v for k, v in food.items() if k not in keys}


# ---------------------------------------------------------------------------
# update_recipe: metadata-only edits
# ---------------------------------------------------------------------------


def test_rename_resends_recipe_unchanged_apart_from_name(tmp_path):
    """A name-only edit re-sends the fetched recipe with its id; only the name
    and its matching translation change, the sync messages are stripped, and
    no ingredient lookup happens."""
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    result = client.update_recipe(RECIPE_ID, name="Renamed")

    assert endpoints(state) == ["get_food", "add_food"]
    assert state["payloads"][0]["id"] == RECIPE_ID
    data = saved(state)
    assert data["id"] == RECIPE_ID
    assert data["name"] == "Renamed"
    assert [t["name"] for t in data["translations"]] == ["Renamed"]
    assert "messages" not in data
    expected = without(recipe(), "messages", "name", "translations")
    assert without(data, "name", "translations") == expected
    assert result == {
        "food_id": RECIPE_ID,
        "name": "Renamed",
        "total_grams": 152,
        "ingredient_count": 2,
        "serving": {
            "measure_id": 313167825,
            "name": "serving (1/2 batch)",
            "quantity": 1,
            "grams": 76,
        },
    }


def test_rename_leaves_other_language_translations_alone(tmp_path):
    translations = [
        {"translationId": 1, "name": "Test Recipe", "languageCode": "en"},
        {"translationId": 2, "name": "Recette", "languageCode": "fr"},
    ]
    client, state = make_cold_client(tmp_path, [recipe(translations=translations), OK])

    client.update_recipe(RECIPE_ID, name="Renamed")

    assert [t["name"] for t in saved(state)["translations"]] == [
        "Renamed",
        "Recette",
    ]


@pytest.mark.parametrize("comments", ["new notes", ""])
def test_comments_set_or_cleared(tmp_path, comments):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    client.update_recipe(RECIPE_ID, comments=comments)

    data = saved(state)
    assert data["comments"] == comments
    assert without(data, "comments") == without(recipe(), "messages", "comments")


def test_comments_none_keeps_existing(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    client.update_recipe(RECIPE_ID, name="X")

    assert saved(state)["comments"] == "old notes"


# ---------------------------------------------------------------------------
# update_recipe: serving measure
# ---------------------------------------------------------------------------


def test_serving_grams_resizes_default_measure_only(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    result = client.update_recipe(RECIPE_ID, serving_grams=50.5)

    data = saved(state)
    by_id = measures_by_id(data)
    assert by_id[313167825] == {
        **measures_by_id(recipe())[313167825],
        "value": 50.5,
        "amount": 1.0,
    }
    assert by_id[313167825]["type"] == "Weight"  # recipes stay weight-based
    for mid in (313167828, 313167827, 313167826):
        assert by_id[mid] == measures_by_id(recipe())[mid]
    assert data["defaultMeasureId"] == 313167825
    assert data["nutrients"] == recipe()["nutrients"]
    assert data["ingredients"] == recipe()["ingredients"]
    assert result["serving"]["grams"] == 50.5


def test_serving_name_is_sent_verbatim_and_resets_quantity(tmp_path):
    """The server parsed a *new* measure named "serving (1/2 batch)" into
    amount 0.5 / name "batch)". Re-sending the existing measure stores the
    name verbatim (verified live), so this is also how such a recipe gets
    repaired: the quantity goes back to 1."""
    mangled = recipe()
    serving = measures_by_id(mangled)[313167825]
    serving.update(name="batch)", amount=0.5)
    client, state = make_cold_client(tmp_path, [mangled, OK])

    client.update_recipe(RECIPE_ID, serving_name="serving (1/6 batch)")

    m = measures_by_id(saved(state))[313167825]
    assert m["name"] == "serving (1/6 batch)"
    assert m["amount"] == 1.0
    assert m["value"] == 76  # weight untouched


@pytest.mark.parametrize(
    ("serving_name", "name", "quantity"),
    [
        ("2 bowls", "bowls", 2.0),
        ("1.5 cups", "cups", 1.5),
        ("  3   plates", "plates", 3.0),
    ],
)
def test_quantity_led_serving_name_moves_number_into_quantity(
    tmp_path, serving_name, name, quantity
):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    result = client.update_recipe(RECIPE_ID, serving_name=serving_name)

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"]) == (name, quantity)
    assert result["serving"]["quantity"] == quantity


def test_serving_name_and_grams_together(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    client.update_recipe(RECIPE_ID, serving_name="bowl", serving_grams=300)

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["value"]) == ("bowl", 1.0, 300)


@pytest.mark.parametrize("fixed", ["full recipe", "g", "oz"])
def test_serving_edit_refused_when_default_is_a_fixed_measure(tmp_path, fixed):
    """Renaming or resizing the "g"/"oz"/"full recipe" measure would corrupt
    it (the server keeps "full recipe" equal to the batch weight)."""
    mid = next(m["id"] for m in recipe()["measures"] if m["name"] == fixed)
    client, state = make_cold_client(tmp_path, [recipe(defaultMeasureId=mid), OK])

    with pytest.raises(CronometerError, match="can't be renamed or resized"):
        client.update_recipe(RECIPE_ID, serving_grams=10)

    assert endpoints(state) == ["get_food"]


def test_fixed_default_measure_still_allows_other_edits(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(defaultMeasureId=313167828), OK])

    client.update_recipe(RECIPE_ID, name="Renamed")

    assert saved(state)["name"] == "Renamed"


def test_missing_default_measure_raises(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(defaultMeasureId=1), OK])

    with pytest.raises(CronometerError, match="defaultMeasureId"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


# ---------------------------------------------------------------------------
# update_recipe: ingredients
# ---------------------------------------------------------------------------


def test_ingredients_replace_list_and_rederive_nutrients(tmp_path):
    """The new list replaces the old one with id-0 rows (the server assigns
    fresh row ids), nutrients are re-aggregated per-100g of the new batch,
    and "full recipe" follows the batch weight while the serving keeps its
    weight."""
    client, state = make_cold_client(
        tmp_path, [recipe(), foods_response(BANANA, EGG, PB), OK]
    )

    result = client.update_recipe(
        RECIPE_ID, ingredients=[(450856, 150), (1234, 50), (619, 16)]
    )

    assert endpoints(state) == ["get_food", "get_foods", "add_food"]
    assert state["payloads"][1]["ids"] == [450856, 1234, 619]
    data = saved(state)
    assert data["id"] == RECIPE_ID
    assert data["ingredients"] == [
        {
            "id": 0,
            "foodId": 450856,
            "measureId": 998949,  # the gram measure
            "translationId": 398492,
            "grams": 150.0,
            "value": 150.0,
        },
        {
            "id": 0,
            "foodId": 1234,
            "measureId": 3776,
            "translationId": 1893,
            "grams": 50.0,
            "value": 50.0,
        },
        {
            "id": 0,
            "foodId": 619,
            "measureId": 1960,  # no gram measure: default measure
            "translationId": 0,
            "grams": 16.0,
            "value": 16.0,
        },
    ]

    total = 216.0
    amounts = {n["id"]: n["amount"] for n in data["nutrients"]}
    assert amounts == {
        203: round((1.1 * 1.5 + 12.6 * 0.5) * 100 / total, 6),
        204: round(50.0 * 0.16 * 100 / total, 6),
        208: round((89 * 1.5 + 143 * 0.5 + 588 * 0.16) * 100 / total, 6),
    }
    # Nutrients the new ingredients don't carry (512 here) are gone, not stale.
    assert 512 not in amounts

    by_id = measures_by_id(data)
    assert by_id[313167828]["value"] == total  # full recipe
    assert by_id[313167825] == measures_by_id(recipe())[313167825]  # serving
    assert by_id[313167827] == measures_by_id(recipe())[313167827]
    assert by_id[313167826] == measures_by_id(recipe())[313167826]

    # Everything else is untouched.
    untouched = ("ingredients", "nutrients", "measures")
    assert without(data, *untouched) == without(recipe(), "messages", *untouched)
    assert result["total_grams"] == total
    assert result["ingredient_count"] == 3
    assert result["serving"]["grams"] == 76


def test_ingredient_measure_override(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), foods_response(EGG), OK])

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 100, 3770)])

    assert saved(state)["ingredients"][0]["measureId"] == 3770


def test_ingredients_and_serving_together(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), foods_response(EGG), OK])

    result = client.update_recipe(
        RECIPE_ID,
        ingredients=[(1234, 600)],
        serving_name="serving (1/6 batch)",
        serving_grams=100,
        name="Eggs",
        comments="",
    )

    data = saved(state)
    by_id = measures_by_id(data)
    assert by_id[313167828]["value"] == 600
    assert (by_id[313167825]["name"], by_id[313167825]["value"]) == (
        "serving (1/6 batch)",
        100,
    )
    assert data["name"] == "Eggs"
    assert data["comments"] == ""
    assert result["total_grams"] == 600


def test_recipe_without_full_recipe_measure_still_updates(tmp_path):
    """App-made recipes may lack the measure; the server derives the batch
    weight anyway, so its absence is not an error."""
    food = recipe()
    food["measures"] = [m for m in food["measures"] if m["name"] != "full recipe"]
    client, state = make_cold_client(tmp_path, [food, foods_response(EGG), OK])

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 100)])

    assert [m["name"] for m in saved(state)["measures"]] == [
        "serving (1/2 batch)",
        "oz",
        "g",
    ]


def test_missing_ingredient_food_raises_before_saving(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), foods_response(EGG), OK])

    with pytest.raises(CronometerError, match=r"not found: \[9999\]"):
        client.update_recipe(RECIPE_ID, ingredients=[(1234, 100), (9999, 10)])

    assert "add_food" not in endpoints(state)


# ---------------------------------------------------------------------------
# update_recipe: validation and refusals
# ---------------------------------------------------------------------------


def test_no_fields_is_an_error(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    with pytest.raises(ValueError, match="at least one field"):
        client.update_recipe(RECIPE_ID)

    assert state["payloads"] == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"serving_grams": 0}, "serving_grams"),
        ({"serving_grams": -5}, "serving_grams"),
        ({"serving_grams": math.nan}, "serving_grams"),
        ({"serving_grams": math.inf}, "serving_grams"),
        ({"serving_name": ""}, "serving_name"),
        ({"serving_name": "   "}, "serving_name"),
        ({"serving_name": "0 bowls"}, "serving_name quantity"),
        ({"name": ""}, "name"),
        ({"name": "  "}, "name"),
        ({"ingredients": []}, "at least one ingredient"),
        ({"ingredients": [(1234,)]}, "Each ingredient"),
        ({"ingredients": [(1234, 10, 1, 2)]}, "Each ingredient"),
        ({"ingredients": [(1234, 0)]}, "positive"),
        ({"ingredients": [(1234, -1)]}, "positive"),
        ({"ingredients": [(1234, math.nan)]}, "positive"),
        ({"ingredients": [(1234, math.inf)]}, "positive"),
    ],
)
def test_bad_arguments_fail_before_any_request(tmp_path, kwargs, match):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    with pytest.raises(ValueError, match=match):
        client.update_recipe(RECIPE_ID, **kwargs)

    assert state["payloads"] == []


@pytest.mark.parametrize("source", ["USDA", "NCCDB:4974", "CRDB", None])
def test_refuses_non_custom_foods(tmp_path, source):
    client, state = make_cold_client(tmp_path, [recipe(source=source), OK])

    with pytest.raises(CronometerError, match="not a user-created recipe"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


@pytest.mark.parametrize("ingredients", [None, []])
def test_refuses_plain_custom_foods(tmp_path, ingredients):
    food = recipe(ingredients=ingredients)
    if ingredients is None:
        del food["ingredients"]
    client, state = make_cold_client(tmp_path, [food, OK])

    with pytest.raises(CronometerError, match="plain custom food"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


def test_refuses_serving_based_recipes(tmp_path):
    """Serving-based recipes ("Recipe"-type measures) store nutrients per
    batch, not per-100g; editing them is untested, so they are refused."""
    food = recipe()
    food["measures"][1]["type"] = "Recipe"
    client, state = make_cold_client(tmp_path, [food, OK])

    with pytest.raises(CronometerError, match="serving-based"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


@pytest.mark.parametrize(
    "response",
    [
        {"result": "SUCCESS", "id": 999},  # some other food
        {"result": "SUCCESS"},  # no id at all
        {"unexpected": True},
    ],
)
def test_unexpected_save_response_raises(tmp_path, response):
    client, _ = make_cold_client(tmp_path, [recipe(), response])

    with pytest.raises(CronometerError, match="Failed to save food"):
        client.update_recipe(RECIPE_ID, name="X")


def test_save_failure_body_raises_after_one_retry(tmp_path):
    """A FAIL body is retried once after re-login, then surfaces."""
    fail = {"result": "FAIL", "error": "nope"}
    client, state = make_cold_client(tmp_path, [recipe(), fail, fail])

    with pytest.raises(CronometerError, match="nope"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food", "add_food", "add_food"]


def test_update_custom_food_still_refuses_recipes(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    with pytest.raises(CronometerError, match="update_recipe"):
        client.update_custom_food(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


# ---------------------------------------------------------------------------
# retire_recipe
# ---------------------------------------------------------------------------


def test_retire_recipe_resends_with_retired_flag_only(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    result = client.retire_recipe(RECIPE_ID)

    data = saved(state)
    assert data["retired"] is True
    assert without(data, "retired") == without(recipe(), "messages", "retired")
    assert result == {"food_id": RECIPE_ID, "name": "Test Recipe", "retired": True}


def test_retire_recipe_is_idempotent(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(retired=True), OK])

    assert client.retire_recipe(RECIPE_ID)["retired"] is True
    assert saved(state)["retired"] is True


def test_retire_recipe_refuses_database_foods(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(source="USDA"), OK])

    with pytest.raises(CronometerError, match="not a user-created recipe"):
        client.retire_recipe(RECIPE_ID)

    assert endpoints(state) == ["get_food"]


def test_retire_recipe_refuses_plain_custom_foods(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(ingredients=[]), OK])

    with pytest.raises(CronometerError, match="delete_custom_food"):
        client.retire_recipe(RECIPE_ID)

    assert endpoints(state) == ["get_food"]


def test_retire_custom_food_still_refuses_recipes(tmp_path):
    client, state = make_cold_client(tmp_path, [recipe(), OK])

    with pytest.raises(CronometerError, match="delete_recipe"):
        client.retire_custom_food(RECIPE_ID)

    assert endpoints(state) == ["get_food"]


def test_recipe_edits_hold_the_food_lock(tmp_path):
    """get_food, get_foods, and add_food all run under the food's lock."""
    client, _ = make_cold_client(
        tmp_path, [recipe(), foods_response(EGG), OK, recipe(), OK]
    )
    lock = client._food_lock(RECIPE_ID)
    seen: list[tuple[str, bool]] = []
    real_post = client._http.post

    def spying_post(endpoint, json=None):
        seen.append((endpoint, lock.locked()))
        return real_post(endpoint, json=json)

    client._http.post = spying_post  # type: ignore[method-assign]

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 100)])
    client.retire_recipe(RECIPE_ID)

    assert seen == [
        ("/api/v2/get_food", True),
        ("/api/v2/get_foods", True),
        ("/api/v2/add_food", True),
        ("/api/v2/get_food", True),
        ("/api/v2/add_food", True),
    ]
    assert not lock.locked()


# ---------------------------------------------------------------------------
# create_recipe shares the ingredient parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grams", [math.nan, math.inf])
def test_create_recipe_rejects_non_finite_grams(tmp_path, grams):
    client, state = make_cold_client(tmp_path, [OK])

    with pytest.raises(ValueError, match="positive"):
        client.create_recipe("R", ingredients=[(1234, grams)])

    assert state["payloads"] == []


# ---------------------------------------------------------------------------
# Server tools
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, food: dict | None = None) -> None:
        self.calls: list[tuple] = []
        self.food = food

    def update_recipe(self, food_id, **kwargs):
        self.calls.append(("update_recipe", food_id, kwargs))
        return {"food_id": food_id, "name": "R", "total_grams": 1.0}

    def retire_recipe(self, food_id):
        self.calls.append(("retire_recipe", food_id))
        return {"food_id": food_id, "name": "R", "retired": True}

    def create_recipe(self, name, **kwargs):
        self.calls.append(("create_recipe", name, kwargs))
        return {"food_id": 1, "total_grams": 1.0, "ingredient_count": 1}

    def get_food(self, food_id):
        if self.food is None:
            raise AssertionError("get_food not expected")
        return self.food


@pytest.fixture
def fake_client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(server, "_get_client", lambda: fake)
    return fake


def test_update_recipe_tool_converts_ingredients(fake_client):
    out = json.loads(
        server.update_recipe(
            7,
            ingredients=[
                {"food_id": 1, "grams": 10},
                {"food_id": 2, "grams": 20, "measure_id": 5},
                {"food_id": 3, "grams": 30, "measure_id": None},
            ],
            serving_grams=50,
        )
    )

    assert out["status"] == "success"
    assert out["food_id"] == 7
    assert fake_client.calls == [
        (
            "update_recipe",
            7,
            {
                "ingredients": [(1, 10), (2, 20, 5), (3, 30)],
                "name": None,
                "serving_name": None,
                "serving_grams": 50,
                "comments": None,
            },
        )
    ]


def test_update_recipe_tool_passes_none_ingredients_through(fake_client):
    server.update_recipe(7, name="New")

    assert fake_client.calls[0][2]["ingredients"] is None
    assert fake_client.calls[0][2]["name"] == "New"


@pytest.mark.parametrize(
    ("item", "match"),
    [
        ({"food_id": 1}, "needs 'food_id' and 'grams'"),
        ({"grams": 1}, "needs 'food_id' and 'grams'"),
        ({"food_id": 1, "gram": 1}, "needs 'food_id' and 'grams'"),
        ({"food_id": 1, "grams": 1, "measure": 5}, "Unknown ingredient keys"),
    ],
)
@pytest.mark.parametrize("tool", ["update_recipe", "add_recipe"])
def test_recipe_tools_reject_malformed_ingredients(fake_client, item, match, tool):
    if tool == "update_recipe":
        raw = server.update_recipe(7, ingredients=[item])
    else:
        raw = server.add_recipe("R", ingredients=[item])

    out = json.loads(raw)
    assert out["status"] == "error"
    assert match in out["message"]
    assert fake_client.calls == []


def test_update_recipe_tool_reports_client_errors(monkeypatch):
    class Failing(FakeClient):
        def update_recipe(self, food_id, **kwargs):
            raise CronometerError("is a plain custom food")

    monkeypatch.setattr(server, "_get_client", lambda: Failing())

    out = json.loads(server.update_recipe(7, name="X"))
    assert out == {
        "status": "error",
        "message": "CronometerError: is a plain custom food",
    }


def test_delete_recipe_tool(fake_client):
    out = json.loads(server.delete_recipe(7))

    assert out == {"status": "success", "food_id": 7, "name": "R", "retired": True}
    assert fake_client.calls == [("retire_recipe", 7)]


def test_get_food_details_lists_recipe_ingredients(monkeypatch):
    fake = FakeClient(food=recipe(retired=True))
    monkeypatch.setattr(server, "_get_client", lambda: fake)

    out = json.loads(server.get_food_details(RECIPE_ID))

    assert out["ingredients"] == [
        {"food_id": 450856, "grams": 120, "measure_id": 998949},
        {"food_id": 619, "grams": 32, "measure_id": 1960},
    ]
    assert out["comments"] == "old notes"
    assert out["retired"] is True


def test_get_food_details_plain_food_has_no_recipe_fields(monkeypatch):
    fake = FakeClient(food=recipe(ingredients=[], comments=None))
    monkeypatch.setattr(server, "_get_client", lambda: fake)

    out = json.loads(server.get_food_details(RECIPE_ID))

    assert "ingredients" not in out
    assert "comments" not in out
    assert "retired" not in out
