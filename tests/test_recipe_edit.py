"""update_recipe (client and tool), and retiring recipes via retire_custom_food.

Fixtures mirror get_food bodies captured from throwaway recipes on the live API.
"""

from __future__ import annotations

import copy
import json
import math

import httpx
import pytest
from test_client import (
    LEADING_QUANTITY_NAMES,
    FakeFoodServer,
    make_server_client,
    server_reparse,
)

from cronometer_api_mcp import server
from cronometer_api_mcp.client import CronometerError

RECIPE_ID = 85008439
OK = {"result": "SUCCESS", "messages": [], "id": RECIPE_ID}

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
# Live: a water-only recipe reads back with energy 0, not without it.
WATER = {
    "id": 455493,
    "name": "Tap Water",
    "defaultMeasureId": 1,
    "measures": [{"id": 1, "name": "g", "value": 1, "amount": 1, "type": "Weight"}],
    "nutrients": [{"id": 208, "amount": 0}, {"id": 307, "amount": 0}],
}
PB = {
    "id": 619,
    "name": "Peanut Butter",
    "defaultMeasureId": 1960,
    "measures": [
        {"id": 1960, "name": "tbsp", "value": 16, "amount": 1, "type": "Atomic"}
    ],
    "nutrients": [{"id": 208, "amount": 588.0}, {"id": 204, "amount": 50.0}],
}


def foods_response(*foods: dict) -> dict:
    return {"foods": [copy.deepcopy(f) for f in foods]}


def recipe(**overrides) -> dict:
    """Banana 120 g + peanut butter 32 g; serving 76 g, batch 152 g."""
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
        "messages": [{"id": 1, "type": "FOOD_CHANGED", "text": "{}"}],
    }
    food.update(overrides)
    return food


def recipe_client(tmp_path, food=None, *, save_responses=(), server_cls=FakeFoodServer):
    srv = server_cls(
        foods=[recipe() if food is None else food],
        ingredient_foods=[BANANA, EGG, PB, WATER],
        save_responses=save_responses,
    )
    return make_server_client(tmp_path, srv)


def saved(state) -> dict:
    """The food object sent with the last add_food."""
    i = len(state["endpoints"]) - 1 - state["endpoints"][::-1].index("add_food")
    payload = state["payloads"][i]
    assert payload["config"] == {"call_version": 1}
    return payload["data"]


def endpoints(state) -> list[str]:
    return state["endpoints"]


SAVE = ["get_food", "add_food", "get_food"]  # fetch, save, read back


def add_food_payloads(state) -> list[dict]:
    return [
        p["data"]
        for e, p in zip(state["endpoints"], state["payloads"], strict=True)
        if e == "add_food"
    ]


def measures_by_id(food: dict) -> dict[int, dict]:
    return {m["id"]: m for m in food["measures"]}


def without(food: dict, *keys: str) -> dict:
    return {k: v for k, v in food.items() if k not in keys}


# ---------------------------------------------------------------------------
# update_recipe: metadata-only edits
# ---------------------------------------------------------------------------


def test_rename_resends_recipe_unchanged_apart_from_name(tmp_path):
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(RECIPE_ID, name="Renamed")

    assert endpoints(state) == SAVE
    assert state["payloads"][0]["id"] == RECIPE_ID
    data = saved(state)
    assert data["id"] == RECIPE_ID
    assert data["name"] == "Renamed"
    assert [t["name"] for t in data["translations"]] == ["Renamed"]
    assert "messages" not in data
    expected = without(recipe(), "messages", "name", "translations", "nutrients")
    assert without(data, "name", "translations", "nutrients") == expected
    assert data["nutrients"] == []  # the server must derive them anew
    assert result == {
        "food_id": RECIPE_ID,
        "name": "Renamed",
        "total_grams": 152,
        "ingredient_count": 2,
    }  # the serving wasn't edited, so it isn't resolved or reported


def test_rename_leaves_other_language_translations_alone(tmp_path):
    translations = [
        {"translationId": 1, "name": "Test Recipe", "languageCode": "en"},
        {"translationId": 2, "name": "Recette", "languageCode": "fr"},
    ]
    client, state = recipe_client(tmp_path, recipe(translations=translations))

    client.update_recipe(RECIPE_ID, name="Renamed")

    assert [t["name"] for t in saved(state)["translations"]] == [
        "Renamed",
        "Recette",
    ]


@pytest.mark.parametrize("comments", ["new notes", ""])
def test_comments_set_or_cleared(tmp_path, comments):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, comments=comments)

    data = saved(state)
    assert data["comments"] == comments
    assert without(data, "comments", "nutrients") == without(
        recipe(), "messages", "comments", "nutrients"
    )


def test_comments_none_keeps_existing(tmp_path):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, name="X")

    assert saved(state)["comments"] == "old notes"


# ---------------------------------------------------------------------------
# update_recipe: serving measure
# ---------------------------------------------------------------------------


def test_serving_grams_resizes_default_measure_only(tmp_path):
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(RECIPE_ID, serving_grams=50.5)

    data = saved(state)
    by_id = measures_by_id(data)
    assert by_id[313167825] == {
        **measures_by_id(recipe())[313167825],
        "value": 50.5,
    }
    assert by_id[313167825]["type"] == "Weight"  # recipes stay weight-based
    for mid in (313167828, 313167827, 313167826):
        assert by_id[mid] == measures_by_id(recipe())[mid]
    assert data["defaultMeasureId"] == 313167825
    assert data["nutrients"] == []
    assert data["ingredients"] == recipe()["ingredients"]
    assert result["serving"]["grams"] == 50.5


def test_serving_grams_alone_keeps_quantity(tmp_path):
    """Resizing "2 bowls" must not turn it into "1 bowls"."""
    food = recipe()
    measures_by_id(food)[313167825].update(name="bowls", amount=2)
    client, state = recipe_client(tmp_path, food)

    result = client.update_recipe(RECIPE_ID, serving_grams=150)

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["value"]) == ("bowls", 2, 150)
    assert result["serving"]["quantity"] == 2


def test_serving_name_with_explicit_quantity_repairs_mangled_serving(tmp_path):
    """Repairs a measure the server mangled on create ("batch)", amount 0.5)."""
    mangled = recipe()
    serving = measures_by_id(mangled)[313167825]
    serving.update(name="batch)", amount=0.5)
    client, state = recipe_client(tmp_path, mangled)

    client.update_recipe(
        RECIPE_ID, serving_name="serving (1/6 batch)", serving_quantity=1
    )

    m = measures_by_id(saved(state))[313167825]
    assert m["name"] == "serving (1/6 batch)"
    assert m["amount"] == 1
    assert m["value"] == 76  # weight untouched


def test_bare_rename_keeps_mangled_quantity_without_serving_quantity(tmp_path):
    mangled = recipe()
    measures_by_id(mangled)[313167825].update(name="batch)", amount=0.5)
    client, state = recipe_client(tmp_path, mangled)

    client.update_recipe(RECIPE_ID, serving_name="serving (1/6 batch)")

    assert measures_by_id(saved(state))[313167825]["amount"] == 0.5


@pytest.mark.parametrize(
    ("serving_name", "stored"),
    [
        ("serving (1/6 batch)", "serving (1/6 batch)"),
        ("serving (1/2 batch)", "serving (1/2 batch)"),
        ("  bowl  ", "bowl"),
    ],
)
def test_serving_name_is_sent_exactly_and_keeps_quantity(
    tmp_path, serving_name, stored
):
    """No client-side parsing: the name goes out stripped but otherwise as
    given (a re-sent existing measure is not re-parsed), quantity untouched."""
    food = recipe()
    measures_by_id(food)[313167825].update(amount=2)
    client, state = recipe_client(tmp_path, food)

    result = client.update_recipe(RECIPE_ID, serving_name=serving_name)

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["type"]) == (stored, 2, "Weight")
    assert result["serving"]["quantity"] == 2


def test_bare_rename_keeps_quantity(tmp_path):
    """Renaming "2 cookies" to "biscuits" gives "2 biscuits" at the same
    weight; the same rule as update_custom_food."""
    food = recipe()
    measures_by_id(food)[313167825].update(name="cookies", amount=2)
    client, state = recipe_client(tmp_path, food)

    client.update_recipe(RECIPE_ID, serving_name="biscuits")

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["value"], m["type"]) == (
        "biscuits",
        2,
        76,
        "Weight",
    )


def test_serving_name_and_grams_together(tmp_path):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, serving_name="bowl", serving_grams=100)

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["value"]) == ("bowl", 1, 100)


@pytest.mark.parametrize("fixed", ["full recipe", "g", "oz"])
def test_serving_found_when_default_drifted_to_a_fixed_measure(tmp_path, fixed):
    """Logging a recipe by "g" made "g" its default (seen live); the serving
    edit must still land on the one non-fixed measure."""
    mid = next(m["id"] for m in recipe()["measures"] if m["name"] == fixed)
    client, state = recipe_client(tmp_path, recipe(defaultMeasureId=mid))

    result = client.update_recipe(RECIPE_ID, serving_grams=10)

    by_id = measures_by_id(saved(state))
    assert by_id[313167825]["value"] == 10
    assert by_id[mid] == measures_by_id(recipe())[mid]
    assert saved(state)["defaultMeasureId"] == mid  # not changed by us
    assert result["serving"]["measure_id"] == 313167825


def test_ambiguous_serving_raises(tmp_path):
    food = recipe(defaultMeasureId=313167826)  # "g"
    food["measures"].append(
        {"id": 5, "name": "bowl", "amount": 1, "value": 300, "type": "Weight"}
    )
    client, state = recipe_client(tmp_path, food)

    with pytest.raises(CronometerError, match=r"several servings \['serving"):
        client.update_recipe(RECIPE_ID, serving_grams=50)

    assert endpoints(state) == ["get_food"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"serving_grams": 80},
        {"serving_name": "cup"},
        {"serving_quantity": 2},
        {"servings_per_batch": 2},
    ],
)
@pytest.mark.parametrize("default", [313167825, 5, 313167826])
def test_serving_edit_refused_when_recipe_has_several_servings(
    tmp_path, kwargs, default
):
    """Whatever the default measure, two non-fixed measures make a serving
    edit ambiguous."""
    food = recipe(defaultMeasureId=default)
    food["measures"].append(
        {"id": 5, "name": "bowl", "amount": 1, "value": 100, "type": "Weight"}
    )
    client, state = recipe_client(tmp_path, food)

    with pytest.raises(CronometerError, match="several servings.*ambiguous"):
        client.update_recipe(RECIPE_ID, **kwargs)

    assert endpoints(state) == ["get_food"]


def test_several_servings_still_allow_non_serving_edits(tmp_path):
    food = recipe()
    food["measures"].append(
        {"id": 5, "name": "bowl", "amount": 1, "value": 100, "type": "Weight"}
    )
    client, state = recipe_client(tmp_path, food)

    client.update_recipe(RECIPE_ID, name="Renamed")

    assert saved(state)["name"] == "Renamed"


def drifted_recipe(kind: str) -> dict:
    """A recipe whose default drifted to "g", with several named servings
    ("several") or none at all ("none")."""
    food = recipe(defaultMeasureId=313167826)  # "g"
    if kind == "several":
        food["measures"].append(
            {"id": 5, "name": "bowl", "amount": 1, "value": 100, "type": "Weight"}
        )
    else:
        food["measures"] = [
            m for m in food["measures"] if m["name"] != "serving (1/2 batch)"
        ]
    return food


@pytest.mark.parametrize("kind", ["several", "none"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Renamed"},
        {"comments": "new"},
        {"ingredients": [(1234, 152)]},  # same batch weight
    ],
)
def test_non_serving_edits_ignore_the_serving(tmp_path, kind, kwargs):
    food = drifted_recipe(kind)
    client, state = recipe_client(tmp_path, food)

    result = client.update_recipe(RECIPE_ID, **kwargs)

    assert endpoints(state) == [
        e
        for e in ["get_food", "get_foods", "add_food", "get_food"]
        if e != "get_foods" or "ingredients" in kwargs
    ]
    assert saved(state)["measures"] == food["measures"]
    assert "serving" not in result and "servings_per_batch" not in result


def test_batch_change_without_a_serving_needs_no_sizing(tmp_path):
    client, _ = recipe_client(tmp_path, drifted_recipe("none"))

    result = client.update_recipe(RECIPE_ID, ingredients=[(1234, 400)])

    assert result["total_grams"] == 400
    assert "serving" not in result


def test_serving_edit_without_a_serving_raises(tmp_path):
    client, state = recipe_client(tmp_path, drifted_recipe("none"))

    with pytest.raises(CronometerError, match="has no serving measure"):
        client.update_recipe(RECIPE_ID, serving_grams=50)

    assert endpoints(state) == ["get_food"]


def test_missing_default_measure_uses_sole_serving(tmp_path):
    client, state = recipe_client(tmp_path, recipe(defaultMeasureId=1))

    client.update_recipe(RECIPE_ID, serving_grams=40)

    assert measures_by_id(saved(state))[313167825]["value"] == 40


# ---------------------------------------------------------------------------
# update_recipe: ingredients
# ---------------------------------------------------------------------------


def test_ingredients_replace_list_leaving_derivation_to_server(tmp_path):
    """The server derives nutrients and the "full recipe" weight on save
    (verified live, also with `nutrients: []`), so both are re-sent as
    fetched."""
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(
        RECIPE_ID, ingredients=[(450856, 150), (1234, 50), (619, 16)], serving_grams=76
    )

    assert endpoints(state) == ["get_food", "get_foods", "add_food", "get_food"]
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

    assert without(data, "ingredients", "nutrients") == without(
        recipe(), "messages", "ingredients", "nutrients"
    )
    assert data["nutrients"] == []
    assert result["total_grams"] == 216.0
    assert result["ingredient_count"] == 3
    assert result["serving"]["grams"] == 76


def test_ingredient_measure_override(tmp_path):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 152, 3770)])

    assert saved(state)["ingredients"][0]["measureId"] == 3770


def test_ingredients_and_serving_together(tmp_path):
    client, state = recipe_client(tmp_path)

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
    assert by_id[313167828]["value"] == 152  # the server re-derives it
    assert (by_id[313167825]["name"], by_id[313167825]["value"]) == (
        "serving (1/6 batch)",
        100,
    )
    assert data["name"] == "Eggs"
    assert data["comments"] == ""
    assert result["total_grams"] == 600


def test_ingredients_with_serving_name_quantity_and_grams(tmp_path):
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(
        RECIPE_ID,
        ingredients=[(1234, 600)],
        serving_name="bowls",
        serving_quantity=3,
        serving_grams=200,
    )

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["value"], m["type"]) == (
        "bowls",
        3,
        200,
        "Weight",
    )
    assert result["serving"] == {
        "measure_id": 313167825,
        "name": "bowls",
        "quantity": 3,
        "grams": 200,
    }


def test_recipe_without_full_recipe_measure_still_updates(tmp_path):
    """get_food reports a virtual id-0 "full recipe" when none is stored;
    re-sending it makes the server store a real one (seen live)."""
    food = recipe()
    food["measures"] = [m for m in food["measures"] if m["name"] != "full recipe"]
    client, state = recipe_client(tmp_path, food)

    result = client.update_recipe(RECIPE_ID, ingredients=[(1234, 152)])

    sent = saved(state)["measures"]
    assert [(m["id"], m["name"]) for m in sent] == [
        (0, "full recipe"),
        (313167825, "serving (1/2 batch)"),
        (313167827, "oz"),
        (313167826, "g"),
    ]
    assert result["total_grams"] == 152


def test_missing_ingredient_food_raises_before_saving(tmp_path):
    client, state = recipe_client(tmp_path)

    with pytest.raises(CronometerError, match=r"not found: \[9999\]"):
        client.update_recipe(RECIPE_ID, ingredients=[(1234, 100), (9999, 10)])

    assert "add_food" not in endpoints(state)


def test_get_food_details_ingredients_round_trip(tmp_path, monkeypatch):
    """Feeding get_food_details' ingredients back into update_recipe yields
    the same rows (bar the row ids the server reassigns)."""
    food = recipe(
        ingredients=[
            {
                "id": 1,
                "foodId": 450856,
                "measureId": 998949,
                "translationId": 398492,
                "grams": 120,
                "value": 120,
            },
            {
                "id": 2,
                "foodId": 1234,
                # A non-gram (50 g "large") display measure. `value` still
                # equals grams: Colin's app-made recipe stores 33 g of milk on
                # a 100 g "cup" measure as value 33, not 0.33 (read live).
                "measureId": 3770,
                "translationId": 1893,
                "grams": 100,
                "value": 100,
            },
        ]
    )
    measures_by_id(food)[313167828]["value"] = 220  # full recipe = ingredients
    client, state = recipe_client(tmp_path, food)
    monkeypatch.setattr(server, "_get_client", lambda: client)

    details = json.loads(server.get_food_details(RECIPE_ID))
    out = json.loads(
        server.update_recipe(RECIPE_ID, ingredients=details["ingredients"])
    )

    assert out["status"] == "success"
    assert saved(state)["ingredients"] == [
        {**row, "id": 0} for row in food["ingredients"]
    ]


# ---------------------------------------------------------------------------
# update_recipe: validation and refusals
# ---------------------------------------------------------------------------


def test_no_fields_is_an_error(tmp_path):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="at least one field"):
        client.update_recipe(RECIPE_ID)

    assert state["endpoints"] == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"serving_grams": 0}, "serving_grams"),
        ({"serving_grams": -5}, "serving_grams"),
        ({"serving_grams": math.nan}, "serving_grams"),
        ({"serving_grams": math.inf}, "serving_grams"),
        ({"serving_name": ""}, "serving_name"),
        ({"serving_name": "   "}, "serving_name"),
        ({"serving_name": "full recipe"}, "reserved"),
        ({"serving_name": "Full Recipe "}, "reserved"),
        ({"serving_name": "g"}, "reserved"),
        ({"serving_name": "OZ"}, "reserved"),
        ({"serving_quantity": 0}, "serving_quantity"),
        ({"serving_quantity": -2}, "serving_quantity"),
        ({"serving_quantity": math.nan}, "serving_quantity"),
        ({"serving_quantity": math.inf}, "serving_quantity"),
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
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match=match):
        client.update_recipe(RECIPE_ID, **kwargs)

    assert state["endpoints"] == []


@pytest.mark.parametrize("source", ["USDA", "NCCDB:4974", "CRDB", None])
def test_refuses_non_custom_foods(tmp_path, source):
    client, state = recipe_client(tmp_path, recipe(source=source))

    with pytest.raises(CronometerError, match="not a custom food or recipe"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


@pytest.mark.parametrize("ingredients", [None, []])
def test_refuses_plain_custom_foods(tmp_path, ingredients):
    food = recipe(ingredients=ingredients)
    if ingredients is None:
        del food["ingredients"]
    client, state = recipe_client(tmp_path, food)

    with pytest.raises(CronometerError, match="plain custom food.*update_custom_food"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


def test_refuses_retired_recipes(tmp_path):
    client, state = recipe_client(tmp_path, recipe(retired=True))

    with pytest.raises(CronometerError, match="deleted"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


@pytest.mark.parametrize("action", ["update", "retire"])
def test_refuses_serving_based_recipes(tmp_path, action):
    """Serving-based recipes store nutrients per batch; no path re-sends them."""
    food = recipe()
    food["measures"][1]["type"] = "Recipe"
    client, state = recipe_client(tmp_path, food)

    with pytest.raises(CronometerError, match="serving-based"):
        if action == "update":
            client.update_recipe(RECIPE_ID, name="X")
        else:
            client.retire_custom_food(RECIPE_ID)

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
    client, _ = recipe_client(tmp_path, save_responses=[response])

    with pytest.raises(CronometerError, match="Failed to save food"):
        client.update_recipe(RECIPE_ID, name="X")


def test_save_failure_body_raises_after_one_retry(tmp_path):
    fail = {"result": "FAIL", "error": "nope"}
    client, state = recipe_client(tmp_path, save_responses=[fail, fail])

    with pytest.raises(CronometerError, match="nope"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food", "add_food", "add_food"]  # retried once


def test_update_custom_food_still_refuses_recipes(tmp_path):
    client, state = recipe_client(tmp_path)

    with pytest.raises(CronometerError, match="update_recipe"):
        client.update_custom_food(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


# ---------------------------------------------------------------------------
# retire_recipe
# ---------------------------------------------------------------------------


def test_retire_custom_food_retires_recipe_with_flag_only(tmp_path):
    client, state = recipe_client(tmp_path)

    result = client.retire_custom_food(RECIPE_ID)

    data = saved(state)
    assert data["retired"] is True
    assert without(data, "retired") == without(recipe(), "messages", "retired")
    assert result == {"food_id": RECIPE_ID, "name": "Test Recipe", "retired": True}


def test_retire_recipe_is_idempotent(tmp_path):
    client, state = recipe_client(tmp_path, recipe(retired=True))

    assert client.retire_custom_food(RECIPE_ID)["retired"] is True
    assert saved(state)["retired"] is True


def test_retire_still_refuses_database_recipes(tmp_path):
    client, state = recipe_client(tmp_path, recipe(source="USDA"))

    with pytest.raises(CronometerError, match="not a custom food or recipe"):
        client.retire_custom_food(RECIPE_ID)

    assert endpoints(state) == ["get_food"]


def test_recipe_edits_hold_the_food_lock(tmp_path):
    client, _ = recipe_client(tmp_path)
    lock = client._food_lock(RECIPE_ID)
    seen: list[tuple[str, bool]] = []
    real_post = client._http.post

    def spying_post(endpoint, json=None):
        seen.append((endpoint, lock.locked()))
        return real_post(endpoint, json=json)

    client._http.post = spying_post  # type: ignore[method-assign]

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 152)])
    client.retire_custom_food(RECIPE_ID)

    assert seen == [
        ("/api/v2/get_food", True),
        ("/api/v2/get_foods", True),
        ("/api/v2/add_food", True),
        ("/api/v2/get_food", True),  # read back
        ("/api/v2/get_food", True),
        ("/api/v2/add_food", True),
    ]
    assert not lock.locked()


# ---------------------------------------------------------------------------
# create_recipe shares the ingredient parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grams", [math.nan, math.inf])
def test_create_recipe_rejects_non_finite_grams(tmp_path, grams):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        client.create_recipe("R", ingredients=[(1234, grams)])

    assert state["endpoints"] == []


# ---------------------------------------------------------------------------
# Serving names the server re-parses on create
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("serving_name", "quantity", "resent"),
    [
        # Kept by the server as sent (verified live).
        ("serving (1/6 batch)", 1.0, False),
        ("cups", 1.5, False),
        # Re-parsed by the server when the quantity is 1 (live: "batch)" x 0.5),
        # so the client re-sends the existing measure verbatim.
        ("serving (1/2 batch)", 1.0, True),
        ("serving (1/3 batch)", 1.0, True),
        ("half (1/4 batch) portion", 1.0, True),
        # Not re-parsed: the quantity isn't 1.
        ("bowls", 3.0, False),
    ],
)
def test_create_recipe_serving_reads_back_exactly(
    tmp_path, serving_name, quantity, resent
):
    srv = FakeFoodServer(ingredient_foods=[EGG])
    client, state = make_server_client(tmp_path, srv)

    result = client.create_recipe(
        "R",
        ingredients=[(1234, 100)],
        serving_name=serving_name,
        serving_grams=40,
        serving_quantity=quantity,
    )

    first = state["payloads"][state["endpoints"].index("add_food")]["data"]
    assert (first["measures"][0]["name"], first["measures"][0]["amount"]) == (
        serving_name,
        quantity,
    )
    assert state["endpoints"].count("add_food") == (2 if resent else 1)
    m = measures_by_id(srv.foods[result["food_id"]])[result["measure_id"]]
    assert (m["name"], m["amount"], m["value"]) == (serving_name, quantity, 40)


def test_settle_compares_quantity_with_tolerance(tmp_path):
    """A read-back 1/3 that differs in the last bit is not a mismatch."""

    class RoundingServer(FakeFoodServer):
        def _save(self, food):
            resp = super()._save(food)
            for m in self.foods[food["id"]]["measures"]:
                if m["name"] == "thirds":
                    m["amount"] = 0.33333333333333337
            return resp

    client, state = make_server_client(tmp_path, RoundingServer(ingredient_foods=[EGG]))

    client.create_recipe(
        "R", ingredients=[(1234, 100)], serving_name="thirds", serving_quantity=1 / 3
    )

    assert state["endpoints"].count("add_food") == 1


@pytest.mark.parametrize("action", ["create", "update"])
def test_indivisible_batch_serving_is_rounded(tmp_path, action):
    """151 g / 6 = 25.1666... is sent as 25.17 g."""
    if action == "create":
        srv = FakeFoodServer(ingredient_foods=[EGG])
        client, state = make_server_client(tmp_path, srv)
        result = client.create_recipe(
            "R", ingredients=[(1234, 151)], servings_per_batch=6
        )
        assert result["serving_grams"] == 25.17
    else:
        client, state = recipe_client(tmp_path)
        result = client.update_recipe(
            RECIPE_ID, ingredients=[(1234, 151)], servings_per_batch=6
        )
        assert result["serving"]["grams"] == 25.17
        # Reported from the rounded grams actually stored: 151 / 25.17.
        assert result["servings_per_batch"]["after"] == 5.999
    sent = state["payloads"][state["endpoints"].index("add_food")]["data"]
    assert 25.17 in [m["value"] for m in sent["measures"]]
    assert state["endpoints"].count("add_food") == 1


@pytest.mark.parametrize(("stored", "resent"), [(25.166667, False), (25.19, True)])
def test_settle_compares_grams_with_an_absolute_tolerance(tmp_path, stored, resent):
    """Within 0.01 g of what was sent is a match; further off is re-sent."""

    class GramsRoundingServer(FakeFoodServer):
        saves = 0

        def _save(self, food):
            resp = super()._save(food)
            self.saves += 1
            if self.saves == 1:
                for m in self.foods[food["id"]]["measures"]:
                    if m["name"] == "Serving":
                        m["value"] = stored
            return resp

    srv = GramsRoundingServer(ingredient_foods=[EGG])
    client, state = make_server_client(tmp_path, srv)

    client.create_recipe("R", ingredients=[(1234, 151)], servings_per_batch=6)

    assert state["endpoints"].count("add_food") == (2 if resent else 1)


class StubbornServer(FakeFoodServer):
    """Re-parses existing measures too, so a verbatim re-send can't stick."""

    def _save(self, food):
        resp = super()._save(food)
        for m in self.foods[food["id"]]["measures"]:
            m["name"], m["amount"] = server_reparse(m["name"], m["amount"])
        return resp


def test_create_recipe_raises_if_serving_name_wont_stick(tmp_path):
    client, state = make_server_client(tmp_path, StubbornServer(ingredient_foods=[EGG]))

    with pytest.raises(CronometerError, match="reads back as 'batch\\)'") as err:
        client.create_recipe(
            "R", ingredients=[(1234, 100)], serving_name="serving (1/2 batch)"
        )

    assert state["endpoints"].count("add_food") == 2  # one re-send, then fail
    assert "The recipe exists (id 4242); don't re-create it" in str(err.value)
    assert "update_recipe" not in str(err.value)


def test_create_recipe_read_back_failure_names_the_created_recipe(tmp_path):
    class LostServer(FakeFoodServer):
        def handle(self, endpoint, body):
            if endpoint == "/api/v2/get_food":
                raise CronometerError("boom")
            return super().handle(endpoint, body)

    client, state = make_server_client(tmp_path, LostServer(ingredient_foods=[EGG]))

    with pytest.raises(
        CronometerError,
        match=r"^The recipe exists \(id 4242\); don't re-create it. boom$",
    ):
        client.create_recipe("R", ingredients=[(1234, 100)])

    assert state["endpoints"].count("add_food") == 1


@pytest.mark.parametrize("name", ["", "  "])
def test_create_recipe_rejects_blank_name(tmp_path, name):
    client, state = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[EGG]))

    with pytest.raises(ValueError, match="name must not be empty"):
        client.create_recipe(name, ingredients=[(1234, 100)])

    assert state["endpoints"] == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"serving_name": "g"}, "reserved"),
        ({"serving_name": " Full Recipe "}, "reserved"),
        ({"serving_name": "  "}, "serving_name"),
        ({"serving_quantity": 0}, "serving_quantity"),
        ({"serving_quantity": -1}, "serving_quantity"),
        ({"serving_quantity": math.nan}, "serving_quantity"),
        ({"serving_quantity": math.inf}, "serving_quantity"),
        ({"serving_grams": 0}, "serving_grams"),
    ],
)
def test_create_recipe_rejects_bad_serving_arguments(tmp_path, kwargs, match):
    client, state = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[EGG]))

    with pytest.raises(ValueError, match=match):
        client.create_recipe("R", ingredients=[(1234, 100)], **kwargs)

    assert state["endpoints"] == []


def test_create_recipe_strips_serving_name(tmp_path):
    srv = FakeFoodServer(ingredient_foods=[EGG])
    client, _ = make_server_client(tmp_path, srv)

    result = client.create_recipe(
        "R", ingredients=[(1234, 100)], serving_name="  bowl ", serving_quantity=2
    )

    m = measures_by_id(srv.foods[result["food_id"]])[result["measure_id"]]
    assert (m["name"], m["amount"]) == ("bowl", 2)


def test_update_recipe_resends_serving_if_server_rewrites_it(tmp_path):
    client, state = recipe_client(tmp_path, server_cls=StubbornServer)

    with pytest.raises(CronometerError, match="reads back as 'batch\\)'"):
        client.update_recipe(RECIPE_ID, serving_name="serving (1/2 batch)")

    assert state["endpoints"].count("add_food") == 2
    assert add_food_payloads(state)[1]["nutrients"] == []  # re-proves derivation


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

    def retire_custom_food(self, food_id):
        self.calls.append(("retire_custom_food", food_id))
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
                "servings_per_batch": None,
                "serving_quantity": None,
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


def test_delete_custom_food_tool_returns_client_result(fake_client):
    out = json.loads(server.delete_custom_food(7))

    assert out == {"status": "success", "food_id": 7, "name": "R", "retired": True}
    assert fake_client.calls == [("retire_custom_food", 7)]


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


def test_get_food_details_recipe_without_comments_key(monkeypatch):
    """A recipe saved without comments has no "comments" key (seen live)."""
    food = recipe()
    del food["comments"]
    monkeypatch.setattr(server, "_get_client", lambda: FakeClient(food=food))

    out = json.loads(server.get_food_details(RECIPE_ID))

    assert out["comments"] is None


def test_get_food_details_plain_food_has_no_recipe_fields(monkeypatch):
    fake = FakeClient(food=recipe(ingredients=[], comments=None))
    monkeypatch.setattr(server, "_get_client", lambda: fake)

    out = json.loads(server.get_food_details(RECIPE_ID))

    assert "ingredients" not in out
    assert "comments" not in out
    assert "retired" not in out


# ---------------------------------------------------------------------------
# Leading quantities, derived nutrients, serving weight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("serving_name", LEADING_QUANTITY_NAMES)
def test_create_recipe_rejects_leading_quantity(tmp_path, serving_name):
    """The app would show "1 2 bowls"; the number belongs in serving_quantity."""
    client, state = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[EGG]))

    with pytest.raises(ValueError, match="pass the number as serving_quantity"):
        client.create_recipe("R", ingredients=[(1234, 100)], serving_name=serving_name)

    assert state["endpoints"] == []


@pytest.mark.parametrize("serving_name", LEADING_QUANTITY_NAMES)
def test_update_recipe_rejects_leading_quantity(tmp_path, serving_name):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="pass the number as serving_quantity"):
        client.update_recipe(RECIPE_ID, serving_name=serving_name)

    assert state["endpoints"] == []


@pytest.mark.parametrize(
    "serving_name", ["serving (1/6 batch)", "bowl of 2", "half (½) cup", "serving #2"]
)
def test_names_without_a_leading_quantity_are_accepted(tmp_path, serving_name):
    srv = FakeFoodServer(ingredient_foods=[EGG])
    client, _ = make_server_client(tmp_path, srv)

    result = client.create_recipe(
        "R", ingredients=[(1234, 100)], serving_name=serving_name
    )

    m = measures_by_id(srv.foods[result["food_id"]])[result["measure_id"]]
    assert (m["name"], m["amount"]) == (serving_name, 1.0)


def test_create_recipe_raises_when_no_nutrients_were_derived(tmp_path):
    srv = FakeFoodServer(ingredient_foods=[EGG], derive_nutrients=False)
    client, state = make_server_client(tmp_path, srv)

    with pytest.raises(CronometerError) as err:
        client.create_recipe("R", ingredients=[(1234, 100)])

    msg = str(err.value)
    assert msg.startswith("The recipe exists (id 4242); don't re-create it.")
    assert "returned no derived nutrients" in msg
    assert state["endpoints"].count("add_food") == 1
    assert list(srv.foods) == [4242]  # exactly one recipe created


def test_create_recipe_timeout_after_create_names_the_one_recipe(tmp_path):
    """A transport error after the save must still carry the id: recipes
    can't be hard-deleted, so a retry would leave a duplicate."""
    timeout = httpx.ReadTimeout("read timed out")

    class SlowServer(FakeFoodServer):
        def handle(self, endpoint, body):
            if endpoint == "/api/v2/get_food":
                raise timeout
            return super().handle(endpoint, body)

    srv = SlowServer(ingredient_foods=[EGG])
    client, state = make_server_client(tmp_path, srv)

    with pytest.raises(CronometerError) as err:
        client.create_recipe("R", ingredients=[(1234, 100)])

    assert str(err.value) == (
        "The recipe exists (id 4242); don't re-create it. read timed out"
    )
    assert err.value.__cause__ is timeout
    assert state["endpoints"].count("add_food") == 1
    assert list(srv.foods) == [4242]


def test_create_recipe_malformed_read_back_names_the_recipe(tmp_path):
    class BrokenServer(FakeFoodServer):
        def handle(self, endpoint, body):
            out = super().handle(endpoint, body)
            if endpoint == "/api/v2/get_food":
                out.pop("id")  # a KeyError in the read-back, not a CronometerError
            return out

    client, _ = make_server_client(tmp_path, BrokenServer(ingredient_foods=[EGG]))

    with pytest.raises(CronometerError, match=r"^The recipe exists \(id 4242\)") as err:
        client.create_recipe("R", ingredients=[(1234, 100)])

    assert isinstance(err.value.__cause__, KeyError)


def test_create_recipe_post_create_failure_names_the_one_recipe(tmp_path):
    srv = StubbornServer(ingredient_foods=[EGG])
    client, state = make_server_client(tmp_path, srv)

    with pytest.raises(CronometerError) as err:
        client.create_recipe(
            "R", ingredients=[(1234, 100)], serving_name="serving (1/2 batch)"
        )

    msg = str(err.value)
    assert msg.startswith("The recipe exists (id 4242); don't re-create it.")
    assert "reads back as 'batch)'" in msg
    assert "update_recipe" not in msg  # it goes through the same settle
    assert list(srv.foods) == [4242]  # the re-send edited it, nothing new
    assert state["endpoints"].count("add_food") == 2


def test_update_recipe_raises_when_no_nutrients_were_derived(tmp_path):
    srv = FakeFoodServer(
        foods=[recipe()], ingredient_foods=[BANANA, EGG, PB], derive_nutrients=False
    )
    client, _ = make_server_client(tmp_path, srv)

    with pytest.raises(
        CronometerError,
        match=rf"Recipe {RECIPE_ID} was saved, but the server returned no derived",
    ):
        client.update_recipe(RECIPE_ID, name="X")


def test_zero_calorie_recipe_create_passes(tmp_path):
    srv = FakeFoodServer(ingredient_foods=[WATER])
    client, _ = make_server_client(tmp_path, srv)

    result = client.create_recipe("Water", ingredients=[(455493, 500)])

    energy = {n["id"]: n["amount"] for n in srv.foods[result["food_id"]]["nutrients"]}
    assert energy[208] == 0


def test_zero_calorie_recipe_update_passes(tmp_path):
    client, _ = recipe_client(tmp_path)

    result = client.update_recipe(
        RECIPE_ID, ingredients=[(455493, 500)], servings_per_batch=2
    )

    assert result["total_grams"] == 500


@pytest.mark.parametrize("grams", [228, 100])
def test_batch_change_without_sizing_is_refused(tmp_path, grams):
    """A "1/2 batch" serving (76 of 152 g) would silently become another
    fraction of the new batch, so the caller must say which to keep."""
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError) as err:
        client.update_recipe(RECIPE_ID, ingredients=[(1234, grams)], name="X")

    msg = str(err.value)
    assert f"from 152 g to {grams} g" in msg
    assert "'serving (1/2 batch)' (76 g)" in msg
    assert "servings_per_batch=N to keep the fraction" in msg
    assert "serving_grams=76 to keep the weight" in msg
    assert "add_food" not in endpoints(state)


def test_batch_change_with_serving_name_only_is_refused(tmp_path):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="servings_per_batch=N"):
        client.update_recipe(RECIPE_ID, ingredients=[(1234, 228)], serving_name="bowl")

    assert "add_food" not in endpoints(state)


@pytest.mark.parametrize(
    "sizing", [{}, {"servings_per_batch": 2}, {"serving_grams": 76}]
)
def test_batch_change_refused_when_serving_is_ambiguous(tmp_path, sizing):
    client, state = recipe_client(tmp_path, drifted_recipe("several"))

    with pytest.raises(CronometerError, match="several servings.*ambiguous"):
        client.update_recipe(RECIPE_ID, ingredients=[(1234, 228)], **sizing)

    assert "add_food" not in endpoints(state)


def test_batch_change_keeping_the_weight_reports_the_new_fraction(tmp_path):
    client, _ = recipe_client(tmp_path)

    kept = client.update_recipe(RECIPE_ID, ingredients=[(1234, 228)], serving_grams=76)

    assert kept["servings_per_batch"] == {"before": 2.0, "after": 3.0}


def test_tiny_batch_change_counts_as_unchanged(tmp_path):
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(RECIPE_ID, ingredients=[(1234, 152.004)])

    assert "serving" not in result
    assert "add_food" in endpoints(state)


def test_derived_energy_is_checked_on_every_save(tmp_path):
    """The happy path: the fake server derives energy, and both calls pass."""
    srv = FakeFoodServer(foods=[recipe()], ingredient_foods=[BANANA, EGG, PB])
    client, _ = make_server_client(tmp_path, srv)

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 152)])
    energy = {n["id"]: n["amount"] for n in srv.foods[RECIPE_ID]["nutrients"]}[208]
    assert energy == pytest.approx(143.0)


class GramsServer(FakeFoodServer):
    """Stores a new serving with the wrong weight, as a mangled save might."""

    def _save(self, food):
        new = [m for m in food["measures"] if not m["id"]]
        resp = super()._save(food)
        for m in new[:1]:
            stored = next(x for x in self.foods[food["id"]]["measures"] if x is m)
            stored["value"] = 1
        return resp


def test_create_recipe_resends_serving_with_mangled_grams(tmp_path):
    srv = GramsServer(ingredient_foods=[EGG])
    client, state = make_server_client(tmp_path, srv)

    result = client.create_recipe(
        "R", ingredients=[(1234, 100)], serving_name="bowl", serving_grams=40
    )

    assert state["endpoints"].count("add_food") == 2
    assert add_food_payloads(state)[1]["nutrients"] == []  # re-proves derivation
    assert result["serving_grams"] == 40
    m = measures_by_id(srv.foods[result["food_id"]])[result["measure_id"]]
    assert m["value"] == 40


def test_update_recipe_raises_if_grams_wont_stick(tmp_path):
    class AlwaysWrongGrams(FakeFoodServer):
        def _save(self, food):
            resp = super()._save(food)
            measures_by_id(self.foods[food["id"]])[313167825]["value"] = 1
            return resp

    client, state = recipe_client(tmp_path, server_cls=AlwaysWrongGrams)

    with pytest.raises(CronometerError, match=r"\(1 g\) instead of .* \(50 g\)"):
        client.update_recipe(RECIPE_ID, serving_grams=50)

    assert state["endpoints"].count("add_food") == 2


def test_create_recipe_rejects_serving_heavier_than_batch(tmp_path):
    client, state = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[EGG]))

    with pytest.raises(ValueError, match="more than the whole batch"):
        client.create_recipe("R", ingredients=[(1234, 100)], serving_grams=100.5)

    assert state["endpoints"] == []


def test_create_recipe_allows_serving_equal_to_batch(tmp_path):
    client, _ = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[EGG]))

    result = client.create_recipe("R", ingredients=[(1234, 100)], serving_grams=100)

    assert result["serving_grams"] == 100


def test_update_recipe_rejects_serving_heavier_than_batch(tmp_path):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match=r"153 is more than the whole batch \(152 g\)"):
        client.update_recipe(RECIPE_ID, serving_grams=153)

    assert "add_food" not in endpoints(state)


def test_update_recipe_rejects_ingredients_that_shrink_below_serving(tmp_path):
    """The 76 g serving no longer fits a 50 g batch."""
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="76 is more than the whole batch"):
        client.update_recipe(RECIPE_ID, ingredients=[(1234, 50)], serving_grams=76)

    assert "add_food" not in endpoints(state)


# ---------------------------------------------------------------------------
# servings_per_batch
# ---------------------------------------------------------------------------


def test_create_recipe_servings_per_batch_sizes_the_serving(tmp_path):
    srv = FakeFoodServer(ingredient_foods=[EGG, BANANA])
    client, _ = make_server_client(tmp_path, srv)

    result = client.create_recipe(
        "R",
        ingredients=[(1234, 300), (450856, 180)],
        serving_name="serving (1/8 batch)",
        servings_per_batch=8,
    )

    assert result["serving_grams"] == 60
    m = measures_by_id(srv.foods[result["food_id"]])[result["measure_id"]]
    assert (m["name"], m["amount"], m["value"]) == ("serving (1/8 batch)", 1.0, 60)


def test_update_recipe_servings_per_batch_keeps_the_fraction(tmp_path):
    """Colin's "serving (1/N batch)": new ingredients, still 1/N."""
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(
        RECIPE_ID, ingredients=[(1234, 400)], servings_per_batch=8
    )

    assert measures_by_id(saved(state))[313167825]["value"] == 50
    assert result["servings_per_batch"] == {"before": 2.0, "after": 8.0}
    assert result["serving"]["grams"] == 50


def test_update_recipe_servings_per_batch_without_new_ingredients(tmp_path):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, servings_per_batch=4)

    assert measures_by_id(saved(state))[313167825]["value"] == 38  # 152 / 4


@pytest.mark.parametrize("value", [0, -2, math.nan, math.inf])
@pytest.mark.parametrize("action", ["create", "update"])
def test_servings_per_batch_must_be_positive_and_finite(tmp_path, value, action):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="servings_per_batch"):
        if action == "create":
            client.create_recipe(
                "R", ingredients=[(1234, 100)], servings_per_batch=value
            )
        else:
            client.update_recipe(RECIPE_ID, servings_per_batch=value)

    assert state["endpoints"] == []


@pytest.mark.parametrize("action", ["create", "update"])
def test_serving_grams_and_servings_per_batch_are_exclusive(tmp_path, action):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="not both"):
        if action == "create":
            client.create_recipe(
                "R", ingredients=[(1234, 100)], serving_grams=10, servings_per_batch=2
            )
        else:
            client.update_recipe(RECIPE_ID, serving_grams=10, servings_per_batch=2)

    assert state["endpoints"] == []


def test_servings_per_batch_below_one_is_heavier_than_the_batch(tmp_path):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="more than the whole batch"):
        client.update_recipe(RECIPE_ID, servings_per_batch=0.5)

    assert "add_food" not in endpoints(state)


def test_tools_pass_servings_per_batch_through(fake_client):
    server.update_recipe(7, servings_per_batch=6)
    assert fake_client.calls[0][2]["servings_per_batch"] == 6


@pytest.mark.parametrize(
    ("kwargs", "note"),
    [
        ({"serving_grams": 50}, "from 76 g to 50 g"),
        ({"ingredients": [(1234, 600)], "servings_per_batch": 2}, "from 76 g to 300 g"),
        ({"servings_per_batch": 4}, "from 76 g to 38 g"),
        ({"servings_per_batch": 2}, None),  # 152 / 2 is still 76 g
        ({"serving_grams": 76.004}, None),  # within the 0.01 g tolerance
        ({"ingredients": [(1234, 600)], "serving_grams": 76}, None),
        ({"serving_name": "bowl"}, None),
        ({"serving_quantity": 2}, None),
        ({"name": "Renamed"}, None),
    ],
)
def test_note_when_serving_grams_change(tmp_path, kwargs, note):
    client, _ = recipe_client(tmp_path)

    result = client.update_recipe(RECIPE_ID, **kwargs)

    if note is None:
        assert "note" not in result
    else:
        assert note in result["note"]
        assert result["note"].endswith(
            "Diary entries logged by this serving follow it, so 1 serving is now "
            f"{result['serving']['grams']:g} g; entries logged by g keep their grams."
        )


@pytest.mark.parametrize("action", ["create", "update"])
def test_one_serving_per_batch_fits_a_batch_that_rounds_up(tmp_path, action):
    """523.3868 g / 1 rounds to 523.39 g, 0.0032 g over the batch: it fits."""
    ingredients = [(1234, 500), (450856, 23.3868)]
    if action == "create":
        srv = FakeFoodServer(ingredient_foods=[EGG, BANANA])
        client, state = make_server_client(tmp_path, srv)
        result = client.create_recipe(
            "R", ingredients=ingredients, servings_per_batch=1
        )
        assert result["serving_grams"] == 523.39
    else:
        client, state = recipe_client(tmp_path)
        result = client.update_recipe(
            RECIPE_ID, ingredients=ingredients, servings_per_batch=1
        )
        assert result["serving"]["grams"] == 523.39
    assert state["endpoints"].count("add_food") == 1


def test_serving_more_than_the_tolerance_over_the_batch_is_refused(tmp_path):
    client, state = recipe_client(tmp_path)

    with pytest.raises(ValueError, match="more than the whole batch"):
        client.update_recipe(RECIPE_ID, serving_grams=152.02)

    assert "add_food" not in endpoints(state)


@pytest.mark.parametrize("full_id", [313167828, 0])
def test_batch_weight_set_in_app_is_refused(tmp_path, full_id):
    """A stored "full recipe" that isn't the ingredient total was set in the
    app; an id-0 one is the server's placeholder and is ignored."""
    food = recipe()
    full = measures_by_id(food)[313167828]
    full["id"], full["value"] = full_id, 500
    client, state = recipe_client(tmp_path, food)

    if full_id:
        with pytest.raises(
            CronometerError,
            match=r"batch weight \(500 g\) was set in the Cronometer app and "
            r"doesn't match its ingredients \(152 g\); edit it there",
        ):
            client.update_recipe(RECIPE_ID, name="X")
        assert endpoints(state) == ["get_food"]
    else:
        client.update_recipe(RECIPE_ID, name="X")
        assert "add_food" in endpoints(state)


def test_batch_weight_within_tolerance_is_accepted(tmp_path):
    food = recipe()
    measures_by_id(food)[313167828]["value"] = 152.004
    client, state = recipe_client(tmp_path, food)

    client.update_recipe(RECIPE_ID, name="X")

    assert "add_food" in endpoints(state)


def test_update_recipe_timeout_on_read_back_says_it_was_saved(tmp_path):
    class TimeoutOnReadBack(FakeFoodServer):
        calls = 0

        def handle(self, endpoint, payload):
            if endpoint == "/api/v2/get_food":
                self.calls += 1
                if self.calls == 2:
                    raise self.timeout
            return super().handle(endpoint, payload)

    srv = TimeoutOnReadBack(foods=[recipe()], ingredient_foods=[BANANA, EGG, PB])
    srv.timeout = httpx.ReadTimeout("read timed out")
    client, state = make_server_client(tmp_path, srv)

    with pytest.raises(CronometerError) as err:
        client.update_recipe(RECIPE_ID, name="X")

    assert str(err.value) == (
        f"Recipe {RECIPE_ID} was saved; verify with get_food_details. read timed out"
    )
    assert err.value.__cause__ is srv.timeout
    assert state["endpoints"].count("add_food") == 1
    assert srv.foods[RECIPE_ID]["name"] == "X"
