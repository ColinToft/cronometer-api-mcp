"""update_recipe (client and tool), and retiring recipes via retire_custom_food.

Fixtures mirror get_food bodies captured from throwaway recipes on the live API.
"""

from __future__ import annotations

import copy
import json
import math

import pytest
from test_client import FakeFoodServer, make_server_client, server_reparse

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
        ingredient_foods=[BANANA, EGG, PB],
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
    assert without(data, "comments") == without(recipe(), "messages", "comments")


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
    assert data["nutrients"] == recipe()["nutrients"]
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


def test_serving_name_is_sent_verbatim_and_resets_quantity(tmp_path):
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
    ("serving_name", "name", "quantity"),
    [
        ("2 bowls", "bowls", 2.0),
        ("1.5 cups", "cups", 1.5),
        ("1/2 cup", "cup", 0.5),
        ("3/4 batch", "batch", 0.75),
        ("  3   plates", "plates", 3.0),
    ],
)
def test_quantity_led_serving_name_moves_number_into_quantity(
    tmp_path, serving_name, name, quantity
):
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(RECIPE_ID, serving_name=serving_name)

    m = measures_by_id(saved(state))[313167825]
    # The server stores a new "2 bowls" recipe measure the same way (live).
    assert (m["name"], m["amount"], m["type"]) == (name, quantity, "Weight")
    assert result["serving"]["quantity"] == quantity


@pytest.mark.parametrize("serving_name", ["0 bowls", "0 calorie syrup"])
def test_non_positive_leading_number_stays_in_name(tmp_path, serving_name):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, serving_name=serving_name)

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"]) == (serving_name, 1.0)


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


@pytest.mark.parametrize("unit", ["grams", "Ounces", "ml"])
def test_bare_rename_of_unit_count_measure_resets_quantity(tmp_path, unit):
    food = recipe()
    measures_by_id(food)[313167825].update(name=unit, amount=76)
    client, state = recipe_client(tmp_path, food)

    client.update_recipe(RECIPE_ID, serving_name="portion")

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"]) == ("portion", 1.0)


def test_serving_name_and_grams_together(tmp_path):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, serving_name="bowl", serving_grams=300)

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["value"]) == ("bowl", 1.0, 300)


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

    with pytest.raises(CronometerError, match="Can't tell which measure"):
        client.update_recipe(RECIPE_ID, name="X")

    assert endpoints(state) == ["get_food"]


def test_non_fixed_default_wins_over_other_servings(tmp_path):
    food = recipe()
    food["measures"].append(
        {"id": 5, "name": "bowl", "amount": 1, "value": 300, "type": "Weight"}
    )
    client, state = recipe_client(tmp_path, food)

    client.update_recipe(RECIPE_ID, serving_grams=80)

    by_id = measures_by_id(saved(state))
    assert by_id[313167825]["value"] == 80
    assert by_id[5]["value"] == 300


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
        RECIPE_ID, ingredients=[(450856, 150), (1234, 50), (619, 16)]
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

    assert without(data, "ingredients") == without(recipe(), "messages", "ingredients")
    assert result["total_grams"] == 216.0
    assert result["ingredient_count"] == 3
    assert result["serving"]["grams"] == 76


def test_ingredient_measure_override(tmp_path):
    client, state = recipe_client(tmp_path)

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 100, 3770)])

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


def test_ingredients_with_quantity_led_serving_and_grams(tmp_path):
    client, state = recipe_client(tmp_path)

    result = client.update_recipe(
        RECIPE_ID, ingredients=[(1234, 600)], serving_name="3 bowls", serving_grams=200
    )

    m = measures_by_id(saved(state))[313167825]
    assert (m["name"], m["amount"], m["value"], m["type"]) == (
        "bowls",
        3.0,
        200,
        "Weight",
    )
    assert result["serving"] == {
        "measure_id": 313167825,
        "name": "bowls",
        "quantity": 3.0,
        "grams": 200,
    }


def test_recipe_without_full_recipe_measure_still_updates(tmp_path):
    """get_food reports a virtual id-0 "full recipe" when none is stored;
    re-sending it makes the server store a real one (seen live)."""
    food = recipe()
    food["measures"] = [m for m in food["measures"] if m["name"] != "full recipe"]
    client, state = recipe_client(tmp_path, food)

    result = client.update_recipe(RECIPE_ID, ingredients=[(1234, 100)])

    sent = saved(state)["measures"]
    assert [(m["id"], m["name"]) for m in sent] == [
        (0, "full recipe"),
        (313167825, "serving (1/2 batch)"),
        (313167827, "oz"),
        (313167826, "g"),
    ]
    assert result["total_grams"] == 100


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
        ({"serving_name": "9" * 400 + " bowls"}, "serving_name quantity"),
        ({"serving_name": "full recipe"}, "reserved"),
        ({"serving_name": "Full Recipe "}, "reserved"),
        ({"serving_name": "g"}, "reserved"),
        ({"serving_name": "OZ"}, "reserved"),
        ({"serving_name": "1 g"}, "reserved"),
        ({"serving_name": "2 full recipe"}, "reserved"),
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

    with pytest.raises(CronometerError, match="not a user-created food"):
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


def test_refuses_serving_based_recipes(tmp_path):
    food = recipe()
    food["measures"][1]["type"] = "Recipe"
    client, state = recipe_client(tmp_path, food)

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

    with pytest.raises(CronometerError, match="not a user-created food"):
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

    client.update_recipe(RECIPE_ID, ingredients=[(1234, 100)])
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


class OddWeightServer(FakeFoodServer):
    """Derives a batch weight the client could not have computed."""

    def _save(self, food):
        resp = super()._save(food)
        for m in self.foods[food["id"]]["measures"]:
            if m["name"] == "full recipe":
                m["value"] = 999
        return resp


def test_update_reports_the_servers_batch_weight(tmp_path):
    client, _ = recipe_client(tmp_path, server_cls=OddWeightServer)

    assert client.update_recipe(RECIPE_ID, name="X")["total_grams"] == 999


def test_create_reports_the_servers_batch_weight(tmp_path):
    srv = OddWeightServer(ingredient_foods=[EGG])
    client, _ = make_server_client(tmp_path, srv)

    assert client.create_recipe("R", ingredients=[(1234, 100)])["total_grams"] == 999


# ---------------------------------------------------------------------------
# Serving names the server re-parses on create
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("serving_name", "sent", "resent", "stored"),
    [
        # Kept by the server as sent (verified live).
        ("serving (1/6 batch)", ("serving (1/6 batch)", 1.0), False, None),
        ("half (1/2)", ("half (1/2)", 1.0), False, None),
        # Re-parsed by the server (live: name "batch)", amount 0.5), so the
        # client re-sends the existing measure verbatim.
        ("serving (1/2 batch)", ("serving (1/2 batch)", 1.0), True, None),
        # A leading fraction is split client-side, like a leading number.
        ("1/2 cup", ("cup", 0.5), False, None),
        ("0 calorie syrup", ("0 calorie syrup", 1.0), True, None),
        # Split client-side and sent with its quantity, which the server
        # leaves alone (live: amount != 1 is never re-parsed).
        ("2 bowls", ("bowls", 2.0), False, None),
    ],
)
def test_create_recipe_serving_name_reads_back_verbatim(
    tmp_path, serving_name, sent, resent, stored
):
    srv = FakeFoodServer(ingredient_foods=[EGG])
    client, state = make_server_client(tmp_path, srv)

    result = client.create_recipe(
        "R", ingredients=[(1234, 100)], serving_name=serving_name, serving_grams=40
    )

    first = state["payloads"][state["endpoints"].index("add_food")]["data"]
    assert (first["measures"][0]["name"], first["measures"][0]["amount"]) == sent
    assert state["endpoints"].count("add_food") == (2 if resent else 1)
    m = measures_by_id(srv.foods[result["food_id"]])[result["measure_id"]]
    assert (m["name"], m["amount"], m["value"]) == (*sent, 40)


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
    assert "Recipe 4242 ('R') was created" in str(err.value)
    assert "update_recipe(4242" in str(err.value)


def test_create_recipe_read_back_failure_names_the_created_recipe(tmp_path):
    class LostServer(FakeFoodServer):
        def handle(self, endpoint, body):
            if endpoint == "/api/v2/get_food":
                raise CronometerError("boom")
            return super().handle(endpoint, body)

    client, state = make_server_client(tmp_path, LostServer(ingredient_foods=[EGG]))

    with pytest.raises(CronometerError, match=r"Recipe 4242 \('R'\) was created.*boom"):
        client.create_recipe("R", ingredients=[(1234, 100)])

    assert state["endpoints"].count("add_food") == 1


@pytest.mark.parametrize("name", ["", "  "])
def test_create_recipe_rejects_blank_name(tmp_path, name):
    client, state = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[EGG]))

    with pytest.raises(ValueError, match="name must not be empty"):
        client.create_recipe(name, ingredients=[(1234, 100)])

    assert state["endpoints"] == []


@pytest.mark.parametrize("serving_name", ["g", "Full Recipe", "1 oz", "  "])
def test_create_recipe_rejects_reserved_or_blank_serving_names(tmp_path, serving_name):
    client, state = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[EGG]))

    with pytest.raises(ValueError, match="serving_name"):
        client.create_recipe("R", ingredients=[(1234, 100)], serving_name=serving_name)

    assert state["endpoints"] == []


def test_update_recipe_resends_serving_if_server_rewrites_it(tmp_path):
    client, state = recipe_client(tmp_path, server_cls=StubbornServer)

    with pytest.raises(CronometerError, match="reads back as 'batch\\)'"):
        client.update_recipe(RECIPE_ID, serving_name="serving (1/2 batch)")

    assert state["endpoints"].count("add_food") == 2


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


def test_get_food_details_plain_food_has_no_recipe_fields(monkeypatch):
    fake = FakeClient(food=recipe(ingredients=[], comments=None))
    monkeypatch.setattr(server, "_get_client", lambda: fake)

    out = json.loads(server.get_food_details(RECIPE_ID))

    assert "ingredients" not in out
    assert "comments" not in out
    assert "retired" not in out
