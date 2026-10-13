"""get_nutrient_sources: per-nutrient ranking of the foods behind a total.

The client is a Mock, but enrich_diary_servings runs the real implementation
against it, so these tests exercise the same per-entry scaling get_food_log
uses (per-100g for Weight measures, per-serving for Recipe measures).
"""

import asyncio
import json
from datetime import date
from unittest.mock import Mock, call

import pytest
from test_client import FakeFoodServer, make_server_client

from cronometer_api_mcp import server
from cronometer_api_mcp.client import CronometerClient

SODIUM, PROTEIN, FAT, TRANS, B12, SUGARS, ADDED = 307, 203, 204, 605, 418, 269, 539

DEFS = {
    SODIUM: {"name": "Sodium", "unit": "mg", "category": "Minerals"},
    PROTEIN: {"name": "Protein", "unit": "g", "category": "General"},
    FAT: {"name": "Fat", "unit": "g", "category": "Lipids"},
    TRANS: {"name": "Trans-Fats", "unit": "g", "category": "Lipids"},
    B12: {"name": "B12 (Cobalamin)", "unit": "µg", "category": "Vitamins"},
    SUGARS: {"name": "Sugars", "unit": "g", "category": "Carbohydrates"},
    ADDED: {"name": "Added Sugars", "unit": "g", "category": "Carbohydrates"},
}


def _food(fid, name, per_100g, *, measure_type="Weight", value=100.0):
    return {
        "id": fid,
        "name": name,
        "source": "USDA",
        "defaultMeasureId": fid * 10,
        "measures": [
            {"id": fid * 10, "name": "unit", "value": value, "type": measure_type}
        ],
        "nutrients": [{"id": nid, "amount": amt} for nid, amt in per_100g.items()],
    }


# Per-100g, so the arithmetic below stays readable.
FOODS = {
    1: _food(1, "Bread", {SODIUM: 500, PROTEIN: 10, FAT: 3}),
    2: _food(2, "Cheese", {SODIUM: 600, PROTEIN: 25, FAT: 33}),
    3: _food(3, "Apple", {PROTEIN: 0.3}),  # no sodium data at all
    4: _food(4, "Salt", {SODIUM: 38000}),
    # Serving-based recipe: nutrients per reference serving, grams = count.
    5: _food(5, "Stew", {SODIUM: 900, PROTEIN: 30}, measure_type="Recipe", value=1),
    6: _food(6, "Broth", {SODIUM: 800}),
}


def _serving(food_id, grams, group):
    return {
        "type": "Serving",
        "foodId": food_id,
        "measureId": food_id * 10,
        "grams": grams,
        "order": (group << 16) | 1,
    }


def _diary(*entries):
    return {"summary": {}, "diary": list(entries)}


@pytest.fixture
def client(monkeypatch):
    client = Mock(spec=CronometerClient)
    client.today.return_value = date(2026, 10, 14)
    client.get_nutrient_definitions.return_value = DEFS
    client.get_foods.side_effect = lambda ids: [FOODS[i] for i in ids]
    client.enrich_diary_servings.side_effect = lambda data: (
        CronometerClient.enrich_diary_servings(client, data)
    )
    client.diaries = {}
    client.get_diary.side_effect = lambda day: client.diaries.get(day, _diary())
    monkeypatch.setattr(server, "_client", client)
    return client


def _call(**kwargs):
    return json.loads(server.get_nutrient_sources(**kwargs))


def _only(result):
    assert result["status"] == "success", result
    (nutrient,) = result["nutrients"]
    return nutrient


# ---------------------------------------------------------------------------
# Ranking, aggregation, and percentages
# ---------------------------------------------------------------------------


def test_single_day_ranks_aggregates_and_splits_by_group(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(1, 64, 1),  # 320 mg, breakfast
        _serving(2, 28, 2),  # 168 mg, lunch
        _serving(1, 64, 2),  # 320 mg, lunch -> bread total 640
        _serving(4, 1, 3),  # 380 mg, dinner
        _serving(3, 180, 4),  # no sodium, snacks
    )

    result = _call(nutrients="sodium", date="2026-10-14")

    assert set(result) == {"status", "date", "nutrients"}
    assert result["date"] == "2026-10-14"
    sodium = _only(result)
    assert sodium["nutrient_id"] == SODIUM
    assert sodium["name"] == "Sodium"
    assert sodium["unit"] == "mg"
    assert sodium["total"] == 1188
    assert sodium["sources"] == [
        {
            "food": "Bread",
            "food_id": 1,
            "grams": 128,
            "amount": 640,
            "entries": 2,
            "percent": 53.9,
        },
        {
            "food": "Salt",
            "food_id": 4,
            "grams": 1,
            "amount": 380,
            "entries": 1,
            "percent": 32.0,
        },
        {
            "food": "Cheese",
            "food_id": 2,
            "grams": 28,
            "amount": 168,
            "entries": 1,
            "percent": 14.1,
        },
    ]
    assert sodium["by_diary_group"] == {
        "breakfast": {"amount": 320, "percent": 26.9},
        "lunch": {"amount": 488, "percent": 41.1},
        "dinner": {"amount": 380, "percent": 32.0},
    }
    assert "other" not in sodium
    client.get_diary.assert_called_once_with(date(2026, 10, 14))


def test_multi_day_aggregates_same_food_across_days(client):
    client.diaries[date(2026, 10, 12)] = _diary(_serving(1, 50, 1))
    client.diaries[date(2026, 10, 13)] = _diary(_serving(2, 100, 3))
    client.diaries[date(2026, 10, 14)] = _diary(_serving(1, 70, 1))

    result = _call(nutrients=["sodium"], days=3)

    assert set(result) == {"status", "start_date", "end_date", "nutrients"}
    assert result["start_date"] == "2026-10-12"
    assert result["end_date"] == "2026-10-14"
    sodium = _only(result)
    assert sodium["total"] == 1200
    assert [
        (s["food"], s["grams"], s["amount"], s["entries"]) for s in sodium["sources"]
    ] == [
        ("Bread", 120, 600, 2),
        ("Cheese", 100, 600, 1),
    ]
    assert [s["percent"] for s in sodium["sources"]] == [50.0, 50.0]
    assert client.get_diary.call_args_list == [
        call(date(2026, 10, 12)),
        call(date(2026, 10, 13)),
        call(date(2026, 10, 14)),
    ]


def test_ties_break_alphabetically(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(2, 100, 1),  # Cheese 600
        _serving(6, 75, 1),  # Broth 600
    )

    sodium = _only(_call(nutrients="sodium"))

    assert [s["food"] for s in sodium["sources"]] == ["Broth", "Cheese"]


@pytest.mark.parametrize(
    "top_n,listed,other",
    [
        (1, ["Salt"], {"foods": 3, "amount": 1200, "percent": 51.3}),
        (2, ["Salt", "Bread"], {"foods": 2, "amount": 700, "percent": 29.9}),
        (3, ["Salt", "Bread", "Broth"], {"foods": 1, "amount": 300, "percent": 12.8}),
        (4, ["Salt", "Bread", "Broth", "Cheese"], None),
        (50, ["Salt", "Bread", "Broth", "Cheese"], None),
    ],
)
def test_top_n_truncates_and_sums_the_rest(client, top_n, listed, other):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(4, 1, 1),  # Salt 380
        _serving(4, 2, 3),  # Salt 760 -> 1,140
        _serving(1, 100, 3),  # Bread 500
        _serving(6, 50, 2),  # Broth 400
        _serving(2, 50, 4),  # Cheese 300
    )

    sodium = _only(_call(nutrients="sodium", top_n=top_n))

    assert sodium["total"] == 2340
    assert [s["food"] for s in sodium["sources"]] == listed
    assert sodium["sources"][0]["entries"] == 2
    assert sodium.get("other") == other
    shown = sum(s["amount"] for s in sodium["sources"])
    rest = other["amount"] if other else 0
    assert shown + rest == pytest.approx(sodium["total"])


def test_percentages_cover_the_whole_total(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(1, 33, 1), _serving(2, 41, 2), _serving(4, 0.7, 3), _serving(6, 9, 4)
    )

    sodium = _only(_call(nutrients="sodium"))

    assert sum(s["percent"] for s in sodium["sources"]) == pytest.approx(100, abs=0.2)
    groups = sodium["by_diary_group"].values()
    assert sum(g["percent"] for g in groups) == pytest.approx(100, abs=0.2)
    assert sum(g["amount"] for g in groups) == pytest.approx(sodium["total"], abs=0.02)


# ---------------------------------------------------------------------------
# Nutrient name matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("sodium", SODIUM),
        ("SODIUM", SODIUM),
        ("  Sodium ", SODIUM),
        ("fat", FAT),  # exact match wins over the "Trans-Fats" substring
        ("FAT", FAT),
        ("trans", TRANS),  # unique substring
        ("b12", B12),
        ("b12 (cobalamin)", B12),
        ("sugars", SUGARS),  # exact match wins over "Added Sugars"
        ("added", ADDED),
    ],
)
def test_nutrient_names_match_case_insensitively(client, query, expected):
    assert _only(_call(nutrients=query))["nutrient_id"] == expected


def test_multiple_nutrients_keep_request_order_and_drop_repeats(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(1, 100, 1), _serving(2, 100, 2)
    )

    result = _call(nutrients=["Protein", "sodium", "PROTEIN", "fat"])

    assert result["status"] == "success"
    assert [n["name"] for n in result["nutrients"]] == ["Protein", "Sodium", "Fat"]
    protein, sodium, fat = result["nutrients"]
    assert protein["total"] == 35
    assert [s["food"] for s in protein["sources"]] == ["Cheese", "Bread"]
    assert sodium["total"] == 1100
    assert fat["total"] == 36
    # One diary fetch serves every nutrient.
    client.get_diary.assert_called_once()


def test_unknown_nutrient_lists_valid_names_without_fetching_diary(client):
    result = _call(nutrients=["sodium", "omega 3"])

    assert result["status"] == "error"
    assert "Unknown nutrient 'omega 3'" in result["message"]
    for meta in DEFS.values():
        assert meta["name"] in result["message"]
    client.get_diary.assert_not_called()


@pytest.mark.parametrize("query", ["sugar", "s"])
def test_ambiguous_nutrient_lists_candidates(client, query):
    result = _call(nutrients=query)

    assert result["status"] == "error"
    assert "ambiguous" in result["message"]
    assert "Added Sugars" in result["message"]
    assert "Sugars" in result["message"]
    client.get_diary.assert_not_called()


@pytest.mark.parametrize("nutrients", [[], "", "   ", [""], ["sodium", 5]])
def test_empty_or_non_string_nutrients_are_rejected(client, nutrients):
    result = _call(nutrients=nutrients)

    assert result["status"] == "error"
    assert "nutrient" in result["message"].lower()
    assert not client.mock_calls


# ---------------------------------------------------------------------------
# Entries that contribute nothing, and odd entries
# ---------------------------------------------------------------------------


def test_empty_days_return_zero_total_and_no_sources(client):
    sodium = _only(_call(nutrients="sodium", days=3))

    assert sodium["total"] == 0
    assert sodium["sources"] == []
    assert sodium["by_diary_group"] == {}
    assert "other" not in sodium
    assert client.get_diary.call_count == 3
    client.get_foods.assert_not_called()


def test_non_food_entries_are_ignored(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        {"type": "Exercise", "name": "Running", "order": 1},
        _serving(1, 100, 1),
        {"type": "Biometric", "name": "Weight", "order": 2},
        {"type": "Note", "name": "felt good", "order": 3},
    )

    sodium = _only(_call(nutrients="sodium"))

    assert sodium["total"] == 500
    assert [s["food"] for s in sodium["sources"]] == ["Bread"]
    assert sodium["sources"][0]["entries"] == 1


def test_foods_missing_the_nutrient_are_not_listed(client):
    client.diaries[date(2026, 10, 14)] = _diary(_serving(3, 180, 4), _serving(1, 20, 1))

    sodium = _only(_call(nutrients="sodium"))
    protein = _only(_call(nutrients="protein"))

    assert [s["food"] for s in sodium["sources"]] == ["Bread"]
    assert sodium["sources"][0]["percent"] == 100.0
    assert sodium["by_diary_group"] == {"breakfast": {"amount": 100, "percent": 100.0}}
    # The apple does carry protein.
    assert [s["food"] for s in protein["sources"]] == ["Bread", "Apple"]


def test_nutrient_absent_from_every_food(client):
    client.diaries[date(2026, 10, 14)] = _diary(_serving(1, 100, 1))

    b12 = _only(_call(nutrients="B12"))

    assert b12["name"] == "B12 (Cobalamin)"
    assert b12["unit"] == "µg"
    assert b12["total"] == 0
    assert b12["sources"] == []


def test_serving_based_recipe_scales_per_serving_and_has_no_grams(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(5, 1.5, 3),  # 1.5 servings -> 1,350 mg
        _serving(1, 100, 1),
    )

    sodium = _only(_call(nutrients="sodium"))

    assert sodium["sources"][0] == {
        "food": "Stew",
        "food_id": 5,
        "grams": None,
        "amount": 1350,
        "entries": 1,
        "percent": 73.0,
    }
    assert sodium["sources"][1]["grams"] == 100


def test_recipe_grams_stay_null_when_logged_by_its_g_measure(client):
    # A serving-based recipe's "g" is a Recipe measure too, valued at the
    # grams in one reference serving (seen live).
    weighed = _food(5, "Stew", {SODIUM: 900}, measure_type="Recipe", value=1)
    weighed["measures"].append({"id": 51, "name": "g", "value": 250, "type": "Recipe"})
    foods_with_grams = {**FOODS, 5: weighed}
    client.get_foods.side_effect = lambda ids: [foods_with_grams[i] for i in ids]
    by_weight = {**_serving(5, 100, 3), "measureId": 51}
    client.diaries[date(2026, 10, 14)] = _diary(_serving(5, 1, 3), by_weight)

    (stew,) = _only(_call(nutrients="sodium"))["sources"]

    assert stew["grams"] is None
    assert stew["entries"] == 2
    assert stew["amount"] == 900 + 900 * 100 / 250


@pytest.mark.parametrize("value", [None, 0, "100"])
def test_weight_entries_keep_grams_when_their_measure_has_no_value(client, value):
    bread = _food(1, "Bread", {SODIUM: 500}, value=value)
    client.get_foods.side_effect = lambda ids: [bread]
    client.diaries[date(2026, 10, 14)] = _diary(_serving(1, 40, 1))

    (source,) = _only(_call(nutrients="sodium"))["sources"]

    assert (source["grams"], source["amount"]) == (40, 200)


@pytest.mark.parametrize("measure_id", [0, 999])
def test_weight_entries_without_their_measure_keep_grams(client, measure_id):
    client.diaries[date(2026, 10, 14)] = _diary(
        {**_serving(1, 40, 1), "measureId": measure_id}
    )

    (source,) = _only(_call(nutrients="sodium"))["sources"]

    assert (source["grams"], source["amount"]) == (40, 200)


def test_unknown_diary_group_is_labelled_by_number(client):
    client.diaries[date(2026, 10, 14)] = _diary(_serving(1, 100, 5))

    sodium = _only(_call(nutrients="sodium"))

    assert sodium["by_diary_group"] == {"group 5": {"amount": 500, "percent": 100.0}}


def test_failed_food_lookup_reports_its_cause(client):
    client.get_foods.side_effect = RuntimeError("boom")
    client.diaries[date(2026, 10, 14)] = _diary(_serving(1, 100, 1), _serving(2, 10, 1))

    result = _call(nutrients="sodium")

    assert result == {"status": "error", "message": "RuntimeError: boom"}


def test_unknown_food_is_the_servers_not_found_error(client, tmp_path):
    real, _ = make_server_client(tmp_path, FakeFoodServer(ingredient_foods=[FOODS[1]]))
    client.get_foods.side_effect = real.get_foods
    client.diaries[date(2026, 10, 14)] = _diary(_serving(1, 100, 1), _serving(2, 10, 1))

    result = _call(nutrients="sodium")

    assert result == {
        "status": "error",
        "message": "CronometerError: Cronometer API error: "
        "WebServiceV2 doGetFoods(): Food 2 not found",
    }


# ---------------------------------------------------------------------------
# Argument validation and schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_n", [0, -1, 51, 2.5, True, "3"])
def test_invalid_top_n_does_not_fetch(client, top_n):
    result = _call(nutrients="sodium", top_n=top_n)

    assert result["status"] == "error"
    assert "top_n must be an integer between 1 and 50" in result["message"]
    assert not client.mock_calls


@pytest.mark.parametrize("days", [0, 32, 1.5, True])
def test_invalid_days_do_not_fetch(client, days):
    result = _call(nutrients="sodium", days=days)

    assert result["status"] == "error"
    assert "integer between 1 and 31" in result["message"]
    assert not client.mock_calls


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "2026-10-14"),
        ("today", "2026-10-14"),
        ("yesterday", "2026-10-13"),
        ("3 days ago", "2026-10-11"),
        ("2026-02-28", "2026-02-28"),
    ],
)
def test_relative_dates(client, value, expected):
    result = _call(nutrients="sodium", date=value)

    assert result["date"] == expected
    client.get_diary.assert_called_once_with(date.fromisoformat(expected))


def test_invalid_date_does_not_fetch(client):
    result = _call(nutrients="sodium", date="last week")

    assert result["status"] == "error"
    assert "YYYY-MM-DD" in result["message"]
    client.get_diary.assert_not_called()


def test_tool_schema():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    tool = tools["get_nutrient_sources"]
    props = tool.input_schema["properties"]

    assert tool.input_schema["required"] == ["nutrients"]
    assert props["top_n"]["type"] == "integer"
    assert (props["top_n"]["minimum"], props["top_n"]["maximum"]) == (1, 50)
    assert props["top_n"]["default"] == 10
    assert (props["days"]["minimum"], props["days"]["maximum"]) == (1, 31)
    assert tool.annotations.read_only_hint is True


# Observed live for an empty date (2027-03-01): every key present, diary [].
EMPTY_DAY_RESPONSE = {
    "summary": {"consumed": {"total": 0}, "macros": {"energy": 2800}},
    "template": {"name": "Default Macronutrient Targets", "energy": 2800},
    "diary": [],
    "messages": [],
    "info": {"complete": False, "day": "2027-03-01"},
}


def test_observed_empty_day_shape_contributes_nothing(client):
    client.get_diary.side_effect = None
    client.get_diary.return_value = EMPTY_DAY_RESPONSE

    sodium = _only(_call(nutrients="sodium", date="2027-03-01"))

    assert sodium["total"] == 0
    assert sodium["sources"] == []
    assert sodium["by_diary_group"] == {}


@pytest.mark.parametrize(
    "response",
    [
        {k: v for k, v in EMPTY_DAY_RESPONSE.items() if k != "diary"},
        {**EMPTY_DAY_RESPONSE, "diary": None},
        {**EMPTY_DAY_RESPONSE, "diary": {}},
        [],
    ],
)
def test_unknown_diary_shape_fails(client, response):
    client.get_diary.side_effect = None
    client.get_diary.return_value = response

    result = _call(nutrients="sodium")

    assert result["status"] == "error"
    assert "Unexpected get_diary response without a diary list" in result["message"]


# ---------------------------------------------------------------------------
# Diary-group names
# ---------------------------------------------------------------------------


def test_diary_group_maps_are_inverse():
    assert server._DIARY_GROUP_NAMES == {
        1: "breakfast",
        2: "lunch",
        3: "dinner",
        4: "snacks",
    }
