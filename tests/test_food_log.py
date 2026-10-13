"""get_food_log: per-entry essentials of the diary, not full nutrient profiles.

As in test_nutrient_sources, the client is a Mock whose enrich_diary_servings
runs the real implementation, so the per-entry nutrient scaling is the real one.
"""

import json
from datetime import date
from unittest.mock import Mock, call

import pytest

from cronometer_api_mcp import server
from cronometer_api_mcp.client import CronometerClient

ENERGY, PROTEIN, FAT, SAT, CARBS, FIBER, SODIUM, CALCIUM = (
    208, 203, 204, 606, 205, 291, 307, 301,
)  # fmt: skip
IRON, B12 = 303, 418
DEFS = {
    ENERGY: {"name": "Energy", "unit": "kcal"},
    PROTEIN: {"name": "Protein", "unit": "g"},
    FAT: {"name": "Fat", "unit": "g"},
    SAT: {"name": "Saturated", "unit": "g"},
    CARBS: {"name": "Carbs", "unit": "g"},
    FIBER: {"name": "Fiber", "unit": "g"},
    SODIUM: {"name": "Sodium", "unit": "mg"},
    CALCIUM: {"name": "Calcium", "unit": "mg"},
    IRON: {"name": "Iron", "unit": "mg"},
    B12: {"name": "B12 (Cobalamin)", "unit": "µg"},
}
DEFAULT_NAMES = ["Energy", "Protein", "Fat", "Saturated", "Carbs", "Fiber"]
DEFAULT_NAMES += ["Sodium", "Calcium"]

FOODS = {
    # Per 100 g; egg has a "large" (50 g) and a "g" measure.
    1: {
        "id": 1,
        "name": "Egg",
        "defaultMeasureId": 10,
        "measures": [
            {"id": 10, "name": "large", "value": 50, "amount": 1, "type": "Atomic"},
            {"id": 11, "name": "g", "value": 1, "amount": 1, "type": "Weight"},
        ],
        "nutrients": [
            {"id": ENERGY, "amount": 143},
            {"id": PROTEIN, "amount": 12.6},
            {"id": SODIUM, "amount": 142},
            {"id": IRON, "amount": 1.75},
        ],
    },
    # Serving-based recipe: per reference serving; the diary "grams" is a count.
    2: {
        "id": 2,
        "name": "Stew",
        "measures": [
            {"id": 20, "name": "bowl", "value": 1, "amount": 1, "type": "Recipe"}
        ],
        "nutrients": [{"id": ENERGY, "amount": 400}, {"id": PROTEIN, "amount": 30}],
    },
    # No energy data at all.
    3: {
        "id": 3,
        "name": "Vitamin D3",
        "measures": [{"id": 30, "name": "tablet", "value": 100, "amount": 1}],
        "nutrients": [{"id": B12, "amount": 0.0}],
    },
}


def _serving(serving_id, food_id, measure_id, grams, group):
    return {
        "servingId": serving_id,
        "type": "Serving",
        "foodId": food_id,
        "measureId": measure_id,
        "grams": grams,
        "order": (group << 16) | 1,
        "day": "2026-10-14",
        "time": "8:0:0",
        "userId": 42,
    }


def _diary(*entries, target=2800, consumed=1000.4, complete=False):
    return {
        "summary": {"macros": {"energy": target}, "consumed": {"total": consumed}},
        "info": {"complete": complete, "day": "2026-10-14"},
        "diary": list(entries),
    }


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
    return json.loads(server.get_food_log(**kwargs))


def test_entries_carry_ids_measure_and_default_nutrients(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(501, 1, 10, 150, 1),  # 3 large eggs
        _serving(502, 1, 11, 25, 4),  # 25 g egg
    )

    result = _call(date="2026-10-14")

    assert result == {
        "status": "success",
        "date": "2026-10-14",
        "complete": False,
        "energy_summary": {
            "total_target_kcal": 2800,
            "consumed_kcal": 1000.4,
            "remaining_kcal": 1800,
        },
        "totals": {
            **dict.fromkeys(DEFAULT_NAMES, 0.0),
            "Energy": 250.25,
            "Protein": 22.05,
            "Sodium": 248.5,
        },
        "units": {
            "Energy": "kcal",
            "Protein": "g",
            "Fat": "g",
            "Saturated": "g",
            "Carbs": "g",
            "Fiber": "g",
            "Sodium": "mg",
            "Calcium": "mg",
        },
        "entries": [
            {
                "serving_id": 501,
                "meal": "breakfast",
                "food": "Egg",
                "food_id": 1,
                "measure_id": 10,
                "grams": 150,
                "measure": "large",
                "quantity": 3.0,
                "nutrients": {"Energy": 214.5, "Protein": 18.9, "Sodium": 213.0},
            },
            {
                "serving_id": 502,
                "meal": "snacks",
                "food": "Egg",
                "food_id": 1,
                "measure_id": 11,
                "grams": 25,
                "measure": "g",
                "quantity": 25.0,
                "nutrients": {"Energy": 35.75, "Protein": 3.15, "Sodium": 35.5},
            },
        ],
        "other_entries": [],
    }
    assert client.get_diary.call_args_list == [call(date(2026, 10, 14))]
    client.get_consumed_nutrients.assert_not_called()
    client.get_nutrition_scores.assert_not_called()


def test_output_is_one_line_of_minified_json(client):
    client.diaries[date(2026, 10, 14)] = _diary(_serving(501, 1, 10, 150, 1))

    raw = server.get_food_log()

    assert "\n" not in raw
    assert ", " not in raw.replace("Egg, ", "")
    assert json.loads(raw)["entries"][0]["serving_id"] == 501


def test_serving_based_recipe_has_null_grams_and_a_count(client):
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 2, 20, 1.5, 3))

    (entry,) = _call()["entries"]

    assert entry["grams"] is None
    assert (entry["measure"], entry["quantity"]) == ("bowl", 1.5)
    assert entry["nutrients"] == {"Energy": 600.0, "Protein": 45.0}


def test_a_measure_the_food_no_longer_has_is_null_not_the_default(client):
    # enrich_diary_servings describes it with the default measure ("large").
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 1, 999, 100, 2))

    (entry,) = _call()["entries"]

    assert entry["measure_id"] == 999
    assert entry["grams"] == 100
    assert entry["measure"] is None
    assert entry["quantity"] is None
    assert entry["nutrients"]["Energy"] == 143.0


@pytest.mark.parametrize("value", [0, None, "50"])
def test_a_measure_without_a_weight_has_no_quantity(client, value):
    egg = {
        **FOODS[1],
        "id": 8,
        "measures": [{"id": 80, "name": "large", "value": value}],
    }
    client.get_foods.side_effect = lambda ids: [egg]
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 8, 80, 150, 1))

    (entry,) = _call()["entries"]

    assert (entry["grams"], entry["measure"], entry["quantity"]) == (150, "large", None)


def test_food_without_measures_keeps_grams(client):
    bare = {**FOODS[1], "id": 9, "measures": []}
    client.get_foods.side_effect = lambda ids: [bare]
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 9, 0, 40, 2))

    (entry,) = _call()["entries"]

    assert (entry["grams"], entry["measure"], entry["quantity"]) == (40, None, None)


def test_missing_nutrients_are_left_out_and_add_nothing(client):
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 3, 30, 100, 1))

    result = _call()

    assert result["entries"][0]["nutrients"] == {}
    assert result["entries"][0]["quantity"] == 1.0
    assert result["totals"] == dict.fromkeys(DEFAULT_NAMES, 0.0)


def test_meals_decode_order_and_unknown_groups_are_numbered(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        *(_serving(g, 1, 11, 1, g) for g in (4, 3, 2, 1, 5))
    )

    meals = [e["meal"] for e in _call()["entries"]]

    assert meals == ["snacks", "dinner", "lunch", "breakfast", "group 5"]


def test_non_food_entries_pass_through(client):
    run = {"type": "Exercise", "name": "Running", "order": 1, "servingId": 9}
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 1, 10, 50, 1), run)

    result = _call()

    assert [e["serving_id"] for e in result["entries"]] == [7]
    assert result["other_entries"] == [run]


def test_energy_summary_is_null_without_a_target(client):
    client.diaries[date(2026, 10, 14)] = {**_diary(), "summary": {}}

    result = _call()

    assert result["energy_summary"] is None
    assert result["entries"] == []


@pytest.mark.parametrize(
    "nutrients,names",
    [
        (["iron", "B12", "Energy"], ["Iron", "B12 (Cobalamin)", "Energy"]),
        ("Iron", ["Iron"]),
        (["Iron", "iron"], ["Iron"]),
    ],
)
def test_nutrients_replace_the_default_set(client, nutrients, names):
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 1, 10, 100, 1))

    result = _call(nutrients=nutrients)

    assert list(result["totals"]) == names
    assert list(result["units"]) == names
    assert list(result["entries"][0]["nutrients"]) == [
        n for n in names if n != "B12 (Cobalamin)"
    ]
    if "Iron" in names:
        assert result["entries"][0]["nutrients"]["Iron"] == 1.75


@pytest.mark.parametrize(
    "nutrients,message",
    [
        ("zinc", "Unknown nutrient 'zinc'"),
        (["Fat", "at"], "Nutrient 'at' is ambiguous; it matches Fat, Saturated"),
        ([], "nutrients must name at least one nutrient"),
        ([""], "Nutrient names must be non-empty strings"),
        ([3], "Nutrient names must be non-empty strings"),
    ],
)
def test_bad_nutrients_fail_without_fetching_the_diary(client, nutrients, message):
    result = _call(nutrients=nutrients)

    assert result["status"] == "error"
    assert message in result["message"]
    client.get_diary.assert_not_called()


def test_multiple_days_fetch_each_day_once(client):
    client.diaries[date(2026, 10, 13)] = _diary(_serving(1, 1, 10, 50, 1))
    client.diaries[date(2026, 10, 14)] = _diary(_serving(2, 1, 10, 100, 1))

    result = _call(days=2)

    assert (result["start_date"], result["end_date"]) == ("2026-10-13", "2026-10-14")
    assert [d["totals"]["Energy"] for d in result["days"]] == [71.5, 143.0]
    assert client.get_diary.call_args_list == [
        call(date(2026, 10, 13)),
        call(date(2026, 10, 14)),
    ]
    assert client.get_foods.call_count == 2


def test_failed_enrichment_is_an_error_not_bare_entries(client):
    client.get_foods.side_effect = RuntimeError("boom")
    client.diaries[date(2026, 10, 14)] = _diary(_serving(7, 1, 10, 50, 1))

    result = _call()

    assert result["status"] == "error"
    assert "food_ids 1;" in result["message"]


def test_unknown_diary_shape_fails(client):
    client.get_diary.side_effect = None
    client.get_diary.return_value = {"summary": {}}

    result = _call()

    assert result["status"] == "error"
    assert "Unexpected get_diary response without a diary list" in result["message"]


def test_invalid_date_does_not_fetch(client):
    assert _call(date="2026-13-01")["status"] == "error"
    client.get_diary.assert_not_called()


ENTRY_KEYS = {
    "serving_id",
    "meal",
    "food",
    "food_id",
    "measure_id",
    "grams",
    "measure",
    "quantity",
    "nutrients",
}
DAY_KEYS = {
    "date",
    "complete",
    "energy_summary",
    "totals",
    "units",
    "entries",
    "other_entries",
}


def test_entries_carry_no_full_profile_or_raw_fields(client):
    client.diaries[date(2026, 10, 14)] = _diary(
        _serving(1, 1, 10, 50, 1), _serving(2, 2, 20, 1, 3), _serving(3, 3, 30, 5, 4)
    )

    result = _call()

    assert set(result) == {"status"} | DAY_KEYS
    assert [set(e) for e in result["entries"]] == [ENTRY_KEYS] * 3


@pytest.mark.parametrize("complete", [True, False])
def test_complete_reports_the_days_flag(client, complete):
    client.diaries[date(2026, 10, 14)] = _diary(complete=complete)

    assert _call()["complete"] is complete


def test_a_diary_without_info_fails_rather_than_guessing_complete(client):
    client.get_diary.side_effect = None
    client.get_diary.return_value = {"summary": {}, "diary": []}

    result = _call()

    assert result["status"] == "error"
    assert "KeyError" in result["message"] and "info" in result["message"]


def test_empty_day_has_zero_totals_and_no_entries(client):
    result = _call()

    assert result["entries"] == [] and result["other_entries"] == []
    assert result["totals"] == dict.fromkeys(DEFAULT_NAMES, 0.0)
    assert list(result["units"]) == DEFAULT_NAMES
    # An empty day needs no food lookups.
    client.get_foods.assert_not_called()


def test_totals_are_the_unrounded_sum_of_entries(client):
    # Seven 7 g eggs: each entry rounds to 10.01 kcal, the true total is 70.07.
    client.diaries[date(2026, 10, 14)] = _diary(
        *(_serving(i, 1, 11, 7, 1 + i % 4) for i in range(7))
    )

    result = _call(nutrients=["Energy", "Sodium", "Iron"])

    assert [e["nutrients"]["Energy"] for e in result["entries"]] == [10.01] * 7
    assert result["totals"] == {"Energy": 70.07, "Sodium": 69.58, "Iron": 0.86}


def test_multiple_days_keep_each_days_shape_and_skip_empty_lookups(client):
    run = {"type": "Exercise", "name": "Running", "order": 1, "servingId": 9}
    client.diaries[date(2026, 10, 12)] = _diary(run, complete=True)
    client.diaries[date(2026, 10, 14)] = _diary(_serving(2, 2, 20, 2, 3))

    result = _call(days=3, nutrients="Protein")

    assert set(result) == {"status", "start_date", "end_date", "days"}
    days = result["days"]
    assert [d["date"] for d in days] == ["2026-10-12", "2026-10-13", "2026-10-14"]
    assert [set(d) for d in days] == [DAY_KEYS] * 3
    assert [d["complete"] for d in days] == [True, False, False]
    assert [d["other_entries"] for d in days] == [[run], [], []]
    assert [d["totals"] for d in days] == [
        {"Protein": 0.0},
        {"Protein": 0.0},
        {"Protein": 60.0},
    ]
    assert client.get_foods.call_args_list == [call([2])]
