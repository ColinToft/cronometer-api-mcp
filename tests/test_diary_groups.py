"""Diary group names from CRONOMETER_DIARY_GROUPS.

The API exposes no group names, so they're account configuration. Each test
starts from the defaults (conftest.py clears the variable) and sets its own
value through monkeypatch, so nothing leaks between tests.
"""

import json

import pytest
import test_diary_batch
from test_diary_batch import D1, _add, _add_one, _at_hour, _item, _logged
from test_food_log import DEFS as LOG_DEFS

from cronometer_api_mcp import server

fake = test_diary_batch.fake

COLIN = "breakfast,lunch,afternoon snack,dinner,evening snack"
COLIN_GROUPS = {
    "breakfast": 1,
    "lunch": 2,
    "afternoon snack": 3,
    "dinner": 4,
    "evening snack": 5,
}


@pytest.fixture
def groups(monkeypatch):
    def configure(value):
        monkeypatch.setenv("CRONOMETER_DIARY_GROUPS", value)

    return configure


# ---------------------------------------------------------------------------
# Parsing the variable
# ---------------------------------------------------------------------------


def test_unset_means_the_default_four():
    assert server._diary_groups() == {
        "breakfast": 1,
        "lunch": 2,
        "dinner": 3,
        "snacks": 4,
    }


def test_names_number_the_groups_in_order(groups):
    groups(COLIN)
    assert server._diary_groups() == COLIN_GROUPS


def test_a_single_name(groups):
    groups("meals")
    assert server._diary_groups() == {"meals": 1}


def test_names_are_stripped_lowercased_and_inner_whitespace_collapsed(groups):
    groups("  Breakfast ,LUNCH,Afternoon   Snack ,\tdinner,\nEvening\t Snack ")
    assert server._diary_groups() == COLIN_GROUPS


@pytest.mark.parametrize("value", ["group", "group snack", "snack group 2", "groups"])
def test_names_that_merely_contain_group_are_fine(groups, value):
    groups(value)
    assert server._diary_groups() == {value: 1}


@pytest.mark.parametrize(
    "value,problem",
    [
        ("", "name 1 is empty"),
        ("   ", "name 1 is empty"),
        (",lunch", "name 1 is empty"),
        ("breakfast,,dinner", "name 2 is empty"),
        ("breakfast, ,dinner", "name 2 is empty"),
        ("breakfast,lunch,", "name 3 is empty"),
        ("breakfast,breakfast", "'breakfast' is listed twice"),
        ("lunch,dinner, LUNCH ", "'lunch' is listed twice"),
        ("afternoon snack,Afternoon  Snack", "'afternoon snack' is listed twice"),
        ("auto", '"auto" is reserved'),
        ("breakfast, Auto ", '"auto" is reserved'),
        ("group 5", "'group 5' looks like a \"group N\" label"),
        ("breakfast,Group  6", "'group 6' looks like a \"group N\" label"),
        ("group5", "'group5' looks like a \"group N\" label"),
        ("group 0", "'group 0' looks like a \"group N\" label"),
        ("group 05", "'group 05' looks like a \"group N\" label"),
    ],
)
def test_bad_values_fail_naming_the_variable(groups, value, problem):
    groups(value)
    with pytest.raises(ValueError) as e:
        server._diary_groups()
    message = str(e.value)
    assert message.startswith("CRONOMETER_DIARY_GROUPS must be comma-separated")
    assert problem in message
    assert repr(value) in message


def test_a_bad_value_fails_every_tool_that_names_groups(groups, fake):
    f = fake()
    groups("breakfast,,dinner")

    result = _add([_item()])

    assert result["status"] == "error"
    assert "CRONOMETER_DIARY_GROUPS" in result["message"]
    assert f.state["endpoints"] == []


def test_main_refuses_a_bad_value_before_serving(groups, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env to load
    ran = []
    monkeypatch.setattr(server.mcp, "run", lambda **kw: ran.append(kw))
    groups("auto")

    with pytest.raises(ValueError, match="CRONOMETER_DIARY_GROUPS"):
        server.main()
    assert ran == []


def test_main_serves_with_a_good_value(groups, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ran = []
    monkeypatch.setattr(server.mcp, "run", lambda **kw: ran.append(kw))
    groups(COLIN)

    server.main()

    assert ran == [{"transport": "stdio"}]


# ---------------------------------------------------------------------------
# Read labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number,label",
    [
        (0, "group 0"),
        (1, "breakfast"),
        (2, "lunch"),
        (3, "dinner"),
        (4, "snacks"),
        (5, "group 5"),
    ],
)
def test_default_labels(number, label):
    assert server._group_name(number) == label


@pytest.mark.parametrize(
    "number,label",
    [
        (0, "group 0"),
        (1, "breakfast"),
        (2, "lunch"),
        (3, "afternoon snack"),
        (4, "dinner"),
        (5, "evening snack"),
        (6, "group 6"),
        (12, "group 12"),
    ],
)
def test_configured_labels(groups, number, label):
    groups(COLIN)
    assert server._group_name(number) == label


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,number",
    [
        ("breakfast", 1),
        ("lunch", 2),
        ("afternoon snack", 3),
        ("dinner", 4),
        ("evening snack", 5),
        ("Evening  Snack", 5),
        (" AFTERNOON\tsnack ", 3),
        ("group 6", 6),
        ("Group 9", 9),
    ],
)
def test_writes_by_every_configured_name(groups, fake, label, number):
    f = fake()
    groups(COLIN)

    assert _add([_item(diary_group=label)])["count"] == 1
    assert _logged(f, D1) == [(1234, number, 100)]


@pytest.mark.parametrize(
    "label",
    [
        "snacks",  # a default name Colin doesn't use
        "group 5",  # named: "evening snack"
        "group 4",
        "group 1",
        "group 0",
        "group 06",
        "eveningsnack",
        "auto",
        "",
    ],
)
def test_writes_refuse_labels_outside_the_configuration(groups, fake, label):
    f = fake()
    groups(COLIN)

    result = _add([_item(diary_group=label)])

    assert result["status"] == "error"
    assert (
        "diary_group must be breakfast, lunch, afternoon snack, dinner, "
        'evening snack, or "group N" (N >= 6, a group past the names in '
        "CRONOMETER_DIARY_GROUPS)"
    ) in result["message"]
    assert f.state["endpoints"] == []


def test_add_food_entry_writes_by_a_configured_name(groups, fake):
    f = fake()
    groups(COLIN)

    assert _add_one(diary_group="Evening Snack")["count"] == 1
    assert _logged(f, D1) == [(1234, 5, 100)]


# ---------------------------------------------------------------------------
# "auto"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hour,number", [(4, 1), (9, 1), (10, 2), (13, 2), (14, 4), (20, 4)]
)
def test_auto_resolves_through_the_configured_names(
    groups, fake, monkeypatch, hour, number
):
    """Colin's dinner is group 4, so "auto" at 18:00 must write 4, not the
    default dinner's 3."""
    f = fake()
    groups(COLIN)
    _at_hour(monkeypatch, hour)

    assert _add_one(diary_group="auto")["count"] == 1
    assert _logged(f, D1) == [(1234, number, 100)]


@pytest.mark.parametrize("hour", [0, 3, 21, 23])
def test_auto_fails_when_its_name_is_not_configured(groups, fake, monkeypatch, hour):
    f = fake()
    groups(COLIN)
    _at_hour(monkeypatch, hour)

    result = _add_one(diary_group="auto")

    assert result["status"] == "error"
    assert result["message"] == (
        f"ValueError: diary_group \"auto\" picks 'snacks' at hour {hour}, but "
        "CRONOMETER_DIARY_GROUPS has no group named 'snacks' (its names are "
        "breakfast, lunch, afternoon snack, dinner, evening snack); pass "
        "diary_group explicitly"
    )
    assert f.state["endpoints"] == []


@pytest.mark.parametrize(
    "value,hour,missing",
    [
        ("lunch,dinner,snacks", 8, "breakfast"),
        ("breakfast,dinner,snacks", 12, "lunch"),
        ("breakfast,lunch,snacks", 15, "dinner"),
    ],
)
def test_auto_names_whichever_name_is_missing(
    groups, fake, monkeypatch, value, hour, missing
):
    fake()
    groups(value)
    _at_hour(monkeypatch, hour)

    result = _add_one(diary_group="auto")

    assert result["status"] == "error"
    assert f"has no group named {missing!r}" in result["message"]


@pytest.mark.parametrize("hour,number", [(4, 1), (10, 2), (14, 3), (21, 4), (3, 4)])
def test_auto_with_the_defaults_is_unchanged(fake, monkeypatch, hour, number):
    f = fake()
    _at_hour(monkeypatch, hour)

    assert _add_one(diary_group="auto")["count"] == 1
    assert _logged(f, D1) == [(1234, number, 100)]


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_every_label_round_trips_through_get_food_log(groups, fake, monkeypatch):
    """Entries written by each configured name and by "group 6" read back
    under the same label."""
    fake()
    groups(COLIN)
    labels = [*COLIN_GROUPS, "group 6"]
    assert _add([_item(diary_group=label) for label in labels])["count"] == 6

    client = server._client
    fetch = client.get_diary
    monkeypatch.setattr(
        client,
        "get_diary",
        lambda day: {
            **fetch(day),
            "info": {"complete": False},
            "summary": {"macros": {"energy": 2800}, "consumed": {"total": 0}},
        },
    )
    monkeypatch.setattr(client, "get_nutrient_definitions", lambda: LOG_DEFS)
    log = json.loads(server.get_food_log(date=D1))

    assert log["status"] == "success", log
    assert [e["meal"] for e in log["entries"]] == labels
