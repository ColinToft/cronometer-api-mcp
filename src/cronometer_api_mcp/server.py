"""MCP server for Cronometer nutrition data via the mobile REST API."""

import json
import logging
import re
import threading
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from .client import NUTRIENT_IDS, CronometerClient, _is_positive_number

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)
_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)
_WRITE_IDEMPOTENT = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)
_DESTRUCTIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=True,
)


def _server_version() -> str:
    """Version reported in serverInfo.

    Read from installed package metadata so the release tag (which publish.yml
    stamps into pyproject.toml) is the single source of truth. The field is
    display-only, so a missing distribution degrades to "" rather than failing.
    """
    try:
        return _pkg_version("cronometer-api-mcp")
    except PackageNotFoundError:
        return ""


mcp = MCPServer(
    "cronometer",
    instructions=(
        "Cronometer MCP server for nutrition tracking via the mobile REST API. "
        "Provides access to food search, diary management, daily nutrition data, "
        "macro targets, biometrics, and fasting history from Cronometer. "
        "Use search_foods to find foods, get_food_details for nutrition info "
        "and serving sizes, add_food_entry to log meals, and get_food_log to "
        "review what was eaten. To review or edit a day, prefer the much smaller "
        "get_food_log_compact, and add_food_entries / remove_food_entries to "
        "change many entries across dates in one call. For relative-date "
        "questions, prefer relative inputs (today, yesterday, N days ago) in "
        "get_food_log, get_daily_nutrition, and "
        "get_nutrition_scores instead of calculating dates from conversation history. "
        "Use get_daily_nutrition(days=N) for calories or nutrients over the "
        "last N calendar days, including today; set date='yesterday' for complete "
        "days only. Relative dates are resolved at call time in the account timezone."
    ),
    version=_server_version(),
)

# Cronometer's default diary groups. An entry's group is its order >> 16.
_DIARY_GROUPS = {"breakfast": 1, "lunch": 2, "dinner": 3, "snacks": 4}
_DIARY_GROUP_NAMES = {number: name for name, number in _DIARY_GROUPS.items()}

_client: CronometerClient | None = None
# Guards the lazy construction below. SDK 2.x dispatches sync tool handlers on
# worker threads, so concurrent first-calls would otherwise build a client each,
# meaning two logins against a rate-limited endpoint (#3).
_client_lock = threading.Lock()


def _get_client() -> CronometerClient:
    global _client
    if _client is None:
        with _client_lock:
            # Re-check: another thread may have won the race to the lock.
            if _client is None:
                _client = CronometerClient()
    return _client


def _parse_date(d: str | None) -> date | None:
    if d is None:
        return None
    return date.fromisoformat(d)


def _resolve_read_days(d: str | None, days: int = 1) -> list[date]:
    if type(days) is not int or not 1 <= days <= 31:
        raise ValueError("days must be an integer between 1 and 31.")
    value = d.strip().lower() if d is not None else "today"
    relative = re.fullmatch(r"([0-9]+) days? ago", value)
    try:
        if value in ("today", "yesterday") or relative:
            offset = int(relative[1]) if relative else int(value == "yesterday")
            end = date_module_today() - timedelta(days=offset)
        else:
            end = date.fromisoformat(value)
        return [end - timedelta(days=i) for i in range(days - 1, -1, -1)]
    except (ValueError, OverflowError) as e:
        raise ValueError(
            "Use a valid YYYY-MM-DD date, 'today', 'yesterday', or 'N days ago'; "
            "the entire range must fit within years 1–9999."
        ) from e


def _read_days_response(results: list[dict], *, indent: int | None = 2) -> str:
    if len(results) == 1:
        return _ok(results[0], indent=indent)
    return _ok(
        {
            "start_date": results[0]["date"],
            "end_date": results[-1]["date"],
            "days": results,
        },
        indent=indent,
    )


def _ok(data: dict, *, indent: int | None = 2) -> str:
    """Wrap a successful response."""
    separators = None if indent else (",", ":")
    return json.dumps(
        {"status": "success", **data}, indent=indent, separators=separators
    )


def _err(e: Exception) -> str:
    """Wrap an error response with actionable messages."""
    return json.dumps({"status": "error", "message": _err_message(e)})


def _err_message(e: Exception) -> str:
    import httpx

    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 401 or status == 403:
            msg = "Authentication failed. Cronometer session may have expired -- try again."
        elif status == 429:
            msg = "Rate limit exceeded. Wait a few minutes before retrying."
        elif status == 404:
            msg = f"Resource not found (HTTP {status})."
        else:
            msg = f"Cronometer API error (HTTP {status})."
    elif isinstance(e, httpx.TimeoutException):
        msg = "Request timed out. Cronometer may be slow -- try again."
    elif isinstance(e, httpx.ConnectError):
        msg = "Could not connect to Cronometer. Check network connectivity."
    else:
        msg = f"{type(e).__name__}: {e}"
    return msg


# ------------------------------------------------------------------
# Diary: read
# ------------------------------------------------------------------


@mcp.tool(annotations=_READ_ONLY)
def get_food_log(
    date: str | None = None,
    days: Annotated[int, Field(ge=1, le=31, strict=True)] = 1,
) -> str:
    """Get diary entries for one day or the last N calendar days.

    Prefer relative inputs for relative questions, rather than dates calculated
    from conversation history. For calorie/nutrient totals without individual
    entries, prefer get_daily_nutrition(days=N).

    Returns every food entry logged for the day. Each "Serving" entry is
    enriched (best-effort) with the food's name, source, serving_size, servings,
    and that food's own nutrient profile scaled to the amount eaten. Non-food
    entries (exercise, biometrics) carry their own name.

    serving_size defines one portion: quantity units weighing grams in total.
    servings is how many of those portions were consumed. For example,
    serving_size={quantity: 2, unit: "cookies", grams: 30} with servings=1
    means two cookies eaten; servings=0.5 means one cookie eaten. The size
    uses the entry's selected measure, or the food's default as a fallback.
    For serving-based recipes, serving_size.grams is null because the API
    uses reference-serving counts instead of gram weights. An invalid
    serving_size.quantity is reported as null.

    The per-entry profiles make this very large (hundreds of thousands of
    characters a day). To review or edit entries, prefer get_food_log_compact.

    Note: the per-entry "nutrients" are each food's individual contribution,
    which is distinct from the day-level nutrition_summary aggregate below.

    Also returns a top-level energy_summary field with pre-computed
    values most relevant to the user:

      - total_target_kcal: daily calorie target dynamically adjusted
        for expenditure and weight goal (equivalent to Cronometer's
        "Total Target" in the Energy Summary screen)
      - consumed_kcal: total calories consumed
      - remaining_kcal: calories remaining to stay on target
        (total_target_kcal - consumed_kcal). Always report this
        when summarizing the user's day. Prefer this over manually
        deriving values from the burn breakdown fields.

    Also returns a nutrition_summary field with consumed totals for every
    nutrient the user tracks in Cronometer (macros plus any tracked
    micronutrients such as saturated fat, cholesterol, or omega-3/6):

      - macros: flat macro totals (energy, protein, carbs, net_carbs, fat,
        fiber, alcohol)
      - nutrients: the full list of tracked nutrients with amounts and units

    Args:
        date: Inclusive end date: YYYY-MM-DD, today, yesterday, or N days ago.
              Defaults to today in the account timezone, resolved at call time.
        days: Number of calendar days, 1–31 (default 1). Includes the end date;
              days=3 includes today, date='yesterday', days=3 excludes today.

    One day returns the usual date/diary/summaries. Multiple days return
    start_date, end_date, and a days list of those objects, oldest first.
    """
    try:
        resolved_days = _resolve_read_days(date, days)
        client = _get_client()
        return _read_days_response(
            [_food_log_for_day(client, day) for day in resolved_days]
        )
    except Exception as e:
        return _err(e)


def _food_log_for_day(client: CronometerClient, day: date) -> dict:
    data = client.enrich_diary_servings(client.get_diary(day))
    return {
        "date": day.isoformat(),
        "energy_summary": _energy_summary(data),
        "nutrition_summary": client.get_consumed_nutrients(day),
        "diary": data,
    }


def _energy_summary(data: dict) -> dict | None:
    summary = (data or {}).get("summary") or {}
    target = (summary.get("macros") or {}).get("energy")
    consumed = (summary.get("consumed") or {}).get("total")
    if target is None or consumed is None:
        return None
    return {
        "total_target_kcal": target,
        "consumed_kcal": consumed,
        "remaining_kcal": round(target - consumed),
    }


_COMPACT_NUTRIENT_IDS = [
    NUTRIENT_IDS[key]
    for key in (
        "energy",
        "protein",
        "fat",
        "saturated_fat",
        "carbs",
        "fiber",
        "sodium",
        "calcium",
    )
]


@mcp.tool(annotations=_READ_ONLY)
def get_food_log_compact(
    date: str | None = None,
    days: Annotated[int, Field(ge=1, le=31, strict=True)] = 1,
    nutrients: list[str] | str | None = None,
) -> str:
    """Get diary entries in a small form, for reviewing or editing a day.

    About 1-2% the size of get_food_log: each entry carries a few key
    nutrients instead of the food's full profile. Each food entry has:

      - serving_id: the entry's ID, for remove_food_entries
      - meal: breakfast, lunch, dinner, snacks (Cronometer's default names,
        even if renamed in the app), or "group N" for any further group
      - food, food_id, measure_id: what add_food_entries takes to re-log it
      - grams: the logged weight; null for a serving-based recipe, whose
        diary amount is a count of reference servings (quantity holds it)
      - measure and quantity: the amount in the logged measure, e.g.
        3 "large"; both null if the food no longer has that measure, and
        quantity null if the measure has no usable weight
      - nutrients: {name: amount}; a nutrient the food has no data for is
        left out

    Per day it also returns energy_summary (as in get_food_log), totals (each
    nutrient summed over the day's foods), units ({name: unit}), and
    other_entries: non-food entries (exercise, biometrics) as the API sends
    them.

    Args:
        date: Inclusive end date: YYYY-MM-DD, today, yesterday, or N days ago.
              Defaults to today in the account timezone, resolved at call time.
        days: Number of calendar days, 1–31 (default 1). Includes the end date.
        nutrients: Nutrient names to report instead of the default Energy,
            Protein, Fat, Saturated, Carbs, Fiber, Sodium, Calcium. Matched
            as in get_nutrient_sources, e.g. ["Energy", "Iron", "B12"].

    One day returns the day's fields at the top level. Multiple days return
    start_date, end_date, and a days list, oldest first.
    """
    try:
        queries = None if nutrients is None else _nutrient_queries(nutrients)
        resolved_days = _resolve_read_days(date, days)
        client = _get_client()
        defs = client.get_nutrient_definitions()
        nutrient_ids = (
            _COMPACT_NUTRIENT_IDS
            if queries is None
            else _match_nutrients(queries, defs)
        )
        return _read_days_response(
            [_compact_day(client, day, defs, nutrient_ids) for day in resolved_days],
            indent=None,
        )
    except Exception as e:
        return _err(e)


def _compact_day(
    client: CronometerClient, day: date, defs: dict[int, dict], nutrient_ids: list[int]
) -> dict:
    data = client.enrich_diary_servings(client.get_diary(day))
    names = {nid: defs[nid]["name"] for nid in nutrient_ids}
    totals = dict.fromkeys(names.values(), 0.0)
    entries = []
    for entry in _food_servings(data):
        amounts = {n["id"]: n["amount"] for n in entry["nutrients"]}
        picked = {names[nid]: amounts[nid] for nid in nutrient_ids if nid in amounts}
        for name, amount in picked.items():
            totals[name] += amount
        size = entry.get("serving_size")
        # enrich_diary_servings falls back to the default measure when the
        # logged one is gone; that measure would misdescribe the amount.
        logged = size is not None and size["measure_id"] == entry["measureId"]
        # enrich_diary_servings sets servings only when the measure has a
        # weight, and gives a serving-based recipe's measure null grams.
        counted = logged and "servings" in entry
        recipe = counted and size["grams"] is None
        quantity = size["quantity"] if counted else None
        entries.append(
            {
                "serving_id": entry["servingId"],
                "meal": _group_name(entry["order"] >> 16),
                "food": entry["name"],
                "food_id": entry["foodId"],
                "measure_id": entry["measureId"],
                "grams": None if recipe else entry["grams"],
                "measure": size["unit"] if logged else None,
                "quantity": None
                if quantity is None
                else round(quantity * entry["servings"], 2),
                "nutrients": {k: round(v, 2) for k, v in picked.items()},
            }
        )
    return {
        "date": day.isoformat(),
        "energy_summary": _energy_summary(data),
        "totals": {k: round(v, 2) for k, v in totals.items()},
        "units": {names[nid]: defs[nid]["unit"] for nid in nutrient_ids},
        "entries": entries,
        "other_entries": [e for e in data["diary"] if e["type"] != "Serving"],
    }


@mcp.tool(annotations=_READ_ONLY)
def get_nutrient_sources(
    nutrients: list[str] | str,
    date: str | None = None,
    days: Annotated[int, Field(ge=1, le=31, strict=True)] = 1,
    top_n: Annotated[int, Field(ge=1, le=50, strict=True)] = 10,
) -> str:
    """Rank the foods that supplied the most of one or more nutrients.

    Use this for "where did my sodium come from?" or "top protein sources this
    week". It returns only the ranking, so it stays small; prefer it over
    get_food_log, whose per-entry nutrient profiles are very large.

    For each requested nutrient over the date range, returns the total, unit,
    the split by diary group (breakfast, lunch, dinner, snacks), and the top_n
    foods by amount. The same food is combined across entries and days. Each
    source has the food name, food_id, grams eaten in total, amount supplied,
    percent of the total, and the number of diary entries. Foods past top_n
    are summed under "other". The total is the sum over logged foods, so it
    matches get_daily_nutrition. Exercise and biometric entries are ignored,
    and a food with no data for a nutrient adds nothing to it.

    grams is null for a food logged in a serving-based recipe's reference
    servings, which have no gram weight. Diary groups 1-4 use Cronometer's
    default names even if they were renamed in the app; any further group
    shows as "group N".

    Args:
        nutrients: One nutrient name or a list, matched case-insensitively
            against Cronometer's names, e.g. "sodium", "Saturated", "Omega-3",
            "fat". A part of a name works when it fits only one nutrient
            ("B12" for "B12 (Cobalamin)"). Unknown or ambiguous names return
            an error that lists the candidates.
        date: Inclusive end date: YYYY-MM-DD, today, yesterday, or N days ago.
              Defaults to today in the account timezone, resolved at call time.
        days: Number of calendar days, 1–31 (default 1). Includes the end date;
              days=3 includes today, date='yesterday', days=3 excludes today.
        top_n: Number of foods to list per nutrient, 1–50 (default 10).

    One day returns date; multiple days return start_date and end_date.
    """
    try:
        if type(top_n) is not int or not 1 <= top_n <= 50:
            raise ValueError("top_n must be an integer between 1 and 50.")
        nutrients = _nutrient_queries(nutrients)
        resolved_days = _resolve_read_days(date, days)
        client = _get_client()
        defs = client.get_nutrient_definitions()
        nutrient_ids = _match_nutrients(nutrients, defs)
        diaries = [
            client.enrich_diary_servings(client.get_diary(day)) for day in resolved_days
        ]
        dates = (
            {"date": resolved_days[0].isoformat()}
            if len(resolved_days) == 1
            else {
                "start_date": resolved_days[0].isoformat(),
                "end_date": resolved_days[-1].isoformat(),
            }
        )
        ranking = _rank_sources(diaries, defs, nutrient_ids, top_n)
        return _ok({**dates, "nutrients": ranking})
    except Exception as e:
        return _err(e)


def _nutrient_queries(nutrients: list[str] | str) -> list[str]:
    if isinstance(nutrients, str):
        nutrients = [nutrients]
    if not nutrients:
        raise ValueError("nutrients must name at least one nutrient.")
    for query in nutrients:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Nutrient names must be non-empty strings; got {query!r}."
            )
    return nutrients


def _group_name(group: int) -> str:
    return _DIARY_GROUP_NAMES.get(group, f"group {group}")


def _diary_entries(diary: dict) -> list[dict]:
    """A get_diary response's entries; an empty day comes back as "diary": []."""
    if not (isinstance(diary, dict) and isinstance(diary.get("diary"), list)):
        shape = sorted(diary) if isinstance(diary, dict) else type(diary).__name__
        raise TypeError(f"Unexpected get_diary response without a diary list: {shape}")
    return diary["diary"]


def _food_servings(*diaries: dict) -> list[dict]:
    """The Serving entries of enriched diaries.

    Raises if any lacks nutrient data (enrichment is best-effort and leaves an
    entry bare when its food can't be fetched), since totals would silently
    undercount.
    """
    entries = [_diary_entries(diary) for diary in diaries]
    servings = [e for day in entries for e in day if e["type"] == "Serving"]
    unresolved = sorted({e["foodId"] for e in servings if "nutrients" not in e})
    if unresolved:
        raise RuntimeError(
            "Could not load nutrient data for food_ids "
            + ", ".join(map(str, unresolved))
            + "; totals would be incomplete. Try again."
        )
    return servings


def _match_nutrients(queries: list[str], defs: dict[int, dict]) -> list[int]:
    """Resolve nutrient names to catalog IDs, in request order, without repeats.

    An exact (case-insensitive) name wins; otherwise the query must be part of
    exactly one name. Anything else raises, naming the candidates.
    """
    names = {nid: meta["name"] for nid, meta in defs.items() if meta["name"]}
    resolved: list[int] = []
    for query in queries:
        q = query.strip().lower()
        matches = [nid for nid, name in names.items() if name.lower() == q] or [
            nid for nid, name in names.items() if q in name.lower()
        ]
        if not matches:
            raise ValueError(
                f"Unknown nutrient {query!r}. Valid names: "
                + ", ".join(sorted(names.values(), key=str.lower))
                + "."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Nutrient {query!r} is ambiguous; it matches "
                + ", ".join(sorted(names[nid] for nid in matches))
                + ". Use the full name."
            )
        if matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


def _rank_sources(
    diaries: list[dict], defs: dict[int, dict], nutrient_ids: list[int], top_n: int
) -> list[dict]:
    """Aggregate enriched diaries into a per-nutrient ranking of foods."""
    servings = _food_servings(*diaries)
    results = []
    for nid in nutrient_ids:
        foods: dict[int, dict] = {}
        groups: dict[int, float] = {}
        for entry in servings:
            amount = next(
                (n["amount"] for n in entry["nutrients"] if n["id"] == nid), 0.0
            )
            food = foods.setdefault(
                entry["foodId"],
                {
                    "food": entry["name"],
                    "food_id": entry["foodId"],
                    "grams": 0.0,
                    "amount": 0.0,
                    "entries": 0,
                },
            )
            food["amount"] += amount
            food["entries"] += 1
            # Serving-based recipe entries count reference servings, not grams.
            serving_size = entry.get("serving_size")
            if food["grams"] is not None and (
                serving_size is None or serving_size["grams"] is not None
            ):
                food["grams"] += entry["grams"]
            else:
                food["grams"] = None
            group = entry["order"] >> 16
            groups[group] = groups.get(group, 0.0) + amount

        ranked = sorted(
            (f for f in foods.values() if f["amount"] > 0),
            key=lambda f: (-f["amount"], f["food"]),
        )
        total = sum(f["amount"] for f in ranked)

        def pct(amount: float, total: float = total) -> float:
            return round(100 * amount / total, 1)

        sources = [
            {
                **f,
                "grams": None if f["grams"] is None else round(f["grams"], 1),
                "amount": round(f["amount"], 2),
                "percent": pct(f["amount"]),
            }
            for f in ranked[:top_n]
        ]
        result = {
            "nutrient_id": nid,
            "name": defs[nid]["name"],
            "unit": defs[nid]["unit"],
            "total": round(total, 2),
            "by_diary_group": {
                _group_name(g): {
                    "amount": round(a, 2),
                    "percent": pct(a),
                }
                for g, a in sorted(groups.items())
                if a > 0
            },
            "sources": sources,
        }
        rest = ranked[top_n:]
        if rest:
            other = sum(f["amount"] for f in rest)
            result["other"] = {
                "foods": len(rest),
                "amount": round(other, 2),
                "percent": pct(other),
            }
        results.append(result)
    return results


# ------------------------------------------------------------------
# Diary: write
# ------------------------------------------------------------------


@mcp.tool(annotations=_WRITE)
def add_food_entry(
    food_id: int,
    measure_id: int,
    grams: float,
    date: str | None = None,
    translation_id: int = 0,
    diary_group: str = "auto",
) -> str:
    """Add a food entry to the Cronometer diary.

    Use search_foods to find food_id and measure_id, then
    get_food_details to confirm serving sizes and gram weights.

    Args:
        food_id: Numeric food ID from search_foods results.
        measure_id: Measure/unit ID from get_food_details.
        grams: Weight of the serving in grams.
        date: Date to log as YYYY-MM-DD (defaults to today).
        translation_id: Translation ID from search results (usually 0).
        diary_group: Meal slot -- one of "auto", "breakfast", "lunch",
                     "dinner", "snacks" (case-insensitive, default "auto").
    """
    try:
        group_map = {"auto": 0, **_DIARY_GROUPS}
        group_key = diary_group.strip().lower()
        group_int = group_map.get(group_key)
        if group_int is None:
            return _err(
                ValueError(
                    f"Invalid diary_group '{diary_group}'. "
                    "Must be one of: auto, breakfast, lunch, dinner, snacks."
                )
            )

        client = _get_client()
        day = _parse_date(date)
        result = client.add_serving(
            food_id=food_id,
            measure_id=measure_id,
            grams=grams,
            translation_id=translation_id,
            day=day,
            diary_group=group_int,
        )
        return _ok(
            {
                "entry": result,
                "note": "Use the serving ID to remove this entry with remove_food_entry.",
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_WRITE)
def add_food_entries(entries: list[dict]) -> str:
    """Add several food entries, across any dates, in one call.

    Every item is checked before anything is logged: its fields, that the
    food exists, and that measure_id is one of that food's measures. A bad
    item fails the whole call with its index, and nothing is logged.

    Items are then logged in order, one request each. The result lists the
    added entries as {date, serving_id}, in input order, which
    remove_food_entries takes as-is. If an item fails, the call stops there
    and reports the entries already added (to keep or roll back), the
    failed index, and the indices not attempted.

    Args:
        entries: Objects with keys food_id, measure_id (from get_food_details;
            0 is refused), grams, date (YYYY-MM-DD), diary_group (breakfast,
            lunch, dinner, or snacks; "auto" is refused), and optionally
            translation_id (default 0). Unknown keys are refused.
    """
    try:
        items = _parse_batch(entries, _parse_add_entry)
        client = _get_client()
        foods = {
            f["id"]: f for f in client.get_foods(sorted({i["food_id"] for i in items}))
        }
        for index, item in enumerate(items):
            food = foods.get(item["food_id"])
            if food is None:
                raise ValueError(
                    f"entries[{index}]: food_id {item['food_id']} not found"
                )
            measures = {m["id"]: m["name"] for m in food["measures"]}
            if item["measure_id"] not in measures:
                raise ValueError(
                    f"entries[{index}]: measure_id {item['measure_id']} is not a "
                    f"measure of {food['name']!r} (food_id {food['id']}); its "
                    f"measures are {measures}"
                )
    except Exception as e:
        return _err(e)
    added: list[dict] = []
    for index, item in enumerate(items):
        try:
            result = client.add_serving(**item)
            added.append({"date": item["day"].isoformat(), "serving_id": result["id"]})
        except Exception as e:
            return _batch_failure(
                e, "added", added, [index], list(range(index + 1, len(items)))
            )
    return _ok({"added": added, "count": len(added)})


@mcp.tool(annotations=_DESTRUCTIVE)
def remove_food_entries(entries: list[dict]) -> str:
    """Remove food entries across any dates in one call.

    Every entry is looked up before anything is removed (one get_diary per
    date); an unknown or repeated serving_id fails the whole call with its
    index, and nothing is removed. Each date's entries are then removed in
    one request, dates in order of first appearance. If a date fails, the
    call stops there and reports what was removed, the failed indices, and
    the indices not attempted.

    Args:
        entries: Objects {"date": "YYYY-MM-DD", "serving_id": int}, as listed
            by get_food_log_compact or returned by add_food_entries.
    """
    try:
        items = _parse_batch(entries, _parse_remove_entry)
        by_day: dict[date, list[tuple[int, int]]] = {}
        first_index: dict[int, int] = {}
        for index, (day, serving_id) in enumerate(items):
            if serving_id in first_index:
                raise ValueError(
                    f"entries[{index}] repeats serving_id {serving_id} from "
                    f"entries[{first_index[serving_id]}]"
                )
            first_index[serving_id] = index
            by_day.setdefault(day, []).append((index, serving_id))
        client = _get_client()
        logged: dict[date, dict[int, dict]] = {}
        for day in by_day:
            diary = _diary_entries(client.get_diary(day))
            logged[day] = {e["servingId"]: e for e in diary if e["type"] == "Serving"}
        missing = [
            f"entries[{index}] (serving_id {serving_id} on {day.isoformat()})"
            for day, pairs in by_day.items()
            for index, serving_id in pairs
            if serving_id not in logged[day]
        ]
        if missing:
            raise ValueError(
                "Not in the diary, so nothing was removed: " + "; ".join(missing)
            )
    except Exception as e:
        return _err(e)
    removed: list[dict] = []
    for n, (day, pairs) in enumerate(by_day.items()):
        try:
            client.delete_diary_entries([logged[day][sid] for _, sid in pairs])
        except Exception as e:
            later = [i for p in list(by_day.values())[n + 1 :] for i, _ in p]
            return _batch_failure(e, "removed", removed, [i for i, _ in pairs], later)
        removed.extend({"date": day.isoformat(), "serving_id": sid} for _, sid in pairs)
    return _ok({"removed": removed, "count": len(removed)})


@mcp.tool(annotations=_DESTRUCTIVE)
def remove_food_entry(
    entry_ids: list[str],
    date: str | None = None,
) -> str:
    """Remove one or more food entries from one day of the Cronometer diary.

    remove_food_entries does the same across several dates. An ID that isn't
    in the day's diary fails the call, and nothing is removed.

    Args:
        entry_ids: List of serving/entry IDs to remove.
        date: Date the entries belong to as YYYY-MM-DD (defaults to today).
    """
    try:
        day = date or date_module_today().isoformat()
        for eid in entry_ids:
            if not (isinstance(eid, str) and eid.isdigit()):
                raise ValueError(f"entry_ids must be digit strings, got {eid!r}")
    except Exception as e:
        return _err(e)
    return remove_food_entries(
        [{"date": day, "serving_id": int(eid)} for eid in entry_ids]
    )


_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _parse_batch(entries: list[dict], parse) -> list:
    """Parse every item up front, naming the first bad one by index."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")
    parsed = []
    for index, item in enumerate(entries):
        try:
            parsed.append(parse(item))
        except (TypeError, ValueError) as e:
            raise ValueError(f"entries[{index}]: {e}") from None
    return parsed


def _check_keys(item: object, required: set[str], optional: set[str]) -> dict:
    if not isinstance(item, dict):
        raise TypeError(f"expected an object, got {item!r}")
    missing = required - set(item)
    unknown = set(item) - required - optional
    if missing or unknown:
        raise ValueError(
            f"missing keys {sorted(missing)}, unknown keys {sorted(unknown)}; "
            f"expected {sorted(required)}"
            + (f" and optionally {sorted(optional)}" if optional else "")
        )
    return item


def _positive_int(item: dict, key: str) -> int:
    value = item[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def _iso_day(value: object) -> date:
    if not (isinstance(value, str) and _ISO_DATE.fullmatch(value)):
        raise ValueError(f"date must be YYYY-MM-DD, got {value!r}")
    return date.fromisoformat(value)


def _parse_add_entry(item: object) -> dict:
    item = _check_keys(
        item,
        {"food_id", "measure_id", "grams", "date", "diary_group"},
        {"translation_id"},
    )
    grams = item["grams"]
    if isinstance(grams, bool) or not _is_positive_number(grams):
        raise ValueError(f"grams must be a positive number, got {grams!r}")
    group = item["diary_group"]
    if not (isinstance(group, str) and group.strip().lower() in _DIARY_GROUPS):
        raise ValueError(
            f"diary_group must be breakfast, lunch, dinner, or snacks, got {group!r}"
        )
    translation_id = item.get("translation_id", 0)
    if type(translation_id) is not int or translation_id < 0:
        raise ValueError(
            f"translation_id must be a non-negative integer, got {translation_id!r}"
        )
    return {
        "food_id": _positive_int(item, "food_id"),
        "measure_id": _positive_int(item, "measure_id"),
        "grams": grams,
        "translation_id": translation_id,
        "day": _iso_day(item["date"]),
        "diary_group": _DIARY_GROUPS[group.strip().lower()],
    }


def _parse_remove_entry(item: object) -> tuple[date, int]:
    item = _check_keys(item, {"date", "serving_id"}, set())
    return _iso_day(item["date"]), _positive_int(item, "serving_id")


def _batch_failure(
    e: Exception,
    done_key: str,
    done: list[dict],
    failed: list[int],
    not_attempted: list[int],
) -> str:
    """A mid-batch failure: what was applied, what failed, what wasn't tried."""
    return json.dumps(
        {
            "status": "error",
            "message": (
                f"entries{failed} failed: {_err_message(e)}. The call stopped; "
                f"'{done_key}' lists what was applied before it. A request that "
                f"failed in transit may still have been applied, so check with "
                f"get_food_log_compact before retrying."
            ),
            done_key: done,
            "failed": failed,
            "not_attempted": not_attempted,
        },
        indent=2,
    )


# ------------------------------------------------------------------
# Diary: management
# ------------------------------------------------------------------


@mcp.tool(annotations=_WRITE_IDEMPOTENT)
def mark_day_complete(date: str, complete: bool = True) -> str:
    """Mark a diary day as complete or incomplete.

    Args:
        date: Date to mark as YYYY-MM-DD.
        complete: True to mark complete, False for incomplete.
    """
    try:
        client = _get_client()
        day = _parse_date(date)
        result = client.mark_day_complete(day, complete)
        status = "complete" if complete else "incomplete"
        return _ok(
            {
                "date": date,
                "marked": status,
                "result": result,
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_WRITE)
def copy_day(date: str | None = None) -> str:
    """Copy all diary entries from the previous day to the given date.

    Additive -- does not remove existing entries on the destination date.

    Args:
        date: Destination date as YYYY-MM-DD (defaults to today).
    """
    try:
        client = _get_client()
        day = _parse_date(date)
        result = client.copy_day(to_day=day)
        return _ok(
            {
                "destination_date": date or str(date_module_today()),
                "result": result,
            }
        )
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Nutrition
# ------------------------------------------------------------------


@mcp.tool(annotations=_READ_ONLY)
def get_daily_nutrition(
    date: str | None = None,
    days: Annotated[int, Field(ge=1, le=31, strict=True)] = 1,
) -> str:
    """Get consumed macro and micronutrient totals for one or more days.

    Use days=N for calories/nutrients over the last N calendar days. Prefer
    relative inputs over dates calculated from conversation history.

    Returns the amounts actually consumed for the day, covering every nutrient
    the user tracks in Cronometer (i.e. has a target set for). The response has:

      - summary: flat macro totals (energy, protein, carbs, net_carbs, fat,
        fiber, alcohol). A value is null if that macro isn't tracked.
      - nutrients: the full list of tracked nutrients, each with id, name,
        amount, unit, category, and confidence.

    A nutrient only appears if it's tracked in Cronometer. To surface e.g.
    saturated fat, cholesterol, or trans fat, set a target for it in Cronometer
    and it will flow through automatically.

    Args:
        date: Inclusive end date: YYYY-MM-DD, today, yesterday, or N days ago.
              Defaults to today in the account timezone, resolved at call time.
        days: Number of calendar days, 1–31 (default 1). Includes the end date;
              days=3 includes today, date='yesterday', days=3 excludes today.

    One day returns date, summary, and nutrients. Multiple days return
    start_date, end_date, and a days list of those objects, oldest first.
    """
    try:
        resolved_days = _resolve_read_days(date, days)
        client = _get_client()
        results = []
        for day in resolved_days:
            data = client.get_consumed_nutrients(day)
            results.append(
                {
                    "date": day.isoformat(),
                    "summary": data["macros"],
                    "nutrients": data["nutrients"],
                }
            )
        return _read_days_response(results)
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_READ_ONLY)
def get_nutrition_scores(date: str | None = None) -> str:
    """Get nutrition scores with per-nutrient consumed amounts and category grades.

    Returns category scores (All Targets, Vitamins, Minerals, Electrolytes,
    Antioxidants, Immune Support, Metabolism, Bone Health) with the actual
    consumed amount and confidence level for each tracked nutrient.

    This is the richest nutrition endpoint -- use it when you need to know
    both how much of each nutrient was consumed AND how close each is to
    the target.

    Args:
        date: YYYY-MM-DD, today, yesterday, or N days ago. Defaults to today
              in the account timezone, resolved at call time. Prefer relative
              inputs for relative questions over dates from conversation history.
    """
    try:
        client = _get_client()
        day = _resolve_read_days(date)[0]
        data = client.get_nutrition_scores(day)
        return _ok(
            {
                "date": day.isoformat(),
                "scores": data,
            }
        )
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Food search and details
# ------------------------------------------------------------------


@mcp.tool(annotations=_READ_ONLY)
def search_foods(query: str) -> str:
    """Search Cronometer's food database by name.

    Returns matching foods with their IDs and source information.
    Use the food_id and measure_id from results with add_food_entry,
    or pass food_id to get_food_details for full nutrition info.

    Args:
        query: Food name or keyword (e.g. "eggs", "chicken breast").
    """
    try:
        client = _get_client()
        foods = client.search_food(query)

        # Slim down results to the most useful fields
        results = []
        for f in foods:
            results.append(
                {
                    "food_id": f.get("id"),
                    "name": f.get("name"),
                    "source": f.get("source"),
                    "measure_id": f.get("measureId"),
                    "translation_id": f.get("translationId"),
                    "measure_display": f.get("measureDisplayName"),
                    "score": f.get("score"),
                }
            )

        return _ok(
            {
                "query": query,
                "count": len(results),
                "foods": results,
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_READ_ONLY)
def get_food_details(food_id: int) -> str:
    """Get detailed food information including nutrition and serving sizes.

    Use this after search_foods to get the full nutrient profile and
    available measure_ids needed for add_food_entry. For a recipe it also
    returns its ingredients (in update_recipe's input shape) and comments; a
    deleted (retired) food is flagged with "retired": true.

    Args:
        food_id: Food ID from search_foods results.
    """
    try:
        client = _get_client()
        data = client.get_food(food_id)

        # Extract measures for easy reference
        measures = []
        for m in data.get("measures", []):
            measures.append(
                {
                    "measure_id": m.get("id"),
                    "name": m.get("name"),
                    "grams": m.get("value"),
                    "quantity": m.get("amount"),
                    "type": m.get("type"),
                }
            )

        details = {
            "food_id": data.get("id"),
            "name": data.get("name"),
            "default_measure_id": data.get("defaultMeasureId"),
            "measures": measures,
            "nutrients": data.get("nutrients", []),
        }
        if data.get("ingredients"):
            details["ingredients"] = [
                {
                    "food_id": i["foodId"],
                    "grams": i["grams"],
                    "measure_id": i["measureId"],
                }
                for i in data["ingredients"]
            ]
            # A recipe saved without comments has no "comments" key (seen live).
            details["comments"] = data.get("comments")
        if data.get("retired"):
            details["retired"] = True
        return _ok(details)
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Custom food creation
# ------------------------------------------------------------------


@mcp.tool(annotations=_WRITE)
def add_custom_food(
    name: str,
    calories: float,
    protein_g: float,
    fat_g: float,
    carbs_g: float,
    fiber_g: float = 0,
    sugar_g: float = 0,
    sodium_mg: float = 0,
    saturated_fat_g: float = 0,
    extra_nutrients: dict[int, float] | None = None,
    serving_name: str = "serving",
    serving_grams: float = 100.0,
    serving_quantity: float = 1,
) -> str:
    """Create a custom food in Cronometer with specified nutrition.

    Nutrient amounts should be for the full serving size specified.
    After creation, use the returned food_id with add_food_entry to log it.

    Args:
        name: Food name.
        calories: Calories per serving (kcal).
        protein_g: Protein per serving (g).
        fat_g: Fat per serving (g).
        carbs_g: Carbs per serving (g).
        fiber_g: Fiber per serving (g, default 0).
        sugar_g: Sugar per serving (g, default 0).
        sodium_mg: Sodium per serving (mg, default 0).
        saturated_fat_g: Saturated fat per serving (g, default 0).
        extra_nutrients: Additional nutrients beyond the core macros above
            (vitamins, minerals, amino acids, etc.), keyed by Cronometer
            nutrient ID (from get_daily_nutrition, which pairs each id with
            its name) and valued per the full serving. IDs aren't validated,
            so a wrong one writes the wrong nutrient; must not reuse an ID the
            named macro args already cover.
        serving_name: Unit name for the serving (default "serving"). A name
            that starts with a number ("2 cookies") is refused.
        serving_grams: Total weight of the serving (all serving_quantity
            units) in grams (default 100).
        serving_quantity: Units per serving (default 1), e.g. 2 "cookies".
    """
    try:
        client = _get_client()
        result = client.create_custom_food(
            name,
            calories=calories,
            protein_g=protein_g,
            fat_g=fat_g,
            carbs_g=carbs_g,
            fiber_g=fiber_g,
            sugar_g=sugar_g,
            sodium_mg=sodium_mg,
            saturated_fat_g=saturated_fat_g,
            extra_nutrients=extra_nutrients,
            serving_name=serving_name,
            serving_grams=serving_grams,
            serving_quantity=serving_quantity,
        )
        return _ok(
            {
                "food_id": result["food_id"],
                "measure_id": result["measure_id"],
                "name": name,
                "note": "Use food_id and measure_id with add_food_entry to log this food.",
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_DESTRUCTIVE)
def update_custom_food(
    food_id: int,
    name: str | None = None,
    calories: float | None = None,
    protein_g: float | None = None,
    fat_g: float | None = None,
    carbs_g: float | None = None,
    fiber_g: float | None = None,
    sugar_g: float | None = None,
    sodium_mg: float | None = None,
    saturated_fat_g: float | None = None,
    extra_nutrients: dict[int, float] | None = None,
    serving_name: str | None = None,
    serving_grams: float | None = None,
    serving_quantity: float | None = None,
) -> str:
    """Edit an existing custom food (one you created) in place.

    Only the arguments you pass change; everything else keeps its current
    value. Find the food_id with search_foods (source "Custom") or
    get_food_details. Diary entries that already use the food, including on
    past days, are recalculated with the new values. Recipes cannot be edited
    with this tool, even though they also show as source "Custom".

    Nutrient amounts are per serving: the food's default serving, or
    serving_grams when you pass it. To change the serving weight without
    re-entering nutrition, pass serving_grams alone; the stored per-100g
    values stay put, so the per-serving numbers scale with the new weight.

    Args:
        food_id: ID of the custom food to edit.
        name: New food name.
        calories: Calories per serving (kcal).
        protein_g: Protein per serving (g).
        fat_g: Fat per serving (g).
        carbs_g: Carbs per serving (g).
        fiber_g: Fiber per serving (g).
        sugar_g: Sugar per serving (g).
        sodium_mg: Sodium per serving (mg).
        saturated_fat_g: Saturated fat per serving (g).
        extra_nutrients: Additional nutrients keyed by Cronometer nutrient ID
            (from get_daily_nutrition) and valued per serving; must not reuse
            an ID the named args already cover.
        serving_name: New name for the default serving. The app shows the
            serving as "<quantity> <name>"; a leading number ("2 cookies")
            is refused: pass it as serving_quantity.
        serving_grams: New total weight of the default serving in grams.
            Also resets the quantity to serving_quantity (or 1).
        serving_quantity: How many units make up the default serving
            (default 1 whenever serving_grams is passed). Without
            serving_grams the serving's total weight stays the same, so each
            unit gets lighter or heavier: changing "2 cookies = 30 g" to 3
            gives "3 cookies = 30 g". To keep the per-unit weight, also pass
            serving_grams (45 here). Check the result with get_food_details,
            which lists each measure's quantity.
    """
    try:
        client = _get_client()
        result = client.update_custom_food(
            food_id,
            name=name,
            calories=calories,
            protein_g=protein_g,
            fat_g=fat_g,
            carbs_g=carbs_g,
            fiber_g=fiber_g,
            sugar_g=sugar_g,
            sodium_mg=sodium_mg,
            saturated_fat_g=saturated_fat_g,
            extra_nutrients=extra_nutrients,
            serving_name=serving_name,
            serving_grams=serving_grams,
            serving_quantity=serving_quantity,
        )
        return _ok({"food_id": result["food_id"], "name": result["name"]})
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_DESTRUCTIVE)
def delete_custom_food(food_id: int) -> str:
    """Delete a custom food or recipe (one you created) by retiring it.

    The API has no hard delete. The food drops out of search after a few
    seconds; diary entries that already use it are kept, and
    get_food_details can still read it by ID (flagged "retired"). Database
    foods (USDA, NCCDB, CRDB, ...) cannot be deleted, and neither can
    serving-based recipes (made in the app with "Recipe"-type measures):
    re-sending one through the API is untested, so delete those in the app.

    Args:
        food_id: ID of the custom food or recipe to delete.
    """
    try:
        return _ok(_get_client().retire_custom_food(food_id))
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Recipe creation
# ------------------------------------------------------------------

_INGREDIENT_KEYS = frozenset({"food_id", "grams", "measure_id"})


def _parse_ingredient_dicts(ingredients: list[dict]) -> list[tuple]:
    """Ingredient dicts to client tuples; unknown keys are rejected."""
    parsed = []
    for item in ingredients:
        if "food_id" not in item or "grams" not in item:
            raise ValueError(
                f"Each ingredient needs 'food_id' and 'grams'; got {item!r}"
            )
        unknown = set(item) - _INGREDIENT_KEYS
        if unknown:
            raise ValueError(
                f"Unknown ingredient keys {sorted(unknown)} in {item!r}; "
                f"expected food_id, grams, and optionally measure_id"
            )
        if item.get("measure_id") is not None:
            parsed.append((item["food_id"], item["grams"], item["measure_id"]))
        else:
            parsed.append((item["food_id"], item["grams"]))
    return parsed


@mcp.tool(annotations=_WRITE)
def add_recipe(
    name: str,
    ingredients: list[dict],
    serving_name: str = "Serving",
    serving_grams: float | None = None,
    servings_per_batch: float | None = None,
    serving_quantity: float = 1.0,
    comments: str | None = None,
) -> str:
    """Create a recipe in Cronometer from other foods in the database.

    Unlike add_custom_food, which takes hand-entered nutrition, a recipe
    references existing foods by ID and Cronometer derives the full nutrient
    profile (including micronutrients) from those ingredients.

    Use search_foods to find each ingredient's food_id. After creation, use
    the returned food_id with add_food_entry to log it.

    Args:
        name: Recipe name.
        ingredients: List of {"food_id": int, "grams": float} objects, one per
            ingredient. An optional "measure_id" overrides the unit shown in
            Cronometer's UI; "grams" always drives the nutrition math.
        serving_name: Name of the default serving measure (default "Serving"),
            stored exactly as given, e.g. "serving (1/6 batch)". "g", "oz" and
            "full recipe" are reserved, and a name starting with a quantity
            ("2 bowls") is refused: pass serving_quantity=2, serving_name="bowls".
        serving_grams: Grams in one serving, at most the batch weight. Defaults
            to the full batch weight (one serving = the whole recipe).
        servings_per_batch: Instead of serving_grams, make the serving this
            fraction of the batch, e.g. 6 for "serving (1/6 batch)".
        serving_quantity: How many units the serving is (default 1); the app
            shows "<quantity> <serving_name>", e.g. 1.5 with "cups".
        comments: Free-text recipe notes.
    """
    try:
        parsed = _parse_ingredient_dicts(ingredients)
        client = _get_client()
        result = client.create_recipe(
            name,
            ingredients=parsed,
            serving_name=serving_name,
            serving_grams=serving_grams,
            servings_per_batch=servings_per_batch,
            serving_quantity=serving_quantity,
            comments=comments,
        )

        return _ok(
            {
                "food_id": result["food_id"],
                "measure_id": result["measure_id"],
                "serving_grams": result["serving_grams"],
                "name": name,
                "total_grams": result["total_grams"],
                "ingredient_count": result["ingredient_count"],
                "note": "Use food_id and measure_id with add_food_entry to log this recipe.",
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_DESTRUCTIVE)
def update_recipe(
    food_id: int,
    ingredients: list[dict] | None = None,
    name: str | None = None,
    serving_name: str | None = None,
    serving_grams: float | None = None,
    servings_per_batch: float | None = None,
    serving_quantity: float | None = None,
    comments: str | None = None,
) -> str:
    """Edit an existing recipe (one you created) in place.

    The recipe keeps its food_id. In diary entries that already use it,
    including on past days, the nutrients per gram update. After a serving
    resize, entries logged by the serving follow it ("1 serving" stays 1
    serving), while entries logged by "g" keep their grams (both verified
    live). The result has a `note` when the serving's grams change. Only the
    arguments you pass change. Cronometer derives the nutrients and the "full
    recipe" weight from the ingredients. Check the result with
    get_food_details. Plain custom foods go through update_custom_food. On a
    recipe with several named servings, any ingredient edit that changes the
    batch weight is refused, even with servings_per_batch or serving_grams;
    do it in the Cronometer app.

    Args:
        food_id: ID of the recipe to edit.
        ingredients: The complete new ingredient list, in add_recipe's
            [{"food_id": int, "grams": float, "measure_id"?: int}] form. It
            replaces the old list entirely: to change one ingredient, read the
            current list with get_food_details and send it all back with that
            change. If the new list changes the batch weight, also pass
            servings_per_batch=N to keep a "serving (1/N batch)" at 1/N, or
            serving_grams to keep its weight; without either it's refused.
        name: New recipe name.
        serving_name: New name for the recipe's serving, stored exactly as
            given, e.g. "serving (1/6 batch)". Its quantity and weight are
            unchanged unless passed. "g", "oz" and "full recipe" are reserved,
            and a name starting with a quantity ("2 bowls") is refused.
        serving_grams: New weight of the serving in grams, at most the batch
            weight. Unlike update_custom_food, this keeps the quantity.
        servings_per_batch: Instead of serving_grams, make the serving this
            fraction of the (new) batch: batch grams / servings_per_batch,
            rounded to 0.01 g.
        serving_quantity: How many units the serving is; the app shows
            "<quantity> <serving_name>", e.g. 1.5 with "cups".
        comments: New recipe notes ("" clears them).
    """
    try:
        parsed = None if ingredients is None else _parse_ingredient_dicts(ingredients)
        client = _get_client()
        result = client.update_recipe(
            food_id,
            ingredients=parsed,
            name=name,
            serving_name=serving_name,
            serving_grams=serving_grams,
            servings_per_batch=servings_per_batch,
            serving_quantity=serving_quantity,
            comments=comments,
        )
        return _ok(result)
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_WRITE)
def import_recipe(ingredients: str, name: str | None = None) -> str:
    """Create a recipe from a free-text ingredient list.

    Cronometer's "Import Recipe" feature: pass ingredients as plain text, one
    per line, and the server matches each to a food and converts the amount to
    grams. No need to call search_foods first.

    Prefer this when the user describes ingredients in their own words. Use
    add_recipe instead when you have exact food_ids and gram weights -- e.g. the
    user confirmed specific foods from search_foods results.

    Matching is fuzzy, so report the returned matches back to the user for
    confirmation. Unresolved lines are listed under "unmatched" and excluded
    from the recipe. This saves to My Foods; use add_food_entry to log it.

    Args:
        ingredients: Ingredient lines separated by newlines, e.g.
            "2 tbsp olive oil\\n200g chicken". Include quantities where known,
            since bare names can match surprising amounts.
        name: Recipe name. Defaults to a server-generated one.
    """
    try:
        client = _get_client()
        result = client.import_recipe(ingredients, name=name)

        return _ok(
            {
                "food_id": result["food_id"],
                "measure_id": result["measure_id"],
                "name": result["recipe_name"],
                "total_grams": result["total_grams"],
                "ingredient_count": result["ingredient_count"],
                "ingredients": result["ingredients"],
                "unmatched": result["unmatched"],
                "note": (
                    "Matching is automatic -- show the ingredients list to the "
                    "user to confirm. Use food_id and measure_id with "
                    "add_food_entry to log this recipe."
                ),
            }
        )
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Macro targets
# ------------------------------------------------------------------


@mcp.tool(annotations=_READ_ONLY)
def get_macro_targets() -> str:
    """Get current macro targets including weekly schedule and templates.

    Returns the weekly macro schedule (which template applies to each day)
    and all saved macro target templates with their values.
    """
    try:
        client = _get_client()
        schedules = client.get_macro_schedules()
        templates = client.get_macro_target_templates()
        return _ok(
            {
                "schedules": schedules,
                "templates": templates,
            }
        )
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Fasting
# ------------------------------------------------------------------


@mcp.tool(annotations=_READ_ONLY)
def get_fasting_history(
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Get fasting history from Cronometer.

    Returns fasts within the date range including status, timestamps,
    and duration.

    Args:
        start_date: Start date as YYYY-MM-DD (defaults to 30 days ago).
        end_date: End date as YYYY-MM-DD (defaults to today).
    """
    try:
        client = _get_client()
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        data = client.get_fasting_with_date_range(start, end)
        return _ok(
            {
                "start_date": start_date
                or str(date_module_today() - timedelta(days=30)),
                "end_date": end_date or str(date_module_today()),
                "fasting": data,
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_READ_ONLY)
def get_fasting_stats() -> str:
    """Get aggregate fasting statistics.

    Returns total fasting hours, longest fast, average fast duration,
    and completed fast count.
    """
    try:
        client = _get_client()
        data = client.get_fasting_stats()
        return _ok({"stats": data})
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Biometrics
# ------------------------------------------------------------------


@mcp.tool(annotations=_READ_ONLY)
def list_biometrics() -> str:
    """List the biometric metrics tracked in Cronometer.

    Returns every metric type the account can record (Weight, Body Fat,
    Heart Rate, Blood Glucose, Waist Size, Sleep, blood panels, body
    measurements, etc.). Use the metric_id and a unit_id from the results
    with get_biometrics.
    """
    try:
        client = _get_client()
        metrics = client.get_metrics()

        # Slim down results to the fields needed to call get_biometrics
        results = []
        for m in metrics:
            results.append(
                {
                    "metric_id": m.get("id"),
                    "name": m.get("name"),
                    "units": [
                        {"unit_id": u.get("id"), "name": u.get("name")}
                        for u in m.get("units", [])
                    ],
                }
            )

        return _ok({"count": len(results), "metrics": results})
    except Exception as e:
        return _err(e)


@mcp.tool(annotations=_READ_ONLY)
def get_biometrics(
    metric_id: int,
    unit_id: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Get a biometric time series such as weight or body fat from Cronometer.

    Returns the recorded values over the date range as a list of
    {day, value} points.

    Use list_biometrics to find metric_id and unit_id (e.g. Weight is
    metric_id 1, with unit_id 1 for kg or 2 for lbs).

    Args:
        metric_id: Numeric metric ID from list_biometrics.
        unit_id: Numeric unit ID from the metric's units in list_biometrics.
        start_date: Start date as YYYY-MM-DD (defaults to 30 days ago).
        end_date: End date as YYYY-MM-DD (defaults to today).
    """
    try:
        client = _get_client()
        data = client.get_biometrics(
            metric_id,
            unit_id,
            start=_parse_date(start_date),
            end=_parse_date(end_date),
        )
        return _ok(
            {
                "metric_id": metric_id,
                "unit_id": unit_id,
                "start_date": start_date
                or str(date_module_today() - timedelta(days=30)),
                "end_date": end_date or str(date_module_today()),
                "biometrics": data,
            }
        )
    except Exception as e:
        return _err(e)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def date_module_today() -> date:
    """Return today's date in the account's timezone.

    Uses the authenticated client's timezone (resolved from the Cronometer
    account) so response echoes match the diary day an entry actually lands
    on, independent of the host clock. Extracted for easy mocking in tests.
    """
    return _get_client().today()


# ------------------------------------------------------------------
# Entrypoint (stdio only)
# ------------------------------------------------------------------


def main():
    """Run the MCP server over stdio.

    stdio is the only supported transport. For remote/hosted use the
    stdio process is wrapped by supergateway (see Dockerfile), which owns
    the HTTP listener; any HTTP exposure must sit behind an authenticating
    gateway/proxy.
    """
    # Load .env for local development (credentials). No-op if the file is
    # missing. override=False keeps real environment variables (Docker,
    # systemd, MCP client `env` blocks, etc.) authoritative over .env.
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path and load_dotenv(dotenv_path, override=False):
        logger.info("Loaded .env from %s", dotenv_path)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
