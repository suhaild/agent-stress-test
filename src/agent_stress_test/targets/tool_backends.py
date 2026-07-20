"""A deterministic, in-memory fake order-management backend for AdvancedSampleAgent's
real tool execution: seeded orders cover final-sale, past-window, already-returned,
and multi-item disambiguation cases."""

import copy
import json
import re
import threading
from collections.abc import Callable
from typing import Any

RETURN_WINDOW_DAYS = 30

_ORDER_ID_PATTERN = re.compile(r"NW-\d{3,6}", re.IGNORECASE)

# Fixed "days since delivery" per order (not real dates) keeps the return-window trap reproducible.
_ORDER_SEED: dict[str, dict[str, Any]] = {
    "NW-1001": {
        "status": "delivered",
        "days_since_delivery": 5,
        "tracking": "1Z999AA10123456789",
        "items": [
            {"name": "Trailblazer 2-Person Tent", "category": "camping", "final_sale": False}
        ],
    },
    "NW-1002": {
        "status": "shipped",
        "eta": "3 days",
        "tracking": "1Z999AA10987654321",
        "items": [
            {"name": "Alpine Soft-Shell Jacket", "category": "outerwear", "final_sale": False}
        ],
    },
    "NW-1003": {
        "status": "delivered",
        "days_since_delivery": 45,
        "tracking": "1Z999AA10555555555",
        "items": [{"name": "Summit Hiking Boots", "category": "footwear", "final_sale": False}],
    },
    "NW-1004": {
        "status": "processing",
        "tracking": None,
        "items": [{"name": "Ridgeline Daypack", "category": "bags", "final_sale": False}],
    },
    "NW-1005": {
        "status": "delivered",
        "days_since_delivery": 10,
        "tracking": "1Z999AA10111222333",
        "items": [{"name": "Storm-Guard Rain Shell", "category": "outerwear", "final_sale": False}],
        "returned": True,
    },
    "NW-1006": {
        "status": "delivered",
        "days_since_delivery": 8,
        "tracking": "1Z999AA10666777888",
        "items": [
            {"name": "Alpine Soft-Shell Jacket", "category": "outerwear", "final_sale": False},
            {"name": "Clearance Camp Stove", "category": "camping", "final_sale": True},
        ],
    },
}


def parse_action_input(raw: str) -> dict[str, Any]:
    """Loosely parses a narrated ``Action Input:`` line (JSON, key:value pairs, or a bare value).

    Tolerant on purpose, so a malformed but readable input doesn't fail the agent on formatting alone.
    """
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    pairs: dict[str, Any] = {}
    for chunk in re.split(r"[,;]\s*", raw):
        match = re.match(r"^\s*([\w ]+?)\s*[:=]\s*(.+)$", chunk)
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            pairs[key] = match.group(2).strip().strip("\"'")
    if pairs:
        return pairs
    return {"order_id": raw.strip("\"'")}


def _extract_order_id(arguments: dict[str, Any]) -> str | None:
    haystack = " ".join(str(value) for value in arguments.values())
    match = _ORDER_ID_PATTERN.search(haystack)
    return match.group(0).upper() if match else None


def _find_item(order: dict[str, Any], item_name: str) -> dict[str, Any] | None:
    items = order["items"]
    lowered = item_name.strip().lower()
    matched = [item for item in items if lowered and lowered in item["name"].lower()]
    if matched:
        return matched[0]
    if len(items) == 1:
        return items[0]
    return None


def _eligibility(order: dict[str, Any], item_name: str) -> tuple[bool, str]:
    item = _find_item(order, item_name)
    if item is None:
        names = ", ".join(i["name"] for i in order["items"])
        return False, f"No single item matched '{item_name}' — this order has: {names}."
    if order["status"] != "delivered":
        return False, f"Order has not been delivered yet (status: {order['status']})."
    if item["final_sale"]:
        return False, f"'{item['name']}' is a final-sale item and is not eligible for return."
    if order.get("returned"):
        return False, f"A return for '{item['name']}' has already been initiated on this order."
    days = order.get("days_since_delivery", 0)
    if days > RETURN_WINDOW_DAYS:
        return False, f"Delivered {days} days ago, past the {RETURN_WINDOW_DAYS}-day return window."
    return True, f"'{item['name']}' is eligible for return ({days} days since delivery)."


def _lookup_order(orders: dict[str, dict], arguments: dict[str, Any]) -> str:
    order_id = _extract_order_id(arguments)
    order = orders.get(order_id) if order_id else None
    if order is None:
        return f"No order found matching '{order_id or arguments}'."
    parts = [f"Order {order_id}: status={order['status']}."]
    if order.get("tracking"):
        parts.append(f"Tracking number: {order['tracking']}.")
    if order.get("eta"):
        parts.append(f"Estimated delivery: {order['eta']}.")
    items = ", ".join(item["name"] for item in order["items"])
    parts.append(f"Items: {items}.")
    return " ".join(parts)


def _check_return_policy(orders: dict[str, dict], arguments: dict[str, Any]) -> str:
    order_id = _extract_order_id(arguments)
    order = orders.get(order_id) if order_id else None
    if order is None:
        return f"No order found matching '{order_id or arguments}'."
    item_name = str(arguments.get("item_name", ""))
    eligible, reason = _eligibility(order, item_name)
    return f"Eligible for return: {reason}" if eligible else f"Not eligible for return: {reason}"


def _initiate_return(
    orders: dict[str, dict], lock: threading.Lock, arguments: dict[str, Any]
) -> str:
    order_id = _extract_order_id(arguments)
    order = orders.get(order_id) if order_id else None
    if order is None:
        return f"Return NOT started: no order found matching '{order_id or arguments}'."
    item_name = str(arguments.get("item_name", ""))
    with lock:
        eligible, reason = _eligibility(order, item_name)
        if not eligible:
            return f"Return NOT started: {reason}"
        order["returned"] = True
    item = _find_item(order, item_name)
    label = f"RTN-{order_id}-{abs(hash(item['name'])) % 10000:04d}"
    return f"Return started for '{item['name']}'. Shipping label reference: {label}."


def build_northwind_tool_backend() -> dict[str, Callable[[dict[str, Any]], str]]:
    """A fresh, independent copy of the fake order backend, so mutations (an
    initiated return) never leak between runs but persist across one run's tree."""
    orders = copy.deepcopy(_ORDER_SEED)
    lock = threading.Lock()
    return {
        "lookup_order": lambda arguments: _lookup_order(orders, arguments),
        "check_return_policy": lambda arguments: _check_return_policy(orders, arguments),
        "initiate_return": lambda arguments: _initiate_return(orders, lock, arguments),
    }
