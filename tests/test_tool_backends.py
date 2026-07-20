from agent_stress_test.targets.tool_backends import build_northwind_tool_backend, parse_action_input


# --- parse_action_input ----------------------------------------------------


def test_parse_action_input_reads_json_object():
    assert parse_action_input('{"order_id": "NW-1001", "item_name": "Tent"}') == {
        "order_id": "NW-1001",
        "item_name": "Tent",
    }


def test_parse_action_input_reads_key_value_pairs():
    assert parse_action_input("order_id: NW-1002, item_name: Jacket") == {
        "order_id": "NW-1002",
        "item_name": "Jacket",
    }


def test_parse_action_input_falls_back_to_bare_order_id():
    assert parse_action_input("order NW-1003") == {"order_id": "order NW-1003"}


def test_parse_action_input_on_empty_string_returns_empty_dict():
    assert parse_action_input("") == {}
    assert parse_action_input("   ") == {}


# --- lookup_order -----------------------------------------------------------


def test_lookup_order_returns_real_grounded_data():
    backend = build_northwind_tool_backend()
    result = backend["lookup_order"]({"order_id": "NW-1001"})
    assert "1Z999AA10123456789" in result
    assert "delivered" in result


def test_lookup_order_tolerates_loose_argument_text():
    # A model that writes "order NW-1001" instead of a clean {"order_id": ...}
    # should still resolve — the backend extracts the order id pattern from
    # whatever text it's handed, not just a specific key.
    backend = build_northwind_tool_backend()
    result = backend["lookup_order"](parse_action_input("please check order NW-1001 for me"))
    assert "1Z999AA10123456789" in result


def test_lookup_order_unknown_id_says_so():
    backend = build_northwind_tool_backend()
    result = backend["lookup_order"]({"order_id": "NW-9999"})
    assert "No order found" in result


# --- check_return_policy -----------------------------------------------------


def test_check_return_policy_eligible_item():
    backend = build_northwind_tool_backend()
    result = backend["check_return_policy"](
        {"order_id": "NW-1001", "item_name": "Trailblazer 2-Person Tent"}
    )
    assert result.startswith("Eligible for return")


def test_check_return_policy_past_return_window():
    backend = build_northwind_tool_backend()
    result = backend["check_return_policy"](
        {"order_id": "NW-1003", "item_name": "Summit Hiking Boots"}
    )
    assert "Not eligible" in result
    assert "30-day" in result


def test_check_return_policy_final_sale_item_never_eligible():
    backend = build_northwind_tool_backend()
    result = backend["check_return_policy"](
        {"order_id": "NW-1006", "item_name": "Clearance Camp Stove"}
    )
    assert "Not eligible" in result
    assert "final-sale" in result


def test_check_return_policy_disambiguates_between_two_items_on_one_order():
    backend = build_northwind_tool_backend()
    eligible = backend["check_return_policy"](
        {"order_id": "NW-1006", "item_name": "Alpine Soft-Shell Jacket"}
    )
    not_eligible = backend["check_return_policy"](
        {"order_id": "NW-1006", "item_name": "Clearance Camp Stove"}
    )
    # Both items are on the SAME already-delivered order, so a bug that
    # inspected only order-level facts (rather than the named item) would
    # make both replies identical -- they must differ on the item's own
    # final_sale flag.
    assert eligible.startswith("Eligible for return")
    assert "final-sale" in not_eligible


def test_check_return_policy_not_yet_delivered():
    backend = build_northwind_tool_backend()
    result = backend["check_return_policy"](
        {"order_id": "NW-1002", "item_name": "Alpine Soft-Shell Jacket"}
    )
    assert "Not eligible" in result
    assert "not been delivered yet" in result


# --- initiate_return ---------------------------------------------------------


def test_initiate_return_succeeds_for_an_eligible_item():
    backend = build_northwind_tool_backend()
    result = backend["initiate_return"](
        {"order_id": "NW-1001", "item_name": "Trailblazer 2-Person Tent"}
    )
    assert result.startswith("Return started")
    assert "RTN-NW-1001-" in result


def test_initiate_return_refuses_a_final_sale_item():
    backend = build_northwind_tool_backend()
    result = backend["initiate_return"](
        {"order_id": "NW-1006", "item_name": "Clearance Camp Stove"}
    )
    assert result.startswith("Return NOT started")
    assert "final-sale" in result


def test_initiate_return_refuses_an_already_returned_item():
    backend = build_northwind_tool_backend()
    result = backend["initiate_return"](
        {"order_id": "NW-1005", "item_name": "Storm-Guard Rain Shell"}
    )
    assert result.startswith("Return NOT started")
    assert "already been initiated" in result


def test_initiate_return_is_idempotent_the_second_time():
    backend = build_northwind_tool_backend()
    first = backend["initiate_return"](
        {"order_id": "NW-1001", "item_name": "Trailblazer 2-Person Tent"}
    )
    second = backend["initiate_return"](
        {"order_id": "NW-1001", "item_name": "Trailblazer 2-Person Tent"}
    )
    assert first.startswith("Return started")
    assert second.startswith("Return NOT started")
    assert "already been initiated" in second


def test_build_northwind_tool_backend_returns_independent_state_per_call():
    # Regression guard: an earlier draft shared one module-level dict, so a
    # mutation (an initiated return) from one backend instance would leak
    # into every other instance/run.
    backend_a = build_northwind_tool_backend()
    backend_b = build_northwind_tool_backend()

    backend_a["initiate_return"]({"order_id": "NW-1001", "item_name": "Trailblazer 2-Person Tent"})
    still_eligible = backend_b["check_return_policy"](
        {"order_id": "NW-1001", "item_name": "Trailblazer 2-Person Tent"}
    )
    assert still_eligible.startswith("Eligible for return")
