from agent_stress_test.models import Step
from agent_stress_test.targets.react_parsing import parse_react_completion, parse_react_step

# --- parse_react_completion (relocated from test_targets.py's SampleAgent
# tests unchanged; re-asserted here directly against the shared module so a
# future edit to this parser is caught at the source, not only indirectly
# through SampleAgent's behavior). ------------------------------------------


def test_parse_react_completion_with_no_labels_is_a_plain_reply():
    response = parse_react_completion("Just a plain reply, no labels at all.")
    assert response.final_reply == "Just a plain reply, no labels at all."
    assert response.trace is None


def test_parse_react_completion_captures_trace_and_final_answer():
    text = (
        "Thought: check the order.\n"
        "Action: lookup_order\n"
        "Action Input: 12345\n"
        "Observation: shipped\n"
        "Final Answer: It shipped."
    )
    response = parse_react_completion(text)
    assert response.final_reply == "It shipped."
    assert response.trace == [
        Step(
            thought="check the order.",
            action="lookup_order",
            action_input="12345",
            observation="shipped",
        )
    ]


# --- parse_react_step -------------------------------------------------------


def test_parse_react_step_returns_a_step_with_no_final_answer():
    step, final = parse_react_step(
        'Thought: look it up.\nAction: lookup_order\nAction Input: {"order_id": "NW-1001"}'
    )
    assert final is None
    assert step == Step(
        thought="look it up.", action="lookup_order", action_input='{"order_id": "NW-1001"}'
    )


def test_parse_react_step_recognizes_final_answer_and_multi_paragraph_reply():
    text = "Final Answer:\nLine one.\n\nLine two."
    step, final = parse_react_step(text)
    assert step is None
    assert final == "Line one.\n\nLine two."


def test_parse_react_step_keeps_a_trailing_thought_alongside_the_final_answer():
    text = "Thought: I now know enough.\nFinal Answer: All set."
    step, final = parse_react_step(text)
    assert step == Step(thought="I now know enough.")
    assert final == "All set."


def test_parse_react_step_tolerates_markdown_bold_final_answer_label():
    step, final = parse_react_step("**Final Answer:** All set.")
    assert step is None
    assert final == "All set."


def test_parse_react_step_falls_back_to_plain_text_when_unlabeled():
    step, final = parse_react_step("no labels here whatsoever")
    assert step is None
    assert final == "no labels here whatsoever"


def test_parse_react_step_second_thought_closes_out_the_step():
    # A model that narrates two Thoughts with no Action or Final Answer in
    # between should still terminate this loop iteration with a step (not
    # hang waiting for a label that never comes), so the caller can nudge it.
    text = "Thought: first idea.\nThought: second idea, no action yet."
    step, final = parse_react_step(text)
    assert final is None
    assert step == Step(thought="first idea.")


def test_parse_react_step_ignores_a_self_narrated_observation():
    # The whole point of the advanced agent's per-step parsing: even if the
    # model disobeys the "wait for a real Observation" instruction and
    # narrates its own, the caller only reads step.action/.action_input to
    # decide what to execute -- the fabricated observation is captured but
    # never trusted as ground truth.
    text = "Action: lookup_order\nAction Input: NW-1001\nObservation: shipped yesterday (I made this up)"
    step, final = parse_react_step(text)
    assert final is None
    assert step.action == "lookup_order"
    assert step.observation == "shipped yesterday (I made this up)"
