"""Phase C3, part 1 — live per-metric cost measurement.

NOT part of the pytest suite (see CLAUDE.md's Testing Policy: live smoke
tests are optional, off by default, and never required to run the main
suite offline). Run by hand, once, whenever the Phase-C metric stack or the
cheap model backing it changes:

    python scripts/measure_metric_costs.py

Requires a real ANTHROPIC_API_KEY (loaded from .env via
``config.load_settings``) — every call below is a genuine, billed request to
``composition.DEFAULT_SIM_MODEL`` (the same cheap Haiku model
``build_runner``'s callers already default the adversarial simulator to, and
now the Phase-C metrics to as well — see runner.py's ``sim_provider`` reuse).

Builds one fresh ``LiteLLMProvider`` per metric (each wrapped to also count
calls, since ``UsageMeter`` tracks tokens/cost but not call count), runs ONE
realistic conversation through every Phase-C metric judge, and prints the
real prompt/completion/total tokens, cost, and call count each one made —
the numbers ``build_runner``'s on-by-default vs opt-in defaults are based on.
See ``orchestration/runner.py``'s ``build_runner`` docstring for the
recorded findings from the last real run of this script and the resulting
on/opt-in decision per metric.
"""

import json

from deepeval.test_case import ConversationalTestCase, Turn

from agent_stress_test.composition import DEFAULT_SIM_MODEL
from agent_stress_test.config import load_agent_spec, load_settings
from agent_stress_test.models import AgentResponse, Message
from agent_stress_test.ports import LLMProvider
from agent_stress_test.providers.litellm_provider import LiteLLMProvider
from agent_stress_test.reasoning.judge import (
    ConversationCompletenessJudge,
    ConversationRuleJudge,
    KnowledgeRetentionJudge,
    RoleAdherenceJudge,
    TaskCompletionJudge,
    ToolArgumentJudge,
    TurnRelevancyJudge,
)
from agent_stress_test.targets.tool_calling_verification_agent import (
    tool_calling_verification_agent,
)

_SPEC_PATH = "config/agents/sample_support_advanced.yaml"

_CONVERSATION = [
    Message(role="user", content="Hi, I ordered a jacket last week and it still hasn't shipped."),
    Message(
        role="assistant",
        content="I'm sorry about the delay! Could you share your order ID so I can check?",
    ),
    Message(role="user", content="It's ORD-88213."),
    Message(
        role="assistant",
        content=(
            "Thanks — I found it. Your jacket is still in the warehouse and is "
            "expected to ship tomorrow."
        ),
    ),
]
_TOOL_QUERY = [Message(role="user", content="Where is order 12345?")]


class _CountingProvider(LLMProvider):
    """Wraps a real ``LLMProvider`` to also count calls — measurement-only,
    not a production adapter. ``UsageMeter`` (the A5 meter) already tracks
    tokens/cost as a side effect of the wrapped provider's own calls; this
    just adds the "how many calls" dimension this script needs, which
    ``TokenUsage`` doesn't carry."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.meter = inner.meter  # share the real meter, don't shadow it
        self.calls = 0

    def complete(self, messages: list[Message]) -> str:
        self.calls += 1
        return self._inner.complete(messages)

    def sample_n(self, messages: list[Message], n: int) -> list[str]:
        raise NotImplementedError("not needed for this measurement")


def _cheap_provider() -> _CountingProvider:
    return _CountingProvider(LiteLLMProvider(model=DEFAULT_SIM_MODEL))


def _report(name: str, provider: _CountingProvider) -> dict:
    usage = provider.meter.total()
    row = {
        "metric": name,
        "calls": provider.calls,
        "total_tokens": usage.total_tokens,
        "cost_usd": usage.cost_usd,
    }
    print(
        f"{name:34s} calls={row['calls']:2d}  tokens={row['total_tokens']:5d}  "
        f"cost=${row['cost_usd']:.6f}"
    )
    return row


def main() -> None:
    load_settings()
    agent_spec = load_agent_spec(_SPEC_PATH)
    print(f"Measuring Phase-C metric costs on {DEFAULT_SIM_MODEL} ...\n")
    results = []

    provider = _cheap_provider()
    ToolArgumentJudge(provider).judge(
        tool_calling_verification_agent(_TOOL_QUERY),
        run_id="measure",
        node_id="n",
        conversation=_TOOL_QUERY,
    )
    results.append(_report("tool_argument_correctness", provider))

    provider = _cheap_provider()
    TaskCompletionJudge(provider).judge(
        AgentResponse(final_reply=_CONVERSATION[-1].content),
        run_id="measure",
        node_id="n",
        conversation=_CONVERSATION[:-1],
    )
    results.append(_report("task_completion", provider))

    test_case = ConversationalTestCase(
        turns=[Turn(role=m.role, content=m.content) for m in _CONVERSATION],
        chatbot_role=agent_spec.purpose or agent_spec.system_prompt,
    )
    conversation_judges = {
        "role_adherence": RoleAdherenceJudge,
        "knowledge_retention": KnowledgeRetentionJudge,
        "conversation_completeness": ConversationCompletenessJudge,
        "turn_relevancy": TurnRelevancyJudge,
    }
    for name, judge_cls in conversation_judges.items():
        provider = _cheap_provider()
        judge_cls(provider).judge_conversation(test_case, run_id="measure", node_id="n")
        results.append(_report(name, provider))

    provider = _cheap_provider()
    ConversationRuleJudge(provider, agent_spec).judge_conversation(
        test_case, run_id="measure", node_id="n"
    )
    results.append(_report(f"conversation_rule_geval (x{len(agent_spec.rules)} rules)", provider))

    total_cost = sum(r["cost_usd"] for r in results)
    total_calls = sum(r["calls"] for r in results)
    print(f"\nTotal: {total_calls} calls, ${total_cost:.6f} for one full metric pass.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
