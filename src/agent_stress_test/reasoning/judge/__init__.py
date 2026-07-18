"""Two-tier failure judge.

Tier 1 is deterministic: a set of typed rule checks (Strategy pattern) derived
from an AgentSpec, each producing a Verdict with a stable rule_id, a
human-readable reason, and tier="rules". Tier 2 is an LLM-as-judge: when tier 1
does not fire, a GEval metric per rule (DeepEval's `GEval`, scored by an
LLMProvider via `LLMProviderAsDeepEvalLLM`) evaluates the reply against the
rule's own text and returns a reason, a rough confidence, and a severity. Both
tiers plug into the same `Judge` interface, and `TwoTierJudge` runs tier 1
first — rules decide when they fire, the LLM is consulted only when they
don't.

Split across this package by responsibility, re-exported here so
``from agent_stress_test.reasoning.judge import X`` keeps working unchanged:
``base`` (the ``Judge`` interface), ``rules`` (tier 1 + deflection detection),
``llm`` (tier 2 + the two-tier composition), ``tool_task`` (Phase C node-level
tool/task metrics + ``CompositeJudge``), ``conversation`` (Phase C2
whole-conversation metrics).
"""

from agent_stress_test.reasoning.judge.base import Judge
from agent_stress_test.reasoning.judge.conversation import (
    ConversationCompletenessJudge,
    ConversationJudge,
    ConversationMetricJudge,
    ConversationRuleJudge,
    KnowledgeRetentionJudge,
    RoleAdherenceJudge,
    TurnRelevancyJudge,
    build_conversation_judge,
)
from agent_stress_test.reasoning.judge.llm import LLMJudge, TwoTierJudge, build_two_tier_judge
from agent_stress_test.reasoning.judge.rules import (
    BannedToolUseCheck,
    CheckResult,
    DETERMINISTIC_CONFIDENCE,
    ForbiddenOutputCheck,
    FormatViolationCheck,
    RequiredDisclaimerCheck,
    RuleCheck,
    RulesJudge,
    UngroundedClaimCheck,
    build_checks,
    is_deflection,
)
from agent_stress_test.reasoning.judge.tool_task import (
    CompositeJudge,
    TaskCompletionJudge,
    ToolArgumentJudge,
)

__all__ = [
    "Judge",
    "RuleCheck",
    "CheckResult",
    "DETERMINISTIC_CONFIDENCE",
    "BannedToolUseCheck",
    "ForbiddenOutputCheck",
    "RequiredDisclaimerCheck",
    "FormatViolationCheck",
    "UngroundedClaimCheck",
    "is_deflection",
    "RulesJudge",
    "build_checks",
    "LLMJudge",
    "TwoTierJudge",
    "build_two_tier_judge",
    "ToolArgumentJudge",
    "TaskCompletionJudge",
    "CompositeJudge",
    "ConversationMetricJudge",
    "RoleAdherenceJudge",
    "KnowledgeRetentionJudge",
    "ConversationCompletenessJudge",
    "TurnRelevancyJudge",
    "ConversationRuleJudge",
    "ConversationJudge",
    "build_conversation_judge",
]
