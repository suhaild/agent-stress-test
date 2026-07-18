"""The ``Judge`` interface, and the confidence proxy every tier shares."""

from abc import ABC, abstractmethod

from deepeval.metrics.base_metric import BaseMetric

from agent_stress_test.models import AgentResponse, Message, Verdict


class Judge(ABC):
    """A failure judge (Strategy). Tier 1 is deterministic; tier 2 is an LLM.

    ``conversation`` is the messages leading up to (and including) the user
    probe this ``response`` answered — optional so a caller with no context
    handy can still judge (rule/tier-2 judging only reads the reply), but the
    Phase-C tool/task metrics use it as their ``input`` (the request a tool
    call's arguments are judged against). Judges that don't need it ignore it.
    """

    @abstractmethod
    def judge(
        self,
        response: AgentResponse,
        *,
        run_id: str,
        node_id: str,
        conversation: list[Message] | None = None,
    ) -> list[Verdict]: ...


def _confidence_from_score(metric: BaseMetric) -> float:
    """A rough confidence proxy: distance from the pass/fail threshold, scaled
    to [0, 1] — 0 at the threshold (maximally ambiguous), 1 at either extreme.
    GEval's score is a compliance degree, not a confidence; real calibration
    of this mapping is deferred (see prompt C3)."""
    return max(0.0, min(1.0, 2 * abs(metric.score - metric.threshold)))
