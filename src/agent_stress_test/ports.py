"""The interfaces: LLMProvider, TargetAgent, Store."""

import threading
from abc import ABC, abstractmethod
from typing import Protocol

from agent_stress_test.models import (
    AgentResponse,
    Capabilities,
    Cluster,
    Message,
    Node,
    RegressionCase,
    Run,
    StressProfile,
    SystemPromptVersion,
    TokenUsage,
    ToolCall,
    Verdict,
)


class UsageMeter:
    """A live, thread-safe spend accumulator one ``LLMProvider`` instance
    writes into as a side effect of each real call (see
    ``LiteLLMProvider._record_usage``/``FakeLLMProvider.complete``) — never
    part of the abstract ``complete()``/``sample_n()`` contract itself.
    ``.total()`` returns an immutable ``TokenUsage`` snapshot, which is what
    ``Runner.run()`` reads once a run finishes and attaches to ``Run.usage``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._cost_usd = 0.0
        self._pricing_unavailable = False

    def record(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float | None,
    ) -> None:
        """``cost_usd=None`` means a real cost couldn't be computed for this
        call (see the caller) — recorded as "pricing unavailable", not
        silently folded in as free."""
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            if cost_usd is None:
                self._pricing_unavailable = True
            else:
                self._cost_usd += cost_usd

    def total(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                total_tokens=self._total_tokens,
                cost_usd=self._cost_usd,
                pricing_unavailable=self._pricing_unavailable,
            )


class ProviderError(Exception):
    """Base for errors an ``LLMProvider`` raises out of ``complete()``/
    ``sample_n()``/``complete_with_tools()`` — carries a human-readable
    message plus a ``retryable`` flag, so a caller (CLI, dashboard) can
    react sensibly without importing or inspecting a raw provider SDK
    exception (e.g. litellm's, which ``LiteLLMProvider`` translates into
    this hierarchy — see ``providers/litellm_provider.py``'s
    ``_classify_error``). ``retryable=False`` here since an unclassified
    provider error is treated as fatal by default; the specific subclasses
    below override it where retrying is actually sensible.
    """

    retryable: bool = False


class ProviderAuthError(ProviderError):
    """Bad/missing API key, or the key lacks access to the model. Fatal —
    retrying the exact same request can't succeed."""


class ProviderRateLimitError(ProviderError):
    """The provider is throttling this key/model. Retryable after a pause."""

    retryable = True


class ProviderTimeoutError(ProviderError):
    """The request didn't complete in time. Retryable."""

    retryable = True


class ProviderConnectionError(ProviderError):
    """Couldn't reach the provider (network, DNS, or the provider's own
    infrastructure being down). Retryable."""

    retryable = True


class LLMProvider(ABC):
    """A source of LLM completions. Real impl wraps litellm; fake impl is deterministic."""

    def __init__(self) -> None:
        self.meter = UsageMeter()

    @abstractmethod
    def complete(self, messages: list[Message]) -> str: ...

    @abstractmethod
    def sample_n(self, messages: list[Message], n: int) -> list[str]: ...


class ToolCallingLLM(Protocol):
    """The native tool-calling capability ``ProviderAgent`` needs from its
    backing provider — narrower than the full ``LLMProvider`` port, since
    only a provider that supports real tool-calling (today: ``LiteLLMProvider``)
    can offer it. A structural ``Protocol`` rather than an ABC so a duck-typed
    test double (see ``tests/test_targets.py``'s ``_FakeToolCallingProvider``)
    satisfies it without inheriting anything.
    """

    def complete_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> tuple[str, list[ToolCall]]: ...


class TargetAgent(ABC):
    """The agent under test."""

    @abstractmethod
    def respond(self, conversation: list[Message]) -> AgentResponse: ...

    def capabilities(self) -> Capabilities:
        """What this adapter actually supports. Defaults to the safest
        possible claim — nothing beyond plain stateless ``respond()`` — so an
        adapter that never overrides this can't accidentally overclaim;
        adapters that genuinely support more (e.g. real tool-calling)
        override it."""
        return Capabilities()


class Embedder(ABC):
    """Turns text into vectors for failure clustering.

    A deterministic offline implementation (hashing) backs the tests and the
    default; a semantic implementation (provider or local model) can slot in
    behind the same interface without changing the clusterer.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Store(ABC):
    """Persist and reload runs, nodes, verdicts, clusters."""

    @abstractmethod
    def save_run(self, run: Run) -> None: ...

    @abstractmethod
    def get_run(self, run_id: str) -> Run | None: ...

    @abstractmethod
    def list_runs(self, limit: int = 20) -> list[Run]: ...

    @abstractmethod
    def save_node(self, node: Node) -> None: ...

    @abstractmethod
    def get_nodes(self, run_id: str) -> list[Node]: ...

    @abstractmethod
    def save_verdict(self, verdict: Verdict) -> None: ...

    @abstractmethod
    def get_verdicts(self, run_id: str) -> list[Verdict]: ...

    @abstractmethod
    def save_cluster(self, cluster: Cluster) -> None: ...

    @abstractmethod
    def get_clusters(self, run_id: str) -> list[Cluster]: ...

    @abstractmethod
    def save_regression_case(self, case: RegressionCase) -> None: ...

    @abstractmethod
    def get_regression_case(self, case_id: str) -> RegressionCase | None: ...

    @abstractmethod
    def get_regression_cases(self, agent_spec_name: str) -> list[RegressionCase]: ...

    @abstractmethod
    def save_system_prompt_version(self, version: SystemPromptVersion) -> None: ...

    @abstractmethod
    def get_system_prompt_versions(self, agent_spec_name: str) -> list[SystemPromptVersion]: ...

    @abstractmethod
    def save_stress_profile(self, profile: StressProfile) -> None: ...

    @abstractmethod
    def get_stress_profile(self, agent_spec_name: str) -> StressProfile | None: ...
