"""The interfaces: LLMProvider, TargetAgent, Store."""

from abc import ABC, abstractmethod

from agent_stress_test.models import AgentResponse, Cluster, Message, Node, Run, Verdict


class LLMProvider(ABC):
    """A source of LLM completions. Real impl wraps litellm; fake impl is deterministic."""

    @abstractmethod
    def complete(self, messages: list[Message]) -> str: ...

    @abstractmethod
    def sample_n(self, messages: list[Message], n: int) -> list[str]: ...


class TargetAgent(ABC):
    """The agent under test."""

    @abstractmethod
    def respond(self, conversation: list[Message]) -> AgentResponse: ...


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
