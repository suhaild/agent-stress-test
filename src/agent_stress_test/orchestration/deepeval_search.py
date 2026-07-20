"""Default search strategy: runs one full DeepEval-simulated conversation per
persona against the target, then ingests the resulting turns into the tree
as a linear chain, judged (and optionally consistency-scored) per node.

Each persona starts its own fresh conversation, so there's no single shared
root turn across personas (unlike ``GreedyBestFirstSearch`` in ``search.py``).
An optional ``ConversationJudge`` can also score each persona's whole chain
once ingested, attaching its verdicts to the chain's leaf node.
"""

from agent_stress_test.models import AgentResponse, Message, Node, Verdict
from agent_stress_test.orchestration.search import SearchResult, SearchStrategy, score_and_judge
from agent_stress_test.orchestration.tree import ConversationTree
from agent_stress_test.ports import LLMProvider, TargetAgent
from agent_stress_test.reasoning.consistency import ConsistencyScorer
from agent_stress_test.reasoning.deepeval_simulator import (
    PERSONAS,
    from_deepeval_tool_call,
    simulate_personas,
)
from agent_stress_test.reasoning.judge import ConversationJudge, Judge


class DeepEvalConversationSearch(SearchStrategy):
    """Runs one full simulated conversation per persona and ingests each
    into the tree.

    ``budget`` means turns per persona conversation (DeepEval's
    ``max_user_simulations``), not a branch-expansion count, despite
    sharing ``SearchStrategy.search()``'s signature with
    ``GreedyBestFirstSearch``. ``seed_messages`` is accepted but unused —
    each persona supplies its own opening line. ``Node.tactic`` holds the
    persona name for every node this strategy creates.
    """

    def __init__(
        self,
        target: TargetAgent,
        sim_provider: LLMProvider,
        judge: Judge,
        scorer: ConsistencyScorer | None = None,
        *,
        personas: list[str] | None = None,
        sample_n: int = 3,
        extra_personas: dict[str, object] | None = None,
        conversation_judge: ConversationJudge | None = None,
    ) -> None:
        """``extra_personas`` merges additional name -> ConversationalGolden
        entries beyond the bundled ``PERSONAS`` dict (e.g. an agent's own
        approved ``StressProfile`` personas). Kept loosely typed (not
        ``ConversationalGolden``) so this module never needs to import
        ``deepeval`` itself.

        ``conversation_judge``, when given, scores each persona's whole
        conversation once ingested, separately from the per-node ``judge``.
        """
        self._target = target
        self._sim_provider = sim_provider
        self._judge = judge
        self._scorer = scorer
        self._extra_personas = dict(extra_personas) if extra_personas else {}
        self._personas = personas if personas is not None else [*PERSONAS, *self._extra_personas]
        self._sample_n = sample_n
        self._conversation_judge = conversation_judge

    def search(
        self, tree: ConversationTree, seed_messages: list[Message], *, budget: int
    ) -> SearchResult:
        failures: list[Verdict] = []
        nodes_created = 0

        personas_map = {**PERSONAS, **self._extra_personas}
        test_cases = simulate_personas(
            target=self._target,
            sim_provider=self._sim_provider,
            persona_names=self._personas,
            max_user_simulations=budget,
            personas=personas_map,
        )
        for persona_name, test_case in zip(self._personas, test_cases):
            golden = personas_map.get(persona_name)
            nodes_created += self._ingest(tree, persona_name, test_case.turns, failures, golden)

        return SearchResult(
            expansions=len(self._personas), nodes_created=nodes_created, failures=failures
        )

    def _ingest(
        self,
        tree: ConversationTree,
        persona: str,
        turns,
        failures: list[Verdict],
        golden: object | None = None,
    ) -> int:
        """Convert one persona's flat, alternating user/assistant turns into
        a linear chain of judged nodes, then (if a conversation-level judge
        is wired) judge the whole chain and attach verdicts to its leaf."""
        conversation: list[Message] = []
        parent_id: str | None = None
        created = 0

        for turn in turns:
            conversation.append(Message(role=turn.role, content=turn.content))
            if turn.role != "assistant":
                continue

            tool_calls = [from_deepeval_tool_call(tc) for tc in (turn.tools_called or [])]
            node = Node(
                run_id=tree.run_id,
                parent_id=parent_id,
                messages=list(conversation[:-1]),
                target_reply=turn.content,
                tactic=persona,
                tool_calls=tool_calls,
            )
            response = AgentResponse(final_reply=turn.content, tool_calls=tool_calls)
            verdicts = score_and_judge(
                node,
                response,
                run_id=tree.run_id,
                judge=self._judge,
                scorer=self._scorer,
                sample_n=self._sample_n,
            )

            tree.add(node)
            tree.attach_verdicts(node.id, verdicts)
            failures.extend(v for v in verdicts if not v.passed)
            parent_id = node.id
            created += 1

        if self._conversation_judge is not None and parent_id is not None:
            conversation_verdicts = self._conversation_judge.judge_conversation(
                conversation,
                run_id=tree.run_id,
                node_id=parent_id,
                scenario=getattr(golden, "scenario", None),
                user_description=getattr(golden, "user_description", None),
            )
            tree.attach_verdicts(parent_id, conversation_verdicts)
            failures.extend(v for v in conversation_verdicts if not v.passed)

        return created
