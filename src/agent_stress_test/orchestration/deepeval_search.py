"""DeepEval-conversation-driven search strategy — the new default engine.

Replaces the old per-node judge-driven tactic branching
(``GreedyBestFirstSearch``, see ``search.py``, kept fully intact for direct
use/testing) with running one full DeepEval-simulated conversation per
persona (a ``ConversationalGolden`` — see
``reasoning/deepeval_simulator.py``) against the target, then ingesting the
resulting turns into the tree as a linear chain, judged (and, optionally,
consistency-scored) exactly the way ``GreedyBestFirstSearch`` already does
per node. Each persona is its own root-to-leaf branch: DeepEval's
``ConversationSimulator`` always starts a conversation fresh from its own
persona's opening line, so — unlike the old strategy — there's no single
shared root turn for personas to branch off of.

Optionally (Phase C2), an injected ``ConversationJudge`` also scores each
persona's WHOLE chain once it's fully ingested — a path-level judgment
alongside the per-node one — with its verdicts attached to the chain's leaf
node (see ``_ingest``).
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

    ``budget`` here means how many user turns each persona's conversation
    runs for (DeepEval's ``max_user_simulations``), not a branch-expansion
    count — a different unit than ``GreedyBestFirstSearch``'s ``budget``,
    despite sharing the same ``SearchStrategy.search()`` signature.
    ``seed_messages`` is accepted (required by that shared interface) but
    unused: each persona supplies its own opening line via its own scenario,
    there's no fixed seed turn to start from.

    Every node this strategy creates carries its persona name in
    ``Node.tactic`` (the field predates this engine, back when a node's
    branch was always a bundled tactic — see ``search.py``'s
    ``GreedyBestFirstSearch``). The bundled personas happen to reuse the same
    5 names as the tactic registry, but a profile-sourced persona (see
    ``extra_personas`` below) is a genuinely distinct name with no
    corresponding tactic — either way, ``Node.tactic`` here always holds
    whichever persona produced that node.
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
        """``extra_personas`` merges in additional name -> ConversationalGolden
        entries beyond the bundled ``PERSONAS`` dict (e.g. an agent's own
        approved ``StressProfile`` personas — see ``runner.py``'s
        ``build_runner()``, which converts them via
        ``reasoning/profiler.py``'s ``to_conversational_golden``). Kept
        loosely typed here (not ``ConversationalGolden``) so this module
        never needs to import ``deepeval`` itself — it only ever holds these
        values opaquely and hands them to ``simulate_personas``, exactly like
        it already does with the imported ``PERSONAS`` dict.

        ``conversation_judge`` (Phase C2), when given, scores each persona's
        WHOLE conversation once it's fully ingested — a different unit of
        judgment than ``judge``, which scores each node/turn as it's created.
        Its verdicts attach to the conversation's leaf node (see ``_ingest``).
        Left ``None`` (the default), no conversation-level judging happens.
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
        """Convert one persona's flat, alternating user/assistant Turn list
        into a linear chain of judged Nodes committed onto the tree, then (if
        a conversation-level judge is wired) judge the WHOLE chain once and
        attach those verdicts to its leaf node — the chain's last node, whose
        ``tree.path_to_root()`` reconstructs exactly the conversation judged."""
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
