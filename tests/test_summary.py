from agent_stress_test.providers.fake import FakeLLMProvider
from agent_stress_test.reasoning.summary import RunSummarizer


def test_run_summarizer_calls_the_llm_with_the_deterministic_text():
    llm = FakeLLMProvider()
    summarizer = RunSummarizer(llm)

    result = summarizer.summarize("This run scored 72% reliability.")

    assert result == "fake-reply: This run scored 72% reliability."
    assert llm.calls[-1][-1].content == "This run scored 72% reliability."


def test_run_summarizer_is_opt_in_only_called_when_asked():
    """No RunSummarizer call happens on its own -- this is really just
    documenting the contract (construct-and-call, never auto-invoked from
    deterministic_summary/executive_summary_context), guarded for real by
    the dashboard route only wiring it behind an explicit POST."""
    llm = FakeLLMProvider()
    RunSummarizer(llm)
    assert llm.calls == []
