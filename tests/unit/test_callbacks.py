import os


from agent.callbacks import SafeStdOutCallbackHandler


def test_safe_stdout_handles_none_serialized(capsys):
    handler = SafeStdOutCallbackHandler()
    handler.on_chain_start(None, {"input": "test"}, name="AgentExecutor")
    output = capsys.readouterr().out
    assert "Entering new AgentExecutor chain" in output


def test_safe_stdout_handles_serialized_dict(capsys):
    handler = SafeStdOutCallbackHandler()
    handler.on_chain_start({"name": "TestChain"}, {"input": "test"})
    output = capsys.readouterr().out
    assert "Entering new TestChain chain" in output
