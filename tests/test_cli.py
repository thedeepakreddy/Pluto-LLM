from llm import cli


def test_no_command_prints_usage_and_fails(capsys):
    assert cli.main([]) == 2
    assert "usage" in capsys.readouterr().err.lower()
