import pytest
from src.success import MISSING, SuccessValidationError, normalize_success, should_summarize, validate_qa_output
from src.utils import QAAnswerFormat

YES = ["Yes", "YES", "yes", "True", "true", True, "1", 1, " YES ", "\tTrue\n"]
NO = ["No", "NO", "no", "False", "false", False, "0", 0, " NO ", "\tfalse\n"]


@pytest.mark.parametrize("raw,expected", [(v, "yes") for v in YES] + [(v, "no") for v in NO])
def test_normalization_and_raw_preservation(raw, expected):
    result = normalize_success(raw)
    assert result.require() == expected
    assert result.raw == raw and type(result.raw) is type(raw)
    assert result.fields()["success_raw"] == raw


@pytest.mark.parametrize("raw", [MISSING, None, "", " ", "maybe", "uncertain", "probably", "not sure", "probably yes", 2, -1, 1.0, 0.0, [], {}])
def test_unknown_fails_closed(raw):
    result = normalize_success(raw)
    assert result.normalized is None
    with pytest.raises(SuccessValidationError):
        result.require()


@pytest.mark.parametrize("raw", YES + NO)
def test_routing_and_schema_boundary(raw):
    payload = {"analysis": "a", "answer": "b", "success": raw, "rating": 8}
    result = validate_qa_output(payload, QAAnswerFormat)
    assert result["success"] == raw and type(result["success"]) is type(raw)
    assert payload == {"analysis": "a", "answer": "b", "success": raw, "rating": 8}
    assert should_summarize(["first", "second"], [result]) == (normalize_success(raw).normalized == "no")
    assert should_summarize(["first"], [result])


def test_missing_or_ambiguous_final_step_is_not_silently_complete():
    for payload in [{}, {"success": "maybe"}]:
        with pytest.raises(SuccessValidationError):
            should_summarize(["one"], [payload])
    assert not should_summarize(["one"], [])
    assert should_summarize([], [])


def test_wire_schema_unchanged():
    schema = QAAnswerFormat.model_json_schema()
    assert schema["properties"]["success"]["type"] == "string"
    assert not any(k.startswith("success_") for k in schema["properties"])
