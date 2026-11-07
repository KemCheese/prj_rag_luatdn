"""Offline tests for the portable single-planner diagnostic."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hashlib
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIError
import pytest

from scripts import diagnose_query_plan as diagnostic
from ragdn.query_analysis import validate_plan
from ragdn.query_analysis import _subject_keys
from ragdn.query_models import PlannerOutput

LOAD_ENVIRONMENT = diagnostic._load_environment

QUESTION = "Điều 58 và Điều 59 Luật Doanh nghiệp quy định gì?"


def proposal(question=QUESTION, *, quote=None, references=(), unprocessed=()):
    return PlannerOutput.model_validate({
        "facts": [{"fact_id": "f1", "quote": question if quote is None else quote}],
        "issues": [{"issue_id": "i1", "title": "Nội dung", "fact_ids": ["f1"],
                    "main_query": "nội dung quy định", "backup_query": "quy định được hỏi"}],
        "reference_quotes": list(references), "unprocessed_parts": list(unprocessed),
    })


def article_proposal():
    return PlannerOutput.model_validate({
        "facts": [{"fact_id": f"f{number}", "quote": f"Điều {number}"} for number in (58, 59)],
        "reference_quotes": ["Điều 58", "Điều 59"],
        "issues": [{"issue_id": f"i{number}", "title": f"Nội dung Điều {number}", "fact_ids": [f"f{number}"],
                    "scope": {"doc_ids": ["luat_dn_2020"]},
                    "main_query": f"nội dung Điều {number} Luật Doanh nghiệp",
                    "backup_query": f"Điều {number} Luật Doanh nghiệp quy định gì"} for number in (58, 59)],
    })


class Client:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.options = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=self.result))])


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("RAG_QUERY_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LLM_MODEL", "configured-reader")
    monkeypatch.setattr(diagnostic, "_load_environment", lambda: None)


def run(output, **kwargs):
    client = Client(output.model_dump_json() if isinstance(output, PlannerOutput) else output)
    references = Mock(return_value=[])
    report, code = diagnostic.diagnose(QUESTION, context_tokens=32768, client=client, resolver=references, **kwargs)
    assert len(client.calls) == report["new_completion_calls"] == 1
    assert client.options[0]["max_retries"] == 0
    assert client.calls[0]["model"] == "configured-reader"
    assert client.calls[0]["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "PlannerOutput", "schema": PlannerOutput.model_json_schema()},
    }
    from ragdn.query_llm import llm
    assert client.calls[0]["temperature"] == llm.LLM_TEMPERATURE
    assert client.calls[0]["top_p"] == llm.LLM_TOP_P
    if llm.LLM_SEED:
        assert client.calls[0]["seed"] == int(llm.LLM_SEED)
    references.assert_called_once_with(QUESTION)
    return report, code, client


@pytest.mark.parametrize("quote,matches", [("Điều 60", 0), ("Điều", 2)])
def test_absent_and_repeated_quotes_share_reason_but_report_distinguishes_them(quote, matches):
    report, code, _ = run(proposal(quote=quote))
    assert code == 1 and report["status"] == "validation_failed"
    assert report["validation_reason"] == "fact_quote_not_unique"
    assert report["quote_checks"] == [{"category": "facts", "index": 0, "matches": matches, "quote_length": len(quote)}]


def test_reference_and_unprocessed_items_have_independent_categories_and_all_are_reported():
    report, code, _ = run(proposal(references=["Điều", "not-present"], unprocessed=[""]))
    assert code == 1
    assert [(row["category"], row["index"], row["matches"]) for row in report["quote_checks"]] == [
        ("facts", 0, 1), ("reference_quotes", 0, 2), ("reference_quotes", 1, 0), ("unprocessed_parts", 0, len(QUESTION) + 1)]


@pytest.mark.parametrize("category", ["reference_quotes", "unprocessed_parts"])
@pytest.mark.parametrize("quote,matches", [("Điều 60", 0), ("Điều", 2)])
def test_actual_validator_reaches_invalid_non_fact_quote_categories(category, quote, matches):
    output = proposal(references=[quote] if category == "reference_quotes" else [],
                      unprocessed=[quote] if category == "unprocessed_parts" else [])
    report, code, _ = run(output)
    assert code == 1 and report["validation_reason"] == "fact_quote_not_unique"
    assert report["quote_checks"][1]["category"] == category
    assert report["quote_checks"][1]["matches"] == matches


def test_overlap_counts_and_validation_of_identical_captured_proposal():
    original = "aaaa"
    captured = proposal(original, quote="aa")
    validator = Mock(wraps=validate_plan)

    def completion(system, user, *, budget, **kwargs):
        budget.begin_completion()
        return captured

    report, code = diagnostic.diagnose(original, context_tokens=32768, completion=completion, resolver=lambda _: [], validator=validator)
    assert report["quote_checks"][0]["matches"] == 3
    assert code == 1 and report["new_completion_calls"] == 1
    assert validator.call_args.args[1] is captured


def test_valid_plan_metadata_privacy_and_explicit_quote_opt_in(monkeypatch):
    # Pipeline and high-level analysis would initiate additional work; neither is used.
    import ragdn.query_analysis as analysis
    import ragdn.query_pipeline as pipeline
    monkeypatch.setattr(analysis, "analyze_question", Mock(side_effect=AssertionError("extra stage")))
    monkeypatch.setattr(pipeline, "run_query_pipeline", Mock(side_effect=AssertionError("retrieval")))
    report, code, client = run(proposal())
    assert code == 0 and report["status"] == "valid" and report["validation_reason"] is None
    assert report["question_sha256"] == hashlib.sha256(QUESTION.encode("utf-8")).hexdigest()
    assert report["question_length"] == len(QUESTION) and report["context_tokens"] == 32768
    assert report["context_method"] == "conservative_utf8_bytes"
    assert QUESTION not in json.dumps(report, ensure_ascii=False)
    assert "quote" not in report["quote_checks"][0]
    assert len(client.calls[0]["messages"]) == 2
    shown, _, _ = run(proposal(), show_quotes=True)
    assert shown["quote_checks"][0]["quote"] == QUESTION


@pytest.mark.parametrize("result,reason", [
    ("private invalid JSON", "invalid_json"),
    ('{"private-secret":"not-a-plan"}', "invalid_schema"),
    (APIError("private-upstream-body", request=httpx.Request("POST", "http://private.invalid"), body=None), "structured_service_error"),
])
def test_bad_json_schema_and_service_error_remain_safe_and_count_one_send(result, reason):
    report, code, _ = run(result)
    assert code == 1 and report["status"] == "completion_failed"
    assert report["error_reason"] == reason and report["validation_reason"] is None
    assert "private" not in json.dumps(report)


@pytest.mark.parametrize("field", ["main_query", "backup_query"])
@pytest.mark.parametrize("title,label", [("Luật Doanh nghiệp", "Điều"), ("Luat Doanh nghiep", "Dieu")])
def test_document_title_before_article_is_not_a_company_subject(field, title, label):
    output = article_proposal().model_dump()
    for number, issue in zip((58, 59), output["issues"]):
        issue[field] = f"{title} {label} {number}"
    client = Client(json.dumps(output, ensure_ascii=False))
    report, code = diagnostic.diagnose(QUESTION, context_tokens=32768, client=client)
    assert _subject_keys(output["issues"][0][field], []) == set()
    assert code == 0 and report["status"] == "valid"
    assert report["new_completion_calls"] == len(client.calls) == 1


@pytest.mark.parametrize("field", ["main_query", "backup_query"])
@pytest.mark.parametrize("subject", ["Công ty B", "Doanh nghiệp B"])
def test_document_title_skip_preserves_real_cross_issue_entity_guards(field, subject):
    question = "Công ty A họp năm 2020. Doanh nghiệp B họp năm 2025. Điều 58 Luật Doanh nghiệp quy định gì?"
    output = proposal(question).model_dump()
    output["facts"] = [{"fact_id": "f1", "quote": "Công ty A họp năm 2020."},
                       {"fact_id": "f2", "quote": "Doanh nghiệp B họp năm 2025."},
                       {"fact_id": "f3", "quote": "Điều 58 Luật Doanh nghiệp quy định gì?"}]
    output["issues"][0].update(fact_ids=["f1", "f3"],
                               scope={"subjects": ["Công ty A"], "doc_ids": ["luat_dn_2020"], "time_quote": "2020"})
    output["issues"][0][field] = f"Luật Doanh nghiệp Điều 58 về {subject} năm 2020"
    report, code = diagnostic.diagnose(question, context_tokens=32768, client=Client(json.dumps(output, ensure_ascii=False)))
    assert code == 1 and report["validation_reason"] == "query_subject_scope_drift"
    assert "company:b" in _subject_keys(output["issues"][0][field], [])


def test_show_plan_exposes_same_schema_valid_proposal_on_actual_subject_validation_failure():
    output = article_proposal().model_dump()
    output["issues"][0]["main_query"] = "Công ty Z quy định gì?"
    result = json.dumps(output, ensure_ascii=False)
    client = Client(result)
    default, code = diagnostic.diagnose(QUESTION, context_tokens=32768, client=client)
    assert code == 1 and default["validation_reason"] == "query_subject_scope_drift"
    assert default["parsed_plan_status"] == "schema_valid" and default["schema_errors"] == []
    assert "planner_json" not in default and "Công ty Z" not in json.dumps(default, ensure_ascii=False)
    shown, code = diagnostic.diagnose(QUESTION, context_tokens=32768, client=Client(result), show_plan=True)
    assert code == 1 and shown["planner_json"] == output
    assert shown["new_completion_calls"] == 1


def test_invalid_schema_reports_only_safe_locations_and_types_and_opt_in_captured_json():
    output = article_proposal().model_dump()
    del output["issues"][0]["backup_query"]
    output["issues"][0]["scope"]["company_types"] = "private-input-value"
    output["private-extra-field"] = "private-extra-value"
    response = json.dumps(output, ensure_ascii=False)
    report, code, _ = run(response)
    assert code == 1 and report["error_reason"] == "invalid_schema"
    assert report["parsed_plan_status"] == "schema_invalid"
    assert {"loc": ["issues", 0, "backup_query"], "type": "missing"} in report["schema_errors"]
    assert {"loc": ["issues", 0, "scope", "company_types"], "type": "list_type"} in report["schema_errors"]
    assert {"loc": ["<unknown_field>"], "type": "extra_forbidden"} in report["schema_errors"]
    assert all(set(error) == {"loc", "type"} for error in report["schema_errors"])
    assert "private" not in json.dumps(report) and "planner_json" not in report
    shown, code, _ = run(response, show_plan=True)
    assert code == 1 and shown["planner_json"] == output and shown["new_completion_calls"] == 1


@pytest.mark.parametrize("content", ["```json\n{}\n```", '{"facts":NaN}', "not JSON"])
def test_show_plan_does_not_invent_parsed_data_for_invalid_json(content):
    report, code, _ = run(content, show_plan=True)
    assert code == 1 and report["planner_json"] is None
    assert report["parsed_plan_status"] == "invalid_json" and report["schema_errors"] == []


def test_capture_forwards_to_the_with_options_client_without_extra_send():
    configured = Client(proposal().model_dump_json())
    base = SimpleNamespace(with_options=Mock(return_value=configured))
    report, code = diagnostic.diagnose(QUESTION, context_tokens=32768, client=base, resolver=lambda _: [], show_plan=True)
    assert code == 0 and len(configured.calls) == report["new_completion_calls"] == 1
    base.with_options.assert_called_once()
    assert base.with_options.call_args.kwargs["max_retries"] == 0
    assert configured.calls[0]["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "PlannerOutput", "schema": PlannerOutput.model_json_schema()},
    }
    assert report["planner_json"] == proposal().model_dump(mode="json")


def test_show_plan_never_exposes_upstream_error_body_when_no_response_was_returned():
    error = APIError("private-upstream-body", request=httpx.Request("POST", "http://private.invalid"), body={"private": "body"})
    report, code, _ = run(error, show_plan=True)
    assert code == 1 and report["error_reason"] == "structured_service_error"
    assert report["planner_json"] is None and report["parsed_plan_status"] == "response_unavailable"
    assert "private" not in json.dumps(report)


def test_unknown_context_sends_nothing_and_preserves_expected_reason():
    completion, resolver = Mock(), Mock()
    report, code = diagnostic.diagnose(QUESTION, completion=completion, resolver=resolver)
    assert code == 1 and report["error_reason"] == "context_budget_unknown"
    assert report["new_completion_calls"] == 0 and report["status"] == "preflight_failed"
    completion.assert_not_called()
    resolver.assert_not_called()


def test_output_collision_is_refused_before_environment_or_request(tmp_path, monkeypatch, capsys):
    output = tmp_path / "existing.json"
    output.write_text("preserved", encoding="utf-8")
    environment = Mock(side_effect=AssertionError("must not read config before collision refusal"))
    completion = Mock()
    monkeypatch.setattr(diagnostic, "_load_environment", environment)
    assert diagnostic.main(["--question", QUESTION, "--output", str(output)], completion=completion) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["error_reason"] == "output_exists" and report["new_completion_calls"] == 0
    assert output.read_text(encoding="utf-8") == "preserved"
    environment.assert_not_called()
    completion.assert_not_called()


def test_main_exclusive_utf8_output_matches_stdout_and_returns_validation_exit(tmp_path, capsys):
    output = tmp_path / "new-report.json"
    client = Client(proposal(quote="Điều").model_dump_json())
    code = diagnostic.main(["--question", QUESTION, "--context-tokens", "32768", "--show-quotes", "--output", str(output)],
                           client=client, resolver=lambda _: [])
    stdout = capsys.readouterr().out
    assert code == 1 and output.read_text(encoding="utf-8") == stdout
    assert "Điều" in stdout and "\\u0110" not in stdout
    assert json.loads(stdout)["new_completion_calls"] == 1 and len(client.calls) == 1


def test_main_show_plan_is_explicit_and_does_not_send_again(tmp_path, capsys):
    output = tmp_path / "captured-plan.json"
    client = Client(article_proposal().model_dump_json())
    code = diagnostic.main(["--question", QUESTION, "--context-tokens", "32768", "--show-plan", "--output", str(output)], client=client)
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and len(client.calls) == report["new_completion_calls"] == 1
    assert report["planner_json"] == article_proposal().model_dump(mode="json")
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_unexpected_programming_error_propagates():
    client = Client(ValueError("programmer defect"))
    with pytest.raises(ValueError, match="programmer defect"):
        diagnostic.diagnose(QUESTION, context_tokens=32768, client=client, resolver=lambda _: [])
    assert len(client.calls) == 1


def test_copy_only_script_into_other_checkout_bootstraps_its_src_from_other_cwd(tmp_path):
    other_root = tmp_path / "other checkout"
    scripts = other_root / "scripts"
    scripts.mkdir(parents=True)
    (other_root / "src").mkdir()
    copied = scripts / "diagnose_query_plan.py"
    copied.write_text(Path(diagnostic.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    unrelated = tmp_path / "other working directory"
    unrelated.mkdir()
    probe = ("import runpy,sys; from pathlib import Path; "
             "ns=runpy.run_path(sys.argv[1]); root=Path(sys.argv[1]).resolve().parents[1]; "
             "assert ns['ROOT']==root; assert str(root/'src') in sys.path; "
             "assert 'ragdn.query_llm' not in sys.modules; ns['main'](['--help'])")
    result = subprocess.run([sys.executable, "-B", "-c", probe, str(copied)], cwd=unrelated,
                            capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0 and "--context-tokens" in result.stdout
    assert "--show-plan" in result.stdout and "private" in result.stdout
    assert "Exit codes" in result.stdout and result.stderr == ""


def test_actual_canonical_resolver_and_validation_work_from_other_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = Client(proposal().model_dump_json())
    validator = Mock(wraps=validate_plan)
    report, code = diagnostic.diagnose(QUESTION, context_tokens=32768, client=client, validator=validator)
    assert code == 0 and report["new_completion_calls"] == len(client.calls) == 1
    references = validator.call_args.kwargs["references"]
    assert [(reference.doc_id, reference.dieu) for reference in references] == [("luat_dn_2020", "58"), ("luat_dn_2020", "59")]


def test_main_uses_sys_argv_and_help_requires_no_environment_or_completion(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [diagnostic.__file__, "--help"])
    load = Mock(side_effect=AssertionError("help must be offline"))
    monkeypatch.setattr(diagnostic, "_load_environment", load)
    with pytest.raises(SystemExit) as caught:
        diagnostic.main()
    assert caught.value.code == 0 and "--show-quotes" in capsys.readouterr().out
    load.assert_not_called()


def test_environment_is_loaded_from_script_checkout_without_overriding_process_values(monkeypatch, tmp_path):
    import dotenv
    load = Mock()
    monkeypatch.setattr(dotenv, "load_dotenv", load)
    monkeypatch.chdir(tmp_path)
    LOAD_ENVIRONMENT()
    load.assert_called_once_with(diagnostic.ROOT / ".env", override=False)


def test_test_module_entrypoint_forwards_cli_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", [__file__, "-q", "-p", "no:cacheprovider"])
    monkeypatch.setattr(pytest, "main", lambda args: calls.append(args) or 7)
    assert main() == 7
    assert calls == [[__file__, "-q", "-p", "no:cacheprovider"]]


def main() -> int:
    return int(pytest.main([__file__, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
