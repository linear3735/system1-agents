"""Protocol tests; optional reference tests use real pinned upstream modules."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "intern_decision", Path(__file__).resolve().parents[2] / "benchmarks/intern_decision.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SavedResponseEngine = MODULE.SavedResponseEngine
replay_bundle = MODULE.replay_bundle
summarize = MODULE.summarize
verify_sources = MODULE.verify_sources
SUITES = MODULE.SUITES


def fixture_answer(body):
    answers = {}
    for field, question in body["questions"].items():
        if question["type"] == "noul":
            labels = ["no", "yes"]
        elif question["type"] == "score":
            labels = [str(i) for i in range(len(question["criteria"]))]
        else:
            labels = list(question["criteria"])
        answers[field] = {"type": question["type"], "probabilities": dict.fromkeys(labels, 1 / len(labels))}
    return {"answers": answers, "model": "protocol-fixture-no-model"}


def test_replay_catches_real_pinned_scorer_overflow_without_torch(tmp_path, monkeypatch):
    """A small evaluator adapter calls the real pinned scorer; it is not full official evaluate."""
    from benchmarks.evidence import load_reference

    intern, scorer = references()
    verified = verify_sources(intern, scorer)
    upstream, _, _ = load_reference()
    bundle = intern / "benchmarks/accuracy-v1"
    suite = SUITES[0]
    rows = list(MODULE._rows(bundle / verified["datasets"][suite]["path"]))
    saved = {(suite, line, row["id"]): fixture_answer(MODULE._body(row)) for line, row in rows}
    key = next(iter(saved))
    probs = saved[key]["answers"]["decision"]["probabilities"]
    probs[next(iter(probs))] = 10**400

    def evaluate(engine, data_path, output_path):
        row = rows[0][1]
        answer = engine.predict(row)
        return upstream.scoring.score_task(
            answer["answers"]["decision"]["probabilities"], upstream.tasks.Task.from_dict(row)
        )

    evaluator = SimpleNamespace(
        evaluate=evaluate,
        suite_datasets=lambda test_root: [
            (name, bundle / spec["path"], spec["rows"]) for name, spec in verified["datasets"].items()
        ],
    )
    monkeypatch.setattr(MODULE, "_official", lambda *_: (evaluator, None))
    target = tmp_path / "overflow-report"
    report = MODULE.replay_bundle(
        intern, scorer, saved, checkpoint="fixture", backend="fixture", temperature=1, output=target
    )
    assert report["attempted"] == 48 and report["planned_decisions"] == 12351
    assert report["average_accuracy"] is None and not report["complete"]
    assert "OverflowError at source_line=1" in report["datasets"][suite]["error"]
    assert (target / "report.json").exists() and not (target / "complete.json").exists()
    assert max(probs.values()) == 10**400


def test_actual_a_loopback_output_replays_without_layout_assumptions(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    from benchmarks.legs.leg1_public231 import predict

    intern, scorer = references()
    monkeypatch.setenv("JEVBENCH_ROOT", str(scorer))
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            content = b"not JSON" if len(calls) == 3 else json.dumps(fixture_answer(body)).encode()
            self.send_response(503 if len(calls) == 4 else 200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    environment = tmp_path / "actual-a-environment.json"
    environment.write_text(
        json.dumps(
            {
                "checkpoint": {"repo": "protocol-fixture-no-model", "revision": "a" * 40},
                "backend": {"name": "loopback", "version": "fixture"},
                "sampling": {"temperature": 1},
                "hardware": "CPU loopback protocol fixture",
            }
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        output = predict(
            SimpleNamespace(
                endpoint=f"http://127.0.0.1:{server.server_port}",
                path="/v1/decisions",
                model="fixture",
                timeout=2,
                rounds=1,
                warmup=1,
                limit=3,
                run_meta=environment,
                out=tmp_path / "actual-a-output",
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    saved, evidence, config = MODULE.load_round(intern, scorer, output / "round-1")
    assert len(calls) == 4 and len(saved) == 3
    assert list(saved.values()) == [fixture_answer(calls[1]), None, None]
    assert [row["status"] for row in evidence["raw"]] == ["ok", "parse_error", "http_error"]
    report = MODULE.replay_bundle(
        intern, scorer, saved, **config, output=tmp_path / "actual-a-replay", provenance=evidence
    )
    assert report["planned_decisions"] == 12351 and report["attempted"] == 3
    assert not report["complete"] and report["average_accuracy"] is None


@pytest.fixture
def disk_round(tmp_path, monkeypatch):
    """A's frozen disk contract; protocol fixture, never model evidence."""
    intern, scorer = tmp_path / "intern", tmp_path / "scorer"
    row = {"id": "item", "state": {"text": "original"}, "question": {"type": "choice", "criteria": {"A": "a"}}}
    plan, hashes = {}, {}
    for suite in SUITES:
        relative = f"{suite}/test.jsonl"
        source = intern / "benchmarks/accuracy-v1" / relative
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps(row) + "\n")
        plan[suite] = {"path": relative, "rows": 1, "decisions": 1}
    for tier in ("easy", "original", "hard"):
        source = scorer / f"datasets/public/{tier}.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(row) + "\n")
        hashes[f"datasets/public/{tier}.jsonl"] = MODULE._sha256(source)
    verified = {
        "intern_root": str(intern),
        "scorer_root": str(scorer),
        "datasets": plan,
        "source_revisions": {},
        "manifest_sha256": "fixture",
        "files": {},
    }
    monkeypatch.setattr(MODULE, "verify_sources", lambda *_: verified)
    round_dir = tmp_path / "run/round-1"
    (round_dir / "raw").mkdir(parents=True)
    meta = {
        "checkpoint": {"repo": "example/model", "revision": "a" * 40},
        "backend": {"name": "hf", "version": "fixture"},
        "sampling": {"temperature": 1.0},
        "hardware": "CPU protocol fixture",
        "dtype": "bf16",
        "context_limits": {"max_tokens": 16384},
        "cache_config": {"enabled": False},
        "dataset": {"jevbench_pin": MODULE.SCORER_REVISION, "hashes": hashes, "dataset_hash": "fixture"},
    }
    environment = {key: value for key, value in meta.items() if key != "dataset"}
    (round_dir.parent / "environment.json").write_text(json.dumps(environment))
    meta.update(model_configured=None, env_sha256=MODULE._sha256(round_dir.parent / "environment.json"))
    raw = {
        "request": {"state": row["state"], "questions": {"decision": row["question"]}, "thinking": {"enabled": False}},
        "response_body": json.dumps(response(0.8)),
        "response": response(0.8),
        "http_status": 200,
        "client_error": None,
    }
    record = {
        "suite": SUITES[0],
        "source_line": 1,
        "task_id": "item",
        "source_file": "easy.jsonl",
        "round": 1,
        "kind": "measured",
        "attempted": True,
        "ok": True,
        "status": "ok",
        "status_code": 200,
        "error": None,
    }
    raw_path = round_dir / "raw" / (hashlib.sha256(b"item").hexdigest() + ".json")

    def save(*, records=None):
        (round_dir.parent / "run.meta.json").write_text(json.dumps(meta))
        raw_path.write_text(json.dumps(raw))
        record["raw_sha256"] = MODULE._sha256(raw_path)
        (round_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (records or [record])))

    save()
    return intern, scorer, round_dir, meta, raw, record, raw_path, save


@pytest.fixture(params=[False, True], ids=["public231", "accuracy-v1"])
def raw_inventory_round(disk_round, request):
    intern, scorer, directory, meta, _, record, original_raw, _ = disk_round
    verified = MODULE.verify_sources(intern, scorer)
    extended = request.param
    if extended:
        meta["dataset"].update(intern_revision=MODULE.INTERN_REVISION, dataset_hash=verified["manifest_sha256"])
        meta["dataset"]["hashes"] = verified["files"]
    second_source = intern / "benchmarks/accuracy-v1" / verified["datasets"][SUITES[1]]["path"]
    second = json.loads(second_source.read_text())
    second["id"] = "second"
    second_source.write_text(json.dumps(second) + "\n")
    records = [dict(record), dict(record, suite=SUITES[1], task_id="second")]
    raw_bytes, raw_paths = original_raw.read_bytes(), []
    for row in records:
        row["source_file"] = (
            verified["datasets"][row["suite"]]["path"]
            if extended
            else row["suite"].removeprefix("jevbench-") + ".jsonl"
        )
        relative = (
            f"{row['suite']}/{row['source_line']}.json"
            if extended
            else hashlib.sha256(row["task_id"].encode()).hexdigest() + ".json"
        )
        path = directory / "raw" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)
        row["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        raw_paths.append(path)
    if original_raw not in raw_paths:
        original_raw.unlink()
    (directory.parent / "run.meta.json").write_text(json.dumps(meta))
    (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return intern, scorer, directory, records, raw_paths


@pytest.mark.parametrize(
    "failure", ["drop_row", "drop_ledger", "empty_ledger", "unknown_raw", "nested_raw", "missing_raw", "blank_line"]
)
def test_round_rejects_raw_inventory_mismatch(raw_inventory_round, failure):
    intern, scorer, directory, records, raw_paths = raw_inventory_round
    ledger = directory / "results.jsonl"
    if failure == "drop_row":
        ledger.write_text(json.dumps(records[0]) + "\n")
    elif failure == "drop_ledger":
        ledger.unlink()
    elif failure == "empty_ledger":
        ledger.write_text("")
    elif failure in ("unknown_raw", "nested_raw"):
        extra = directory / "raw" / ("unexpected.txt" if failure == "unknown_raw" else "unknown/nested.json")
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(raw_paths[0].read_bytes())
    elif failure == "missing_raw":
        raw_paths[0].unlink()
    else:
        ledger.write_text("\n")
    with pytest.raises((ValueError, FileNotFoundError)) as error:
        MODULE.load_round(intern, scorer, directory)
    if failure not in ("missing_raw", "blank_line"):
        assert "Raw/record mismatch" in str(error.value)


@pytest.mark.parametrize("state", ["partial", "empty_ledger", "absent_ledger", "absent_round"])
def test_round_matching_raw_inventory_keeps_partial_plan(raw_inventory_round, state):
    import shutil

    intern, scorer, directory, records, raw_paths = raw_inventory_round
    ledger = directory / "results.jsonl"
    if state == "partial":
        ledger.write_text(json.dumps(records[0]) + "\n")
        raw_paths[1].unlink()
    elif state == "absent_round":
        shutil.rmtree(directory)
    else:
        for path in raw_paths:
            path.unlink()
        if state == "empty_ledger":
            ledger.write_text("")
        else:
            ledger.unlink()
    saved, provenance, config = MODULE.load_round(intern, scorer, directory)
    attempted = 1 if state == "partial" else 0
    assert len(saved) == len(provenance["raw"]) == attempted
    report = MODULE.replay_bundle(intern, scorer, saved, **config, output=directory.parent / "partial-replay")
    assert report["planned"] == report["planned_decisions"] == 7
    assert report["attempted"] == attempted and set(report["datasets"]) == set(SUITES)
    assert not report["complete"] and report["average_accuracy"] is None


@pytest.mark.parametrize(
    "field", ["checkpoint", "backend", "sampling", "hardware", "dtype", "context_limits", "cache_config"]
)
def test_replay_rejects_every_environment_field_mismatch(disk_round, field):
    intern, scorer, directory, meta, _, _, _, save = disk_round
    if field == "checkpoint":
        meta[field]["revision"] = "relabeled-revision"
    elif field == "backend":
        meta[field]["version"] = "other"
    elif field == "sampling":
        meta[field]["temperature"] = 2.0
    elif field == "context_limits":
        meta[field]["max_tokens"] = 4096
    elif field == "cache_config":
        meta[field]["enabled"] = True
    else:
        meta[field] = "other"
    save()
    with pytest.raises(ValueError, match="[Ee]nvironment"):
        MODULE.load_round(intern, scorer, directory)


@pytest.mark.parametrize("failure", ["missing", "hash", "corrupt", "deleted_field"])
def test_replay_rejects_missing_or_corrupt_environment(disk_round, failure):
    intern, scorer, directory, meta, _, _, _, save = disk_round
    environment = directory.parent / "environment.json"
    if failure == "missing":
        environment.unlink()
    elif failure == "hash":
        meta["env_sha256"] = "wrong"
    elif failure == "corrupt":
        environment.write_text("not JSON")
        meta["env_sha256"] = MODULE._sha256(environment)
    else:
        del meta["dtype"]
    save()
    with pytest.raises((ValueError, FileNotFoundError)):
        MODULE.load_round(intern, scorer, directory)


@pytest.mark.parametrize(
    "change",
    [
        {"targets": {"decision": "A"}},
        {"labels": ["A"]},
        {"thinking": {"enabled": True}},
        {"thinking": {"enabled": 0}},
        {"model": "different-model"},
    ],
)
def test_replay_rejects_changed_complete_request_envelope(disk_round, change):
    intern, scorer, directory, _, raw, _, _, save = disk_round
    raw["request"].update(change)
    save()  # Recompute the correct raw hash: request semantics still must be rejected.
    with pytest.raises(ValueError, match="request"):
        MODULE.load_round(intern, scorer, directory)


def test_disk_round_keeps_original_response_and_evidence(disk_round):
    intern, scorer, directory, meta, raw, _, raw_path, _ = disk_round
    saved, provenance, config = MODULE.load_round(intern, scorer, directory)
    assert saved == {(SUITES[0], 1, "item"): raw["response"]}
    assert config == {"checkpoint": "example/model@" + "a" * 40, "backend": "hf", "temperature": 1.0}
    assert provenance["run_meta"] == meta
    assert provenance["raw"][0]["path"] == str(raw_path)
    assert provenance["raw"][0]["sha256"] == MODULE._sha256(raw_path)


@pytest.mark.parametrize(
    "change",
    [
        {"suite": "unknown"},
        {"source_line": 2},
        {"source_line": True},
        {"task_id": "forged"},
        {"source_file": "hard.jsonl"},
        {"kind": "warmup"},
        {"round": 2},
        {"round": True},
        {"attempted": False},
        {"attempted": 1},
    ],
)
def test_disk_round_rejects_forged_record_identity(disk_round, change):
    intern, scorer, directory, _, _, record, _, save = disk_round
    record.update(change)
    save()
    with pytest.raises(ValueError):
        MODULE.load_round(intern, scorer, directory)


def test_disk_round_rejects_duplicates_and_raw_hash_mismatch(disk_round):
    intern, scorer, directory, _, _, record, raw_path, save = disk_round
    save(records=[record, record])
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.load_round(intern, scorer, directory)
    save()
    raw_path.write_text(raw_path.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        MODULE.load_round(intern, scorer, directory)


@pytest.mark.parametrize("part", ["state", "questions"])
def test_disk_round_rejects_raw_request_from_another_input(disk_round, part):
    intern, scorer, directory, _, raw, _, _, save = disk_round
    raw["request"][part] = {}
    save()
    with pytest.raises(ValueError, match="request"):
        MODULE.load_round(intern, scorer, directory)


@pytest.mark.parametrize(
    "status,code,error",
    [
        ("http_error", 503, None),
        ("client_error", None, "timeout"),
        ("parse_error", 200, None),
        ("refused", 200, None),
    ],
)
def test_disk_round_does_not_score_failed_attempts(disk_round, status, code, error):
    intern, scorer, directory, _, raw, record, _, save = disk_round
    record.update(status=status, status_code=code, ok=False, error=status)
    raw.update(http_status=code, client_error=error)
    if status == "parse_error":
        raw.update(response_body="not JSON", response=None)
    elif status == "client_error":
        raw.update(response_body=None, response=None)
    save()
    saved, provenance, _ = MODULE.load_round(intern, scorer, directory)
    assert saved == {(SUITES[0], 1, "item"): None}
    assert provenance["raw"][0]["status"] == status


def test_disk_round_body_is_authoritative_and_decoded_is_optional(disk_round):
    intern, scorer, directory, _, raw, _, _, save = disk_round
    raw["response"] = response(1)
    save()
    with pytest.raises(ValueError, match="decoded"):
        MODULE.load_round(intern, scorer, directory)
    del raw["response"]
    save()
    saved, _, _ = MODULE.load_round(intern, scorer, directory)
    assert saved[(SUITES[0], 1, "item")] == response(0.8)


def test_disk_round_preserves_integer_decode_limit_failure(disk_round):
    import sys

    intern, scorer, directory, _, raw, record, raw_path, save = disk_round
    body = '{"answers":{"decision":{"type":"choice","probabilities":{"A":' + "9" * 5001 + "}}}}"
    previous_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        with pytest.raises(ValueError) as exc:
            json.loads(body)
        assert not isinstance(exc.value, json.JSONDecodeError)
        raw.update(response_body=body, response=None)
        record.update(status="parse_error", ok=False, error="parse_error")
        save()
        saved, provenance, _ = MODULE.load_round(intern, scorer, directory)
    finally:
        sys.set_int_max_str_digits(previous_limit)
    assert saved == {(SUITES[0], 1, "item"): None}
    assert provenance["raw"][0]["status"] == "parse_error"
    assert provenance["raw"][0]["sha256"] == MODULE._sha256(raw_path)
    assert json.loads(raw_path.read_text())["response_body"] == body


@pytest.mark.parametrize("failure", ["http", "client", "parse"])
def test_disk_round_does_not_trust_success_columns(disk_round, failure):
    intern, scorer, directory, _, raw, record, _, save = disk_round
    record.update(valid=True, correct=True, probs={"A": 1})
    if failure == "http":
        raw["http_status"] = record["status_code"] = 503
    elif failure == "client":
        raw["client_error"] = "failed read"
    else:
        raw.update(response_body="<html>failure</html>", response=None)
    save()
    saved, _, _ = MODULE.load_round(intern, scorer, directory)
    assert saved[(SUITES[0], 1, "item")] is None


def test_disk_round_preserves_invalid_probabilities_for_official_scorer(disk_round):
    intern, scorer, directory, _, raw, _, _, save = disk_round
    raw["response"] = response(float("nan"))
    raw["response_body"] = json.dumps(raw["response"])
    save()
    saved, _, _ = MODULE.load_round(intern, scorer, directory)
    assert json.dumps(saved[(SUITES[0], 1, "item")]) == raw["response_body"]


def test_a_scorer_error_does_not_replace_original_decoded_response(disk_round):
    intern, scorer, directory, _, raw, record, _, save = disk_round
    raw["response"] = {"answers": {"decision": {"type": "choice", "probabilities": {"unknown": 1}}}}
    raw["response_body"] = json.dumps(raw["response"])
    record.update(valid=False, error="upstream diagnostic: unexpected probability labels")
    save()
    saved, _, _ = MODULE.load_round(intern, scorer, directory)
    assert saved[(SUITES[0], 1, "item")] == raw["response"]


def test_disk_round_rejects_option_reordering(disk_round):
    intern, scorer, directory, _, raw, _, _, save = disk_round
    source = intern / "benchmarks/accuracy-v1/jevbench-easy/test.jsonl"
    row = json.loads(source.read_text())
    row["question"]["criteria"] = {"A": "a", "B": "b"}
    source.write_text(json.dumps(row) + "\n")
    raw["request"]["questions"]["decision"]["criteria"] = {"B": "b", "A": "a"}
    save()
    with pytest.raises(ValueError, match="request"):
        MODULE.load_round(intern, scorer, directory)


@pytest.mark.parametrize("change", ["hash", "pin", "checkpoint", "backend", "temperature"])
def test_disk_round_rejects_unbound_metadata(disk_round, change):
    intern, scorer, directory, meta, _, _, _, save = disk_round
    if change == "hash":
        meta["dataset"]["hashes"]["datasets/public/easy.jsonl"] = "bad"
    elif change == "pin":
        meta["dataset"]["jevbench_pin"] = "main"
    elif change == "temperature":
        meta["sampling"]["temperature"] = 0
    else:
        del meta[change]
    save()
    with pytest.raises(ValueError):
        MODULE.load_round(intern, scorer, directory)


def test_replay_cli_retains_seven_suite_plan(disk_round, monkeypatch, capsys):
    intern, scorer, directory, _, _, _, _, _ = disk_round
    out = directory.parent.parent / "replay"
    # Import failure cannot turn the one attempted fixture row into a score.
    monkeypatch.setattr(MODULE, "_official", lambda *_: (_ for _ in ()).throw(ImportError("CPU fixture")))
    monkeypatch.setattr(
        "sys.argv",
        [
            "intern_decision.py",
            "replay",
            "--intern-root",
            str(intern),
            "--scorer-root",
            str(scorer),
            "--round",
            str(directory),
            "--out",
            str(out),
        ],
    )
    MODULE.main()
    report = json.loads(capsys.readouterr().out)
    assert report["planned"] == 7 and report["attempted"] == 1
    assert not report["complete"] and report["average_accuracy"] is None
    assert len(report["datasets"]) == 7
    assert not (out / "complete.json").exists()
    assert report["input_evidence"]["sha256"] == MODULE._sha256(out / "input.json")
    assert json.loads((out / "input.json").read_text())["raw"][0]["task_id"] == "item"


def references():
    intern = os.environ.get("INTERN_DECISION_ROOT")
    scorer = os.environ.get("JEVBENCH_ROOT")
    if not intern or not scorer:
        pytest.skip("set INTERN_DECISION_ROOT and JEVBENCH_ROOT for pinned reference checks")
    return Path(intern), Path(scorer)


def response(value):
    return {"answers": {"decision": {"probabilities": {"A": value, "B": 1 - value}}}, "timing": {"inference_ms": 7}}


def test_saved_responses_keep_suite_line_and_original_values():
    saved = {
        ("one", 1, "same"): response(0.8),
        ("one", 2, "same"): response(0.2),
        ("two", 1, "same"): response(0.6),
    }
    original = copy.deepcopy(saved)
    engine = SavedResponseEngine("one", saved, checkpoint="model@revision", backend="hf", temperature=1.0)
    assert engine.predict({"id": "same"}) == saved[("one", 1, "same")]
    assert engine.predict({"id": "same"}) == saved[("one", 2, "same")]
    other = SavedResponseEngine("two", saved, checkpoint="model@revision", backend="hf", temperature=1.0)
    assert other.predict({"id": "same"}) == saved[("two", 1, "same")]
    assert saved == original
    assert (engine.checkpoint, engine.backend_name, engine.temperature) == ("model@revision", "hf", 1.0)


def test_saved_response_never_falls_back_to_id_only():
    engine = SavedResponseEngine("one", {("one", 2, "same"): response(1)}, checkpoint="m", backend="hf", temperature=1)
    with pytest.raises(ValueError, match="one.*1.*same"):
        engine.predict({"id": "same"})


def test_engine_does_not_repair_or_drop_invalid_answers():
    raw = {"answers": {"decision": {"probabilities": {"A": float("nan")}}}, "usage": {"input_tokens": 4}}
    engine = SavedResponseEngine("s", {("s", 1, "x"): raw}, checkpoint="m", backend="hf", temperature=1)
    assert engine.predict({"id": "x"}) is raw


def test_summary_uses_all_seven_planned_denominators():
    plan = {name: {"rows": 1, "decisions": 2} for name in SUITES}
    plan[SUITES[-1]] = {"rows": 10, "decisions": 100}
    metrics = {name: {"rows": spec["rows"], "total": spec["decisions"], "correct": 1} for name, spec in plan.items()}
    metrics[SUITES[-1]]["correct"] = 90
    report = summarize(plan, metrics, {}, {})
    assert report["complete"] is True
    assert report["average_accuracy"] == pytest.approx((6 * 0.5 + 0.9) / 7)
    assert report["planned_decisions"] == 112
    assert report["correct"] == 96


def test_summary_missing_suite_or_short_result_cannot_be_complete():
    plan = {name: {"rows": 2, "decisions": 2} for name in SUITES}
    metrics = {name: {"rows": 2, "total": 2, "correct": 1} for name in plan}
    missing = SUITES[-1]
    del metrics[missing]
    report = summarize(plan, metrics, {missing: "missing response"}, {missing: 1})
    assert report["complete"] is False
    assert report["planned_decisions"] == 14
    assert report["datasets"][missing]["planned_decisions"] == 2
    assert report["datasets"][missing]["attempted"] == 1
    assert report["datasets"][missing]["correct"] is None
    assert report["average_accuracy"] is None
    metrics[missing] = {"rows": 1, "total": 1, "correct": 1}
    with pytest.raises(ValueError, match="count"):
        summarize(plan, metrics, {}, {})


def test_summary_requires_exactly_seven_suites():
    with pytest.raises(ValueError, match="seven"):
        summarize({"one": {"rows": 1, "decisions": 1}}, {}, {}, {})
    with pytest.raises(ValueError, match="seven"):
        summarize({f"unknown{i}": {"rows": 1, "decisions": 1} for i in range(7)}, {}, {}, {})


def test_real_bundle_is_pinned_and_cpu_only():
    intern, scorer = references()
    result = verify_sources(intern, scorer)
    assert result["rows"] == 10751
    assert result["decisions"] == 12351
    assert len(result["datasets"]) == 7
    assert result["datasets"]["typed_decisions-test"]["decisions"] == 2000
    assert result["datasets"]["agnews-test"]["upstream_license_metadata"] == ["unknown"]


def test_wrong_source_pin_is_rejected():
    intern, scorer = references()
    with pytest.raises(ValueError, match="revision"):
        verify_sources(scorer, intern)


def test_hard_bundle_has_no_tvd_reference_distributions():
    intern, scorer = references()
    counts = []
    for path in (
        intern / "benchmarks/accuracy-v1/jevbench/hard.jsonl",
        scorer / "datasets/public/hard.jsonl",
    ):
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        counts.append(sum(bool(row.get("gold_probs") or row.get("provenance", {}).get("gold_probs")) for row in rows))
    assert counts == [0, 10]


def test_empty_replay_keeps_full_plan_without_torch(tmp_path):
    intern, scorer = references()
    out = tmp_path / "replay"
    report = replay_bundle(intern, scorer, {}, checkpoint="model@revision", backend="hf", temperature=1, output=out)
    assert report["complete"] is False
    assert report["planned_decisions"] == 12351
    assert report["attempted"] == 0
    assert report["average_accuracy"] is None
    assert not (out / "complete.json").exists()
    assert json.loads((out / "report.json").read_text())["complete"] is False
    assert not (out / "accuracy.md").exists()
    with pytest.raises(FileExistsError):
        replay_bundle(intern, scorer, {}, checkpoint="m", backend="hf", temperature=1, output=out)


def test_replay_rejects_orphan_identity_before_output(tmp_path):
    intern, scorer = references()
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match="identity"):
        replay_bundle(
            intern, scorer, {("unknown", 1, "x"): response(1)}, checkpoint="m", backend="hf", temperature=1, output=out
        )
    assert not out.exists()


def test_official_replay_of_protocol_responses(tmp_path):
    """Exercise real scoring, never model quality: probabilities are uniform fixtures."""
    intern, scorer = references()
    if importlib.util.find_spec("torch") is None:
        pytest.skip("official evaluate imports Torch; run this check in the prepared CPU environment")
    verified = verify_sources(intern, scorer)
    evaluator, _ = MODULE._official(intern.resolve(), scorer.resolve())
    saved = {}
    for suite, entry in verified["datasets"].items():
        source = intern / "benchmarks/accuracy-v1" / entry["path"]
        with source.open(encoding="utf-8") as stream:
            for line, text in enumerate(stream, 1):
                row = json.loads(text)
                questions = row.get("questions") or {"decision": row["question"]}
                answers = {}
                for field, question in questions.items():
                    labels = [value for value, _ in evaluator._options(question)]
                    answers[field] = {"type": question["type"], "probabilities": dict.fromkeys(labels, 1 / len(labels))}
                saved[(suite, line, row["id"])] = {"answers": answers}
    out = tmp_path / "uniform-protocol-fixture"
    report = replay_bundle(
        intern, scorer, saved, checkpoint="protocol-fixture-no-model", backend="hf", temperature=1, output=out
    )
    assert report["complete"] is True, json.dumps(report, indent=2)
    assert report["valid"] == report["planned_decisions"] == 12351
    assert report["datasets"]["jevbench-hard"]["tvd_n"] == 0
    assert report["datasets"]["jevbench-hard"]["tvd"] is None
    assert (out / "complete.json").is_file()
    assert (out / "accuracy.md").is_file()
    assert report["average_accuracy"] == pytest.approx(sum(v["accuracy"] for v in report["datasets"].values()) / 7)
    direct = evaluator.evaluate(
        SavedResponseEngine(
            "typed_decisions-test", saved, checkpoint="protocol-fixture-no-model", backend="hf", temperature=1
        ),
        intern / "benchmarks/accuracy-v1/typed_decisions/test.jsonl",
        tmp_path / "direct.predictions.jsonl",
    )
    for key in ("correct", "total", "brier", "ece"):
        assert direct[key] == report["datasets"]["typed_decisions-test"][key]

    # A complete response ledger with an invalid probability must still fail closed.
    easy = {key: copy.deepcopy(value) for key, value in saved.items() if key[0] == "jevbench-easy"}
    first = next(iter(easy.values()))["answers"]["decision"]["probabilities"]
    first[next(iter(first))] = float("nan")
    failed = replay_bundle(
        intern,
        scorer,
        easy,
        checkpoint="protocol-fixture-no-model",
        backend="hf",
        temperature=1,
        output=tmp_path / "invalid",
    )
    assert failed["complete"] is False
    assert failed["datasets"]["jevbench-easy"]["attempted"] == 48
    assert failed["datasets"]["jevbench-easy"]["correct"] is None
    assert "not finite" in failed["datasets"]["jevbench-easy"]["error"]
    assert not (tmp_path / "invalid/complete.json").exists()
