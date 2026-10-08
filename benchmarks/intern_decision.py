"""Verify, collect and replay pinned Intern-Decision suites with the official evaluator."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

INTERN_REVISION = "3572c8a68b5df5dafe02d0e093989ba8ec0183bc"
SCORER_REVISION = "7ce310c7262ed49cc85853339a8a42459298e3f3"
SUITES = (
    "jevbench-easy",
    "jevbench-original",
    "jevbench-hard",
    "agnews-test",
    "toolace-test",
    "typed_decisions-test",
    "wildjailbreak-test",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_checkout(root, revision):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("rev-parse", "HEAD") != revision:
        raise ValueError(f"Source revision mismatch: {root}; expected {revision}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Source checkout is not clean: {root}")


def verify_sources(intern_root, scorer_root):
    """Verify code pins, then call the unmodified upstream CPU bundle verifier."""
    intern_root, scorer_root = Path(intern_root).resolve(), Path(scorer_root).resolve()
    _verify_checkout(intern_root, INTERN_REVISION)
    _verify_checkout(scorer_root, SCORER_REVISION)
    spec = importlib.util.spec_from_file_location("_intern_bundle_verifier", intern_root / "src/eval/verify_bundle.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    bundle = intern_root / "benchmarks/accuracy-v1"
    verifier.verify(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["datasets"]) != set(SUITES) or (manifest["rows"], manifest["decisions"]) != (10751, 12351):
        raise ValueError("Expected all seven accuracy-v1 suites")
    return {
        **manifest,
        "intern_root": str(intern_root),
        "scorer_root": str(scorer_root),
        "source_revisions": {"intern_decision": INTERN_REVISION, "jevbench": SCORER_REVISION},
        "manifest_sha256": _sha256(bundle / "manifest.json"),
    }


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank input record: {path}:{number}")
            yield number, json.loads(line)


def _body(row):
    questions = row["questions"] if "questions" in row else {"decision": row["question"]}
    return {
        "state": row["state"],
        "questions": {
            field: {key: value for key, value in question.items() if key in ("type", "instructions", "criteria")}
            for field, question in questions.items()
        },
        "thinking": {"enabled": False},
    }


def _raw_path(directory, key, extended):
    name = f"{key[0]}/{key[1]}.json" if extended else hashlib.sha256(key[2].encode("utf-8")).hexdigest() + ".json"
    return directory / "raw" / name


def load_round(intern_root, scorer_root, round_dir):
    """Read A public231 or B accuracy-v1 evidence without trusting scored columns."""
    from benchmarks.client import decode
    from benchmarks.evidence import digest, dump
    from benchmarks.legs.leg1_public231 import validate_environment

    verified = verify_sources(intern_root, scorer_root)
    directory = Path(round_dir).resolve()
    if directory.name not in ("round-1", "round-2", "round-3"):
        raise ValueError("Expected a measured round-1, round-2 or round-3 directory")
    meta_path, results_path = directory.parent / "run.meta.json", directory / "results.jsonl"
    metadata = decode(meta_path.read_text(encoding="utf-8"))
    validate_environment(metadata)
    environment = (directory.parent / "environment.json").read_bytes()
    if digest(environment) != metadata.get("env_sha256"):
        raise ValueError("Environment source hash mismatch")
    if any(key not in metadata or metadata[key] != value for key, value in decode(environment).items()):
        raise ValueError("Run metadata differs from supplied environment")
    try:
        checkpoint, backend, dataset = metadata["checkpoint"], metadata["backend"], metadata["dataset"]
        temperature = metadata["sampling"]["temperature"]
        if not dataset["dataset_hash"]:
            raise ValueError("Run metadata requires data hash")
        if dataset["jevbench_pin"] != SCORER_REVISION:
            raise ValueError("Run metadata scorer pin mismatch")
        extended = "intern_revision" in dataset
        if extended and (dataset["intern_revision"], dataset["dataset_hash"]) != (
            INTERN_REVISION,
            verified["manifest_sha256"],
        ):
            raise ValueError("Run metadata Intern bundle pin mismatch")
        hashes = (
            verified["files"]
            if extended
            else {
                name: _sha256(Path(verified["scorer_root"]) / name)
                for name in (f"datasets/public/{tier}.jsonl" for tier in ("easy", "original", "hard"))
            }
        )
        if any(dataset["hashes"].get(path) != expected for path, expected in hashes.items()):
            raise ValueError("Run metadata data hash mismatch")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or malformed run metadata: {exc}") from exc
    bundle = Path(verified["intern_root"]) / "benchmarks/accuracy-v1"
    inputs = {
        (suite, line, row["id"]): row
        for suite in (SUITES if extended else SUITES[:3])
        for line, row in _rows(bundle / verified["datasets"][suite]["path"])
    }
    responses, evidence = {}, []
    for number, record in _rows(results_path) if results_path.exists() else ():
        suite, line, identity = record.get("suite"), record.get("source_line"), record.get("task_id")
        if not isinstance(suite, str) or type(line) is not int or not isinstance(identity, str):
            raise ValueError(f"Malformed response identity at results line {number}")
        key = (suite, line, identity)
        source = (
            verified["datasets"].get(suite, {}).get("path") if extended else suite.removeprefix("jevbench-") + ".jsonl"
        )
        if key not in inputs or record.get("source_file") != source:
            raise ValueError(f"Saved response identity is outside the pinned public input plan: {key}")
        if key in responses:
            raise ValueError(f"Duplicate saved response identity: {key}")
        if (
            record.get("kind") != "measured"
            or record.get("attempted") is not True
            or type(record.get("round")) is not int
            or record["round"] != int(directory.name[-1])
        ):
            raise ValueError(f"Not a measured attempt from this round: {key}")
        raw_path = _raw_path(directory, key, extended)
        raw_bytes = raw_path.read_bytes()
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        if raw_hash != record.get("raw_sha256"):
            raise ValueError(f"Raw response hash mismatch: {key}")
        raw = json.loads(raw_bytes)
        expected = _body(inputs[key])
        if metadata["model_configured"] is not None:
            expected["model"] = metadata["model_configured"]
        if dump(raw.get("request")) != dump(expected):
            raise ValueError(f"Raw request differs from frozen task and configuration: {key}")
        status, body = raw["http_status"], raw["response_body"]
        if status != record.get("status_code") or (status is not None and type(status) is not int):
            raise ValueError(f"HTTP status mismatch: {key}")
        if record.get("status") not in ("ok", "http_error", "client_error", "parse_error", "refused"):
            raise ValueError(f"Unknown attempt status: {key}")
        if body is not None and not isinstance(body, str):
            raise ValueError(f"Raw response body must be text or null: {key}")
        try:
            decoded = json.loads(body) if body is not None else None
        except ValueError:
            decoded = None
        if raw.get("response") is not None and json.dumps(raw["response"], sort_keys=True) != json.dumps(
            decoded, sort_keys=True
        ):
            raise ValueError(f"Raw decoded response differs from body: {key}")
        usable = (
            status is not None
            and 200 <= status < 300
            and raw["client_error"] is None
            and record.get("status") == "ok"
            and record.get("ok") is True
        )
        responses[key] = decoded if usable else None
        evidence.append(
            {
                "suite": suite,
                "source_line": line,
                "task_id": identity,
                "path": str(raw_path),
                "sha256": raw_hash,
                "status": record["status"],
                "http_status": status,
                "error": record.get("error"),
                "client_error": raw["client_error"],
            }
        )
    expected_raw = {item["path"] for item in evidence}
    actual_raw = {str(path) for path in (directory / "raw").rglob("*") if path.is_file()}
    if expected_raw != actual_raw:
        raise ValueError(
            f"Raw/record mismatch in {directory}: missing={sorted(expected_raw - actual_raw)}, "
            f"orphan={sorted(actual_raw - expected_raw)}"
        )
    provenance = {
        "round_directory": str(directory),
        "run_meta": metadata,
        "run_meta_sha256": _sha256(meta_path),
        "results_sha256": _sha256(results_path) if results_path.exists() else None,
        "raw": evidence,
    }
    config = {
        "checkpoint": f"{checkpoint['repo']}@{checkpoint['revision']}",
        "backend": backend["name"],
        "temperature": temperature,
    }
    return responses, provenance, config


def collect(args, *, transport=None):
    """Collect one complete accuracy-v1 round, with five separate warmup requests."""
    from benchmarks.client import DecisionClient, decode
    from benchmarks.evidence import dump, outside_repo, write_json
    from benchmarks.legs.leg1_public231 import validate_environment

    verified = verify_sources(args.intern_root, args.scorer_root)
    environment = args.run_meta.read_bytes()
    meta = decode(environment)
    validate_environment(meta)
    controlled = {
        "endpoint",
        "path",
        "model_configured",
        "env_sha256",
        "dataset",
        "plan",
        "stop_reason",
        "attempted",
        "collected",
    }
    if controlled.intersection(meta):
        raise ValueError(f"run-meta contains collector-controlled fields: {sorted(controlled.intersection(meta))}")
    bundle = Path(verified["intern_root"]) / "benchmarks/accuracy-v1"
    items = [
        (suite, line, row) for suite in SUITES for line, row in _rows(bundle / verified["datasets"][suite]["path"])
    ]
    output = outside_repo(args.out)
    meta.update(
        endpoint=args.endpoint,
        path=args.path,
        model_configured=args.model,
        env_sha256=hashlib.sha256(environment).hexdigest(),
        dataset=dict(
            intern_revision=INTERN_REVISION,
            jevbench_pin=SCORER_REVISION,
            hashes=verified["files"],
            dataset_hash=verified["manifest_sha256"],
        ),
        plan=dict(datasets=verified["datasets"], warmup=5, rounds=1),
        stop_reason=None,
    )
    with closing(
        DecisionClient(args.endpoint, path=args.path, model=args.model, timeout_s=args.timeout, transport=transport)
    ) as client:
        output.mkdir(parents=True, exist_ok=False)
        (output / "environment.json").write_bytes(environment)
        write_json(output / "run.meta.json", meta)
        stop, errors, attempted = None, 0, 0
        try:
            for number, selected in ((0, items[:5]), (1, items)):
                directory = output / ("warmup" if number == 0 else "round-1")
                directory.mkdir()
                with (directory / ("records.jsonl" if number == 0 else "results.jsonl")).open(
                    "x", encoding="utf-8"
                ) as stream:
                    for suite, line, row in selected:
                        body = _body(row)
                        raw = client.request(body)
                        path = _raw_path(directory, (suite, line, row["id"]), True)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        write_json(path, raw)
                        code, response = raw["http_status"], raw["response"]
                        answers = response.get("answers") if isinstance(response, dict) else None
                        parsed = isinstance(answers, dict) and all(
                            isinstance(answers.get(field), dict) and answers[field].get("type") == question["type"]
                            for field, question in body["questions"].items()
                        )
                        status = "ok" if parsed else "parse_error"
                        if raw["client_error"] is not None:
                            status = "client_error"
                        elif code is None or not 200 <= code < 300:
                            status = "refused" if code == 422 else "http_error"
                        record = dict(
                            suite=suite,
                            source_line=line,
                            task_id=row["id"],
                            source_file=verified["datasets"][suite]["path"],
                            round=number,
                            kind="warmup" if number == 0 else "measured",
                            attempted=True,
                            ok=status == "ok",
                            status=status,
                            status_code=code,
                            error=raw["client_error"] or (status if status != "ok" else None),
                            ts=raw["ts"],
                            latency_s=raw["latency_s"],
                            raw_sha256=_sha256(path),
                        )
                        stream.write(dump(record) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        attempted += number
                        errors = errors + 1 if status != "ok" and code != 422 else 0
                        if raw["client_error"] == "KeyboardInterrupt" or code in (401, 403, 429) or errors >= 3:
                            stop = raw["client_error"] or "access/rate limit or three consecutive infrastructure errors"
                            break
                if stop:
                    break
        except KeyboardInterrupt:
            stop = "interrupted"
        except Exception as exc:
            stop = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            meta.update(stop_reason=stop, attempted=attempted, collected=attempted == len(items) and stop is None)
            write_json(output / "run.meta.final.json", meta)
            (output / "run.meta.final.json").replace(output / "run.meta.json")
    return output


class SavedResponseEngine:
    """Sequential, unsharded evaluate() adapter; never normalizes saved answers."""

    def __init__(self, suite, responses, *, checkpoint, backend, temperature):
        self.suite, self.responses = suite, responses
        self.checkpoint, self.backend_name, self.temperature = checkpoint, backend, temperature
        self.source_line = 0

    def predict(self, row):
        self.source_line += 1
        key = (self.suite, self.source_line, row["id"])
        if key not in self.responses:
            raise ValueError(f"Missing saved response: {key}")
        return self.responses[key]


def _official(intern_root, scorer_root):
    """Refuse an already-imported foreign package instead of silently mixing sources."""
    roots = {"src": Path(intern_root), "jevbench": Path(scorer_root)}
    for name, module in tuple(sys.modules.items()):
        prefix = name.split(".")[0]
        if prefix in roots and module is not None:
            source = getattr(module, "__file__", None)
            locations = [source] if source else list(getattr(module, "__path__", []))
            if not locations or any(not Path(path).resolve().is_relative_to(roots[prefix]) for path in locations):
                raise ValueError(f"Foreign reference module already imported: {name}")
    configured = os.environ.get("JEVBENCH_ROOT")
    if configured and Path(configured).resolve() != roots["jevbench"]:
        raise ValueError("JEVBENCH_ROOT differs from the verified scorer checkout")
    previous_path = sys.path.copy()
    sys.path[:0] = [str(roots["src"]), str(roots["jevbench"])]
    try:
        evaluator = importlib.import_module("src.eval.jev")
        tables = importlib.import_module("src.eval.tables")
    finally:
        sys.path[:] = previous_path
    return evaluator, tables


def summarize(plan, metrics, errors, attempted):
    """Only complete official suites have scores; every planned denominator remains."""
    if set(plan) != set(SUITES):
        raise ValueError("Expected seven planned suites")
    if (set(metrics) | set(errors) | set(attempted)) - set(plan):
        raise ValueError("Unknown evaluation suite")
    datasets = {}
    for name, expected in plan.items():
        result = metrics.get(name)
        if result and (result["rows"], result["total"]) != (expected["rows"], expected["decisions"]):
            raise ValueError(f"{name}: official count mismatch")
        complete = result is not None and name not in errors
        datasets[name] = {
            **(result or {}),
            "planned": expected["rows"],
            "planned_decisions": expected["decisions"],
            "attempted": attempted.get(name, expected["rows"] if complete else 0),
            "valid": expected["decisions"] if complete else None,
            "correct": result["correct"] if complete else None,
            "accuracy": result["correct"] / expected["decisions"] if complete else None,
            "complete": complete,
            "error": errors.get(name, None if complete else "Suite not evaluated"),
        }
    complete = all(value["complete"] for value in datasets.values())
    return {
        "complete": complete,
        "planned": sum(value["rows"] for value in plan.values()),
        "planned_decisions": sum(value["decisions"] for value in plan.values()),
        "attempted": sum(value["attempted"] for value in datasets.values()),
        "valid": sum(value["valid"] for value in datasets.values()) if complete else None,
        "correct": sum(value["correct"] for value in datasets.values()) if complete else None,
        "average_accuracy": sum(value["accuracy"] for value in datasets.values()) / 7 if complete else None,
        "datasets": datasets,
    }


def replay_bundle(intern_root, scorer_root, responses, *, checkpoint, backend, temperature, output, provenance=None):
    """Replay A's attempted responses keyed by (suite, physical line, id).

    The caller retains raw HTTP evidence and rejects duplicate keys before building
    the mapping. Missing keys are unattempted; error/invalid bodies stay unchanged.
    This function makes no model requests and does not measure inference latency.
    """
    if not checkpoint or not backend or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Checkpoint, backend and finite positive temperature are required")
    verified = verify_sources(intern_root, scorer_root)
    bundle = Path(verified["intern_root"]) / "benchmarks/accuracy-v1"
    plan = verified["datasets"]
    identities = {
        name: [(name, line, row["id"]) for line, row in _rows(bundle / entry["path"])] for name, entry in plan.items()
    }
    expected_keys = {key for keys in identities.values() for key in keys}
    if set(responses) - expected_keys:
        raise ValueError("Saved response identity is outside the pinned input plan")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if provenance is not None:
        (output / "input.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics, errors, attempted = {}, {}, {}
    for name, entry in plan.items():
        missing = [key for key in identities[name] if key not in responses]
        attempted[name] = entry["rows"] - len(missing)
        if missing:
            errors[name] = f"Missing {len(missing)} responses; first: {missing[0]}"
            continue
        engine = None
        try:
            evaluator, _ = _official(verified["intern_root"], verified["scorer_root"])
            paths = {name: Path(path).resolve() for name, path, _ in evaluator.suite_datasets(test_root=bundle)}
            if paths != {suite: (bundle / spec["path"]).resolve() for suite, spec in plan.items()}:
                raise ValueError("Official suite paths differ from the verified bundle")
            engine = SavedResponseEngine(
                name, responses, checkpoint=checkpoint, backend=backend, temperature=temperature
            )
            destination = output / f"{name}.predictions.jsonl"
            result = evaluator.evaluate(engine, paths[name], destination)
            actual = [(name, row["source_line"], row["id"]) for _, row in _rows(destination)]
            if actual != identities[name] or (result["rows"], result["total"]) != (entry["rows"], entry["decisions"]):
                raise ValueError("Official output identity/count mismatch")
            metrics[name] = result
        except (ValueError, KeyError, TypeError, AttributeError, ImportError, OverflowError) as exc:
            location = f" at source_line={engine.source_line}" if engine is not None else ""
            errors[name] = f"{type(exc).__name__}{location}: {exc}"
    report = {
        **summarize(plan, metrics, errors, attempted),
        "source_revisions": verified["source_revisions"],
        "manifest_sha256": verified["manifest_sha256"],
        "data_sha256": verified["files"],
        "checkpoint": checkpoint,
        "inference_backend": backend,
        "temperature": temperature,
        "scope": "accuracy-v1; seven-suite mean, not the full JevBench leaderboard",
        "scoring_policy": "Incomplete suites have no official score; planned denominators are retained.",
    }
    if provenance is not None:
        report["input_evidence"] = {"path": "input.json", "sha256": _sha256(output / "input.json")}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if report["complete"]:
        _, tables = _official(verified["intern_root"], verified["scorer_root"])
        table = tables.accuracy_table([(checkpoint, report)])
        (output / "accuracy.md").write_text(table, encoding="utf-8")
        (output / "complete.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "replay", "collect"):
        child = commands.add_parser(command)
        child.add_argument("--intern-root", type=Path, required=True)
        child.add_argument("--scorer-root", type=Path, required=True)
        if command == "replay":
            child.add_argument("--round", type=Path, required=True, help="A's saved round-N directory")
            child.add_argument("--out", type=Path, required=True, help="New replay output directory")
        if command == "collect":
            child.add_argument("--endpoint", required=True)
            child.add_argument("--path", choices=("/v1/decisions", "/v1/systemone"), required=True)
            child.add_argument("--model")
            child.add_argument("--timeout", type=float, default=120)
            child.add_argument("--run-meta", type=Path, required=True)
            child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        report = verify_sources(args.intern_root, args.scorer_root)
    elif args.command == "collect":
        report = {"run": str(collect(args))}
    else:
        responses, provenance, config = load_round(args.intern_root, args.scorer_root, args.round)
        report = replay_bundle(
            args.intern_root, args.scorer_root, responses, **config, output=args.out, provenance=provenance
        )
    print(json.dumps(report, indent=2))
    if (
        args.command == "collect"
        and not json.loads((Path(report["run"]) / "run.meta.json").read_text(encoding="utf-8"))["collected"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
