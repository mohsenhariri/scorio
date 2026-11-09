# ruff: noqa: E402

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("inspect_ai", minversion="0.3.288")

from inspect_ai import Epochs, Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import (
    EvalError,
    EvalLog,
    recompute_metrics,
)
from inspect_ai.model import CachePolicy, ModelName
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState

from scorio.integrations.inspect import bayes_at_n, record_epochs, validate_bayes_at_n


def _sample(value, sample_id="q1", epoch=1) -> SampleScore:
    return SampleScore(
        sample_id=sample_id,
        score=Score(value=value, metadata={"scorio.bayes_at_n": {"epoch": epoch}}),
    )


def _compute(metric: Metric, scores: list[SampleScore]) -> dict[str, float]:
    return cast(dict[str, float], metric(scores))


@pytest.mark.parametrize(
    ("values", "expected_sd"),
    [
        (["I", "I", "C", "C"], math.sqrt(3 / 160)),
        (["I", "C", "I", "C"], math.sqrt(1 / 40)),
    ],
)
def test_question_posteriors_are_not_pooled(values, expected_sd) -> None:
    scores = [
        _sample(value, sample_id, epoch)
        for value, (sample_id, epoch) in zip(
            values, [("q1", 1), ("q1", 2), ("q2", 1), ("q2", 2)], strict=True
        )
    ]
    result = _compute(bayes_at_n(["q1", "q2"], 2), scores)
    assert result == {
        "posterior_mean": pytest.approx(0.5),
        "posterior_sd": pytest.approx(expected_sd),
    }
    assert _compute(bayes_at_n(["q2", "q1"], 2), scores[::-1]) == result


@pytest.mark.parametrize(("value", "mean"), [("I", 1 / 3), ("C", 2 / 3)])
def test_one_trial_keeps_posterior_uncertainty(value, mean) -> None:
    assert _compute(bayes_at_n(["q1"], 1), [_sample(value)]) == {
        "posterior_mean": pytest.approx(mean),
        "posterior_sd": pytest.approx(math.sqrt(1 / 18)),
    }


@pytest.mark.parametrize(
    ("ids", "epochs"),
    [
        ([], 1),
        (["q1", "q1"], 1),
        ([1, "1"], 1),
        ([True], 1),
        ([1.0], 1),
        ("q1", 1),
        (["q1"], 0),
        (["q1"], True),
        (["q1"], 2.0),
    ],
)
def test_invalid_experiment_is_rejected(ids, epochs) -> None:
    with pytest.raises(ValueError):
        bayes_at_n(ids, epochs)


@pytest.mark.parametrize(
    "value", ["P", "N", "0", 0, 1, True, 0.5, math.nan, math.inf, ["C"], {"grade": "C"}]
)
def test_only_literal_correct_incorrect_scores_are_accepted(value) -> None:
    with pytest.raises(ValueError, match="C/I"):
        _compute(bayes_at_n(["q1"], 1), [_sample(value)])


@pytest.mark.parametrize(
    "scores", [[], [_sample("C")], [_sample("C"), _sample("I", "q2")]]
)
def test_incomplete_live_cohort_has_no_finite_posterior(scores) -> None:
    result = _compute(bayes_at_n(["q1", "q2"], 2), scores)
    assert set(result) == {"posterior_mean", "posterior_sd"}
    assert all(math.isnan(value) for value in result.values())


def test_epoch_identity_detects_duplicate_replacing_missing_trial() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        _compute(bayes_at_n(["q1"], 2), [_sample("C"), _sample("I")])


@pytest.mark.parametrize(
    "score",
    [
        SampleScore(sample_id="q1", score=Score(value="C")),
        _sample("C", "unknown"),
        _sample("C", None),
        _sample("C", epoch=0),
        _sample("C", epoch=3),
        _sample("C", epoch=True),
    ],
)
def test_missing_or_unexpected_identity_is_rejected(score) -> None:
    with pytest.raises(ValueError):
        _compute(bayes_at_n(["q1"], 2), [score])


def test_wrapper_preserves_score_fields_without_mutating_the_original() -> None:
    original = Score(
        value="C", answer="answer", explanation="graded", metadata={"rubric": "binary"}
    )

    @scorer(metrics=[])
    def fixed_score() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return original

        return score

    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="q1",
        epoch=2,
        input="question",
        messages=[],
    )
    result = asyncio.run(record_epochs(fixed_score())(state, Target("answer")))
    assert result is not original
    assert original.metadata == {"rubric": "binary"}
    assert result.model_dump(exclude={"metadata"}) == original.model_dump(
        exclude={"metadata"}
    )
    assert result.metadata == {"rubric": "binary", "scorio.bayes_at_n": {"epoch": 2}}


@scorer(metrics=[])
def _binary_scorer(style: str = "separated", delay: float = 0) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        if delay:
            await asyncio.sleep(delay)
        correct = (
            state.sample_id == "q2" if style == "separated" else state.epoch % 2 == 0
        )
        return Score(value="C" if correct else "I", metadata={"grader": "fixture"})

    return score


def _task(*, style="separated", epochs=2, ids=None, delay=0, reducers=None) -> Task:
    ids = ids or ["q1", "q2"]
    return Task(
        dataset=[
            Sample(id=sample_id, input="question", target="answer") for sample_id in ids
        ],
        solver=[],
        scorer=record_epochs(_binary_scorer(style, delay)),
        metrics=[accuracy(), bayes_at_n(sample_ids=ids, epochs=epochs)],
        epochs=Epochs(epochs, reducers or ["mean", "max"]),
    )


def _run(task: Task, path: Path, **kwargs) -> EvalLog:
    log = eval(
        task,
        model="mockllm/model",
        display="none",
        log_dir=str(path),
        ctl_server=False,
        acp_server=False,
        **kwargs,
    )[0]
    assert log.status == "success", log.error
    return log


def _posterior(log: EvalLog) -> dict[str, float]:
    assert log.results is not None
    results = [
        score for score in log.results.scores if "posterior_mean" in score.metrics
    ]
    assert len(results) == 1
    assert results[0].reducer is None
    return {key: metric.value for key, metric in results[0].metrics.items()}


@pytest.fixture(autouse=True)
def _inspect_paths(monkeypatch, tmp_path) -> None:
    for variable in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable))


@pytest.fixture
def complete_log(tmp_path) -> EvalLog:
    return _run(_task(), tmp_path / "logs")


@pytest.mark.parametrize(
    ("style", "expected_sd"),
    [("separated", math.sqrt(3 / 160)), ("mixed", math.sqrt(1 / 40))],
)
def test_mock_eval_and_multiple_reducers_preserve_question_moments(
    tmp_path, style, expected_sd
) -> None:
    log = _run(_task(style=style), tmp_path / "logs")
    expected = {
        "posterior_mean": pytest.approx(0.5),
        "posterior_sd": pytest.approx(expected_sd),
    }
    assert _posterior(log) == expected
    assert validate_bayes_at_n(log, ["q1", "q2"], 2) == expected
    assert {score.reducer for score in log.results.scores} == {None, "mean", "max"}
    for sample in log.samples:
        metadata = sample.scores["record_epochs"].metadata
        assert metadata["scorio.bayes_at_n"]["epoch"] == sample.epoch
        assert metadata["grader"] == "fixture"


def test_live_partial_updates_do_not_abort_evaluation(tmp_path, monkeypatch) -> None:
    import scorio.integrations.inspect as integration

    observed = []
    original = integration._outcomes

    def capture(scores, experiment, *, complete):
        observed.append((len(scores), complete))
        return original(scores, experiment, complete=complete)

    monkeypatch.setattr(integration, "_outcomes", capture)
    log = _run(_task(delay=0.35), tmp_path / "logs", max_samples=1, score_display=True)
    assert any(0 < count < 4 and not final for count, final in observed)
    assert validate_bayes_at_n(log, ["q1", "q2"], 2)["posterior_mean"] == pytest.approx(
        0.5
    )


def test_one_epoch_and_selected_subset_work_through_eval(tmp_path) -> None:
    task = _task(epochs=1, ids=["q1", "q2", "q3"])
    task.metrics = [bayes_at_n(sample_ids=["q2"], epochs=1)]
    log = _run(task, tmp_path / "logs", sample_id="q2")
    assert validate_bayes_at_n(log, ["q2"], 1) == {
        "posterior_mean": pytest.approx(2 / 3),
        "posterior_sd": pytest.approx(math.sqrt(1 / 18)),
    }
    task = _task(ids=["q1", "q2", "q3"])
    task.metrics = [bayes_at_n(sample_ids=["q1"], epochs=2)]
    limited = _run(task, tmp_path / "limited", limit=1)
    assert validate_bayes_at_n(limited, ["q1"], 2)["posterior_mean"] == pytest.approx(
        0.25
    )


@pytest.mark.parametrize("missing", ["one", "all", "question"])
def test_nan_filtering_and_missing_question_cannot_become_a_posterior(
    complete_log, missing
) -> None:
    log = complete_log
    if missing == "question":
        log.samples = [sample for sample in log.samples if sample.id != "q1"]
    else:
        for sample in log.samples if missing == "all" else log.samples[:1]:
            sample.scores["record_epochs"].value = math.nan
    recompute_metrics(log, trust_log=True)
    assert all(math.isnan(value) for value in _posterior(log).values())
    with pytest.raises(ValueError):
        validate_bayes_at_n(log, ["q1", "q2"], 2)


def test_declared_cohort_cannot_be_shrunk_after_recompute(complete_log) -> None:
    log = complete_log
    log.samples = [sample for sample in log.samples if sample.id == "q2"]
    recompute_metrics(log, trust_log=True)
    with pytest.raises(ValueError, match="selected dataset IDs"):
        validate_bayes_at_n(log, ["q2"], 2)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "stale",
        "legacy",
        "error",
        "log_error",
        "no_score",
        "header",
        "status",
        "partial_count",
        "invalidated",
        "adaptive",
        "shared_cache",
        "changed_generation",
    ],
)
def test_final_log_validation_rejects_invalid_experiments(complete_log, damage) -> None:
    log = complete_log
    if damage == "duplicate":
        log.samples[-1] = log.samples[0].model_copy(deep=True)
    elif damage == "stale":
        log.samples[0].scores["record_epochs"].metadata["scorio.bayes_at_n"][
            "epoch"
        ] = 99
    elif damage == "legacy":
        log.samples[0].scores["record_epochs"].metadata = None
    elif damage == "error":
        log.samples[0].error = EvalError(
            message="failed", traceback="", traceback_ansi=""
        )
    elif damage == "log_error":
        log.error = EvalError(message="failed", traceback="", traceback_ansi="")
    elif damage == "no_score":
        log.samples[0].scores = {}
    elif damage == "header":
        log.samples = None
    elif damage == "status":
        log.status = "cancelled"
    elif damage == "partial_count":
        log.results.total_samples += 1
    elif damage == "invalidated":
        log.invalidated = True
    elif damage == "adaptive":
        data = log.model_dump()
        data["results"]["early_stopping"] = {
            "manager": "fixture",
            "early_stops": [],
            "metadata": {},
        }
        log = EvalLog.model_validate(data)
    elif damage == "shared_cache":
        log.plan.config.cache = CachePolicy(per_epoch=False)
    elif damage == "changed_generation":
        data = log.model_dump()
        data["config_updates"] = [
            {
                "scope": "task",
                "changes": [
                    {"config": "generate", "name": "temperature", "value": 0.9}
                ],
                "provenance": {"author": "test"},
            }
        ]
        log = EvalLog.model_validate(data)
    with pytest.raises(ValueError):
        validate_bayes_at_n(log, ["q1", "q2"], 2)


def test_per_epoch_cache_and_explicit_scorer_selection(complete_log) -> None:
    log = complete_log
    log.plan.config.cache = CachePolicy(per_epoch=True)
    for sample in log.samples:
        sample.scores["chosen"] = sample.scores.pop("record_epochs")
        sample.scores["unrelated"] = Score(value="P")
    assert validate_bayes_at_n(log, ["q1", "q2"], 2, scorer="chosen")[
        "posterior_mean"
    ] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Missing scorer"):
        validate_bayes_at_n(log, ["q1", "q2"], 2)


def test_public_registry_save_load_recompute_and_scorer_replay(tmp_path) -> None:
    path = tmp_path / "roundtrip.eval"
    create = """
import json, sys
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate
from inspect_ai.util import registry_info
from scorio.integrations.inspect import bayes_at_n, record_epochs, validate_bayes_at_n
output = ModelOutput.from_content(model='mockllm', content='answer')
output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
wrapped = record_epochs(match())
assert registry_info(wrapped).name == 'scorio/record_epochs'
task = Task(
    dataset=[Sample(id='q1', input='question', target='answer')],
    solver=generate(), scorer=wrapped,
    metrics=[bayes_at_n(sample_ids=['q1'], epochs=2)], epochs=2,
)
model = get_model('mockllm/model', custom_outputs=[output.model_copy(deep=True) for _ in range(2)])
log = eval(task, model=model, display='none', log_dir=sys.argv[2], ctl_server=False, acp_server=False)[0]
assert log.status == 'success', log.error
write_eval_log(log, sys.argv[1])
loaded = read_eval_log(sys.argv[1])
assert loaded.eval.metrics[0].name == 'scorio/bayes_at_n'
assert loaded.eval.metrics[0].options == {'sample_ids': ['q1'], 'epochs': 2}
assert loaded.eval.scorers[0].name == 'record_epochs'
assert 'match' in json.dumps(loaded.eval.scorers[0].options)
expected = validate_bayes_at_n(log, ['q1'], 2)
assert validate_bayes_at_n(loaded, ['q1'], 2) == expected
stored = [s for s in loaded.results.scores if 'posterior_mean' in s.metrics]
assert len(stored) == 1 and stored[0].reducer is None
assert {k: v.value for k, v in stored[0].metrics.items()} == expected
"""
    created = subprocess.run(
        [sys.executable, "-I", "-c", create, str(path), str(tmp_path / "logs")],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=60,
    )
    assert created.returncode == 0, created.stderr
    script = """
import json, sys
from inspect_ai import score
from inspect_ai.log import read_eval_log, recompute_metrics
from inspect_ai.util import registry_create, registry_info
assert 'scorio.integrations.inspect' not in sys.modules
log = read_eval_log(sys.argv[1])
recompute_metrics(log)
from scorio.integrations.inspect import validate_bayes_at_n
first = validate_bayes_at_n(log, ['q1'], 2)
stored = [s for s in log.results.scores if 'posterior_mean' in s.metrics]
assert len(stored) == 1 and stored[0].reducer is None
assert {k: v.value for k, v in stored[0].metrics.items()} == first
restored = registry_create('scorer', 'scorio/record_epochs', **log.eval.scorers[0].options)
assert registry_info(restored).name == 'scorio/record_epochs'
rescored = score(log, restored, action='overwrite', display='none')
second = validate_bayes_at_n(rescored, ['q1'], 2)
assert first == second
stored = [s for s in rescored.results.scores if 'posterior_mean' in s.metrics]
assert len(stored) == 1 and stored[0].reducer is None
assert {k: v.value for k, v in stored[0].metrics.items()} == second
print(json.dumps(second))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "posterior_mean": pytest.approx(0.75),
        "posterior_sd": pytest.approx(math.sqrt(3 / 80)),
    }
