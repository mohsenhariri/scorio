"""Bayes@N for fixed-cohort binary evaluations in Inspect AI.

Install ``scorio[inspect]`` on Python 3.11 or later. Wrap the task's scorer
with ``record_epochs`` and attach ``bayes_at_n`` through the task's metrics.
Call ``validate_bayes_at_n`` on the completed log before using the result.
"""

from importlib.metadata import version
from typing import Any

import numpy as np

from scorio.eval import bayes

try:
    from inspect_ai.log import EvalLog
    from inspect_ai.model import CachePolicy
    from inspect_ai.scorer import (
        CORRECT,
        INCORRECT,
        Metric,
        SampleScore,
        Score,
        Scorer,
        Target,
    )
    from inspect_ai.scorer import metric as inspect_metric
    from inspect_ai.scorer import scorer as inspect_scorer
    from inspect_ai.solver import TaskState
    from packaging.version import Version
except ModuleNotFoundError as exc:
    if exc.name in {"inspect_ai", "packaging"}:
        raise ImportError(
            "The Inspect AI integration requires Python 3.11 or later and "
            "the optional dependencies: pip install 'scorio[inspect]'."
        ) from exc
    raise

if not Version("0.3.288") <= Version(version("inspect-ai")) < Version("0.4"):
    raise ImportError(
        "The Inspect AI integration requires inspect-ai>=0.3.288,<0.4. "
        "Install compatible dependencies with pip install 'scorio[inspect]'."
    )

__all__ = ["bayes_at_n", "record_epochs", "validate_bayes_at_n"]

_EPOCH_KEY = "scorio.bayes_at_n"


class _Experiment:
    def __init__(self, sample_ids: list[str | int], epochs: int) -> None:
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError(
                "sample_ids must be a nonempty list of strings or integers"
            )
        if any(type(sample_id) not in (str, int) for sample_id in sample_ids):
            raise ValueError("sample_ids must contain only strings or integers")
        if len({str(sample_id) for sample_id in sample_ids}) != len(sample_ids):
            raise ValueError("sample_ids must be unique, including their string forms")
        self.sample_ids = tuple(sample_ids)
        self.rows = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        self.epochs = epochs

    def position(self, sample_id: str | int | None, epoch: Any) -> tuple[int, int]:
        if type(sample_id) not in (str, int) or sample_id not in self.rows:
            raise ValueError(f"Unexpected or missing sample ID: {sample_id!r}")
        if type(epoch) is not int or not 1 <= epoch <= self.epochs:
            raise ValueError(f"Unexpected epoch {epoch!r} for sample {sample_id!r}")
        return self.rows[sample_id], epoch - 1


def _epoch(score: Score) -> int:
    metadata = (score.metadata or {}).get(_EPOCH_KEY)
    if not isinstance(metadata, dict) or type(metadata.get("epoch")) is not int:
        raise ValueError(
            "Missing epoch metadata; use the record_epochs() scorer wrapper"
        )
    return metadata["epoch"]


def _outcomes(
    scores: list[SampleScore], experiment: _Experiment, *, complete: bool
) -> np.ndarray | None:
    outcomes = np.full((len(experiment.sample_ids), experiment.epochs), -1, dtype=int)
    for sample in scores:
        row, column = experiment.position(sample.sample_id, _epoch(sample.score))
        if outcomes[row, column] != -1:
            raise ValueError(
                f"Duplicate score for sample {sample.sample_id!r}, epoch {column + 1}"
            )
        value = sample.score.value
        if not isinstance(value, str) or value not in (CORRECT, INCORRECT):
            raise ValueError(
                f"Bayes@N requires C/I scores; sample {sample.sample_id!r}, "
                f"epoch {column + 1} has {value!r}"
            )
        outcomes[row, column] = int(value == CORRECT)
    missing = np.argwhere(outcomes == -1)
    if len(missing):
        if complete:
            row, column = missing[0]
            raise ValueError(
                f"Incomplete Bayes@N cohort: missing score for sample "
                f"{experiment.sample_ids[row]!r}, epoch {column + 1}"
            )
        return None
    return outcomes


def _moments(outcomes: np.ndarray) -> dict[str, float]:
    mean, sd = bayes(outcomes)
    return {"posterior_mean": mean, "posterior_sd": sd}


def _shared_cache(cache: Any) -> bool:
    if isinstance(cache, CachePolicy):
        return not cache.per_epoch
    return isinstance(cache, dict) and cache.get("per_epoch") is False


def _validate_log(log: EvalLog, experiment: _Experiment) -> None:
    if log.status != "success" or log.error is not None or log.invalidated:
        raise ValueError("Bayes@N requires a successful, non-invalidated completed log")
    if log.samples is None:
        raise ValueError("Load the log's samples before validating Bayes@N")
    configured_epochs = log.eval.config.epochs
    if (1 if configured_epochs is None else configured_epochs) != experiment.epochs:
        raise ValueError("The log's epoch count does not match the declared experiment")
    selected = log.eval.dataset.sample_ids
    if (
        selected is None
        or len(selected) != len(experiment.sample_ids)
        or set(selected) != set(experiment.sample_ids)
    ):
        raise ValueError("sample_ids do not match the log's selected dataset IDs")
    if log.results is None:
        raise ValueError("Bayes@N requires completed evaluation results")
    expected = len(experiment.sample_ids) * experiment.epochs
    if (
        log.results.total_samples != expected
        or log.results.completed_samples != expected
        or log.results.logged_samples not in (None, expected)
    ):
        raise ValueError(
            "The log's planned or completed count does not match the cohort"
        )
    if log.results.early_stopping is not None:
        raise ValueError("Bayes@N does not support adaptive stopping")
    if any(
        change.config == "generate"
        for update in log.config_updates or []
        for change in update.changes
    ):
        raise ValueError(
            "Bayes@N requires fixed generation settings throughout the run"
        )
    if any(
        _shared_cache(config.cache)
        for config in (log.eval.model_generate_config, log.plan.config)
    ) or any(
        _shared_cache(step.params.get("cache"))
        for step in [*log.plan.steps, *([log.plan.finish] if log.plan.finish else [])]
        if step.solver in ("generate", "inspect_ai/generate")
    ):
        raise ValueError("Bayes@N requires model caches to keep epochs separate")


@inspect_metric(scores="unreduced")
def bayes_at_n(sample_ids: list[str | int], epochs: int) -> Metric:
    """Return a Bayes@N metric for a declared set of binary trials.

    The prior is Beta(1, 1) independently for each question. The outputs are
    the posterior mean and posterior standard deviation of the average
    success probability over these questions. No prior observations are used.
    The caller must use a fixed model, prompt, grader, and generation settings
    with comparable independent generations. The metric cannot establish these
    properties from scores alone.

    Inspect also computes metrics during live progress. An incomplete cohort
    returns NaN for both outputs. Inspect filters unscored NaNs before calling
    metrics, so ``validate_bayes_at_n`` must check the final log separately.

    Args:
        sample_ids: Expected question IDs, after selecting any dataset subset.
            IDs must be unique strings or integers. Their string forms must
            also be unique, as required by Inspect.
        epochs: Expected number of independent generations per question.

    Returns:
        An unreduced Inspect metric with ``posterior_mean`` and
        ``posterior_sd`` outputs, both in units of binary correctness.

    Raises:
        ValueError: Factory arguments are invalid. During computation, also
            raised for non-C/I scores, unexpected IDs or epochs, duplicate
            trials, or missing epoch metadata.
    """
    experiment = _Experiment(sample_ids, epochs)

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        outcomes = _outcomes(scores, experiment, complete=False)
        if outcomes is None:
            return {"posterior_mean": float("nan"), "posterior_sd": float("nan")}
        return _moments(outcomes)

    return compute


@inspect_scorer(metrics=[])
def record_epochs(scorer: Scorer) -> Scorer:
    """Wrap an Inspect scorer to preserve epoch identity in score metadata.

    The wrapper copies each returned score and adds
    ``metadata["scorio.bayes_at_n"]["epoch"]``. Existing metadata is preserved.
    It carries no default metrics; attach ``bayes_at_n`` through task metrics.
    The wrapped scorer must be registered with Inspect for log replay.

    Args:
        scorer: Binary C/I scorer to call for each sample and epoch.

    Returns:
        A registered Inspect scorer that records the epoch on each score.

    Raises:
        ValueError: A score already uses the reserved metadata key.
    """

    async def score(state: TaskState, target: Target) -> Score | None:
        result = await scorer(state, target)
        if result is None:
            return None
        metadata = dict(result.metadata or {})
        if _EPOCH_KEY in metadata:
            raise ValueError(
                f"Score metadata already contains reserved key {_EPOCH_KEY!r}"
            )
        metadata[_EPOCH_KEY] = {"epoch": state.epoch}
        return result.model_copy(update={"metadata": metadata})

    return score


def validate_bayes_at_n(
    log: EvalLog,
    sample_ids: list[str | int],
    epochs: int,
    *,
    scorer: str = "record_epochs",
) -> dict[str, float]:
    """Validate a completed Inspect log and return its Bayes@N moments.

    This checks raw log entries, including unscored NaNs that Inspect removes
    before invoking metrics. Every expected sample must have one C/I score
    for each epoch, with matching metadata from ``record_epochs``. Errored,
    incomplete, and adaptively stopped runs are rejected. Recorded changes to
    generation settings and caches shared across epochs are also rejected.
    Custom callbacks and hidden caching remain the caller's responsibility.

    Args:
        log: Completed log with all samples loaded.
        sample_ids: Expected question IDs for the evaluated dataset subset.
        epochs: Expected number of independent generations per question.
        scorer: Key of the wrapped scorer in each sample's ``scores`` mapping.
            Use the suffixed name if several wrappers occur in one task.

    Returns:
        ``posterior_mean`` and ``posterior_sd`` from ``scorio.eval.bayes``.

    Raises:
        ValueError: The log or declared experiment violates the fixed-cohort
            binary contract, including scores without epoch metadata.
    """
    experiment = _Experiment(sample_ids, epochs)
    _validate_log(log, experiment)
    assert log.samples is not None
    if not isinstance(scorer, str) or not scorer:
        raise ValueError("scorer must name a score in each sample's scores mapping")

    scores = []
    for sample in log.samples:
        experiment.position(sample.id, sample.epoch)
        if sample.error is not None:
            raise ValueError(f"Sample {sample.id!r}, epoch {sample.epoch} has an error")
        if not sample.scores or scorer not in sample.scores:
            raise ValueError(
                f"Missing scorer {scorer!r} for sample {sample.id!r}, epoch {sample.epoch}"
            )
        score = sample.scores[scorer]
        if _epoch(score) != sample.epoch:
            raise ValueError(
                f"Epoch metadata disagrees with log for sample {sample.id!r}, "
                f"epoch {sample.epoch}"
            )
        scores.append(SampleScore(score=score, sample_id=sample.id))
    outcomes = _outcomes(scores, experiment, complete=True)
    assert outcomes is not None
    return _moments(outcomes)
