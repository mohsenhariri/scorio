Inspect AI
==========

Scorio provides an optional `Inspect AI <https://inspect.aisi.org.uk/>`_
extension for Bayes@N evaluation of repeated binary scores. It reports the
posterior mean and posterior standard deviation over a fixed set of questions.
The uncertainty is conditional on those questions; Inspect's population
standard error remains a separate quantity.

Installation
------------

The extension requires Python 3.11 or later and Inspect AI 0.3.288 or later
within the 0.3 series:

.. code-block:: bash

   pip install "scorio[inspect]"

The extra also requires ``agent-client-protocol<1`` for compatibility with
Inspect 0.3.

Installing Scorio without the extra retains its Python 3.10 support. Inspect
discovers the extension through Scorio's ``inspect_ai`` package entry point.
Registered component names are ``scorio/bayes_at_n`` and
``scorio/record_epochs``.

Evaluate a complete run
-----------------------

Declare the expected question IDs and number of epochs before evaluation.
Wrap the binary scorer with ``record_epochs`` and attach ``bayes_at_n`` as
a task metric. After evaluation, call ``validate_bayes_at_n`` to check the
finished run and obtain its posterior moments.

This example uses Inspect's local mock model and makes no model API request:

.. code-block:: python

   from inspect_ai import Task, eval
   from inspect_ai.dataset import Sample
   from inspect_ai.model import ModelOutput, ModelUsage, get_model
   from inspect_ai.scorer import match
   from inspect_ai.solver import generate

   from scorio.integrations.inspect import (
       bayes_at_n,
       record_epochs,
       validate_bayes_at_n,
   )

   sample_ids = ["q1", "q2"]
   epochs = 2
   task = Task(
       dataset=[
           Sample(id="q1", input="What is 2 + 2?", target="4"),
           Sample(id="q2", input="What is 3 + 1?", target="4"),
       ],
       solver=generate(),
       scorer=record_epochs(match()),
       metrics=[bayes_at_n(sample_ids=sample_ids, epochs=epochs)],
       epochs=epochs,
   )
   outputs = [
       ModelOutput.from_content(model="mockllm", content="4")
       for _ in range(len(sample_ids) * epochs)
   ]
   for output in outputs:
       output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
   model = get_model("mockllm/model", custom_outputs=outputs)
   log = eval(task, model=model, display="none")[0]
   moments = validate_bayes_at_n(log, sample_ids=sample_ids, epochs=epochs)
   print(moments)

Both questions receive two correct scores, giving a posterior mean of
``0.75`` and posterior standard deviation of approximately ``0.13693``.
The result mapping uses ``posterior_mean`` and ``posterior_sd``.

The final validation call is part of the supported workflow. Inspect also
computes metrics during live progress, when the declared cohort is usually
incomplete. At that stage, the metric returns unavailable values (``NaN``).
Inspect can also return ``NaN`` for entirely empty score sets. Final validation
raises for incomplete or invalid runs instead of treating those values as
a posterior estimate.

Input scope
-----------

The first version supports Inspect's literal ``C`` and ``I`` scores, with
one score for every declared question and epoch. It uses the same binary
plus-one prior as :func:`scorio.eval.bayes`. It does not convert partial credit,
continuous rewards, refusals, lists or dictionaries into binary scores.

``record_epochs`` adds epoch identity to the score metadata. The metric checks
that identities belong to the declared question/epoch set and are unique.
The final validator checks the raw log, including scores that Inspect may
have omitted from metric inputs. Keep the complete log and use the same
sample IDs and epoch count when validating it.

Use a fixed model, question text, grader and generation configuration with
comparable independent generations. The adapter cannot establish independence
or inspect arbitrary callback behavior. Adaptive stopping, selective omission
of outcomes and incomplete experiments are outside this integration's scope.
The final validator also rejects recorded generation-setting changes and
explicit caches shared across epochs (``CachePolicy(per_epoch=False)``).
Inspect's normal per-epoch caching is supported; custom hidden caching remains
the caller's responsibility.
Categorical rubrics, prior observations and credible intervals remain available
through Scorio's evaluation API but are not exposed by this adapter.

Saved logs and recomputation
----------------------------

Scorio must be installed with the Inspect extra when recomputing registered
metrics. For the locally generated log above:

.. code-block:: python

   from inspect_ai.log import read_eval_log, recompute_metrics

   saved = read_eval_log(log.location)
   validate_bayes_at_n(saved, sample_ids=sample_ids, epochs=epochs)
   recompute_metrics(saved, trust_log=True)
   moments = validate_bayes_at_n(saved, sample_ids=sample_ids, epochs=epochs)

The scorer wrapper and metric options are saved in the evaluation log.
Logs produced without ``record_epochs`` are not supported by this metric.

API
---

.. currentmodule:: scorio.integrations.inspect

.. py:function:: bayes_at_n(sample_ids, epochs)

   Create an unreduced Inspect metric for a fixed set of sample IDs and a
   positive epoch count. It returns ``posterior_mean`` and ``posterior_sd``
   after the complete question/epoch set is present. Incomplete progress
   returns ``NaN`` for both fields.

.. py:function:: record_epochs(scorer)

   Wrap an Inspect scorer to preserve each score's epoch in metadata.
   Attach the Bayes@N metric through ``Task(metrics=[...])``.

.. py:function:: validate_bayes_at_n(log, sample_ids, epochs, *, scorer="record_epochs")

   Validate a finished evaluation log and return its posterior moments.
   ``scorer`` selects the per-sample score key when the log contains
   several scorers. Invalid or incomplete input raises ``ValueError``.

The estimator is described in `Don't Pass@k: A Bayesian Framework for Large
Language Model Evaluation <https://arxiv.org/abs/2510.04265>`_.
