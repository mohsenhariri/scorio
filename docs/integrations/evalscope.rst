EvalScope
=========

EvalScope 1.14.1 added built-in Bayes@N aggregation for binary
``accuracy`` and ``exact_match`` scores.

|evalscope-notebook|

.. include:: notebooks.inc

.. code-block:: bash

   python -m pip install "evalscope>=1.14.1"

Bayes@N
-------

Choose one subset of a benchmark that uses EvalScope's standard aggregation.
Set ``repeats`` on ``TaskConfig`` and ``aggregation="bayes_at_n"`` in the
dataset's ``dataset_args`` entry. When evaluating your own model, draw
independent answers to each question with the same prompt and sampling
settings. Apply the same grading rule to every answer. Each question needs
one binary grade per repeat.

To run a local example, save these three questions as
``questions/quiz_val.jsonl``, one JSON object per line.

.. code-block:: json

   {"id":"planet","question":"Which planet is known as the Red Planet?","A":"Mars","B":"Venus","answer":"A"}
   {"id":"hexagon","question":"How many sides does a hexagon have?","A":"Five","B":"Six","answer":"B"}
   {"id":"current","question":"Which unit measures electric current?","A":"Ampere","B":"Volt","answer":"A"}

Run the following code from the directory containing ``questions/``. The mock
model answers B once, then A eleven times. A batch size of one keeps those
answers in order.

.. code-block:: python

   from evalscope import TaskConfig, run_task
   from evalscope.api.model import ModelOutput

   answers = ["B"] + ["A"] * 11
   outputs = [
       ModelOutput.from_content(model="quiz-model", content=answer)
       for answer in answers
   ]
   config = TaskConfig(
       model="quiz-model",
       eval_type="mock_llm",
       model_args={"custom_outputs": outputs},
       datasets=["general_mcq"],
       dataset_args={
           "general_mcq": {
               "local_path": "questions",
               "subset_list": ["quiz"],
               "aggregation": "bayes_at_n",
           }
       },
       repeats=4,
       eval_batch_size=1,
       judge={"strategy": "rule"},
   )
   report = run_task(config)["general_mcq"]

Read the report
~~~~~~~~~~~~~~~

.. code-block:: python

   for metric in report.metrics:
       print(metric.identity.aggregation, metric.score)

.. code-block:: text

   bayes_at_n 0.5556
   bayes_at_n_sd 0.0891

The three questions had 3, 0 and 4 correct answers out of four. With a
uniform Beta(1, 1) prior per question, their posterior mean success
probabilities are 4/6, 1/6 and 5/6. Their average, 55.56%, is the quiz's
posterior mean (``bayes_at_n``). The posterior standard deviation for the
quiz mean (``bayes_at_n_sd``) is about 8.91 percentage points.

In both rows, ``metric.num`` counts the three questions and
``metric.identity.dimensions["n"]`` records the four repeats per question.
The notebook shows how to recover the full-precision values from the subset
metadata. See also the `EvalScope evaluation FAQ
<https://evalscope.readthedocs.io/en/v1.14.1/get_started/faq.html#evaluation-configuration-parameters>`_.
