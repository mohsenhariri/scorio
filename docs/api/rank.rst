scorio.rank
===========

.. automodule:: scorio.rank
   :no-members:

Notation
--------

All ranking methods use response tensors :math:`R \in \{0,1\}^{L \times M \times N}`,
with :math:`L` models, :math:`M` questions, and :math:`N` trials per question.
Methods compute raw scores :math:`s_l` and then convert scores to ranks with
``scorio.utils.rank_scores``.

Prior Classes
-------------

.. automodule:: scorio.rank.priors
   :members:
   :show-inheritance:

Evaluation-based Ranking Methods
--------------------------------

.. automodule:: scorio.rank.eval_ranking
   :members:
   :show-inheritance:

Pointwise Methods
-----------------

.. automodule:: scorio.rank.pointwise
   :members:
   :show-inheritance:

Rank Centrality Methods
-----------------------

.. automodule:: scorio.rank.rank_centrality
   :members:
   :show-inheritance:

HodgeRank
---------

.. automodule:: scorio.rank.hodge_rank
   :members:
   :show-inheritance:

Serial Rank
-----------

.. automodule:: scorio.rank.serial_rank
   :members:
   :show-inheritance:


Graph-based Methods
-------------------

.. automodule:: scorio.rank.graph
   :members:
   :show-inheritance:


Pairwise Methods
----------------

.. automodule:: scorio.rank.pairwise
   :members:
   :show-inheritance:

Paired-Comparison Probabilistic Models
--------------------------------------

.. automodule:: scorio.rank.bradley_terry
   :members:
   :show-inheritance:

Bayesian Ranking Methods
------------------------

.. automodule:: scorio.rank.bayesian
   :members:
   :show-inheritance:

Item Response Theory Methods
----------------------------

.. automodule:: scorio.rank.irt
   :members:
   :show-inheritance:


Voting Methods
--------------

.. automodule:: scorio.rank.voting
   :members:
   :show-inheritance:

Listwise and Setwise Choice Models
----------------------------------

.. automodule:: scorio.rank.listwise
   :members:
   :show-inheritance:
