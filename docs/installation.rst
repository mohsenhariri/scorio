Installation
============

Requirements
------------

- Python 3.9 - 3.13
- NumPy 2.0+
- SciPy

Install from PyPI
-----------------

The easiest way to install scorio is from PyPI:

.. code-block:: bash

   pip install scorio

Install from Source
-------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/mohsenhariri/scorio.git
   cd scorio
   pip install -e .

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This installs:

- pytest (testing)
- black (code formatting)
- isort (import sorting)
- mypy (type checking)

Verify Installation
-------------------

Verify your installation by importing the package:

.. code-block:: python

   import scorio
   print(scorio.__version__)
