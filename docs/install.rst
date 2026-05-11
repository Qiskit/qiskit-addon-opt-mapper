Installation instructions
=========================

There are two primary ways to run and install the packages:

- :ref:`Option 1`
- :ref:`Option 2`

Pre-installation
^^^^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

Note: If you are using Windows, use the following commands in PowerShell:

.. code:: pwsh

    python3 -m venv c:\path\to\virtual\environment
    c:\path\to\virtual\environment\Scripts\Activate.ps1


.. _Option 1:

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``qiskit-addon-opt-mapper`` package is via ``PyPI``.

.. code:: sh

    pip install 'qiskit-addon-opt-mapper'


.. _Option 2:

Option 2: Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop in the repository or run the notebooks locally, install from source.

The first step is to clone the ``qiskit-addon-opt-mapper`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-opt-mapper.git

Next, upgrade ``pip`` and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-addon-opt-mapper

The next step is to install ``qiskit-addon-opt-mapper`` to the virtual environment. If you plan on running the notebooks, install the
notebook dependencies in order to run all the visualizations in the notebooks. If you plan on developing in the repository, you
should also install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab