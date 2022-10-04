Installation
============

Prerequisites
~~~~~~~~~~~~~~

`mrvi` can be installed via PyPI.

conda prerequisites
###################

1. Install Conda. We typically use the Miniconda_ Python distribution. Use Python version >=3.7.

2. Create a new conda environment::

    conda create -n mrvi-env python=3.7

3. Activate your environment::

    source activate mrvi-env

pip prerequisites:
##################

1. Install Python_, we prefer the `pyenv <https://github.com/pyenv/pyenv/>`_ version management system, along with `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv/>`_.

2. Install PyTorch_. If you have an Nvidia GPU, be sure to install a version of PyTorch that supports it -- scvi-tools runs much faster with a discrete GPU.

.. _Miniconda: https://conda.io/miniconda.html
.. _Python: https://www.python.org/downloads/
.. _PyTorch: http://pytorch.org

mrvi installation
~~~~~~~~~~~~~~~~~~~~~~~

Install mrvi in one of the following ways:

Through **pip**::

    pip install <mrvi>

Through pip with packages to run notebooks. This installs scanpy, etc.::

    pip install <mrvi>[tutorials]

Nightly version - clone this repo and run::

    pip install .

For development - clone this repo and run::

    pip install -e .[dev,docs]
