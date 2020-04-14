============
Contribution
============

.. Docs

Currently, https://readthedocs.org/ is used
for CI of the docs.
Before submitting changes, test the make command in the environment:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate trainer_env

    # Generating docs from code
    sphinx-apidoc.exe ./trainer/ -o ./documentation_source/modules

    # Compiling the docs
    make html

If this throws warnings or errors, https://readthedocs.org/ won`t be able to publish the docs.

.. Tutorials inside the repo

- Do not use jupyter notebooks
- Should be testable without preparing data by hand where possible.


.. Uploading to PyPi by hand

.. code-block:: bash

    python setup.py sdist bdist_wheel
    twine upload dist/* # The asterisk is important

.. Development Web GUI

Install the debugger extension for your browser of choice, we recommend Microsoft Edge:
https://marketplace.visualstudio.com/items?itemName=msjsdiag.debugger-for-edge