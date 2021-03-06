============
Contribution
============

Docs
----

Currently, https://readthedocs.org/ is used for CI of the docs.
Before submitting changes, test the make command in a separate environment:

.. include:: rebuild_docs.sh
   :code: bash

.. code-block:: bash

    conda env create -f environment.yml
    conda activate trainer_env

    # Remove the old auto-generated docs
    rm -rf ./documentation_source/modules

    # Generating docs from code
    sphinx-apidoc.exe ./trainer/ -o ./documentation_source/modules

    # Compiling the docs
    make html

If this throws warnings or errors, https://readthedocs.org/ won't be able to publish the docs.

The following list summarizes a few helpful tips for writing docs.

- `Helpful markup resource <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_
- Inline Math example: :math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}`
- Math example:

.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}



Writing Tests
-------------

- Inline-tests (those defined inside docstrings) should not have side-effects.


Tutorials inside the repo
-------------------------

- Do not use jupyter notebooks because of problems with git versioning
- Should be testable without preparing data by hand where possible.

Uploading to PyPi by hand
-------------------------

.. code-block:: bash

    python setup.py sdist bdist_wheel
    twine upload dist/* # The asterisk is important

Development Web GUI
-------------------

The next version will make a shot at rapid programming using `Flexx <https://flexx.readthedocs.io/en/stable/>`_.

Install the debugger extension for your browser of choice, we recommend Microsoft Edge:
https://marketplace.visualstudio.com/items?itemName=msjsdiag.debugger-for-edge
