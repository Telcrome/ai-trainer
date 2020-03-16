============
Contribution
============

.. Docs

Currently, [Read the Docs](https://readthedocs.org/) is used
for CI of the docs.
Before submitting changes, test the make command in the environment:
```shell script
conda env create -f environment.yml
conda activate trainer_env
make html
```
If this throws warnings or errors, `Read the Docs` won`t publish them.

.. Tutorials inside the repo

- Do not use jupyter notebooks
- Should be testable without preparing data by hand where possible.


.. Uploading to PyPi by hand

```shell script
python setup.py sdist bdist_wheel
twine upload dist/* # The asterisk is important
```