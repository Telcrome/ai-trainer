===============
Getting Started
===============

Trainer currently supports annotating images and videos.
To store data in the database:

1) Install postgresql (From conda or from _here: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)
2) Import data (either exported from trainer or from directories with raw image data)
3) Happy annotating and training using the CLI tools

Create a config json of the following form:

.. code-block:: json

    {
       "db_con": "postgresql+psycopg2://postgres:password@127.0.0.1:5432/db_name"
    }

Generator Usage Workflow
------------------------

Subject generator -> Preprocessing -> Model optimization
