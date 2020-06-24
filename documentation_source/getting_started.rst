===============================
Getting Started With Annotation
===============================

Trainer currently supports annotating images and videos.
First we need to setup the database.
Using a database instead of simply storing files helps working with complex, structured data.

First, install postgresql (From conda or from `here: <https://www.enterprisedb.com/downloads/postgres-postgresql-downloads>`_).
Follow some guide from google to setup an empty database.
We recommend to test the connection using ``psql`` or ``pgadmin4`` now.

Execute some code that tries to use the database:

.. code-block:: bash

    python ./tutorials/db.py

In your user directory a new folder ``.trainer`` should pop up. Update the connection string ``db_con``.

.. code-block:: json

    {
       "big_bin_dir": "C:\\Users\\rapha\\.trainer\\big_bin",
       "db_con": "postgresql+psycopg2://postgres:admin@127.0.0.1:5432/trainer"
    }
    

Now, Import data using the CLI tools.
Data might be exported earlier from someone else or imported using several file formats from disk.
Here we assume you do not already have a dataset which was exported using trainer.

At this point, start the GUI to start annotating:

.. code-block:: bash

    trainer annotate -n testset

Generator Usage Workflow
------------------------

Subject generator -> Preprocessing -> Model optimization
