==================
Command Line Tools
==================

The following commands can be used to find help yourself.
Just using the top-level command without any arguments will return an overview over all commands:

.. code-block:: shell

    $ trainer

The help-switch can be used to gain specific insights into one command:

.. code-block:: shell

    $ trainer [command] --help

Resetting the database
======================

The command removes all content from the database and the big-binary folder on disk (if it exists).

.. code-block:: shell

    $ trainer reset-database

Importing data from a folder exported with trainer
==================================================

At first create the dataset that the data should be appended to:

.. code-block:: shell

    $ trainer init-dataset



.. code-block:: shell

