"""
Produces (smallish) datasets for testing the functionality of the annotator and the machine learning capabilities.

Uses synthetic data that uses tasks solvable by a human to enable simple demonstration of trainer functionality.

To fill your database with one of the datasets, just execute corresponding file.
The relevant code is always at the bottom in the __main__ statement.
"""
from trainer.demo_data.DemoDataset import get_test_logits, build_test_subject, DemoDataset
from trainer.demo_data.arc import ArcDataset
from trainer.demo_data.mnist import MnistDataset
