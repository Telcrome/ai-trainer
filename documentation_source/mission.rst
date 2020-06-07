===========
The Mission
===========

.. What task does trainer solve?

There are still tasks that can be faster explained to a human than to a computer.
This project came to life for the purpose of solving that.
Trainer allows to assemble AI without much experience in the field,
utilizing GUIs and simple python frameworks!

.. Motivation

Take for example the following traditional medical multi-class classification problem:
Given one or more ultrasound videos, an xray and a textual description of the patient's symptoms,
predict the medical decision of care.
From an engineering perspective, this is a difficult task.
With trainer, tasks that involve decisions given numerous different inputs,
should be greatly simplified.

Secondly, if only few datapoints are available it is desirable to encode inductive bias.
Tradionally, inductive bias in deep learning is encoded using data-augmentation.
While trainer allows to use such old, outdated and boring techniques as well,
we encourage to define inductive bias by defining

.. Dataset Format

Furthermore, trainer defines a convenient, subject-centered dataset format.
In the medical case, a subject can be a patient.
A patient can have multiple different studies
(Simple attributes, CT 3D data, ultrasound videos, wholeslide images),
which, in an ideal world, should all be usable by a trained AI.
