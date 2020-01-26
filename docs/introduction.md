# The Mission

<!-- What task does trainer solve? -->
There are still tasks that can be faster explained to a human than to a computer.
This project came to life for the purpose of solving that. 
Using trainer you can train artificial brains by stating a set of inputs and either your desired output or a combination of output and easily to define constraints.

<!-- Motivation -->
Take for example the following traditional medical multi-class classification problem:
Given one or more ultrasound videos, an xray and a textual description of the patient's symptoms,
predict the medical decision of care.
From an engineering perspective, this is a difficult task.
With trainer, tasks that involve decisions given numerous different inputs,
should be greatly simplified.

<!-- Dataset Format -->
Furthermore, trainer defines a convenient, subject-centered dataset format.
In the medical case, a subject can be a patient.
A patient can have multiple different studies
(Simple attributes, CT 3D data, ultrasound videos, wholeslide images),
which should all be usable by a trained AI.
