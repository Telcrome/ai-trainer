from typing import Callable

import PySimpleGUI as sg
from PyQt5 import QtWidgets, QtCore

from trainer.lib import ClassSelectionLevel
from trainer.ml import Dataset, Subject


class TClassBox(QtWidgets.QWidget):
    """
    Takes the information of one class.
    """

    def __init__(self, class_info, callback: Callable[[str, str], None], class_name=None):
        super().__init__()
        self.f_changed = callback
        self.class_name = class_name

        self.label = QtWidgets.QLabel(f"Set Class: {class_name}")

        self.selector = QtWidgets.QComboBox()
        self.selector.addItem("--Removed--")
        self.selector.addItems(class_info['values'])
        self.selector.activated[str].connect(self.selection_changed)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.selector)
        self.setLayout(self.layout)

    def selection_changed(self, text):
        self.f_changed(self.class_name, text)

    def update_values(self, s: Subject, for_binary=""):
        index = self.selector.findText(s.get_class_value(self.class_name, for_binary=for_binary),
                                       QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.selector.setCurrentIndex(index)


class TClassSelector(QtWidgets.QWidget):
    """
    Container for multiple classes, instantiates a TClassBox for every class
    """

    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.class_boxes = []
        self.subject, self.binary_name, self.frame_number = None, '', -1
        self.selection_level = ClassSelectionLevel.SubjectLevel

        self.label = QtWidgets.QLabel("No Selected Subject")

        self.level_selector = QtWidgets.QComboBox()
        self.level_selector.addItems([selection_level.value for selection_level in ClassSelectionLevel])
        self.level_selector.activated[str].connect(self.selection_level_changed)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.level_selector)
        self.setLayout(self.layout)

    def selection_level_changed(self, text):
        next_selection_level = ClassSelectionLevel.SubjectLevel
        for selection_level in ClassSelectionLevel:
            if text == selection_level.value:
                next_selection_level = selection_level

        if next_selection_level == ClassSelectionLevel.SubjectLevel:
            pass
        elif next_selection_level == ClassSelectionLevel.BinaryLevel:
            pass
        elif next_selection_level == ClassSelectionLevel.FrameLevel:
            sg.popup("Frame level class selection not supported yet, go back and hope for the best")

        self.selection_level = next_selection_level
        self.update_label()

    def configure_selection(self, d: Dataset):
        class_names = d.get_class_names()

        def change_handler(class_name: str, class_value: str):
            print(class_name)
            print(class_value)
            if class_value == '--Removed--':
                if self.selection_level == ClassSelectionLevel.BinaryLevel:
                    self.subject.remove_class(class_name, for_binary=self.binary_name)
                elif self.selection_level == ClassSelectionLevel.SubjectLevel:
                    self.subject.remove_class(class_name)
            else:
                if self.selection_level == ClassSelectionLevel.BinaryLevel:
                    self.subject.set_class(class_name, class_value, for_dataset=d, for_binary=self.binary_name)
                elif self.selection_level == ClassSelectionLevel.SubjectLevel:
                    self.subject.set_class(class_name, class_value, for_dataset=d)

        for class_name in class_names:
            class_info = d.get_class(class_name)
            class_box = TClassBox(class_info, callback=change_handler, class_name=class_name)
            self.class_boxes.append(class_box)
            self.layout.addWidget(class_box)

    def set_subject(self, subject: Subject):
        self.subject = subject
        if self.selection_level == ClassSelectionLevel.SubjectLevel:
            self.update_label()

    def set_binary_name(self, binary_name: str):
        self.binary_name = binary_name
        if self.selection_level == ClassSelectionLevel.BinaryLevel:
            self.update_label()

    def set_frame_number(self, frame_number: int):
        self.frame_number = frame_number
        if self.selection_level == ClassSelectionLevel.FrameLevel:
            self.update_label()

    def update_label(self):
        # Update the label to give feedback to the user
        res = f'Selected subject:\n{self.subject.name}\n'
        if self.selection_level == ClassSelectionLevel.BinaryLevel or self.selection_level == ClassSelectionLevel.FrameLevel:
            res += f'Selected binary:\n{self.binary_name}'
        if self.selection_level == ClassSelectionLevel.FrameLevel:
            res += f'Selected frame:\n{self.frame_number}'
        self.label.setText(res)
        # Update the values in the class boxes
        for box in self.class_boxes:
            if self.selection_level == ClassSelectionLevel.BinaryLevel:
                box.update_values(self.subject, for_binary=self.binary_name)
            elif self.selection_level == ClassSelectionLevel.SubjectLevel:
                box.update_values(self.subject)
