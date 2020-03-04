from enum import Enum

from typing import Callable, List

import PySimpleGUI as sg
from PyQt5 import QtWidgets, QtCore

import trainer.lib as lib


class ClassSelectionLevel(Enum):
    SubjectLevel = "Subject Level"
    BinaryLevel = "Binary Level"
    FrameLevel = "Frame Level"


class TClassBox(QtWidgets.QWidget):
    """
    Takes the information of one class.
    """

    def __init__(self, class_info: lib.ClassDefinition, callback: Callable[[str, str], None]):
        super().__init__()
        self.f_changed = callback
        self.class_name = class_info.name

        self.label = QtWidgets.QLabel(f"Set Class: {self.class_name}")

        self.selector = QtWidgets.QComboBox()
        self.selector.addItem("--Removed--")
        self.selector.addItems(class_info.values)
        # noinspection PyUnresolvedReferences
        self.selector.activated[str].connect(self.selection_changed)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.selector)
        self.setLayout(self.layout)

    def selection_changed(self, text):
        self.f_changed(self.class_name, text)

    def update_values(self, cls_target: lib.Classifiable):
        index = self.selector.findText(cls_target.get_class(self.class_name),
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
        self.subject, self.imstack, self.frame_number = None, None, -1
        self.selection_level = ClassSelectionLevel.SubjectLevel

        self.label = QtWidgets.QLabel("No Selected Subject")

        self.level_selector = QtWidgets.QComboBox()
        self.level_selector.addItems([selection_level.value for selection_level in ClassSelectionLevel])
        # noinspection PyUnresolvedReferences
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

    def configure_selection(self, cls_tpls: List[lib.ClassDefinition]):
        # class_names = [c.name for c in cls_tpls]

        def change_handler(class_name: str, class_value: str):
            print(class_name)
            print(class_value)
            if class_value == '--Removed--':
                print(class_value)
                if self.selection_level == ClassSelectionLevel.BinaryLevel:
                    self.imstack.remove_class(class_name)
                elif self.selection_level == ClassSelectionLevel.SubjectLevel:
                    self.subject.remove_class(class_name)
            else:
                print(f'{class_value} ELSE')
                if self.selection_level == ClassSelectionLevel.BinaryLevel:
                    self.imstack.set_class(class_name, class_value)
                elif self.selection_level == ClassSelectionLevel.SubjectLevel:
                    self.subject.set_class(class_name, class_value)

        for cls_def in cls_tpls:
            # class_info = d.get_class(class_name)
            class_box = TClassBox(cls_def, callback=change_handler)
            self.class_boxes.append(class_box)
            self.layout.addWidget(class_box)

    def set_subject(self, subject: lib.Subject):
        self.subject = subject
        if self.selection_level == ClassSelectionLevel.SubjectLevel:
            self.update_label()

    def set_img_stack(self, imstack: lib.ImStack):
        self.imstack = imstack
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
            if self.selection_level == ClassSelectionLevel.SubjectLevel:
                box.update_values(self.subject)
            elif self.selection_level == ClassSelectionLevel.BinaryLevel:
                box.update_values(self.imstack)
