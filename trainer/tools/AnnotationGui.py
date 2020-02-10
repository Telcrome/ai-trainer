"""
The annotation tools takes a subject and allows manual annotation.
For manual annotation the source binary that should be annotated is required.

If a dataset is provided it can be used to select the next subject for annotation,
enabling a conveyor belt like process.
Everything else required should be stored in the subject.
"""

import os
import random
import sys
from typing import Tuple, List, Callable

import PySimpleGUI as sg
import cv2
import imageio
import numpy as np
from PyQt5 import QtWidgets, QtCore

import trainer.lib as lib
from trainer.lib import standalone_foldergrab
from trainer.lib.tgui import TClassSelector, TWindow, run_window, Brushes, SegToolController


def binary_filter(x, for_name, frame_number):
    a = x['binary_type'] == 'img_mask'
    return a and x['meta_data']['mask_of'] == for_name and x['meta_data']['frame_number'] == frame_number


class FrameController(QtWidgets.QWidget):

    def __init__(self, frame_changer: Callable[[int], None]):
        super(FrameController, self).__init__()
        self._current_frame, self._frame_number = -1, -1
        self.frame_changer = frame_changer

        self._layout = QtWidgets.QVBoxLayout()

        self._label = QtWidgets.QLabel("0/0")

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.sliderReleased.connect(self.slider_changed)

        self._layout.addWidget(self._label)
        self._layout.addWidget(self._slider)
        self.setLayout(self._layout)

    def slider_changed(self):
        v = self._slider.value()
        self.set_frame_number(v, self._frame_number)
        self.frame_changer(v)

    def set_frame_number(self, i: int, n: int):
        self._current_frame, self._frame_number = i, n
        self._label.setText(f"{self._current_frame}/{self._frame_number}")
        self._slider.setValue(self._current_frame)
        self._slider.setMaximum(self._frame_number)


class AnnotationGui(TWindow):
    """
    Opens a tools window that can be used to fill one subject with values.
    """

    def __init__(self, te_path: str, d_path: str = ""):
        if d_path:
            self.d = lib.Dataset.from_disk(d_path)
        self.current_subject: lib.Subject = None
        self.seg_structs: List = None
        self.frame_number, self.brush, self._struct_index = 0, Brushes.Standard, -1
        self.img_data, self.mask_data, self.mask_name, self._made_changes = None, None, '', False
        self._selected_source_binary, self._selected_gt_binary, self._struct_name = None, None, None

        self.seg_tool = SegToolController(self.selection_event_handler)

        self.main_view = self.seg_tool.get_graphics_scene()
        super().__init__(self.main_view, title="Annotation Window")

        super().init_ui(actions={
            "Subject": [
                super().create_action_from_tuple("Save",
                                                 "Ctrl+S",
                                                 "Saves the currently opened subject on disk",
                                                 self.save_to_disk),
                super().create_action_from_tuple("Add Image Data from Disk",
                                                 "Ctrl+A",
                                                 "Allows to add standard image and video files",
                                                 self.add_image_data_from_disk),
                super().create_action_from_tuple("Add DICOM image data",
                                                 "Ctrl+Shift+A",
                                                 "Note that this assumes a DICOM file with image data",
                                                 self.add_dicom_image_data),
                super().create_action_from_tuple("Delete Selected Binaries",
                                                 "Ctrl+D",
                                                 'Deletes the selected binaries from the source list view',
                                                 self.delete_selected_binaries),
                super().create_action_from_tuple("Delete currently displayed mask",
                                                 "Ctrl+Shift+D",
                                                 'Deletes the selected binaries from the source list view',
                                                 self.delete_selected_mask)
            ],
            "Navigation": [
                super().create_action_from_tuple("Annotate next example",
                                                 "Ctrl+Right",
                                                 "Selects the next example by rules if possible, if not randomly",
                                                 self.select_next_subject),
                super().create_action_from_tuple("Pick next example for annotation",
                                                 "Ctrl+Shift+Right",
                                                 "Opens a subject picker that allows to pick the subject by name",
                                                 self.pick_next_subject),
                super().create_action_from_tuple("Increment displayed frame",
                                                 "Right",
                                                 "Increment displayed frame",
                                                 lambda: self.change_frame(self.frame_number + 1)),
                super().create_action_from_tuple("Decrement displayed frame",
                                                 "Left",
                                                 "Decrement displayed frame",
                                                 lambda: self.change_frame(self.frame_number - 1))
            ],
            "Debug": [
                super().create_action_from_tuple("Inspect dir on Disk",
                                                 "Ctrl+I",
                                                 "Open the directory of the subject on disk",
                                                 self.inspect_on_disk),
                super().create_action_from_tuple("Inspect json on Disk",
                                                 "Ctrl+Shift+I",
                                                 "Open the json file of the subject on disk",
                                                 lambda: self.inspect_on_disk(open_file=True)),
                super().create_action_from_tuple("Update Log",
                                                 "Ctrl+L",
                                                 "Shows the actually loaded json content in the logging tool",
                                                 self.update_log),
                super().create_action_from_tuple("Find frames with annotation",
                                                 "Ctrl+F",
                                                 "Computes the frames with annotations",
                                                 self.find_annotations),
                super().create_action_from_tuple("Ask Model",
                                                 "Ctrl+M",
                                                 "Ask the model",
                                                 self.ask_model)
            ]
        })
        # Frame Controller
        self.frame_controller = FrameController(self.change_frame)
        self.content_grid.add_tool(self.frame_controller)

        # Class selector
        self.class_selector = TClassSelector()
        self.class_selector.configure_selection(self.d)
        self.content_grid.add_tool(self.class_selector)

        # Source binary selector
        self.content_grid.add_tool(QtWidgets.QLabel("Source Binary Selection:"))
        self.lst_source_binaries = QtWidgets.QListWidget()
        self.lst_source_binaries.currentItemChanged.connect(self.lst_src_binaries_changed)
        self.content_grid.add_tool(self.lst_source_binaries)

        # Mask Selector
        self.content_grid.add_tool(QtWidgets.QLabel("Mask Selection:"))
        self.lst_gt_binaries = QtWidgets.QListWidget()
        self.lst_gt_binaries.currentItemChanged.connect(self.lst_gt_binaries_changed)
        self.content_grid.add_tool(self.lst_gt_binaries)

        self.set_current_subject(lib.Subject.from_disk(te_path))

        self.console.push_to_ipython({'gui': self, 'dataset': self.d})

    def ask_model(self):
        raise NotImplementedError()
        # from trainer.ml.predictor import predict
        # print("The model was consulted")
        # channels = self.img_data.shape[-1]
        # if self.img_data.shape[-1] == 3:
        #     gray_image = cv2.cvtColor(self.img_data[self.frame_number, :, :, :], cv2.COLOR_BGR2GRAY)
        # else:
        #     gray_image = self.img_data[self.frame_number, :, :, :]
        # struct_name = self.seg_structs[self._struct_index]
        # res = predict(gray_image, self.d, struct_name)
        #
        # # If no mask exists yet, create a new
        # self.create_new_mask()
        #
        # self.mask_data[:, :, self._struct_index] = res > 0.3
        # self.update()

    def select_next_subject(self):
        self.save_to_disk()

        next_subject_name = random.choice(self.d.get_subject_name_list())
        self.set_current_subject(lib.Subject.from_disk(os.path.join(self.d.get_working_directory(), next_subject_name)))
        print("Selecting next subject")

    def pick_next_subject(self):
        self.save_to_disk()

        tool_width = 20
        tools = [
            [sg.Button(button_text="Annotate subject", key='open_subject', size=(tool_width, 1))]
        ]
        main_layout = [
            [sg.Text(text="Pick subject for annotation", size=(50, 1), key='lbl')],
            [sg.Listbox(key='ls_s',
                        values=[te_name for te_name in self.d.get_subject_name_list()],
                        size=(60, 20)),
             sg.Column(tools)]
        ]

        window = sg.Window("Pick your next subject", layout=main_layout)

        event, values = window.read()
        window.close()

        if event == "open_subject" and values['ls_s']:
            next_subject_name = values['ls_s'][0]
            self.set_current_subject(
                lib.Subject.from_disk(os.path.join(self.d.get_working_directory(), next_subject_name)))
            print(f"Selecting next subject: {next_subject_name}")

    def save_to_disk(self):
        print(f"Saving to {self.current_subject._get_parent_directory()}")
        self.current_subject.to_disk(self.current_subject._get_parent_directory())

    def add_image_data_from_disk(self):
        layout = [[sg.Text(text="Select from Disk")],
                  [sg.Text(text="Name: "), sg.Input(key='binary_name')],
                  [sg.Input(key='filepath'), sg.FileBrowse()],
                  [sg.Checkbox(text="Reduce to recommended size", key="reduce_size", default=True)],
                  [sg.Submit(), sg.Cancel()]]
        p = sg.Window('Select folder', layout)
        e, v = p.Read()
        p.close()
        if e == "Submit":
            im = imageio.imread(v['filepath'])

            # The annotator cannot deal with big images very well, so resize it if its too big
            if v['reduce_size']:
                dst_pixel = 600
                src_pixel = im.shape[0]
                if dst_pixel < src_pixel:  # Do not resize if the image is small anyways
                    resize_factor = dst_pixel / src_pixel
                    im = cv2.resize(im, None, fx=resize_factor, fy=resize_factor)

            self.current_subject.add_source_image_by_arr(im, binary_name=v['binary_name'], structures={"gt": "blob"})
            self.set_current_subject(self.current_subject)

    def add_dicom_image_data(self):
        dicom_path, ks = standalone_foldergrab(folder_not_file=False,
                                               optional_inputs=[("Binary Name", "binary_name")],
                                               optional_choices=[("Structure Template", "struct_tpl",
                                                                  self.d.get_structure_template_names())],
                                               title="Select DICOM file")
        if dicom_path:
            from trainer.lib.import_utils import append_dicom_to_subject
            tpl_name = ks['struct_tpl']
            if tpl_name in self.d.get_structure_template_names():
                seg_structs = self.d.get_structure_template_by_name(tpl_name)
                append_dicom_to_subject(self.current_subject.get_working_directory(), dicom_path,
                                        seg_structs=seg_structs)
                print(ks['binary_name'])
                print(dicom_path)
                self.set_current_subject(lib.Subject.from_disk(self.current_subject.get_working_directory()))

    def inspect_on_disk(self, open_file=False):
        if open_file:
            os.startfile(self.current_subject._get_json_path())
        else:
            os.startfile(self.current_subject.get_working_directory())

    def lst_src_binaries_changed(self, item, auto_save=True):
        if item is not None:
            src_item = item.text()
            print(f"Selected Source binary: {src_item}")
            self.select_source_binary(src_item)

    def lst_gt_binaries_changed(self, item, auto_save=True):
        if item is not None:
            print(f"Selected Ground Truth: {item.text()}")
            self.select_gt_binary(self.seg_structs[self.lst_gt_binaries.currentIndex().row()],
                                  for_name=self._selected_source_binary,
                                  frame_number=self.frame_number)

    def set_current_subject(self, s: lib.Subject):
        self.current_subject = s
        self.frame_number = 0
        self.mask_data = None

        # Handle GUI elements concerned with classes
        self.class_selector.set_subject(self.current_subject)

        # Load the list of source binaries into GUI
        self.lst_source_binaries.clear()
        self.lst_gt_binaries.clear()
        src_names = self.current_subject._get_binary_list_filtered(lambda x: x['binary_type'] == 'image_stack')
        if src_names:
            for b in src_names:
                self.lst_source_binaries.addItem(str(b))
            self.select_source_binary(src_names[0])

        self.update()
        self.console.push_to_ipython({"current_subject": s})

    def select_source_binary(self, name: str, auto_save=True):
        if auto_save and self.mask_data is not None and self._made_changes:
            self.save_to_disk()
        self.mask_data, self.mask_name, self.frame_number = None, name, 0

        self.img_data, self._selected_source_binary = self.current_subject._get_binary(name), name
        self.seg_tool.set_img_stack(self.img_data)
        # Load the possible structures into lst_gt
        self.lst_gt_binaries.clear()
        meta = self.current_subject._get_binary_model(name)
        self.seg_structs = list(meta["meta_data"]["structures"].keys())
        for s in self.seg_structs:
            self.lst_gt_binaries.addItem(f'{s}: {meta["meta_data"]["structures"][s]}')

        # Preselect a mask if there is only one structure anyway
        if self.current_subject._get_binary_list_filtered(lambda x: binary_filter(x, name, self.frame_number)):
            self.select_gt_binary(self.seg_structs[0], for_name=name)
        else:
            self._selected_gt_binary, self.mask_data = None, None

        # Inform the class selector about the new selected binary
        self.class_selector.set_binary_name(name)

        self.update()

    def select_gt_binary(self, structure_name, for_name: str, frame_number=0, auto_create=False, auto_save=True):
        """
        For every source binary the subject knows which structures can exist by template.
        This function looks for the ground truth and if there is none creates a new fitting one.
        :param structure_name: The selected structure name
        :param for_name: The source binary which the structure should correspond to
        :param frame_number: The frame number which is selected for annotation
        :param auto_create: Automatically write a new ground truth array on disk if there is none
        :param auto_save: Automatically save the previously selected gt and binary first
        :return:
        """

        # See if this structure already is described in one of the masks of this frame number
        gt_names = self.current_subject._get_binary_list_filtered(lambda x: binary_filter(x, for_name, frame_number))

        # Currently assume that only one masks exists for every frame, which describes all structs
        assert len(gt_names) <= 1

        def load_mask(gt_name: str) -> Tuple[str, str, int, np.ndarray]:
            """
            Loads the right mask out of the subject.
            :param gt_name: The name of the ground truth binary that contains masks of all structures.
            :return:
            """
            gt = self.current_subject._get_binary_model(gt_name)
            gt_arr = self.current_subject._get_binary(gt_name)
            i = gt['meta_data']['structures'].index(structure_name)
            return gt_name, structure_name, i, gt_arr

        if gt_names:
            self._selected_gt_binary, self._struct_name, self._struct_index, self.mask_data = load_mask(gt_names[0])
        elif auto_create:
            # Create a new ground truth for this frame
            src_b = self.current_subject._get_binary(for_name)
            new_gt_name = self.current_subject.add_new_gt_by_arr(
                np.zeros((src_b.shape[1], src_b.shape[2], len(self.seg_structs)), dtype=np.bool),
                structure_names=self.seg_structs,
                mask_of=for_name,
                frame_number=self.frame_number)
            self._selected_gt_binary, self._struct_name, self._struct_index, self.mask_data = load_mask(new_gt_name)
        else:
            self._selected_gt_binary, self.mask_data = None, None

        self.update()
        self._made_changes = False

    def update_log(self):
        # self.content_grid.logging_widget.clear()
        # self.content_grid.logging_widget.appendPlainText(f"{str(self.te)}")
        sg.popup(str(self.current_subject))

    def delete_selected_binaries(self):
        ns = [i.text() for i in self.lst_source_binaries.selectedItems()]
        for n in ns:
            print(f"Deleting binary {n}")
            self.current_subject._remove_binary(n)
        self.set_current_subject(self.current_subject)

    def delete_selected_mask(self):
        self.current_subject.delete_gt(mask_of=self._selected_source_binary, frame_number=self.frame_number)
        self._selected_gt_binary, self.mask_data = None, None

        self.update()
        self._made_changes = False

    def selection_event_handler(self, ps: List[Tuple[int, int]], add=True):
        if not self.mask_name:
            sg.popup("Please set a mask name")

        for p in ps:
            self.add_point(p, add=add)

        self.seg_tool.set_mask(self.mask_data, self.current_subject._get_binary_model(self._selected_source_binary))
        self.seg_tool.display_mask(self._struct_name)
        self._made_changes = True

    def change_frame(self, frame_number: int):
        if self.img_data.shape[0] - 1 >= frame_number >= 0:
            self.frame_number = frame_number

            self.select_gt_binary(self._struct_name,
                                  for_name=self._selected_source_binary,
                                  frame_number=frame_number)

            self.update()
        else:
            msg = f"Cannot set frame {self.frame_number} on multi frame image with {self.img_data.shape[0]} frames"
            print(msg)

    def create_new_mask(self):
        if self.mask_data is None:
            self.select_gt_binary(
                self.seg_structs[self.lst_gt_binaries.currentIndex().row()],
                for_name=self._selected_source_binary,
                frame_number=self.frame_number,
                auto_create=True)

    def add_point(self, pos, add=True):
        # If the ground truth does not yet exist, create a new
        self.create_new_mask()

        if self.brush == Brushes.Standard:
            intensity = add * 255
            tmp = cv2.circle(self.mask_data[:, :, self._struct_index].astype(np.uint8) * 255, pos,
                             self.seg_tool.pen_size, intensity, -1)
            self.mask_data[:, :, self._struct_index] = tmp.astype(np.bool)
        # elif self.m.brush == Brushes.AI_Merge:
        #     intensity = add * 255
        #     c = cv2.circle(np.zeros(self.m.foreground.shape, dtype=np.uint8), pos, self.m.brush_size, intensity, -1)
        #     tmp = (c & self.m.proposal) | self.m.foreground
        #     self.m.foreground = tmp.astype(np.bool)

    def update(self):
        self.seg_tool.display_img_stack(frame_number=self.frame_number)
        self.seg_tool.set_mask(self.mask_data,
                               self.current_subject._get_binary_model(self._selected_source_binary))
        self.seg_tool.display_mask(self._struct_name)
        self.frame_controller.set_frame_number(self.frame_number, self.img_data.shape[0])

    def find_annotations(self):
        struct_name = self.seg_structs[self._struct_index]
        annos, _ = self.current_subject.get_manual_struct_segmentations(struct_name)
        if self._selected_source_binary in annos:
            bs = annos[self._selected_source_binary]
            res = f"{struct_name} is segmented in {len(bs)} frames:\n"
            f_str = [f"{self.current_subject._get_binary_model(b_name)['meta_data']['frame_number']}\n" for b_name in
                     bs]
            for f in f_str:
                res += f
            sg.Popup(res)
        else:
            sg.Popup(f"No Annotations made for {struct_name}")


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) < 2:
        folder_path = standalone_foldergrab(folder_not_file=True,
                                            title="Please select the subject that you want to add")
        run_window(AnnotationGui, folder_path)
    else:
        run_window(AnnotationGui, sys.argv[1])
