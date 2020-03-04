"""
Purpose is to display an image with zooming capabilities.
Reports mouse_events using the event handler f
"""
from enum import Enum
from typing import Tuple, Callable, Dict, List

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from skimage.data import astronaut
from skimage.morphology import skeletonize

import trainer.lib as lib


def pos_from_event(e):
    return int(e.scenePos().x()), int(e.scenePos().y())


def arr_to_pixmap(arr: np.ndarray) -> QtGui.QPixmap:
    image = QtGui.QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1] * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap(image)


class Brushes(Enum):
    Standard = 1
    AI_Merge = 2


class IndicatorSceneItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent=None, pos: Tuple[int, int] = None, size=15):
        QtWidgets.QGraphicsPixmapItem.__init__(self, parent)

        # self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        # self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        #
        # # set move restriction rect for the item
        # self.move_restrict_rect = QtCore.QRectF(20, 20, 200, 200)
        # # set item's rectangle
        self.pos, self.size = pos, size
        self.pen = QtGui.QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
        self.setPen(self.pen)

    def set_position(self, pos: Tuple[int, int]):
        self.pos = pos

    def set_size(self, size: int):
        self.size = size

    def draw(self):
        if self.pos[0] >= 0 and self.pos[1] >= 0:
            self.setRect(self.pos[0] - self.size, self.pos[1] - self.size, self.size * 2, self.size * 2)


class TSegToolGraphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, f: Callable[[QtGui.QMouseEvent], None], parent=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)

        self.mouse_event_handler = f
        self.pixmap = QtWidgets.QGraphicsPixmapItem()
        self.addItem(self.pixmap)
        self.mask_pixmap = QtWidgets.QGraphicsPixmapItem()
        self.mask_pixmap.setOpacity(0.25)
        self.addItem(self.mask_pixmap)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.mouse_event_handler(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self.mouse_event_handler(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        self.mouse_event_handler(event)


class TSegToolGraphicsView(QtWidgets.QGraphicsView):

    def __init__(self,
                 pen_size_changed: Callable[[int], None],
                 parent=None):
        self.pen_size_changed = pen_size_changed
        QtWidgets.QGraphicsView.__init__(self, parent)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

    def set_scene(self, scene: TSegToolGraphicsScene):
        self.setScene(scene)

    def mousePressEvent(self, event):

        if event.button() == QtCore.Qt.MidButton:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(QtCore.Qt.ClosedHandCursor)
            # self.original_event = event
            handmade_event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPointF(event.pos()),
                                               QtCore.Qt.LeftButton, event.buttons(),
                                               QtCore.Qt.KeyboardModifiers())
            self.mousePressEvent(handmade_event)
            # Im Sure I have to do something here.

        super(TSegToolGraphicsView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MidButton:
            # for changing back to Qt.OpenHandCursor
            self.viewport().setCursor(QtCore.Qt.OpenHandCursor)
            handmade_event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(event.pos()),
                                               QtCore.Qt.LeftButton,
                                               event.buttons(), QtCore.Qt.KeyboardModifiers())
            self.mouseReleaseEvent(handmade_event)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        super(TSegToolGraphicsView, self).mouseReleaseEvent(event)

    def center_image(self, scale=True):
        rect = QtCore.QRectF(self.scene().pixmap.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self.scale(factor, factor)

    def wheelEvent(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            zoom_in_factor = 1.25
            zoom_out_factor = 1 / zoom_in_factor
            # Zoom
            if event.angleDelta().y() > 0:
                zoom_factor = zoom_in_factor
            else:
                zoom_factor = zoom_out_factor
            self.scale(zoom_factor, zoom_factor)
        elif event.modifiers() == QtCore.Qt.ShiftModifier:
            # Change Brush Size
            self.pen_size_changed(event.angleDelta().y() // 120)
        else:
            return QtWidgets.QGraphicsView.wheelEvent(self, event)


class IndicatorType(Enum):
    Circle = 'circle'


class SegToolController:

    def __init__(self,
                 selection_event_handler: Callable[[List[Tuple[int, int]], bool], None]):
        self._scene = TSegToolGraphicsScene(self.mouse_event_handler)
        self._graphics_view = TSegToolGraphicsView(self.change_pen_size)
        self._graphics_view.set_scene(self._scene)
        self.selection_event_handler = selection_event_handler

        self._img_stack: lib.ImStack = None
        self._mask: lib.SemSegMask = None
        self.pen_size = 15
        # self.set_img_stack(astronaut().reshape([1, astronaut().shape[0], astronaut().shape[1], 3]))
        # self.display_img_stack(frame_number=0)
        self.indicator = IndicatorSceneItem(size=self.pen_size)
        self._scene.addItem(self.indicator)

        self.mouse_down, self.mouse_down_left = False, False
        self.selected_points: List[Tuple[int, int]] = []

    def change_pen_size(self, change: int):
        self.pen_size = self.pen_size + change
        self.indicator.set_size(self.pen_size)
        self.display_indicator()

    def mouse_event_handler(self, e: QtGui.QMouseEvent):
        # Display indicator
        pos = pos_from_event(e)
        self.indicator.set_position(pos)
        self.display_indicator()

        # Handle buttons
        if e.buttons() == QtCore.Qt.LeftButton or e.buttons() == QtCore.Qt.RightButton:
            self.mouse_down_left = e.buttons() == QtCore.Qt.LeftButton
            self.mouse_down = True
            self.selected_points.append(pos)
        elif self.mouse_down:
            self.mouse_down = False
            # Mouse UP!
            self.selection_event_handler(self.selected_points, self.mouse_down_left)
            self.selected_points.clear()

    def get_graphics_scene(self):
        return self._graphics_view

    def set_img_stack(self, img_stack: lib.ImStack) -> None:
        self._img_stack = img_stack
        # self._graphics_view.center_image()

    def set_mask(self, mask: lib.SemSegMask):
        self._mask = mask
        # if mask is not None:
        #     self._mask_meta = struct_meta['meta_data']['structures']

    def display_indicator(self):
        self.indicator.draw()

    def display_img_stack(self, frame_number: int) -> None:
        if self._img_stack.get_ndarray().shape[3] == 1:
            # Assumption: Grayscale
            image_data = self._img_stack.get_ndarray()[frame_number, :, :, 0]
        else:
            # Assumption: RGB
            image_data = np.dot(self._img_stack.get_ndarray()[frame_number, :, :, :], [0.2989, 0.5870, 0.1140])

        blend_im = image_data.astype(np.uint8)
        res = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
        res[:, :, 0] = blend_im
        res[:, :, 1] = blend_im
        res[:, :, 2] = blend_im

        self._scene.pixmap.setPixmap(arr_to_pixmap(res))

    def display_mask(self, semsegclass: lib.SemSegClass):
        im_arr = self._img_stack.get_ndarray()
        res = np.zeros((im_arr.shape[1], im_arr.shape[2], 3), dtype=np.uint8)
        if self._mask is not None:
            struct_index = self._mask.tpl.ss_classes.index(semsegclass)
            struct_mask = self._mask.get_ndarray()[:, :, struct_index].astype(np.uint8)
            res[:, :, 0] = struct_mask * 255
            if semsegclass.ss_type == lib.MaskType.Line:
                res[:, :, 1] = skeletonize(struct_mask) * 255
            elif semsegclass.ss_type == lib.MaskType.Point:
                raise NotImplementedError()
            res[:, :, 2] = np.zeros((self._mask.get_ndarray().shape[0], self._mask.get_ndarray().shape[1]), dtype=np.uint8)
            for i in range(len(self._mask.tpl.ss_classes)):
                if i != struct_index:
                    res[:, :, 2] |= (self._mask.get_ndarray()[:, :, i].astype(np.uint8) * 255)
                # res[:, :, 2] = self._mask.get_ndarray()[:, :, i].astype(np.uint8) * 255
        self._scene.mask_pixmap.setPixmap(arr_to_pixmap(res))
