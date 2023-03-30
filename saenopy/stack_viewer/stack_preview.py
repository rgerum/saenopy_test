#import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
from qimage2ndarray import array2qimage
import QExtendedGraphicsView
import tifffile
import sys

app = QtWidgets.QApplication(sys.argv)
def view_stack(im):
    global app
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    window = StackPreview(im)
    window.show()
    app.exec_()


class StackPreview(QtWidgets.QWidget):

    def __init__(self, im):
        super().__init__()
        if not isinstance(im, list):
            im = [im]
        layout = QtWidgets.QHBoxLayout(self)
        self.views = []
        for index, i in enumerate(im):
            self.views.append(StackView(i))
            layout.addWidget(self.views[-1])
            self.views[-1].valueChanged.connect(lambda z, y, x, i=index: self.onValueChange(z, y, x, i))
            if index > 0:
                self.views[0].view_xy.link(self.views[-1].view_xy)
                self.views[0].view_xz.link(self.views[-1].view_xz)
                self.views[0].view_yz.link(self.views[-1].view_yz)

    def onValueChange(self, z, y, x, i):
        for index, view in enumerate(self.views):
            if index != i:
                view.update(z, y, x)

def connect_view_to_slider(view, slider, use_y, invert=False):
    mousePressEvent = view.mousePressEvent
    mouseMoveEvent = view.mouseMoveEvent
    mouseReleaseEvent = view.mouseReleaseEvent
    mouse_down = False

    def update(event):
        p = view.mapToScene(event.pos())
        if not use_y:
            value = int(p.x() / view.scaler.transform().m11() - view.translater.transform().dx())
        else:
            value = int(p.y() / view.scaler.transform().m11() - view.translater.transform().dy())
        if not invert:
            slider.setValue(value)
        else:
            slider.setValue(slider.maximum()-value)

    def mousePressEvent2(event):
        nonlocal mouse_down
        if event.button() == 1:
            update(event)
            mouse_down = True

        return mousePressEvent(event)

    def mouseMoveEvent2(event):
        if mouse_down:
            update(event)

        return mouseMoveEvent(event)

    def mouseReleaseEvent2(event):
        nonlocal mouse_down
        if event.button() == 1:
            mouse_down = False
        return mouseReleaseEvent(event)

    view.mousePressEvent = mousePressEvent2
    view.mouseMoveEvent = mouseMoveEvent2
    view.mouseReleaseEvent = mouseReleaseEvent2


class StackView(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int, int, int)

    def __init__(self, im):
        super().__init__()
        im = im.astype(np.float32)-im.min()
        im = im/im.max()*255
        self.im = im

        layout = QtWidgets.QGridLayout(self)

        self.slider_y = QtWidgets.QSlider()
        layout.addWidget(self.slider_y, 1, 0)
        self.slider_y.setRange(0, im.shape[2] - 1)

        self.view_xy = QExtendedGraphicsView.QExtendedGraphicsView()
        layout.addWidget(self.view_xy, 1, 1)
        # self.label2.setMinimumWidth(300)
        self.pixmap_xy = QtWidgets.QGraphicsPixmapItem(self.view_xy.origin)
        self.pen = QtGui.QPen(QtGui.QColor("red"))
        self.pen.setCosmetic(True)
        self.view_xy.line_x = QtWidgets.QGraphicsLineItem(self.view_xy.origin)
        self.view_xy.line_x.setPen(self.pen)
        self.view_xy.line_y = QtWidgets.QGraphicsLineItem(self.view_xy.origin)
        self.view_xy.line_y.setPen(self.pen)

        self.view_yz = QExtendedGraphicsView.QExtendedGraphicsView()
        layout.addWidget(self.view_yz, 1, 2)
        # self.label2.setMinimumWidth(300)
        self.pixmap_yz = QtWidgets.QGraphicsPixmapItem(self.view_yz.origin)
        self.view_yz.line_x = QtWidgets.QGraphicsLineItem(self.view_yz.origin)
        self.view_yz.line_x.setPen(self.pen)
        self.view_yz.line_y = QtWidgets.QGraphicsLineItem(self.view_yz.origin)
        self.view_yz.line_y.setPen(self.pen)

        self.view_xz = QExtendedGraphicsView.QExtendedGraphicsView()
        layout.addWidget(self.view_xz, 2, 1)
        # self.label2.setMinimumWidth(300)
        self.pixmap_xz = QtWidgets.QGraphicsPixmapItem(self.view_xz.origin)
        self.view_xz.line_x = QtWidgets.QGraphicsLineItem(self.view_xz.origin)
        self.view_xz.line_x.setPen(self.pen)
        self.view_xz.line_y = QtWidgets.QGraphicsLineItem(self.view_xz.origin)
        self.view_xz.line_y.setPen(self.pen)

        self.slider_z = QtWidgets.QSlider()
        layout.addWidget(self.slider_z, 1, 3)
        self.slider_z.setRange(0, im.shape[0]-1)

        self.slider_x = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        layout.addWidget(self.slider_x, 0, 1)
        self.slider_x.setRange(0, im.shape[2] - 1)

        self.slider_z.setValue(self.im.shape[0]//2)
        self.slider_y.setValue(self.im.shape[1]//2)
        self.slider_x.setValue(self.im.shape[2]//2)
        self.slider_x.valueChanged.connect(self.update_slider)
        self.slider_y.valueChanged.connect(self.update_slider)
        self.slider_z.valueChanged.connect(self.update_slider)


        spLeft = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        spLeft.setHorizontalStretch(self.im.shape[2])
        spLeft.setVerticalStretch(self.im.shape[1])
        self.view_xy.setSizePolicy(spLeft)

        spLeft = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        spLeft.setHorizontalStretch(self.im.shape[0])
        spLeft.setVerticalStretch(self.im.shape[1])
        self.view_yz.setSizePolicy(spLeft)

        spLeft = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        spLeft.setHorizontalStretch(self.im.shape[0])
        spLeft.setVerticalStretch(self.im.shape[2])
        self.view_xz.setSizePolicy(spLeft)

        self.valueChanged.connect(self.update)
        self.update_slider()
        self.view_xy.fitInView()
        self.view_xz.fitInView()
        self.view_yz.fitInView()

        self.view_xy.link2(self.view_yz, False, True)
        self.view_xy.link2(self.view_xz, True, False)

        connect_view_to_slider(self.view_xy, self.slider_x, False)
        connect_view_to_slider(self.view_xy, self.slider_y, True, True)

        connect_view_to_slider(self.view_yz, self.slider_z, False)
        connect_view_to_slider(self.view_yz, self.slider_y, True, True)

        connect_view_to_slider(self.view_xz, self.slider_x, False)
        connect_view_to_slider(self.view_xz, self.slider_z, True)


    def set_image(self, im):
        im = im.astype(np.float32) - im.min()
        im = im / im.max() * 255
        self.im = im

        self.slider_z.setValue(self.im.shape[0] // 2)
        self.slider_y.setValue(self.im.shape[1] // 2)
        self.slider_x.setValue(self.im.shape[2] // 2)

        for index, slider in enumerate([self.slider_z, self.slider_y, self.slider_x]):
            if slider.maximum() != im.shape[index] - 1:
                slider.setMaximum(im.shape[index]-1)
                slider.setValue(im.shape[index] // 2)

        self.update_slider()

    def update_slider(self):
        self.valueChanged.emit(self.slider_x.value(), self.slider_y.value(), self.slider_z.value())

    def update(self, x, y, z):
        if x != self.slider_x.value():
            self.slider_x.setValue(x)
        if y != self.slider_y.value():
            self.slider_y.setValue(y)
        if z != self.slider_z.value():
            self.slider_z.setValue(z)
        x, y, z = self.slider_x.value(), self.im.shape[1]-1-self.slider_y.value(), self.slider_z.value()
        im = self.im[z, :, :]#.copy()
        #im[y, :] = 0
        #im[:, x] = 0
        self.pixmap_xy.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view_xy.setExtend(im.shape[1], im.shape[0])
        self.view_xy.line_x.setLine(x+0.5, 0, x+0.5, im.shape[0])
        self.view_xy.line_y.setLine(0, y+0.5, im.shape[1], y+0.5)

        im = self.im[:, :, x]#.copy()
        #im[z, :] = 0
        #im[:, y] = 0
        im = im.T
        self.pixmap_yz.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view_yz.setExtend(im.shape[1], im.shape[0])
        self.view_yz.line_x.setLine(z+0.5, 0, z+0.5, im.shape[0])
        self.view_yz.line_y.setLine(0, y+0.5, im.shape[1], y+0.5)

        im = self.im[:, y]#.copy()
        #im[z, :] = 0
        #im[:, x] = 0
        self.pixmap_xz.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view_xz.setExtend(im.shape[1], im.shape[0])
        self.view_xz.line_x.setLine(x+0.5, 0, x+0.5, im.shape[0])
        self.view_xz.line_y.setLine(0, z+0.5, im.shape[1], z+0.5)




def load_stack(filename):
    file = tifffile.TiffFile(filename)
    im = []
    for page in file.pages:
        im.append(page.asarray())
    im = np.asarray(im).astype(np.float32)
    return im


if __name__ == "__main__":
    view_stack([
        load_stack("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Pos004_S001_z{z}_ch02.tif"),
        load_stack("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Pos004_S001_z{z}_ch02.tif"),
    ])
