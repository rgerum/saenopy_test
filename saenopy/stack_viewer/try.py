import saenopy
from stack_preview import view_stack, StackPreview
from conv_functions import get_PSF_torch as get_PSF, wiener_deconvolution_torch, roll
from qtpy import QtWidgets, QtCore, QtGui
import sys
import time
import numpy as np
from saenopy.gui import QtShortCuts

def wiener_deconvolution_torch_wrapped(signal, kernel, lambd):
    import torch
    return wiener_deconvolution_torch(torch.tensor(signal), torch.tensor(kernel), lambd).numpy(force=True)
    return wiener_deconvolution_torch(torch.Tensor(signal), torch.Tensor(kernel.copy()), lambd).numpy(force=True)

class PSFChooser(QtWidgets.QWidget):
    psfChanged = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        with QtShortCuts.QVBoxLayout(self):
            with QtShortCuts.QHBoxLayout():
                self.input_w = QtShortCuts.QInputNumber(None, "w", 21)
                self.input_h = QtShortCuts.QInputNumber(None, "h", 21)
            with QtShortCuts.QHBoxLayout():
                self.input_dz = QtShortCuts.QInputNumber(None, "dz", 0.988, float=True, step=0.1)
                self.input_px = QtShortCuts.QInputNumber(None, "px", 0.7211, float=True, step=0.1)
            with QtShortCuts.QHBoxLayout():
                self.input_na = QtShortCuts.QInputNumber(None, "NA", 0.3, float=True, step=0.1)
                self.input_n = QtShortCuts.QInputNumber(None, "n", 1, float=True, step=0.1)

            self.input_w.valueChanged.connect(self.update)
            self.input_h.valueChanged.connect(self.update)
            self.input_dz.valueChanged.connect(self.update)
            self.input_px.valueChanged.connect(self.update)
            self.input_na.valueChanged.connect(self.update)
            self.input_n.valueChanged.connect(self.update)
        self.update()

    def update(self):
        self.psf = get_PSF(self.input_h.value(), self.input_w.value(), self.input_w.value(),
                           dz=self.input_dz.value(), Pixelsize=self.input_px.value(), NA=self.input_na.value(),
                           magnification=1,
                           n=self.input_n.value())
        self.psfChanged.emit(self.psf)


class Convolver(QtWidgets.QWidget):
    def __init__(self, im):
        super().__init__()
        self.im = im
        with QtShortCuts.QHBoxLayout(self):
            with QtShortCuts.QVBoxLayout():
                self.input_lambda = QtShortCuts.QInputNumber(None, "lambda", 1, min=0.01, max=1000, use_slider=True, log_slider=True)

                self.psf_func = PSFChooser().addToLayout()
                self.input_lambda.valueChanged.connect(self.psf_func.update)

                self.view = StackPreview([self.psf_func.psf.numpy(force=True)]).addToLayout()

                from saenopy.gui.gui_classes import CheckAbleGroup, MatplotlibWidget, NavigationToolbar
                self.canvas = MatplotlibWidget(self).addToLayout()
                self.canvas.setMinimumHeight(300)
                NavigationToolbar(self.canvas, self).addToLayout()
                #self.canvas2 = MatplotlibWidget(self).addToLayout()
            self.view2 = StackPreview([im, im]).addToLayout()
            self.psf_func.psfChanged.connect(self.psf_changed)
        self.psf_changed(self.psf_func.psf)

    def psf_changed(self, psf):
        self.view.views[0].set_image(psf.numpy(force=True))
        t = time.time()
        print("deconvolving")
        im2 = wiener_deconvolution_torch_wrapped(self.im, psf, lambd=self.input_lambda.value())
        im2 = roll(im2, np.array(psf.shape)//2)
        print(im.std(), im.mean(), im.std()/im.mean(), im.min(), im.max(), im.dtype)
        print(im2.std(), im2.mean(), im2.std()/im2.mean(), im2.min(), im2.max(), im2.dtype)
        print(f"time {time.time()-t:.2f}s")
        print(im2.shape, im.shape)
        self.view2.views[1].set_image(im2)
        self.canvas.figure.clf()
        import matplotlib.pyplot as plt

        plt.figure(self.canvas.figure)
        ax = plt.subplot(2, 1, 1)
        plt.plot(im.std(axis=(1, 2)))
        plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.plot(im2.std(axis=(1, 2)))


app = None
def view_stack2(im):
    global app
    app = QtWidgets.QApplication(sys.argv)

    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    print(sys.argv)
    window = Convolver(im)
    window.show()
    print("show")
    app.exec_()
    print("done")

result = saenopy.load("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM_x/example_output/Pos004_S001_z{z}_ch{c00}.npz")
im = result.stack[0][:, :, 0, :, 2].transpose(2, 0, 1)
import pandas as pd

M = result.mesh_piv[0]
length = np.linalg.norm(M.getNodeVar("U_measured"), axis=1)
angle = np.arctan(M.getNodeVar("U_measured")[:, 0], M.getNodeVar("U_measured")[:, 1])
data = pd.DataFrame(np.hstack((M.R, length[:, None], angle[:, None])), columns=["x", "y", "z", "length", "angle"])
data = data.sort_values(by="length")
sort_values
d2 = data.groupby(["x", "y"]).first()
d2 = d2[["vx", "vy", "vz"]]
np.array(d2[["vx", "vy", "vz"]])
np.array([i for i in d2.index])

# optional slice
d2.loc[(slice(None, None, 2), slice(None, None, 2)), :].shape
for i, v in d2.iterrows():
    print(i, v)
#view_stack2(im)

d2.loc[pd.IndexSlice[["x","y"], :, :2], :]


R = np.array([i for i in d2.index])
lengths = np.asarray(d2.length)
angles = d2.angle

max_length = np.max(lengths)


def getarrow(length, angle, scale=1, width=2, headlength=5, headheight=5, offset=None):
    length *= scale
    width *= scale
    headlength *= scale
    headheight *= scale
    print("scale", scale, length)
    headlength = headlength*np.ones(len(lengths))
    headheight = headheight*np.ones(len(lengths))
    index_small = length < headlength
    if np.any(index_small):
        headheight[index_small] = headheight * length[index_small] / headlength
        headlength[index_small] = length[index_small]

    # generate the arrow points
    arrow = [(0, width / 2), (length - headlength, width / 2), (length - headlength, headheight / 2), (length, 0),
            (length - headlength, -headheight / 2), (length - headlength, -width / 2), (0, -width / 2)]
    # and distribute them for each point
    arrows = np.zeros([length.shape[0], 7, 2])
    for p in range(7):
        for i in range(2):
            arrows[:, p, i] = arrow[p][i]

    # rotate the arrow
    angle = np.deg2rad(angle)
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    arrows = np.einsum("ijk,kli->ijl", arrows, rot).shape

    # add the offset
    arrows += offset[:, None, :]

    return arrows

    arrow = arrow @ rot


    arrows[:, 0, :] = (0, width/2)
    arrows[:, 1, 0] = length - headlength; arrows[:, 1, 1] = width / 2

    return [(0, width / 2), (length - headlength, width / 2), (length - headlength, headheight / 2), (length, 0),
            (length - headlength, -headheight / 2), (length - headlength, -width / 2), (0, -width / 2)]

def add_quiver(pil_image, R, lengths, angles, max_length, cmap, alpha=1, scale=1):
    cmap = plt.get_cmap(cmap)

    colors = cmap(lengths / max_length)
    colors[:, 3] = alpha
    colors = (colors*255).astype(np.uint8)

    arrows = getarrow(lengths, angles, scale=1, width=2, headlength=5, headheight=5, offset=None)

    image = ImageDraw.ImageDraw(pil_image, "RGBA")

    for a, c in zip(arrows, colors):
        image.polygon(a.ravel(), fill=c, outline=c)
    return pil_image