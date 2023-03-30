#!/usr/bin/env python

# Simple example of Wiener deconvolution in Python.
# We use a fixed SNR across all frequencies in this example.
#
# Written 2015 by Dan Stowell. Public domain.

import numpy as np
from numpy.fft import fft, ifft, ifftshift
from scipy.signal import convolve2d, fftconvolve
import matplotlib
# matplotlib.use('PDF') # http://www.astrobetter.com/plotting-to-a-file-in-python/
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import tifffile

plt.rcParams.update({'font.size': 6})

##########################
# user config
sonlen = 128
irlen = 64

lambd_est = 1e-3  # estimated noise lev


##########################

def gen_son(length):
    "Generate a synthetic un-reverberated 'sound event' template"
    # (whitenoise -> integrate -> envelope -> normalise)
    son = np.cumsum(np.random.randn(length))
    # apply envelope
    attacklen = length // 8
    env = np.hstack((np.linspace(0.1, 1, attacklen), np.linspace(1, 0.1, length - attacklen)))
    son *= env
    son /= np.sqrt(np.sum(son * son))
    return son


def gen_ir(length):
    "Generate a synthetic impulse response"
    # First we generate a quietish tail
    length = int(length)
    son = np.random.randn(length)
    attacklen = length // 2
    env = np.hstack((np.linspace(0.1, 1, attacklen), np.linspace(1, 0.1, length - attacklen)))
    son *= env
    son *= 0.05
    # Here we add the "direct" signal
    son[0] = 1
    # Now some early reflection spikes
    for _ in range(10):
        son[int(length * (np.random.rand() ** 2))] += np.random.randn() * 0.5
    # Normalise and return
    son /= np.sqrt(np.sum(son * son))
    return son


def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR"
    padding = [signal.shape[i]-kernel.shape[i] for i in range(len(signal.shape))]
    #kernel2 = np.zeros_like(signal)
    #kernel2[tuple(slice(i) for i in kernel.shape)] = kernel
    #kernel = np.pad(kernel, [(p//2, int(np.ceil(p/2))) for p in padding])
    kernel = np.pad(kernel, [(0, p) for p in padding])
    #kernel = roll(kernel, [p//2 for p in kernel.shape])
    H = np.fft.fftn(kernel)
    # kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel))))  # zero pad the kernel to same length
    # H = fft(kernel)
    deconvolved = np.real(np.fft.ifftn(np.fft.fftn(signal) * np.conj(H) / (H * np.conj(H) + lambd ** 2)))
    return deconvolved

def wiener_deconvolution_torch_wrapped(signal, kernel, lambd):
    import torch
    return wiener_deconvolution_torch(torch.Tensor(signal), torch.Tensor(kernel.copy()), lambd).numpy(force=True)

def wiener_deconvolution_torch(signal, kernel, lambd):
    import torch
    "lambd is the SNR"
    if 0:
        p1, p2, p3 = (signal.shape[-1] - kernel.shape[-1]) // 2, (signal.shape[-2] - kernel.shape[-2]) // 2, (
                    signal.shape[-3] - kernel.shape[-3]) // 2
        kernel = torch.nn.functional.pad(kernel, (0, p1, 0, p2, 0, p3))
    else:
        p1, p2, p3 = (signal.shape[-1] - kernel.shape[-1]), (signal.shape[-2] - kernel.shape[-2]), (
                signal.shape[-3] - kernel.shape[-3])
        kernel = torch.nn.functional.pad(kernel, (0, p1, 0, p2, 0, p3))
    H = torch.fft.fftn(kernel)
    #kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel))))  # zero pad the kernel to same length
    #H = fft(kernel)
    deconvolved = torch.real(torch.fft.ifftn(torch.fft.fftn(signal) * torch.conj(H) / (H * torch.conj(H) + lambd ** 2)))
    return deconvolved



def cut_center(im, size):
    if isinstance(size, int):
        size = [size] * len(im.shape)
    return im[tuple(slice(im.shape[i] // 2 - size[i], im.shape[i] // 2 + 1 + size[i]) for i in range(len(im.shape)))]


def roll(im, offset=None):
    if offset is None:
        offset = np.array(im.shape) // 2
    if isinstance(offset, int):
        offset = [offset]*len(im.shape)
    for i in range(len(im.shape)):
        im = np.roll(im, offset[i], i)
    return im


def show3d(im):
    plt.subplot(221)
    plt.imshow(im[:, im.shape[1] // 2, :])
    plt.axhline(im.shape[0] // 2)
    plt.axvline(im.shape[2] // 2)
    plt.subplot(222)
    plt.imshow(im[:, :, im.shape[2] // 2])
    plt.subplot(223)
    plt.imshow(im[im.shape[0] // 2])
    plt.show()


def plot_stacks(im1, im2, im3, im4=None):
    vmin1 = np.percentile(im2, 99)
    vmin2 = vmin1
    vmin3 = np.percentile(im3, 99)
    vmin4 = np.percentile(im4, 99)
    for z in range(im1.shape[0]):
        print(z)
        plt.subplot(221)
        plt.imshow(im1[z], vmin=-vmin1, vmax=vmin1)
        plt.subplot(222)
        plt.imshow(im2[z], vmin=-vmin2, vmax=vmin2)
        plt.subplot(223)
        plt.imshow(im3[z], vmin=-vmin3, vmax=vmin3)
        if im4 is not None:
            plt.subplot(224)
            plt.imshow(im4[z], vmin=-vmin4, vmax=vmin4)
        plt.savefig(f"slice{z}.png", dpi=300)
        plt.clf()


def save_stack(im, filename, percentiles=(0, 100)):
    vmin, vmax = np.percentile(im, percentiles)
    out = im.copy()
    out = out - vmin
    out = out / vmax
    out = np.clip(out, vmin, vmax)
    for z in range(im.shape[0]):
        print(z, filename.format(z=z))
        plt.imsave(filename.format(z=z), out[z])


# compute theoretical PSF
def get_PSF(pz=5, py=5, px=5, dz=5, Pixelsize=3.45, NA=0.3, magnification=10, n=1.0, wl=0.5, dr=0.01):
    from scipy.special import j0
    # pz, py, px: size of PSF
    # dz: y-distance between subsequent images (µm)
    # n: refractive index of objective immersion medium
    # dr: integration step size
    um_per_pix = Pixelsize / magnification
    # wave vector with light wavelength lambda in um
    k = 2 * np.pi / wl

    # generate ranges of x,y,z
    z = np.arange(-pz//2, pz//2)[:, None, None, None] * dz
    y = np.arange(-py//2, py//2)[None, :, None, None] * um_per_pix
    x = np.arange(-px//2, px//2)[None, None, :, None] * um_per_pix
    r = np.arange(0, 1, dr)[None, None, None, :]

    # getting picture where the object is in focus
    Bessel = j0(k * NA * r * np.sqrt(x**2 + y**2))
    sin = np.sin(k * n * z * ((1 - (NA * r / n)**2)**0.5 - 1))
    PSF = np.sum(Bessel * sin * dr, axis=-1)

    # normalization
    PSF = PSF/np.max(PSF)
    PSF = PSF#*np.abs(PSF)*np.abs(PSF)
    return PSF


def get_PSF_torch(pz=5, py=5, px=5, dz=5, Pixelsize=3.45, NA=0.3, magnification=10, n=1.0, wl=0.5, dr=0.01):
    import torch
    # pz, py, px: size of PSF
    # dz: y-distance between subsequent images (µm)
    # n: refractive index of objective immersion medium
    # dr: integration step size
    um_per_pix = Pixelsize / magnification
    # wave vector with light wavelength lambda in um
    k = 2 * np.pi / wl

    z = torch.arange(pz)[:, None, None, None]
    y = torch.arange(py)[None, :, None, None]
    x = torch.arange(px)[None, None, :, None]
    r = torch.arange(0, 1, dr)[None, None, None, :]
    # getting picture where the object is in focus
    Bessel = torch.special.bessel_j0(k * NA * r * torch.sqrt(((x - px//2)*um_per_pix)**2 + ((y - py//2)*um_per_pix)**2))
    sin = torch.sin(k * n * (z - pz//2) * dz * ((1 - (NA * r / n)**2)**0.5 - 1))
    PSF = torch.sum(Bessel * sin * dr, axis=-1)

    # normalization
    PSF = PSF/torch.max(PSF)
    PSF = PSF#*torch.abs(PSF)*torch.abs(PSF)
    return PSF


def get_PSF_torch2(pz=5, py=5, px=5, dz=5, Pixelsize=3.45, NA=0.3, magnification=10, n=1.0, wl=0.5, dr=0.01):
    import torch
    # pz, py, px: size of PSF
    # dz: y-distance between subsequent images (µm)
    # n: refractive index of objective immersion medium
    # dr: integration step size
    um_per_pix = Pixelsize / magnification
    # wave vector with light wavelength lambda in um
    k = 2 * np.pi / wl

    z = torch.arange(pz)[:, None, None, None]
    y = torch.arange(py)[None, :, None, None]
    x = torch.arange(px)[None, None, :, None]
    r = torch.arange(0, 1, dr)[None, None, None, :]
    # getting picture where the object is in focus
    Bessel = torch.special.bessel_j0(k * NA * r * um_per_pix * torch.sqrt((x - px//2)**2 + (y - py//2)**2))
    sin = torch.sin(k * n * (z - pz//2) * dz * ((1 - (NA * r / n)**2)**0.5 - 1))
    PSF = torch.sum(Bessel * sin * dr, axis=-1)

    # normalization
    PSF = PSF/torch.max(PSF)
    PSF = PSF#*torch.abs(PSF)*torch.abs(PSF)
    return PSF


if __name__ == '__main__':
    import sys
    """ some magic to prevent PyQt5 from swallowing exceptions """
    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook
    # Set the exception hook to our wrapping function
    sys.excepthook = lambda *args: sys._excepthook(*args)

    from stack_preview import view_stack
    psf = get_PSF(21, 21, 21, 5, 3.45)
    #show3d(psf)
    #view_stack(psf)
    print("read image")

    im = tifffile.imread("2023_02_03_18_21_01.tif")

    im = im[216-50:216+50, 300-50:300+50]

    #plt.imshow(im)
    #plt.show()
    res = np.zeros([21] + list(im.shape))
    res[res.shape[0]//2] = im

    stack = fftconvolve(res, psf, mode="full")#[10:-10, 10:-10, 10:-10]
    print(res.shape, stack.shape)
    #view_stack(stack)

    im2 = wiener_deconvolution(stack, psf, lambd=0.01)
    print("view stack", res.shape, stack.shape, im2.shape)
    view_stack([res, stack[10:-10, 10:-10, 10:-10], roll(im2, 10)[10:-10, 10:-10, 10:-10]])
