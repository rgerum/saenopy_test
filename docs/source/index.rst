.. saenopy documentation master file

Welcome to the SAENOPY Documentation
====================================

.. figure:: images/Logo.png


SAENOPY is a free open source 3D traction force microscopy software. Its material model is especially well suited for
tissue-mimicking and typically highly non-linear biopolymer matrices such as collagen, fibrin, or Matrigel.

It features a python package to use in scripts and an extensive graphical user interface.

This migration immune cell demonstrated what scientific discoveries you can achieve with saenop:

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="//www.youtube.com/embed/g7vlqs_hT4s" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>
    <br/>

.. toctree::
   :caption: Contents
   :maxdepth: 2

   interface
   boundarycondition.ipynb
   regularization.ipynb
   mesh
   material.ipynb
   auto_examples/index.rst
   api

Installation
------------

Standalone
~~~~~~~~~~
You can download saenopy as a standalone application for windows or for linux:

Windows
https://github.com/rgerum/saenopy/releases/download/v0.8.0/saenopy_windows.exe

Linux
https://github.com/rgerum/saenopy/releases/download/v0.8.0/saenopy_linux

Using Python
~~~~~~~~~~~~

If you are experienced with python or even want to use our Python API, you need to install saenopy as a python package.
Saenopy can be installed directly using pip:

    ``pip install saenopy``

Now you can start the user interface with:

    ``saenopy``


Citing Saenopy
--------------

If you use Saenopy for academic research, you are highly encouraged (though not
required) to cite our preprint:

* *Dynamic traction force measurements of migrating immune cells in 3D matrices*
  David Böhringer, Mar Cóndor, Lars Bischof, Tina Czerwinski, Andreas Bauer, Caroline Voskens, Silvia Budday,
  Christoph Mark, Ben Fabry, Richard Gerum
  **bioRxiv 2022.11.16.516758**; doi: https://doi.org/10.1101/2022.11.16.516758

You can also refer to the previous publications on the predecessor of saenopy:

* Steinwachs J., Metzner C., Skodzek K., Lang N., Thievessen I., Mark C., Munster S., Aifantis K. E., Fabry B. (2016)
  `"Three-dimensional force microscopy of cells in biopolymer networks" <http://dx.doi.org/10.1038/nmeth.3685>`_.
  In Nat Methods, volume 13, 2016. doi.org/10.1038/nmeth.3685

* Condor M., Steinwachs J., Mark C., Garcia-Aznar J. M., Fabry B. (2017)
  `"Traction Force Microscopy in 3-Dimensional Extracellular Matrix Networks" <http://dx.doi.org/10.1002/cpcb.24>`_
  In Curr Protoc Cell Biol, volume 75, doi.org/10.1002/cpcb.24

