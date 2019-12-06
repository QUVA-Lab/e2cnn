
:github_url: https://github.com/QUVA-Lab/e2cnn

e2cnn documentation
=================================

*e2cnn* is a Pytorch based library for equivariant deep learning.

*Equivariant neural networks* guarantee a prespecified transformation behavior of their features under transformations of their input.
This package provides functionality for the equivariant processing of planar images.
It implements the *most general convolutional maps* which are equivariant under the isometries of the plane, that is, under translations, rotations and reflections.


Package Reference
-----------------

The library is structured into four subpackages with different high-level features:

* :doc:`e2cnn.group <api/e2cnn.group>`         implements basic concepts of group and representation theory
    
* :doc:`e2cnn.kernels <api/e2cnn.kernels>`     solves for spaces of equivariant convolution kernels
    
* :doc:`e2cnn.gspaces <api/e2cnn.gspaces>`     defines the image plane and its symmetries
        
* :doc:`e2cnn.nn <api/e2cnn.nn>`               contains equivariant modules to build deep neural networks

Typically, only the high level functionalities provided in :doc:`e2cnn.gspaces <api/e2cnn.gspaces>` and 
:doc:`e2cnn.nn <api/e2cnn.nn>` are needed to build an equivariant model.


To get started, we provide an `introductory tutorial <https://github.com/QUVA-Lab/e2cnn/blob/master/examples/introduction.ipynb>`_
which introduces the basic functionality of the library.
A second `tutorial <https://github.com/QUVA-Lab/e2cnn/blob/master/examples/model.ipynb>`_ goes through building and training 
an equivariant model on the rotated MNIST dataset.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference
   :hidden:

   api/e2cnn.group
   api/e2cnn.kernels
   api/e2cnn.gspaces
   api/e2cnn.nn


Cite Us
-------

The development of this library was part of the work done in `our paper <https://arxiv.org/abs/1911.08251>`_ .
Please, cite us if you use this code in your own work::

    @inproceedings{e2cnn,
        title={{General E(2)-Equivariant Steerable CNNs}},
        author={Weiler, Maurice and Cesa, Gabriele},
        booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
        year={2019},
    }


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`

