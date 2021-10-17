.. automodule::  e2cnn.nn
   :no-members:

e2cnn.nn
========

This subpackage provides implementations of equivariant neural network modules.

In an equivariant network, features are associated with a transformation law under actions of a symmetry group.
The transformation law of a feature field is implemented by its :class:`~e2cnn.nn.FieldType` which can be interpreted as a data type.
A :class:`~e2cnn.nn.GeometricTensor` is wrapping a :class:`torch.Tensor` to endow it with a :class:`~e2cnn.nn.FieldType`.
Geometric tensors are processed by :class:`~e2cnn.nn.EquivariantModule` s which are :class:`torch.nn.Module` s that guarantee the
specified behavior of their output fields given a transformation of their input fields.


This subpackage depends on :doc:`e2cnn.group` and :doc:`e2cnn.gspaces`.


To enable efficient deployment of equivariant networks, many :class:`~e2cnn.nn.EquivariantModule` s implement a
:meth:`~e2cnn.nn.EquivariantModule.export` method which converts a *trained* equivariant module into a pure PyTorch
module, with few or no dependencies with **e2cnn**.
Not all modules support this feature yet, so read each module's documentation to check whether it implements this method
or not.
We provide a simple example::

    # build a simple equivariant model using a SequentialModule

    s = e2cnn.gspaces.Rot2dOnR2(8)
    c_in = e2cnn.nn.FieldType(s, [s.trivial_repr]*3)
    c_hid = e2cnn.nn.FieldType(s, [s.regular_repr]*3)
    c_out = e2cnn.nn.FieldType(s, [s.regular_repr]*1)

    net = SequentialModule(
        R2Conv(c_in, c_hid, 5, bias=False),
        InnerBatchNorm(c_hid),
        ReLU(c_hid, inplace=True),
        PointwiseMaxPool(c_hid, kernel_size=3, stride=2, padding=1),
        R2Conv(c_hid, c_out, 3, bias=False),
        InnerBatchNorm(c_out),
        ELU(c_out, inplace=True),
        GroupPooling(c_out)
    )

    # train the model
    # ...

    # export the model

    net.eval()
    net_exported = net.export()

    print(net)
    > SequentialModule(
    >   (0): R2Conv([8-Rotations: {irrep_0, irrep_0, irrep_0}], [8-Rotations: {regular, regular, regular}], kernel_size=5, stride=1, bias=False)
    >   (1): InnerBatchNorm([8-Rotations: {regular, regular, regular}], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (2): ReLU(inplace=True, type=[8-Rotations: {regular, regular, regular}])
    >   (3): PointwiseMaxPool()
    >   (4): R2Conv([8-Rotations: {regular, regular, regular}], [8-Rotations: {regular}], kernel_size=3, stride=1, bias=False)
    >   (5): InnerBatchNorm([8-Rotations: {regular}], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (6): ELU(alpha=1.0, inplace=True, type=[8-Rotations: {regular}])
    >   (7): GroupPooling([8-Rotations: {regular}])
    > )

    print(net_exported)
    > Sequential(
    >   (0): Conv2d(3, 24, kernel_size=(5, 5), stride=(1, 1), bias=False)
    >   (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (2): ReLU(inplace=True)
    >   (3): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
    >   (4): Conv2d(24, 8, kernel_size=(3, 3), stride=(1, 1), bias=False)
    >   (5): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (6): ELU(alpha=1.0, inplace=True)
    >   (7): MaxPoolChannels(kernel_size=8)
    > )


    # check that the two models are equivalent

    x = torch.randn(10, c_in.size, 31, 31)
    x = GeometricTensor(x, c_in)

    y1 = net(x).tensor
    y2 = net_exported(x.tensor)

    assert torch.allclose(y1, y2)

|


.. contents:: Contents
    :local:
    :backlinks: top
    :depth: 3



Field Type
----------

.. autoclass:: e2cnn.nn.FieldType
    :members:
    :undoc-members:

Geometric Tensor
----------------

.. autoclass:: e2cnn.nn.GeometricTensor
    :members:
    :undoc-members:
    :exclude-members: __add__,__sub__,__iadd__,__isub__,__mul__,__repr__,__rmul__,__imul__


Equivariant Module
------------------

.. autoclass:: e2cnn.nn.EquivariantModule
    :members:


Utils
-----

direct sum
~~~~~~~~~~

.. autofunction:: e2cnn.nn.tensor_directsum


Planar Convolution and Differential Operators
---------------------------------------------

R2Conv
~~~~~~
.. autoclass:: e2cnn.nn.R2Conv
    :members:
    :show-inheritance:

R2ConvTransposed
~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.R2ConvTransposed
    :members:
    :show-inheritance:

R2Diffop
~~~~~~~~
.. autoclass:: e2cnn.nn.R2Diffop
    :members:
    :show-inheritance:

Basis Expansion Modules
~~~~~~~~~~~~~~~~~~~~~~~

Basis Expansion
"""""""""""""""
.. autoclass:: e2cnn.nn.modules.r2_conv.BasisExpansion
    :members:
    :show-inheritance:

BlocksBasisExpansion
""""""""""""""""""""
.. autoclass:: e2cnn.nn.modules.r2_conv.BlocksBasisExpansion
    :members:
    :show-inheritance:

SingleBlockBasisExpansion
"""""""""""""""""""""""""
.. autoclass:: e2cnn.nn.modules.r2_conv.SingleBlockBasisExpansion
    :members:
    :show-inheritance:

block_basisexpansion
""""""""""""""""""""
.. autofunction:: e2cnn.nn.modules.r2_conv.block_basisexpansion


Non Linearities
---------------

ConcatenatedNonLinearity
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.ConcatenatedNonLinearity
    :members:
    :show-inheritance:

ELU
~~~
.. autoclass:: e2cnn.nn.ELU
    :members:
    :show-inheritance:

GatedNonLinearity1
~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.GatedNonLinearity1
   :members:
   :show-inheritance:

GatedNonLinearity2
~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.GatedNonLinearity2
   :members:
   :show-inheritance:

InducedGatedNonLinearity1
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.InducedGatedNonLinearity1
   :members:
   :show-inheritance:

InducedNormNonLinearity
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.InducedNormNonLinearity
   :members:
   :show-inheritance:

NormNonLinearity
~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.NormNonLinearity
   :members:
   :show-inheritance:

PointwiseNonLinearity
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseNonLinearity
   :members:
   :show-inheritance:

ReLU
~~~~
.. autoclass:: e2cnn.nn.ReLU
   :members:
   :show-inheritance:

VectorFieldNonLinearity
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.VectorFieldNonLinearity
   :members:
   :show-inheritance:


Invariant Maps
--------------

GroupPooling
~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.GroupPooling
   :members:
   :show-inheritance:

NormPool
~~~~~~~~
.. autoclass:: e2cnn.nn.NormPool
   :members:
   :show-inheritance:


InducedNormPool
~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.InducedNormPool
   :members:
   :show-inheritance:


Pooling
-------

NormMaxPool
~~~~~~~~~~~
.. autoclass:: e2cnn.nn.NormMaxPool
   :members:
   :show-inheritance:

PointwiseMaxPool
~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseMaxPool
   :members:
   :show-inheritance:

PointwiseMaxPoolAntialiased
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseMaxPoolAntialiased
   :members:
   :show-inheritance:

PointwiseAvgPool
~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseAvgPool
   :members:
   :show-inheritance:

PointwiseAvgPoolAntialiased
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseAvgPoolAntialiased
   :members:
   :show-inheritance:

PointwiseAdaptiveAvgPool
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseAdaptiveAvgPool
   :members:
   :show-inheritance:

PointwiseAdaptiveMaxPool
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseAdaptiveMaxPool
   :members:
   :show-inheritance:


Normalization
-------------------

InnerBatchNorm
~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.InnerBatchNorm
   :members:
   :show-inheritance:

NormBatchNorm
~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.NormBatchNorm
   :members:
   :show-inheritance:

InducedNormBatchNorm
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.InducedNormBatchNorm
   :members:
   :show-inheritance:

GNormBatchNorm
~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.GNormBatchNorm
   :members:
   :show-inheritance:


Dropout
-------

FieldDropout
~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.FieldDropout
   :members:
   :show-inheritance:

PointwiseDropout
~~~~~~~~~~~~~~~~
.. autoclass:: e2cnn.nn.PointwiseDropout
   :members:
   :show-inheritance:


Other Modules
-------------

Sequential
~~~~~~~~~~

.. autoclass:: e2cnn.nn.SequentialModule
    :members:
    :show-inheritance:

ModuleList
~~~~~~~~~~

.. autoclass:: e2cnn.nn.ModuleList
    :members:
    :show-inheritance:

Restriction
~~~~~~~~~~~

.. autoclass:: e2cnn.nn.RestrictionModule
    :members:
    :show-inheritance:

Disentangle
~~~~~~~~~~~

.. autoclass:: e2cnn.nn.DisentangleModule
    :members:
    :show-inheritance:

Upsampling
~~~~~~~~~~

.. autoclass:: e2cnn.nn.R2Upsampling
    :members:
    :show-inheritance:

Multiple
~~~~~~~~

.. autoclass:: e2cnn.nn.MultipleModule
    :members:
    :show-inheritance:
    
Reshuffle
~~~~~~~~~

.. autoclass:: e2cnn.nn.ReshuffleModule
    :members:
    :show-inheritance:

Mask
~~~~

.. autoclass:: e2cnn.nn.MaskModule
    :members:
    :show-inheritance:

Identity
~~~~~~~~

.. autoclass:: e2cnn.nn.IdentityModule
    :members:
    :show-inheritance:


Weight Initialization
---------------------

.. automodule:: e2cnn.nn.init
   :members:


.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   e2cnn.nn.others




