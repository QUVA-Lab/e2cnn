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


.. contents:: Contents
    :local:
    :backlinks: top


Main Classes
------------

Field Type
~~~~~~~~~~

.. autoclass:: e2cnn.nn.FieldType
    :members:
    :undoc-members:

Geometric Tensor
~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.nn.GeometricTensor
    :members:
    :undoc-members:

Equivariant Module
~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.nn.EquivariantModule
    :members:


Utils
-----

direct sum
~~~~~~~~~~

.. autofunction:: e2cnn.nn.tensor_directsum


Planar Convolution
------------------

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

Basis Expansion methods
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.nn.modules.r2_conv.BasisExpansion
    :members:
    :show-inheritance:

.. autoclass:: e2cnn.nn.modules.r2_conv.BlocksBasisExpansion
    :members:
    :show-inheritance:


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



