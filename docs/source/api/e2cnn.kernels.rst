.. automodule:: e2cnn.kernels
   :no-members:

e2cnn.kernels
=============

This subpackage implements the complete analytical solutions of the equivariance constraints on the kernel space as
explained in `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_.
The bases of equivariant kernels should be built through :ref:`factory functions <factory-functions-bases>`.

Typically, the user does not need to interact directly with this subpackage.
Instead, we suggest to use the interface provided in :doc:`e2cnn.gspaces`.

This subpackage depends only on :doc:`e2cnn.group`.


.. contents:: Contents
    :local:
    :backlinks: top


Abstract Classes
----------------

.. autoclass:: e2cnn.kernels.Basis
    :members:
    :undoc-members:

.. autoclass:: e2cnn.kernels.KernelBasis
    :members:
    :undoc-members:
    
.. autoclass:: e2cnn.kernels.EmptyBasisException
    :members:
    :undoc-members:


Kernel Bases
------------

Tensor Product of radial and angular bases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.kernels.PolarBasis
    :members:
    :undoc-members:
    :show-inheritance:


Radial Profile
~~~~~~~~~~~~~~

.. autoclass:: e2cnn.kernels.GaussianRadialProfile
    :members:
    :undoc-members:
    :show-inheritance:

General Steerable Basis for equivariant kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.kernels.SteerableKernelBasis
    :members:
    :undoc-members:
    :show-inheritance:

.. _factory-functions-bases:

Bases for Group Actions on the Plane
------------------------------------

The following factory functions provide an interface to build the bases for kernels equivariant to groups acting on the
two dimensional plane.
The names of the functions follow this convention `kernels_[G]_act_R[d]`, where :math:`G` is the origin-preserving isometry
group while :math:`\R^d` is the space on which it acts, interpreted as the domain of the
kernel :math:`\kappa: \R^d \to \R^{c_\text{out} \times c_\text{in}}`.
Here, :math:`d=2` as we only consider the planar setting.
In the language of `Gauge Equivariant CNNs <https://arxiv.org/abs/1902.04615>`_ , the origin-preserving isometry
:math:`G` is called *structure group* (or, sometimes, *gauge group*).

Reflections and Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: e2cnn.kernels.kernels_O2_act_R2

Continuous Rotations
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: e2cnn.kernels.kernels_SO2_act_R2

Reflections and Discrete Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: e2cnn.kernels.kernels_DN_act_R2

Discrete Rotations
~~~~~~~~~~~~~~~~~~
.. autofunction:: e2cnn.kernels.kernels_CN_act_R2

Reflections
~~~~~~~~~~~
.. autofunction:: e2cnn.kernels.kernels_Flip_act_R2

Trivial Action
~~~~~~~~~~~~~~
.. autofunction:: e2cnn.kernels.kernels_Trivial_act_R2




   
   
