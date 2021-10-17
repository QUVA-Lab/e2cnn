.. automodule:: e2cnn.diffops
   :no-members:

e2cnn.diffops
=============

This subpackage implements the complete analytical solutions of the equivariance constraints partial
differential operators (PDOs) as described in `Steerable Partial Differentail Operators for Equivariant Neural Network <https://arxiv.org/abs/2106.10163>`_.
The bases of equivariant PDOs should be built through :ref:`factory functions <factory-functions-diffops>`.

Typically, the user does not need to interact directly with this subpackage.
Instead, we suggest to use the interface provided in :doc:`e2cnn.gspaces`.

This subpackage depends only on :doc:`e2cnn.group` and :doc:`e2cnn.kernels`.

Note that discretization of the differential operators relies on the `sympy <https://docs.sympy.org/>`_ and `rbf <https://rbf.readthedocs.io>`_ packages.
If these packages are not installed, an error will be thrown when trying to sample the operators.



.. contents:: Contents
    :local:
    :backlinks: top


Abstract Class
--------------

.. autoclass:: e2cnn.diffops.DiffopBasis
    :members:
    :undoc-members:
    
.. autoclass:: e2cnn.diffops.DiscretizationArgs
    :members:
    :undoc-members:


PDO Bases
------------

Tensor Product of Laplacian and angular PDOs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.diffops.TensorBasis
    :members:
    :undoc-members:
    :show-inheritance:


Laplacian Profile
~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.diffops.LaplaceProfile
    :members:
    :undoc-members:
    :show-inheritance:

General Steerable Basis for equivariant PDOs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.diffops.SteerableDiffopBasis
    :members:
    :undoc-members:
    :show-inheritance:

.. _factory-functions-diffops:

Bases for Group Actions on the Plane
------------------------------------

The following factory functions provide an interface to build the bases for PDOs equivariant to groups acting on the
two dimensional plane.
The names of the functions follow this convention `diffops_[G]_act_R[d]`, where :math:`G` is the origin-preserving isometry
group while :math:`\R^d` is the space on which it acts.
Here, :math:`d=2` as we only consider the planar setting.
In the language of `Gauge Equivariant CNNs <https://arxiv.org/abs/1902.04615>`_ , the origin-preserving isometry
:math:`G` is called *structure group* (or, sometimes, *gauge group*).

Reflections and Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: e2cnn.diffops.diffops_O2_act_R2

Continuous Rotations
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: e2cnn.diffops.diffops_SO2_act_R2

Reflections and Discrete Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: e2cnn.diffops.diffops_DN_act_R2

Discrete Rotations
~~~~~~~~~~~~~~~~~~
.. autofunction:: e2cnn.diffops.diffops_CN_act_R2

Reflections
~~~~~~~~~~~
.. autofunction:: e2cnn.diffops.diffops_Flip_act_R2

Trivial Action
~~~~~~~~~~~~~~
.. autofunction:: e2cnn.diffops.diffops_Trivial_act_R2


Utility functions
-----------------
.. autofunction:: e2cnn.diffops.store_cache
.. autofunction:: e2cnn.diffops.load_cache


   
   
