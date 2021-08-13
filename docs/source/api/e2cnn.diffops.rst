.. automodule:: e2cnn.diffops
   :no-members:

e2cnn.diffops
=============

This subpackage implements the complete analytical solutions of the equivariance constraints partial
differential operators (PDOs).
The bases of equivariant PDOs should be built through :ref:`factory functions <factory-functions-bases>`.

Typically, the user does not need to interact directly with this subpackage.
Instead, we suggest to use the interface provided in :doc:`e2cnn.gspaces`.

This subpackage depends only on :doc:`e2cnn.group`.


.. contents:: Contents
    :local:
    :backlinks: top


Abstract Classes
----------------

.. autoclass:: e2cnn.diffops.DiffopBasis
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

.. _factory-functions-bases:

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


   
   
