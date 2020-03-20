
.. automodule:: e2cnn.gspaces
   :no-members:

e2cnn.gspaces
====================================

This subpackage implements G-spaces as a user interface for defining spaces and their symmetries.
The user typically instantiates a subclass of GSpace to specify the symmetries considered and uses
it to instanciate equivariant neural network modules (see :doc:`e2cnn.nn`).


This subpackage depends on :doc:`e2cnn.group` and :doc:`e2cnn.kernels`.



.. contents:: Contents
    :local:
    :backlinks: top


Abstract Group Space
--------------------

.. autoclass:: e2cnn.gspaces.GSpace
    :members:
    :undoc-members:


Group Actions on the Plane
--------------------------

.. autoclass:: e2cnn.gspaces.GeneralOnR2
    :members:
    :undoc-members:
    :show-inheritance:

Reflections and Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.gspaces.FlipRot2dOnR2
    :members:
    :show-inheritance:

Rotations
~~~~~~~~~

.. autoclass:: e2cnn.gspaces.Rot2dOnR2
    :members:
    :show-inheritance:

Reflections
~~~~~~~~~~~

.. autoclass:: e2cnn.gspaces.Flip2dOnR2
    :members:
    :show-inheritance:

Trivial Action
~~~~~~~~~~~~~~

.. autoclass:: e2cnn.gspaces.TrivialOnR2
    :members:
    :show-inheritance:


