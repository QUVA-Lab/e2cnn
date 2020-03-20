
.. automodule:: e2cnn.group
   :no-members:

e2cnn.group
===========

This subpackage implements groups and group representations.

To avoid creating multiple redundant instances of the same group, we suggest using the factory functions in :ref:`Factory Functions <factory-functions>`.
These functions only build a single instance of each different group and return that instance on consecutive calls.

When building equivariant networks, it is not necessary to directly instantiate objects from this subpackage.
Instead, we suggest using the interface provided in :doc:`e2cnn.gspaces`.

This subpackage is not dependent on the others and can be used alone to generate groups and comput their representations.

.. contents:: Contents
    :local:
    :backlinks: top


Groups
------

Group
~~~~~
.. autoclass:: e2cnn.group.Group
    :members:
    :undoc-members:

O(2)
~~~~
.. autoclass:: e2cnn.group.O2
    :members:
    :show-inheritance:

SO(2)
~~~~~
.. autoclass:: e2cnn.group.SO2
    :members:
    :show-inheritance:

Cyclic Group
~~~~~~~~~~~~
.. autoclass:: e2cnn.group.CyclicGroup
    :members:
    :show-inheritance:

Dihedral Group
~~~~~~~~~~~~~~
.. autoclass:: e2cnn.group.DihedralGroup
    :members:
    :show-inheritance:


.. _factory-functions:

Factory Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: e2cnn.group.so2_group

.. autofunction:: e2cnn.group.o2_group

.. autofunction:: e2cnn.group.cyclic_group

.. autofunction:: e2cnn.group.dihedral_group

.. autofunction:: e2cnn.group.trivial_group


Representations
---------------

Representation
~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.group.Representation
    :members:
    :undoc-members:

Irreducible Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: e2cnn.group.IrreducibleRepresentation
    :members:
    :show-inheritance:

Utility Functions
-----------------

.. autofunction:: e2cnn.group.change_basis

.. autofunction:: e2cnn.group.directsum

.. autofunction:: e2cnn.group.disentangle

Subpackages
-----------

.. toctree::
   :maxdepth: 1
   
   e2cnn.group.utils
   
   
