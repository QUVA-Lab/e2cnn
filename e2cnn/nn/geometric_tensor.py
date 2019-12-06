
import torch

from torch import Tensor

from .field_type import FieldType

from typing import List

__all__ = ["GeometricTensor", "tensor_directsum"]


class GeometricTensor:
    
    def __init__(self, tensor: Tensor, type: FieldType):
        r"""
        
        A GeometricTensor can be interpreted as a *typed* tensor.
        It is wrapping a common :class:`torch.Tensor` and endows it with a (compatible) :class:`~e2cnn.nn.FieldType` as
        transformation law.
        
        All neural network operations have :class:`~e2cnn.nn.GeometricTensor` s as inputs and outputs.
        They perform a dynamic typechecking, ensuring that the transformation law of the data and the operation match.
        See also :class:`~e2cnn.nn.EquivariantModule`.
 
        As usual, the first dimension of the tensor is interpreted as the batch dimension. The second is the fiber
        dimension (usually interpreted as the channels dimension). The following dimensions are the base space
        dimensions (eg. the spatial dimension in a conventional CNN).
        
        Args:
            tensor (torch.Tensor): the tensor data
            type (FieldType): the type of the tensor, modeling its transformation law
        
        Attributes:
            ~.tensor (torch.Tensor)
            ~.type (FieldType)
            
        """
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(type, FieldType)
        
        assert len(tensor.shape) > 2
        assert tensor.shape[1] == type.size, (tensor.shape, type.size)
        
        # torch.Tensor: PyTorch tensor containing the data
        self.tensor = tensor
        
        # FieldType: field type of the signal
        self.type = type
    
    def restrict(self, id) -> 'GeometricTensor':
        r"""
        
        Return a new :class:`~e2cnn.nn.GeometricTensor` whose field type is the restriction through the input
        ``id`` of the current tensor's one.
        
        Notice that the underlying :attr:`~e2cnn.nn.GeometricTensor.tensor` instance will be shared between
        the current tensor and the returned one.
        
        .. warning ::
        
            The method builds the new representation on the fly; hence, if this operation is needed at run time,
            we suggest to use :class:`e2cnn.nn.RestrictionModule` which pre-computes the new representation offline.
        
        .. seealso ::
        
            Check the documentation of the :meth:`~e2cnn.gspaces.GSpace.restrict` method in the
            :class:`~e2cnn.gspaces.GSpace` instance used for a description of the parameter ``id``.
        
        
        Args:
            id: the id identifying the subgroup to restrict to

        Returns:
            the geometric tensor with the restricted representation
            
        """
        new_class = self.type.restrict(id)
        return GeometricTensor(self.tensor, new_class)
    
    def split(self, breaks: List[int]):
        r"""
        
        Split this tensor on the channel dimension in a list of smaller tensors.
        The original tensor is split at the *fields* specified by the index list ``breaks``.
        
        If the tensor is associated with the list of fields :math:`\{\rho_i\}_i`
        (see :attr:`e2cnn.nn.FieldType.representations`), the :math:`i`-th output tensor will contain the fields
        :math:`\rho_{\text{breaks}[i-1]}, \dots, \rho_{\text{breaks}[i]-1}` of the original tensor.
        
        Args:
            breaks (list): indices of the fields where to split the tensor

        Returns:
            list of :class:`~e2cnn.nn.GeometricTensor` s into which the original tensor is chunked
            
        """
        
        breaks.append(len(self.type.representations))
        
        # final list of tensors
        tensors = []
        
        # list containing the index of the channels separating consecutive fields in this tensor
        positions = []
        last = 0
        for repr in self.type.representations:
            positions.append(last)
            last += repr.size
        positions.append(last)
        
        last_field = 0
        # for each break point
        for b in breaks:
            assert b > last_field, 'Error! "breaks" must be an increasing list of positive indexes'
            
            # compute the sub-class of the new sub-tensor
            repr = FieldType(self.type.gspace, self.type.representations[last_field: b])
            
            # retrieve the sub-tensor
            data = self.tensor[:, positions[last_field]:positions[b], ...]
            
            tensors.append(GeometricTensor(data, repr))
            
            last_field = b
        
        return tensors

    def transform(self, element) -> 'GeometricTensor':
        r"""
        Transform the current tensor according to the group representation associated to the input element
        and its induced action on the base space

        .. warning ::
            The input tensor is detached before the transformation therefore no gradient is backpropagated
            through this operation

            See :meth:`e2cnn.nn.GeometricTensor.transform_fibers` to transform only the fibers, i.e. not transform
            the base space.

        Args:
            element: an element of the group of symmetries of the fiber.

        Returns:
            the transformed tensor

        """
    
        transformed = self.type.transform(self.tensor, element)
        return GeometricTensor(transformed, self.type)

    def transform_fibers(self, element) -> 'GeometricTensor':
        r"""
        
        Transform the feature vectors of the underlying tensor according to the group representation associated to
        the input element.
        
        Interpreting the tensor as a vector-valued signal :math:`f: X \to \mathbb{R}^c` over a base space :math:`X`
        (where :math:`c` is the number of channels of the tensor), given the input ``element`` :math:`g \in G`
        (:math:`G` fiber group) the method returns the new signal :math:`f'`:
        
        .. math ::
            f'(x) := \rho(g) f(x)
        
        for :math:`x \in X` point in the base space and :math:`\rho` the representation of :math:`G` in the
        field type of this tensor.
        
        
        Notice that the input element has to be an element of the fiber group of this tensor's field type.
        
        .. seealso ::
        
            See :meth:`e2cnn.nn.GeometricTensor.transform` to transform the whole tensor.
        
        Args:
            element: an element of the group of symmetries of the fiber.

        Returns:
            the transformed tensor

        """
        rho = torch.FloatTensor(self.type.representation(element))
        data = torch.einsum("oi,bihw->bohw", (rho, self.tensor.contiguous())).contiguous()
        return GeometricTensor(data, self.type)
    
    @property
    def shape(self):
        r"""
        Alias for ``self.tensor.shape``

        """
        return self.tensor.shape

    def size(self):
        r"""
        Alias for ``self.tensor.size()``

        .. seealso ::
            :meth:`torch.Tensor.size`

        """
        return self.tensor.size()

    def to(self, *args, **kwargs):
        r"""
        
        Alias for ``self.tensor.to(*args, **kwargs)``.
        
        Applies :meth:`torch.Tensor.to` to the underlying tensor and wraps the resulting tensor in a new
        :class:`~e2cnn.nn.GeometricTensor` with the same type.

        """
        tensor = self.tensor.to(*args, **kwargs)
        return GeometricTensor(tensor, self.type)

    def __add__(self, other: 'GeometricTensor') -> 'GeometricTensor':
        r"""
        Add two compatible :class:`~e2cnn.nn.GeometricTensor` using pointwise addition.
        The two tensors needs to have the same shape and be associated to the same field type.

        Args:
            other (GeometricTensor): the other geometric tensor

        Returns:
            the sum

        """
        assert self.type == other.type
        return GeometricTensor(self.tensor + other.tensor, self.type)


def tensor_directsum(tensors: List['GeometricTensor']) -> 'GeometricTensor':
    r"""
    Concatenate a list of :class:`~e2cnn.nn.GeometricTensor` s on the channels dimension (``dim=1``).
    The input tensors have to be compatible: they need to have the same shape except for the channels
    dimension (``dim=1``).
    In the resulting :class:`~e2cnn.nn.GeometricTensor`, the channels dimension will be associated with the direct sum
    representation of the representations of the input tensors.
    
    .. seealso::
        
        :func:`e2cnn.group.directsum`
    
    Args:
        tensors (list): a list of :class:`~e2cnn.nn.GeometricTensor` s

    Returns:
        the direct sum of the inputs
        
    """
    # assert len(tensors) > 1
    
    for i in range(1, len(tensors)):
        assert tensors[0].type.gspace == tensors[i].type.gspace
        assert tensors[0].tensor.ndimension() == tensors[i].tensor.ndimension()
        assert tensors[0].tensor.shape[0] == tensors[i].tensor.shape[0]
        assert tensors[0].tensor.shape[2:] == tensors[i].tensor.shape[2:]
    
    # concatenate all representations from all field types
    reprs = []
    for t in tensors:
        reprs += t.type.representations
    
    # build the new field type
    cls = FieldType(tensors[0].type.gspace, reprs)
    
    # concatenate the underlying tensors
    data = torch.cat([t.tensor for t in tensors], dim=1)
    
    # build the new Geometric Tensor
    return GeometricTensor(data, cls)

