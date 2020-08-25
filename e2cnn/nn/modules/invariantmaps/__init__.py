
from .gpool import GroupPooling, MaxPoolChannels

from .norm import NormPool
from .induced_norm import InducedNormPool

__all__ = [
    "GroupPooling",
    "NormPool",
    "InducedNormPool",
    "MaxPoolChannels"
]
