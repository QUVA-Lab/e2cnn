
from .__about__ import *


__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]


try:
    from e2cnn import group
    __all__ += ['group']
except ImportError:
    pass

try:
    from e2cnn import kernels
    __all__ += ['kernels']
except ImportError:
    pass

try:
    from e2cnn import gspaces
    __all__ += ['gspaces']
except ImportError:
    pass

try:
    from e2cnn import nn
    __all__ += ['nn']
except ImportError:
    pass


