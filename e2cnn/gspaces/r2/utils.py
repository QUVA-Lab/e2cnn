
import numpy as np

from scipy.ndimage import rotate


def rotate_array(x, angle):
    k = 2 * angle / np.pi
    if k.is_integer():
        # Rotations by 180 and 270 degrees seem to be not perfect using `ndimage.rotate` and can therefore
        # make some tests fail.
        # For this reason, we use `np.rot90` to perform rotations by multiples of 90 degrees without interpolation
        return np.rot90(x, k, axes=(-2, -1))
    else:
        return rotate(x, angle * 180.0 / np.pi, (-2, -1), reshape=False, order=2)

